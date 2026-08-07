"""Wikipedia source fetching for the ingestion pipeline.

Two concerns live here:

* :func:`fetch_candidates` powers the UI's "Did you mean?" disambiguation. For
  broad/ambiguous topics the app asks the user which sense they mean instead of
  guessing (e.g. *Mercury* the planet vs the element vs the Roman god).
* :func:`fetch_sources` builds the corpus the retriever grounds against. It
  locks onto a **single, coherent article** — the one Wikipedia itself resolves
  the query to, following redirects (e.g. *AI* -> *Artificial intelligence*) and
  honoring the primary-topic article. We deliberately do *not* stitch together
  multiple loosely name-matched pages, nor hand-pick among search titles (which
  can prefer a minor namesake — grounding *AI* on the chimpanzee *Ai*, or
  *Volcano* on the novel *Under the Volcano*). When a bare title is genuinely
  ambiguous the chosen sense comes from the user via disambiguation, not from a
  guess.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import wikipedia

# Wikipedia's API rejects/limits requests that don't send a descriptive
# User-Agent, replying with non-JSON error pages that surface as
# JSONDecodeErrors (the intermittent "empty facts" failures). Set one up front
# so lookups are reliable. Override via the WIKI_USER_AGENT env var.
_USER_AGENT = os.getenv(
    "WIKI_USER_AGENT",
    "GroundedFacts/1.0 (RAG portfolio demo; https://github.com/grounded-facts)",
)
wikipedia.set_user_agent(_USER_AGENT)

# Be a polite API citizen: space out bursts (a candidate lookup fires several
# requests) so Wikipedia doesn't rate-limit us. Guarded for older versions.
try:
    wikipedia.set_rate_limiting(True)
except Exception:  # pragma: no cover - defensive, older wikipedia builds
    pass

# Character budget for the grounding article. A single article chunks into a
# rich corpus on its own, so we give it plenty of room.
_PRIMARY_CHARS = 9000

# On-disk cache of fetched articles. Wikipedia access here is intermittently
# flaky, so once we successfully pull an article we keep a copy: later requests
# are instant and, crucially, survive a transient upstream failure (the pipeline
# becomes genuinely offline-capable after the first fetch). Disable by setting
# WIKI_CACHE_DIR="".
_CACHE_DIR_ENV = os.getenv("WIKI_CACHE_DIR")
if _CACHE_DIR_ENV == "":
    _CACHE_DIR: Optional[Path] = None  # explicitly disabled
elif _CACHE_DIR_ENV is not None:
    _CACHE_DIR = Path(_CACHE_DIR_ENV)
else:
    _CACHE_DIR = Path(__file__).resolve().parents[2] / ".wiki_cache"


def _cache_path(title: str) -> Optional[Path]:
    if _CACHE_DIR is None:
        return None
    digest = hashlib.md5(title.lower().strip().encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{digest}.json"


def _cache_read(title: str) -> Optional[Dict]:
    path = _cache_path(title)
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_write(data: Dict, key: Optional[str] = None) -> None:
    # Cache under the lookup key (the requested title) so a later request for the
    # same query hits the cache even when the page resolved via a redirect to a
    # different canonical title.
    path = _cache_path(key or data["title"])
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass

# Wikipedia articles end with boilerplate sections ("See also", "References",
# ...) that are navigation/citation noise rather than prose. We cut the article
# at the first such heading so those lines never become "facts". This is
# field-agnostic: every article has these sections.
_TERMINAL_SECTIONS = re.compile(
    r"\n=+\s*(See also|References|External links|Further reading|Notes|"
    r"Citations|Bibliography|Sources|Footnotes|Works cited|Explanatory notes)"
    r"\s*=+",
    re.IGNORECASE,
)

# Wikipedia's public API occasionally returns a non-JSON / rate-limited response.
# These errors are deterministic (not transient), so retrying them is pointless.
_NO_RETRY = (
    wikipedia.exceptions.DisambiguationError,
    wikipedia.exceptions.PageError,
)


def _retry(fn: Callable, *args, attempts: int = 3, delay: float = 0.4, **kwargs):
    """Call ``fn`` with a few retries to ride out transient Wikipedia hiccups.

    The ``wikipedia`` package does no retrying itself, so a single flaky response
    (e.g. an HTML error page that fails JSON decoding) would otherwise surface as
    an empty result. Deterministic errors in ``_NO_RETRY`` are re-raised at once.
    """
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except _NO_RETRY:
            raise
        except Exception as exc:  # noqa: BLE001 - transient network/parse errors
            last = exc
            time.sleep(delay * (attempt + 1))
    raise last  # type: ignore[misc]


def fetch_summary(topic: str, sentences: int = 5) -> str:
    """Fetch a short summary for a topic using the wikipedia package."""
    try:
        return _retry(wikipedia.summary, topic, sentences=sentences)
    except Exception:
        try:
            results = _retry(wikipedia.search, topic)
            if not results:
                return ""
            page = _retry(wikipedia.page, results[0], auto_suggest=False)
            return page.summary[:2000]
        except Exception:
            return ""


def _page_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"


def fetch_candidates(topic: str, limit: int = 6) -> List[Dict]:
    """Return disambiguation candidates as ``[{"title", "description"}]``.

    Used by the frontend to offer follow-up choices on broad topics. Each
    description is the first sentence of that page's summary (best-effort).
    """
    try:
        titles = _retry(wikipedia.search, topic, results=limit)
    except Exception:
        titles = []

    candidates: List[Dict] = []
    for title in titles:
        try:
            description = _retry(
                wikipedia.summary, title, sentences=1, auto_suggest=False
            )
        except Exception:
            description = ""
        candidates.append({"title": title, "description": description})
    return candidates


def _fetch_page(title: str, max_chars: int) -> Optional[Dict]:
    """Fetch a single page as ``{"title", "url", "text"}`` or ``None``.

    The ``wikipedia`` library follows redirects, so a redirect title like *AI*
    resolves to its target (*Artificial intelligence*) and ``data["title"]`` is
    the canonical article name. A bare title that is itself a **disambiguation
    page** returns ``None`` — we never silently pick an arbitrary sense; the
    caller decides what to do (ask the user, or try the ranked search results).

    ``page.content`` / ``page.url`` are *lazy* — accessing them fires additional
    network requests — so the whole load is wrapped in :func:`_retry` to survive
    a transient failure on any of those calls.
    """

    def _load(page_title: str) -> Dict:
        page = wikipedia.page(page_title, auto_suggest=False)
        # Cut boilerplate ("See also"/"References"/...) before applying the char
        # budget so those trailing sections never make it into the corpus.
        content = _TERMINAL_SECTIONS.split(page.content or "", maxsplit=1)[0]
        text = content[:max_chars]
        return {"title": page.title, "url": page.url, "text": text}

    try:
        data: Optional[Dict] = _retry(_load, title)
    except wikipedia.exceptions.DisambiguationError:
        # Ambiguous bare title — refuse to guess a sense.
        return None
    except Exception:
        data = None

    if data and data["text"].strip():
        _cache_write(data, key=title)  # keep a copy for next time / offline use
        return data

    # Upstream fetch failed — fall back to a previously cached copy if we have one.
    return _cache_read(title)


def is_ambiguous(topic: str) -> bool:
    """Best-effort check of whether ``topic`` has several distinct senses.

    The UI uses this to decide whether to *proactively* ask a follow-up question
    (rather than silently committing to one article). A topic is treated as
    ambiguous when Wikipedia has no single primary-topic article for it (i.e. the
    bare title resolves to a disambiguation page). Redirects with a clear primary
    topic (e.g. *AI* -> *Artificial intelligence*) are **not** ambiguous.
    """
    try:
        _retry(wikipedia.page, topic, auto_suggest=False)
        return False
    except wikipedia.exceptions.DisambiguationError:
        return True
    except Exception:
        return False


def fetch_sources(
    topic: str,
    max_pages: int = 1,
    max_chars: int = _PRIMARY_CHARS,
    title: Optional[str] = None,
) -> List[Dict]:
    """Fetch the grounding source(s) for ``topic`` as ``[{title, url, text}]``.

    Grounds on a **single coherent article** so facts can't bleed across senses
    or fields. ``max_pages``/``max_chars`` are kept for API compatibility but the
    default path returns exactly one article.

    Parameters
    ----------
    title:
        When set (the user picked a specific option from the "Did you mean?"
        list), fetch precisely that page — even a parenthetical one.
    """
    # Explicit disambiguation choice -> fetch exactly what was requested.
    if title:
        page = _fetch_page(title, max_chars=_PRIMARY_CHARS)
        return [page] if page else []

    # Primary path: let Wikipedia resolve the query the way it does for readers —
    # following redirects (*AI* -> *Artificial intelligence*) and honoring the
    # primary-topic article. This is more reliable than matching search titles
    # ourselves, which can prefer a minor namesake (e.g. the chimpanzee *Ai*).
    page = _fetch_page(topic, max_chars=_PRIMARY_CHARS)
    if page:
        return [page]

    # Fallback: the bare title was missing or ambiguous. Walk the ranked search
    # results and ground on the first concrete (non-disambiguation) article.
    try:
        results = _retry(wikipedia.search, topic, results=6)
    except Exception:
        results = []
    topic_l = topic.lower().strip()
    for candidate in results:
        if candidate.lower().strip() == topic_l:
            continue  # the bare title was already tried (and disambiguated/failed)
        page = _fetch_page(candidate, max_chars=_PRIMARY_CHARS)
        if page:
            return [page]

    # Last-resort fallback so callers always get something usable.
    summary = fetch_summary(topic, sentences=6)
    if summary.strip():
        return [{"title": topic, "url": _page_url(topic), "text": summary}]
    return []
