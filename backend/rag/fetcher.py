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


# --- Custom (non-Wikipedia) sources ----------------------------------------
# Users can ground on their own link instead of Wikipedia. We fetch the page,
# strip boilerplate, and return the same {title, url, text} shape as a Wikipedia
# source, so the rest of the pipeline (chunk -> embed -> retrieve -> guard) is
# completely unchanged.

_URL_FETCH_TIMEOUT = float(os.getenv("URL_FETCH_TIMEOUT", "10"))
_URL_MAX_BYTES = 2_000_000  # cap the download so a huge page can't exhaust memory


def _is_safe_public_url(url: str) -> bool:
    """SSRF guard: allow only unambiguous http(s) URLs whose host resolves to
    public (globally routable) IP addresses.

    A user-supplied link is untrusted input. Without this check it could reach
    internal services (databases, cloud metadata at 169.254.169.254, localhost
    admin panels). We first reject syntax that URL parsers and HTTP clients
    disagree on (backslashes, embedded credentials), then resolve the host and
    reject any address that is not globally routable.

    Note: this is a point-in-time check. A hostile DNS server could still rebind
    to an internal address between this check and the actual connection
    (TOCTOU / DNS rebinding); :func:`fetch_url_source` therefore re-validates
    every redirect hop. Fully closing rebinding would require pinning the
    resolved IP at connect time, which is out of scope for this app.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    try:
        # Browsers treat "\" as "/", but urlparse/urllib do not. That
        # disagreement enables bypasses like "http://127.0.0.1:80\@evil.com/"
        # (guard sees host evil.com; the client connects to 127.0.0.1). Reject.
        if "\\" in url:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        # Embedded credentials ("http://8.8.8.8@169.254.169.254/") confuse host
        # extraction; a legitimate article URL never carries userinfo.
        if parsed.username or parsed.password:
            return False
        host = parsed.hostname
        if not host:
            return False
        _ = parsed.port  # raises ValueError on a bad port -> rejected below
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            # is_global is False for private/loopback/link-local/reserved/CGNAT/
            # unspecified ranges; the explicit flags belt-and-suspender any gaps.
            if (
                not ip.is_global
                or ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False
        return True
    except Exception:
        return False


def fetch_url_source(url: str, max_chars: int = _PRIMARY_CHARS) -> Optional[Dict]:
    """Fetch an arbitrary web page as a grounding source ``{title, url, text}``.

    Returns ``None`` when the URL is unsafe/unreachable or yields no readable
    text. The extracted text is capped at ``max_chars`` to match the Wikipedia
    path. Used when the user supplies their own source link instead of Wikipedia.
    """
    url = (url or "").strip()
    if not _is_safe_public_url(url):
        return None

    import time
    from urllib.parse import urljoin

    try:
        import requests

        # A single wall-clock budget across every redirect hop and the body read.
        # requests' own timeout is per-read (inactivity) only, so a server that
        # drips one byte at a time (slowloris) could otherwise tie up the worker
        # indefinitely.
        deadline = time.monotonic() + _URL_FETCH_TIMEOUT
        current = url
        resp = None
        for _ in range(5):  # follow a bounded number of redirects manually
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            resp = requests.get(
                current,
                headers={"User-Agent": _USER_AGENT},
                timeout=remaining,
                stream=True,
                allow_redirects=False,  # never auto-follow: re-validate each hop
            )
            if not resp.is_redirect:
                break
            location = resp.headers.get("Location", "")
            resp.close()
            if not location:
                return None
            current = urljoin(current, location)
            # A public URL must not redirect to an internal address: the classic
            # SSRF-via-redirect bypass, since the guard only saw the original URL.
            if not _is_safe_public_url(current):
                return None
        else:
            return None  # too many redirects

        resp.raise_for_status()
        # Bounded read (both size AND wall-clock) so a huge or slow-drip response
        # can't exhaust memory or tie up the worker.
        chunks: List[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=16384):
            if time.monotonic() > deadline:
                break
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= _URL_MAX_BYTES:
                break
        html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
    except Exception:
        return None

    return _extract_readable(html, url, max_chars)


def _extract_readable(html: str, url: str, max_chars: int) -> Optional[Dict]:
    """Strip boilerplate from HTML and return ``{title, url, text}`` or ``None``."""
    try:
        from bs4 import BeautifulSoup
    except Exception:  # pragma: no cover - bs4 ships with the wikipedia dependency
        return None

    soup = BeautifulSoup(html, "html.parser")
    # Remove non-content nodes so we ground on prose, not nav/scripts/styling.
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "svg",
        ]
    ):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url

    # Prefer the main article region when the page marks one up.
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = " ".join(main.get_text(separator=" ", strip=True).split())[:max_chars]
    if not text.strip():
        return None
    return {"title": title[:200], "url": url, "text": text}


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


def fetch_qa_sources(question: str, max_articles: int = 3) -> List[Dict]:
    """Fetch the top few candidate articles for a *question* as grounding sources.

    Unlike :func:`fetch_sources` (which grounds facts on a single coherent
    article about a topic), a question's answer often lives in a *different*
    article than the one its subject names — e.g. "who helped Ghana gain
    independence?" is answered in the **Kwame Nkrumah** article, not **Ghana**.
    So for QA we retrieve across the top ``max_articles`` search hits and let the
    sentence re-ranker pick the best-supported answer from whichever article
    actually contains it.

    Returns ``[{title, url, text}]`` (deduped by title); empty on total failure,
    which makes the caller abstain rather than guess.
    """
    try:
        titles = _retry(wikipedia.search, question, results=max_articles)
    except Exception:
        titles = []

    sources: List[Dict] = []
    seen: set[str] = set()
    for title in titles[:max_articles]:
        page = _fetch_page(title, max_chars=_PRIMARY_CHARS)
        if page and page["title"] not in seen:
            seen.add(page["title"])
            sources.append(page)
    return sources


def fetch_topic_sources(
    topic: str, max_articles: int = 3, title: Optional[str] = None
) -> List[Dict]:
    """Fetch several related articles about ``topic`` as grounding sources.

    A digest curates on the subscriber's behalf, so rather than a single page it
    gathers the primary article plus a few closely related ones — giving facts
    that span multiple sources instead of one. The primary article is always
    first so the lead/definition still anchors the result. Returns
    ``[{title, url, text}]`` deduped by title; an empty list makes the caller
    behave as if there were no facts (rather than guessing).
    """
    # The primary, coherent article (honors an explicit disambiguation choice).
    sources = fetch_sources(topic, title=title)
    seen = {s["title"] for s in sources}
    if len(sources) >= max_articles:
        return sources

    # Broaden with the next-best search hits so the digest isn't confined to one
    # page. Deduped against the primary article and against each other.
    try:
        titles = _retry(wikipedia.search, topic, results=max_articles + 3)
    except Exception:
        titles = []
    for cand in titles:
        if len(sources) >= max_articles:
            break
        # Skip parenthetical-disambiguated titles ("Volcano (1997 film)"): these
        # are namesakes from a different sense, not the subject the subscriber
        # follows. Keeping only clean titles keeps the digest on one coherent
        # sense while still spanning several related articles.
        if "(" in cand:
            continue
        page = _fetch_page(cand, max_chars=_PRIMARY_CHARS)
        if page and page["title"] not in seen:
            seen.add(page["title"])
            sources.append(page)
    return sources
