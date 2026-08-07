"""Grounded fact curation with a hallucination guard.

Given a topic, this module:

1. Retrieves the most relevant chunks from the topic's vector index.
2. Generates candidate facts, either with an LLM constrained to the retrieved
   context (when ``OPENAI_API_KEY`` is set) or with a fully offline extractive
   fallback that selects the most on-topic source sentences.
3. Runs a **hallucination guard**: every candidate fact is embedded and compared
   against the retrieved chunks; facts whose maximum similarity to the supporting
   evidence falls below a threshold are dropped, and the best-matching chunk is
   attached as a citation with a grounding score.

The guard is what turns "the model said so" into "the model said so *and* it is
supported by a retrieved source", which is the core reliability story of the
project.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np

from backend.rag import fetcher, ingest
from backend.rag.chunking import split_sentences
from backend.rag.embeddings import Embedder, get_default_embedder

DEFAULT_MAX_FACTS = 3
DEFAULT_K = 6
# For question answering we ground across the top few candidate articles, not
# just one: a question's answer entity often lives in a *different* article than
# the one its subject names (e.g. "who helped Ghana gain independence?" is
# answered in "Kwame Nkrumah", not "Ghana").
QA_MAX_ARTICLES = 4
# A digest curates on the subscriber's behalf across several related articles
# (multiple sources) rather than one page, so a topic's facts draw on more than a
# single source. Interactive "Explore" stays single-article for tight coherence.
DIGEST_MAX_SOURCES = 3
# Negation cues used to demote answers like "does not produce oxygen" when the
# question itself is affirmative (extractive QA can't tell polarity otherwise).
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|don't|doesn't|didn't|isn't|"
    r"aren't|wasn't|weren't|nor|neither|none)\b"
)
# Minimum cosine similarity between a fact and its best supporting chunk.
# The right bar depends on the embedding backend: the offline bag-of-words
# hashing embedder produces low similarities (genuine matches ~0.2-0.4), while
# neural encoders (sbert/openai) spread scores higher (genuine matches ~0.4-0.7,
# unrelated ~0.1). We therefore pick the default from the active EMBED_BACKEND
# and let GROUNDING_THRESHOLD override it explicitly.
DEFAULT_GROUNDING_THRESHOLD = 0.20
NEURAL_GROUNDING_THRESHOLD = 0.35

# Question answering compares a whole question against individual sentences,
# which share fewer tokens than a sentence does with its broad topic, so genuine
# answers score a little lower than facts. QA therefore uses a slightly lower bar
# than fact extraction on each backend; override with QA_GROUNDING_THRESHOLD.
DEFAULT_QA_THRESHOLD = 0.15
NEURAL_QA_THRESHOLD = 0.30

_HEADER_RE = re.compile(r"={2,}[^=\n]*={2,}")
# Inline reference/footnote markers like "[1]" or "[citation needed]".
_REF_RE = re.compile(r"\[[^\]]*\]")

# A fact that opens with one of these words usually refers back to a previous
# sentence ("He also stated…", "This means…") and reads as a non-sequitur on its
# own, so we drop it from the extractive candidates.
_DANGLING_START_RE = re.compile(
    r"^(he|she|it|they|them|his|her|its|their|this|that|these|those|there|then|"
    r"thus|hence|therefore|however|but|and|so|also|moreover|furthermore|"
    r"additionally|meanwhile|instead|nonetheless|nevertheless|yet|still|such|"
    r"which|who|whom|whose)\b",
    re.IGNORECASE,
)
# Opinion / review / attribution language — not the kind of standalone, factual
# statement we want to surface.
_OPINION_RE = re.compile(
    r"\b(said|says|stated|wrote|writes|reviewer|reviewed|a review|praised|"
    r"criticized|criticised|described it|called it|according to|rated|felt that|"
    r"argued|opined|remarked|commented|in a review)\b",
    re.IGNORECASE,
)


def _is_neural_backend() -> bool:
    """True when the active embedding backend is a neural encoder.

    Reads ``EMBED_BACKEND`` directly for an explicit choice; for ``auto`` (the
    default) it consults the process-wide embedder that was actually selected,
    since ``auto`` resolves to ``sbert`` only when it is installed and loads.
    Neural cosine similarities sit noticeably higher than the hashing embedder's,
    so the grounding bars default higher to keep the hallucination guard strict.
    """
    backend = os.getenv("EMBED_BACKEND", "auto").lower()
    if backend == "auto":
        backend = get_default_embedder().backend
    return backend in {"sbert", "openai"}


def _grounding_threshold() -> float:
    """Minimum grounding score, from GROUNDING_THRESHOLD env or a backend default."""
    default = (
        NEURAL_GROUNDING_THRESHOLD
        if _is_neural_backend()
        else DEFAULT_GROUNDING_THRESHOLD
    )
    raw = os.getenv("GROUNDING_THRESHOLD")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _qa_threshold() -> float:
    """Minimum grounding score for question answering.

    Lower than :func:`_grounding_threshold` because QA matches a full question
    against individual sentences (sparser overlap) rather than a sentence against
    a broad topic. Defaults scale with the embedding backend; override with the
    QA_GROUNDING_THRESHOLD env var.
    """
    default = NEURAL_QA_THRESHOLD if _is_neural_backend() else DEFAULT_QA_THRESHOLD
    raw = os.getenv("QA_GROUNDING_THRESHOLD")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _llm_enabled() -> bool:
    """True when LLM generation is available (API key set) and not disabled."""
    if os.getenv("USE_LLM", "auto").lower() in ("0", "false", "no"):
        return False
    return bool(os.getenv("OPENAI_API_KEY"))


def retrieve(
    topic: str,
    query: Optional[str] = None,
    k: int = DEFAULT_K,
    sources: Optional[List[Dict]] = None,
    embedder: Optional[Embedder] = None,
    title: Optional[str] = None,
    include_lead: bool = False,
) -> List[Dict]:
    """Return the top-``k`` retrieved chunks as ``{score, text, source_*}`` dicts.

    When ``include_lead`` is set, the opening chunk (``order == 0``, the
    Wikipedia lead/definition) of *every* grounded article is guaranteed to be
    present even if it didn't rank in the top-``k`` — so results can open with a
    proper definition and, for multi-article QA, the answer entity's own lead
    (which often names it directly) is always in play.
    """
    embedder = embedder or get_default_embedder()
    store = ingest.get_index(topic, sources=sources, embedder=embedder, title=title)
    if len(store) == 0:
        return []
    query_vec = embedder.embed_one(query or f"Key facts about {topic}")
    hits = store.search(query_vec, k=k)
    results = [
        {
            "score": round(score, 4),
            "text": meta["text"],
            "source_title": meta.get("source_title", topic),
            "source_url": meta.get("source_url", ""),
            "order": meta.get("order", 0),
        }
        for score, meta in hits
    ]
    if include_lead:
        # Force in the lead chunk of each distinct article (source_order == 0 is
        # the first chunk *within* each source), not just the corpus-wide first.
        leads = [m for m in store.metadatas if m.get("source_order") == 0]
        for lead in leads:
            if any(r["text"] == lead["text"] for r in results):
                continue
            lead_score = float(embedder.embed_one(lead["text"]) @ query_vec)
            results.append(
                {
                    "score": round(lead_score, 4),
                    "text": lead["text"],
                    "source_title": lead.get("source_title", topic),
                    "source_url": lead.get("source_url", ""),
                    "order": lead.get("order", 0),
                }
            )
    return results


def curate_facts(
    topic: str,
    max_facts: int = DEFAULT_MAX_FACTS,
    k: int = DEFAULT_K,
    sources: Optional[List[Dict]] = None,
    embedder: Optional[Embedder] = None,
    min_grounding: Optional[float] = None,
    title: Optional[str] = None,
    max_sources: int = 1,
) -> List[Dict]:
    """Return up to ``max_facts`` grounded, cited facts about ``topic``.

    Each fact dict contains: ``fact``, ``source_title``, ``source_url``,
    ``grounding_score`` and ``method`` ("llm" or "extractive"). ``title`` pins a
    specific Wikipedia page chosen via disambiguation.

    ``max_sources`` > 1 curates across several related articles (used by the
    digest) so facts span multiple sources; the default (1) grounds on one
    coherent article, which keeps interactive lookups tightly on-topic.
    """
    embedder = embedder or get_default_embedder()
    threshold = _grounding_threshold() if min_grounding is None else min_grounding

    # Curate across several related articles when asked (the digest), so the
    # result draws on multiple sources rather than a single page. Only when the
    # caller didn't inject their own sources or pin a specific article.
    if sources is None and title is None and max_sources > 1:
        sources = fetcher.fetch_topic_sources(topic, max_articles=max_sources) or None

    # Retrieve enough chunks to actually satisfy larger requests: each chunk
    # yields only a few coherent sentences after filtering, so scale k with the
    # number of facts asked for (retrieve() caps k at the corpus size).
    k = max(k, max_facts * 2)
    # With multiple sources, widen retrieval so every article can contribute.
    if sources:
        k = max(k, DEFAULT_K * len(sources))

    hits = retrieve(
        topic, k=k, sources=sources, embedder=embedder, title=title, include_lead=True
    )
    if not hits:
        return []

    if _llm_enabled():
        candidates = _generate_llm(topic, hits, max_facts) or _generate_extractive(
            topic, hits, max_facts, embedder
        )
        method = "llm" if candidates and candidates[0].get("_llm") else "extractive"
    else:
        candidates = _generate_extractive(topic, hits, max_facts, embedder)
        method = "extractive"

    guarded = _apply_guard(candidates, hits, embedder, threshold, method)
    deduped = _dedupe_candidates(guarded, embedder)

    # Selection favors the best-supported facts, but we always lead with the
    # article's opening line (the definition) when it survived the guard, so a
    # multi-fact result reads as a small story: definition first, then the
    # strongest supporting details, presented in the source's reading order.
    selected: List[Dict] = []
    selected_ids: set[int] = set()

    def _take(cand: Dict) -> None:
        selected.append(cand)
        selected_ids.add(id(cand))

    first_chunk = [c for c in deduped if c.get("_doc_order", 1_000_000) < 1000]
    if first_chunk and max_facts >= 1:
        _take(min(first_chunk, key=lambda c: c["_doc_order"]))

    # ``deduped`` is ordered by grounding strength. When several sources were
    # curated (the digest), first reserve the best still-unused fact from each
    # *other* source so the result visibly spans multiple sources instead of
    # collapsing onto whichever article grounded highest. For a single-source
    # lookup (Explore) every fact shares one source, so this pass is a no-op and
    # behaviour is unchanged.
    used_sources = {c["source_url"] for c in selected}
    for cand in deduped:
        if len(selected) >= max_facts:
            break
        if id(cand) in selected_ids or cand["source_url"] in used_sources:
            continue
        _take(cand)
        used_sources.add(cand["source_url"])

    # Fill any remaining slots purely by grounding strength.
    for cand in deduped:
        if len(selected) >= max_facts:
            break
        if id(cand) not in selected_ids:
            _take(cand)

    selected.sort(key=lambda c: c.get("_doc_order", 0))
    for fact in selected:
        fact.pop("_doc_order", None)  # internal ordering key — not part of the API
    return selected


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------
def _snippet(text: str, limit: int = 300) -> str:
    """Collapse whitespace and truncate a chunk to a readable citation snippet."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _build_citations(
    hits: List[Dict],
    limit: int = 3,
    priority_hits: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Top supporting chunks as citations, deduped.

    Any ``priority_hits`` (typically the chunks the answer was actually drawn
    from) are listed first, so the evidence we show always contains the answer's
    own source; the remaining hits follow ordered by retrieval score.
    """
    ordered = list(priority_hits or [])
    ordered += sorted(hits, key=lambda h: h.get("score", 0.0), reverse=True)
    citations: List[Dict] = []
    seen = set()
    for hit in ordered:
        key = (hit.get("source_url", ""), hit["text"][:80])
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source_title": hit["source_title"],
                "source_url": hit["source_url"],
                "snippet": _snippet(hit["text"]),
                "score": hit.get("score", 0.0),
            }
        )
        if len(citations) >= limit:
            break
    return citations


def answer_question(
    question: str,
    k: int = DEFAULT_K,
    sources: Optional[List[Dict]] = None,
    embedder: Optional[Embedder] = None,
    min_grounding: Optional[float] = None,
    title: Optional[str] = None,
) -> Dict:
    """Answer a free-form question from retrieved sources, or abstain.

    Pipeline: resolve which article the question is about, retrieve the most
    relevant chunks, then pick the retrieved sentence that best matches the
    question. If nothing clears the grounding threshold we deliberately abstain
    (``grounded=False``) instead of guessing — the QA counterpart of the fact
    hallucination guard, and the app's core "don't make things up" promise.

    Returns ``{question, answer (str|None), grounded (bool), confidence (float),
    source_title, source_url, citations:[{source_title, source_url, snippet,
    score}]}``.
    """
    embedder = embedder or get_default_embedder()
    threshold = _qa_threshold() if min_grounding is None else min_grounding

    empty = {
        "question": question,
        "answer": None,
        "grounded": False,
        "confidence": 0.0,
        "source_title": "",
        "source_url": "",
        "citations": [],
        "alternatives": [],
    }
    q = question.strip()
    if not q:
        return empty

    # Figure out which article the question is about. Wikipedia search maps a
    # natural-language question to its most relevant page well; skip this when
    # Figure out which article(s) the question is about. A question's answer
    # often lives in a *different* article than the one its subject names, so we
    # ground across the top few candidates and let the sentence re-ranker pick
    # the best-supported answer from whichever article actually contains it.
    # Skip this when the caller pinned a title or supplied their own sources.
    resolved_title = title
    if title is None and sources is None:
        try:
            fetched = fetcher.fetch_qa_sources(q, max_articles=QA_MAX_ARTICLES)
        except Exception:
            fetched = None
        # Coerce an empty result to None so retrieval falls back to a Wikipedia
        # lookup of the raw question, instead of grounding on an empty index
        # (which build_index treats as "no text" and would force an abstain).
        sources = fetched or None

    topic = resolved_title or q
    # Retrieve enough chunks that each grounded article can contribute evidence.
    n_articles = len(sources) if sources else 1
    hits = retrieve(
        topic,
        query=q,
        k=max(k, DEFAULT_K * n_articles),
        sources=sources,
        embedder=embedder,
        title=resolved_title,
        include_lead=True,
    )
    if not hits:
        return empty

    # Extractive QA: gather coherent candidate sentences from the retrieved
    # chunks and score each against the question; the closest becomes the answer.
    q_vec = embedder.embed_one(q)
    sentences: List[Dict] = []
    seen = set()
    for hit in hits:
        for sent in split_sentences(hit["text"]):
            sent = _clean_sentence(sent)
            key = sent.lower()
            if key in seen or not _is_coherent_fact(sent):
                continue
            seen.add(key)
            sentences.append({"text": sent, "hit": hit})

    citations = _build_citations(hits, limit=3)
    if not sentences:
        return {**empty, "citations": citations}

    vecs = embedder.embed([s["text"] for s in sentences])
    sims = vecs @ q_vec

    # A negated sentence ("… does not produce oxygen") shares keywords with an
    # affirmative question ("What does photosynthesis produce?"), so a weak
    # bag-of-words embedder often ranks it top — yet it is a poor answer. When
    # the question is affirmative, drop negated sentences from contention and
    # only fall back to them if nothing affirmative remains.
    q_is_affirmative = _NEGATION_RE.search(q.lower()) is None
    candidate_idx = list(range(len(sentences)))
    if q_is_affirmative:
        affirmative = [
            i
            for i in candidate_idx
            if not _NEGATION_RE.search(sentences[i]["text"].lower())
        ]
        if affirmative:
            candidate_idx = affirmative

    # Rank the remaining candidates by similarity (descending). The top sentence
    # becomes the answer; the rest become "try another answer" alternatives.
    # Choosing by raw similarity means the reported confidence is exactly that
    # sentence's score, so a grounded answer always has confidence >= threshold
    # (no inflated or borrowed score).
    ranked = sorted(candidate_idx, key=lambda i: float(sims[i]), reverse=True)
    best_i = ranked[0]
    best_score = float(sims[best_i])

    if best_score < threshold:
        # Nothing relevant enough — abstain rather than fabricate an answer.
        return {
            "question": question,
            "answer": None,
            "grounded": False,
            "confidence": round(max(best_score, 0.0), 4),
            "source_title": hits[0]["source_title"],
            "source_url": hits[0]["source_url"],
            "citations": citations,
            "alternatives": [],
        }

    # Keep each answer to a single retrieved sentence: fully grounded and
    # attributable to one source, clearer and safer than stitching sentences.
    primary = sentences[best_i]["hit"]
    answer_text = sentences[best_i]["text"]

    # Gather distinct, above-threshold runners-up so the UI can offer a
    # "try another answer" control without a second request. We return *every*
    # distinct answer that clears the grounding threshold — no fixed cap — so the
    # count honestly reflects how much grounded evidence the sources contain
    # (naturally bounded by how many chunks we retrieved). Near-duplicates (one
    # sentence wholly contained in another) are skipped so alternatives read as
    # genuinely different answers.
    def _norm(text: str) -> str:
        # Drop punctuation for comparison so "…politician." and "…politician!"
        # count as the same answer rather than two near-identical alternatives.
        return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())

    chosen = [_norm(answer_text)]
    alternatives: List[Dict] = []
    for i in ranked[1:]:
        score = float(sims[i])
        if score < threshold:
            break  # ranked is descending: nothing further can qualify
        text = sentences[i]["text"]
        norm = _norm(text)
        if any(norm in c or c in norm for c in chosen):
            continue
        hit = sentences[i]["hit"]
        alternatives.append(
            {
                "answer": text,
                "confidence": round(score, 4),
                "source_title": hit["source_title"],
                "source_url": hit["source_url"],
            }
        )
        chosen.append(norm)

    # Build citations *after* choosing the answer so the shown evidence always
    # includes the chunk the answer actually came from.
    citations = _build_citations(hits, limit=3, priority_hits=[primary])
    return {
        "question": question,
        "answer": answer_text,
        "grounded": True,
        "confidence": round(best_score, 4),
        "source_title": primary["source_title"],
        "source_url": primary["source_url"],
        "citations": citations,
        "alternatives": alternatives,
    }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def _clean_sentence(sentence: str) -> str:
    """Strip section headers and inline reference markers ("[1]") from a sentence."""
    sentence = _HEADER_RE.sub(" ", sentence)
    sentence = _REF_RE.sub("", sentence)  # drop "[1]", "[citation needed]", etc.
    return " ".join(sentence.split()).strip()


def _is_coherent_fact(sentence: str) -> bool:
    """Heuristic: does this sentence stand on its own as an informative fact?

    Rejects fragments, sentences that dangle off a previous one ("He also…"),
    and opinion/review/attribution lines — the things that made the extractive
    output read incoherently.
    """
    words = sentence.split()
    if len(words) < 6 or len(sentence) < 40:
        return False
    if not sentence[0].isupper():
        return False  # likely a mid-sentence fragment
    if sentence[-1] not in '.!?"”’)':
        return False  # truncated at a chunk boundary — not a whole sentence
    if _DANGLING_START_RE.match(sentence):
        return False
    if _OPINION_RE.search(sentence):
        return False
    if '"' in sentence or "“" in sentence or "”" in sentence:
        return False  # usually a quotation rather than a plain fact
    # Require a sensible amount of alphabetic content (skip tables/lists).
    letters = sum(c.isalpha() or c.isspace() for c in sentence)
    return letters / len(sentence) > 0.75


def _generate_extractive(
    topic: str, hits: List[Dict], max_facts: int, embedder: Embedder
) -> List[Dict]:
    """Offline fallback: pick the most on-topic, self-contained source sentences.

    Each candidate records ``_doc_order`` — its position in the source article —
    so the final selection can be re-sorted into reading order (definition first,
    then elaboration) and told as a small story rather than a jumble.
    """
    candidates: List[Dict] = []
    seen = set()
    for hit in hits:
        chunk_order = hit.get("order", 0)
        for sent_idx, sentence in enumerate(split_sentences(hit["text"])):
            sentence = _clean_sentence(sentence)
            key = sentence.lower()
            if key in seen:
                continue
            if not _is_coherent_fact(sentence):
                continue
            seen.add(key)
            candidates.append(
                {
                    "fact": sentence,
                    "source_title": hit["source_title"],
                    "source_url": hit["source_url"],
                    # Sortable position: chunk index dominates, sentence breaks ties.
                    "_doc_order": chunk_order * 1000 + sent_idx,
                }
            )

    if not candidates:
        return []

    # Rank by relevance to the topic, with a small bonus for sentences that
    # actually mention the topic's key terms (more likely to be informative).
    query_vec = embedder.embed_one(f"Key facts about {topic}")
    cand_vecs = embedder.embed([c["fact"] for c in candidates])
    scores = cand_vecs @ query_vec

    topic_terms = set(re.findall(r"[a-z]{4,}", topic.lower()))
    for i, cand in enumerate(candidates):
        if topic_terms:
            fact_terms = set(re.findall(r"[a-z]{4,}", cand["fact"].lower()))
            if topic_terms & fact_terms:
                scores[i] += 0.05
        # Nudge the article's opening lines up: on Wikipedia these are the
        # definition/overview, which make a natural first fact for the "story".
        if cand["_doc_order"] < 1000:  # first chunk
            scores[i] += 0.04

    order = np.argsort(-scores)
    shortlist_idx = list(order[: max(max_facts * 3, max_facts)])
    # Always keep the article's opening lines available so the final story can
    # start with a definition, even if they didn't rank in the relevance cut.
    for i, cand in enumerate(candidates):
        if cand["_doc_order"] < 1000 and i not in shortlist_idx:
            shortlist_idx.append(i)
    return [candidates[i] for i in shortlist_idx]


def _generate_llm(topic: str, hits: List[Dict], max_facts: int) -> Optional[List[Dict]]:
    """Grounded generation constrained to the retrieved context. ``None`` on error."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        context = "\n\n".join(
            f"[{i + 1}] (source: {h['source_title']})\n{h['text']}"
            for i, h in enumerate(hits)
        )
        system = (
            "You are a careful fact curator. Use ONLY the numbered context passages. "
            "Do not use outside knowledge. Return a JSON array; each element must be "
            '{"fact": <1-2 sentence fact>, "source_index": <the [n] passage it came from>}. '
            "Return [] if the context supports no clear facts."
        )
        user = (
            f"Topic: {topic}\n\nContext:\n{context}\n\n"
            f"Extract up to {max_facts} concise, verifiable facts."
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "facts" in parsed:
            parsed = parsed["facts"]

        candidates: List[Dict] = []
        for pos, item in enumerate(parsed):
            if not isinstance(item, dict) or "fact" not in item:
                continue
            idx = item.get("source_index")
            citation = hits[0]
            if isinstance(idx, int) and 1 <= idx <= len(hits):
                citation = hits[idx - 1]
            candidates.append(
                {
                    "fact": str(item["fact"]).strip(),
                    "source_title": citation["source_title"],
                    "source_url": citation["source_url"],
                    "_llm": True,
                    # Keep the order the model chose to present the facts in.
                    "_doc_order": pos,
                }
            )
        return candidates or None
    except Exception as exc:  # pragma: no cover - network/parse failures
        print("LLM curation failed, falling back to extractive:", exc)
        return None


# ---------------------------------------------------------------------------
# Hallucination guard + dedup
# ---------------------------------------------------------------------------
def _apply_guard(
    candidates: List[Dict],
    hits: List[Dict],
    embedder: Embedder,
    threshold: float,
    method: str,
) -> List[Dict]:
    """The hallucination guard: drop candidates unsupported by any retrieved
    chunk, and attach the best-matching chunk as a citation with its score."""
    if not candidates:
        return []
    evidence = embedder.embed([h["text"] for h in hits])  # (k, d)
    fact_vecs = embedder.embed([c["fact"] for c in candidates])

    kept: List[Dict] = []
    for cand, fv in zip(candidates, fact_vecs, strict=True):
        sims = evidence @ fv
        best = int(np.argmax(sims))
        grounding = float(sims[best])
        if grounding < threshold:
            continue  # unsupported by any retrieved chunk -> likely hallucinated
        kept.append(
            {
                "fact": cand["fact"],
                "source_title": cand.get("source_title") or hits[best]["source_title"],
                "source_url": cand.get("source_url") or hits[best]["source_url"],
                "grounding_score": round(grounding, 4),
                "method": method,
                # Preserve reading position so the final list can be re-ordered
                # into a coherent narrative after selection.
                "_doc_order": cand.get("_doc_order", best * 1000),
            }
        )
    kept.sort(key=lambda c: c["grounding_score"], reverse=True)
    return kept


def _norm_for_dedup(text: str) -> str:
    """Lowercase + strip trailing punctuation so containment checks compare the
    substance of two sentences, not their casing or final period."""
    return text.lower().strip().rstrip('.!?"”’) ')


def _dedupe_candidates(
    candidates: List[Dict], embedder: Embedder, threshold: float = 0.9
) -> List[Dict]:
    """Drop candidates that duplicate one already selected.

    Two signals are used: (1) embedding cosine similarity for paraphrases, and
    (2) substring containment for the common case where one sentence is a
    truncated prefix of another (e.g. a lead-chunk sentence cut off at a chunk
    boundary). When containment is found we keep the *fuller* sentence so the
    result never shows both "...supplies most" and the complete version.
    """
    selected: List[Dict] = []
    selected_vecs: List[np.ndarray] = []
    for cand in candidates:
        norm = _norm_for_dedup(cand["fact"])

        # Containment against already-selected facts (whole sentences, so a
        # substring match is a strong duplicate signal — not a coincidence).
        replace_idx = None
        contained = False
        cand_neg = bool(_NEGATION_RE.search(norm))
        for i, sel in enumerate(selected):
            snorm = _norm_for_dedup(sel["fact"])
            # Opposite polarity => different claims, not a fragment/fuller pair
            # ("X is longest" vs "some dispute X is longest"): keep both.
            if cand_neg != bool(_NEGATION_RE.search(snorm)):
                continue
            if norm == snorm or norm in snorm:
                contained = True  # cand is same as, or a fragment of, a selection
                break
            if snorm in norm:
                # cand is the fuller sentence — upgrade to it only if it is at
                # least as well-grounded, so we never trade down on evidence.
                if cand.get("grounding_score", 0.0) >= sel.get("grounding_score", 0.0):
                    replace_idx = i
                else:
                    contained = True
                break
        if contained:
            continue

        vec = embedder.embed_one(cand["fact"])
        if replace_idx is not None:
            selected[replace_idx] = cand
            selected_vecs[replace_idx] = vec
            continue
        if any(float(np.dot(vec, sv)) >= threshold for sv in selected_vecs):
            continue
        selected.append(cand)
        selected_vecs.append(vec)
    return selected
