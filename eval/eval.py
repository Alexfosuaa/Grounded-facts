"""Offline evaluation harness for the RAG pipeline.

Runs fully offline against small, hand-labeled seed corpora (no network, no API
key) so the numbers are deterministic and reproducible in CI. It measures the
two properties that matter most for this project:

1. **Retrieval hit-rate@k** — for a set of queries with a known relevant source,
   how often does a chunk from that source appear in the top-k results?
2. **Hallucination-guard quality** — given facts labeled "supported" (paraphrased
   from a source) or "unsupported" (off-topic / fabricated), how well does the
   grounding guard keep the supported ones and drop the rest? Reported as
   precision / recall / F1 plus the mean grounding score for each group.

Run it with::

    python -m eval.eval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from backend.rag import curator, ingest
from backend.rag.embeddings import Embedder, get_default_embedder


@dataclass
class Corpus:
    """One topic with its sources and the labels we evaluate against."""

    topic: str
    sources: List[Dict]
    # query -> substring expected to appear in a retrieved chunk from the source.
    queries: Dict[str, str] = field(default_factory=dict)
    supported_facts: List[str] = field(default_factory=list)
    unsupported_facts: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Seed corpora. Kept intentionally small and self-contained; the source text is
# plain paraphrase written for the test, not copied from any external work.
# --------------------------------------------------------------------------- #
CORPORA: List[Corpus] = [
    Corpus(
        topic="Photosynthesis",
        sources=[
            {
                "title": "Photosynthesis",
                "url": "https://example.org/photosynthesis",
                "text": (
                    "Photosynthesis is the process used by plants, algae and some "
                    "bacteria to convert light energy into chemical energy stored in "
                    "glucose. In plants it takes place mainly in the chloroplasts, "
                    "which contain the green pigment chlorophyll. The light-dependent "
                    "reactions capture energy from sunlight and split water molecules, "
                    "releasing oxygen as a by-product. The Calvin cycle then uses that "
                    "energy to fix carbon dioxide into sugars. Photosynthesis is the "
                    "primary source of oxygen in Earth's atmosphere."
                ),
            }
        ],
        queries={
            "What pigment captures light in plants?": "chlorophyll",
            "What gas is released by photosynthesis?": "oxygen",
            "Where does photosynthesis happen in plant cells?": "chloroplast",
        },
        supported_facts=[
            "Photosynthesis converts light energy into chemical energy stored in glucose.",
            "Chlorophyll is the green pigment that captures light in the chloroplasts.",
            "The light-dependent reactions split water and release oxygen.",
        ],
        unsupported_facts=[
            "The Eiffel Tower was completed in 1889 in Paris.",
            "Photosynthesis was invented by a Roman emperor in the year 44 BC.",
        ],
    ),
    Corpus(
        topic="Alan Turing",
        sources=[
            {
                "title": "Alan Turing",
                "url": "https://example.org/alan-turing",
                "text": (
                    "Alan Turing was a British mathematician and logician widely "
                    "considered a founder of theoretical computer science. He formalised "
                    "the concepts of algorithm and computation with the Turing machine. "
                    "During the Second World War he worked at Bletchley Park, where he "
                    "helped break the German Enigma cipher, work credited with shortening "
                    "the war. In 1950 he proposed the Turing test as a criterion for "
                    "machine intelligence."
                ),
            }
        ],
        queries={
            "What machine is named after Turing?": "Turing machine",
            "Where did Turing work during the war?": "Bletchley Park",
            "What test did Turing propose?": "Turing test",
        },
        supported_facts=[
            "Alan Turing helped break the German Enigma cipher at Bletchley Park.",
            "Turing proposed the Turing test as a criterion for machine intelligence.",
        ],
        unsupported_facts=[
            "Alan Turing discovered penicillin in a hospital in 1928.",
            "The recipe calls for two cups of flour and a pinch of salt.",
        ],
    ),
]


def evaluate_retrieval(corpus: Corpus, embedder: Embedder, k: int) -> List[bool]:
    """Return one hit/miss per query: did the expected substring make top-k?"""
    outcomes: List[bool] = []
    for query, expected in corpus.queries.items():
        hits = curator.retrieve(
            corpus.topic,
            query=query,
            k=k,
            sources=corpus.sources,
            embedder=embedder,
        )
        joined = " ".join(h["text"].lower() for h in hits)
        outcomes.append(expected.lower() in joined)
    return outcomes


def grounding_scores(
    facts: List[str], evidence_vecs: np.ndarray, embedder: Embedder
) -> List[float]:
    """Max cosine similarity of each fact to any evidence chunk (the guard metric)."""
    if not facts:
        return []
    fact_vecs = embedder.embed(facts)  # already L2-normalised -> dot == cosine
    return [float(np.max(evidence_vecs @ fv)) for fv in fact_vecs]


def main() -> None:
    embedder = get_default_embedder()
    threshold = curator._grounding_threshold()
    k = 5

    print("=" * 68)
    print("RAG pipeline evaluation (offline)")
    print(f"  embedder           : {embedder.name}")
    print(f"  grounding threshold : {threshold}")
    print(f"  retrieval k         : {k}")
    print("=" * 68)

    all_retrieval: List[bool] = []
    tp = fp = fn = tn = 0
    supported_scores: List[float] = []
    unsupported_scores: List[float] = []

    for corpus in CORPORA:
        # --- retrieval ---
        outcomes = evaluate_retrieval(corpus, embedder, k)
        all_retrieval.extend(outcomes)

        # --- guard: embed the retrieved evidence once, then score every fact ---
        hits = curator.retrieve(
            corpus.topic, k=k, sources=corpus.sources, embedder=embedder
        )
        evidence_vecs = embedder.embed([h["text"] for h in hits])

        sup = grounding_scores(corpus.supported_facts, evidence_vecs, embedder)
        uns = grounding_scores(corpus.unsupported_facts, evidence_vecs, embedder)
        supported_scores.extend(sup)
        unsupported_scores.extend(uns)

        # Supported facts should pass the threshold; unsupported should not.
        tp += sum(1 for s in sup if s >= threshold)
        fn += sum(1 for s in sup if s < threshold)
        fp += sum(1 for s in uns if s >= threshold)
        tn += sum(1 for s in uns if s < threshold)

        hit_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
        print(f"\nTopic: {corpus.topic}")
        print(f"  retrieval hit-rate@{k} : {hit_rate:.0%} ({sum(outcomes)}/{len(outcomes)})")
        print(f"  supported grounding    : {[round(s, 3) for s in sup]}")
        print(f"  unsupported grounding  : {[round(s, 3) for s in uns]}")

    # --- aggregate metrics ---
    retrieval_rate = sum(all_retrieval) / len(all_retrieval) if all_retrieval else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_sup = float(np.mean(supported_scores)) if supported_scores else 0.0
    mean_uns = float(np.mean(unsupported_scores)) if unsupported_scores else 0.0

    print("\n" + "=" * 68)
    print("Aggregate metrics")
    print("-" * 68)
    print(f"  retrieval hit-rate@{k}        : {retrieval_rate:.0%}")
    print(f"  guard precision              : {precision:.0%}")
    print(f"  guard recall                 : {recall:.0%}")
    print(f"  guard F1                     : {f1:.2f}")
    print(f"  mean grounding (supported)   : {mean_sup:.3f}")
    print(f"  mean grounding (unsupported) : {mean_uns:.3f}")
    print(f"  separation margin            : {mean_sup - mean_uns:.3f}")
    print("=" * 68)

    # Reset the process cache so repeated runs in the same interpreter are clean.
    ingest.clear_cache()


if __name__ == "__main__":
    main()
