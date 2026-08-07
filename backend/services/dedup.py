"""Semantic de-duplication for spaced-repetition delivery.

Before a batch of curated facts is emailed to a subscriber, this module removes
any fact that is semantically too similar to a fact already delivered to that
same subscription. This prevents the "you keep sending me the same fact" problem
that plain keyword matching cannot catch (paraphrases slip through), and turns
the delivery loop into a lightweight spaced-repetition system.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from backend.services import db
from backend.rag.embeddings import Embedder, get_default_embedder

DEFAULT_DEDUP_THRESHOLD = 0.9


def filter_new_facts(
    subscription_id: int,
    facts: List[Dict],
    embedder: Embedder | None = None,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> Tuple[List[Dict], List[List[float]]]:
    """Return facts not previously sent to ``subscription_id``.

    Returns a tuple ``(kept_facts, kept_embeddings)`` where ``kept_embeddings`` are
    the embeddings of the kept facts, ready to be persisted with
    :func:`db.add_sent_fact` after a successful send.
    """
    embedder = embedder or get_default_embedder()
    if not facts:
        return [], []

    previous = [np.asarray(e, dtype=np.float32) for e in db.get_sent_fact_embeddings(subscription_id)]
    fact_vecs = embedder.embed([f["fact"] for f in facts])

    kept_facts: List[Dict] = []
    kept_embeddings: List[List[float]] = []
    for fact, vec in zip(facts, fact_vecs):
        if _too_similar(vec, previous, threshold) or _too_similar(
            vec, kept_embeddings, threshold
        ):
            continue
        kept_facts.append(fact)
        kept_embeddings.append(vec.tolist())
    return kept_facts, kept_embeddings


def _too_similar(vec: np.ndarray, others, threshold: float) -> bool:
    for other in others:
        other = np.asarray(other, dtype=np.float32)
        if float(np.dot(vec, other)) >= threshold:
            return True
    return False
