"""Text chunking utilities for the retrieval pipeline.

Splitting long source documents into overlapping windows is a standard RAG
preprocessing step: it keeps each embedded unit topically focused while the
overlap preserves context that would otherwise be lost at chunk boundaries.
"""

from __future__ import annotations

import re
from typing import List

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def split_sentences(text: str) -> List[str]:
    """Naive but dependency-free sentence splitter."""
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_words: int = 120, overlap: int = 30) -> List[str]:
    """Split ``text`` into overlapping windows of roughly ``chunk_words`` words.

    Parameters
    ----------
    chunk_words:
        Target number of words per chunk.
    overlap:
        Number of words shared between consecutive chunks.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_words")

    text = _WS_RE.sub(" ", (text or "").strip())
    if not text:
        return []

    words = text.split(" ")
    if len(words) <= chunk_words:
        return [text]

    step = chunk_words - overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks
