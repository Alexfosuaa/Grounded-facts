"""A small cosine-similarity vector store with a pluggable backend.

The corpus for this project (a handful of Wikipedia pages per topic) is tiny, so
an exact search is both correct and fast. Two interchangeable backends are
provided behind one interface:

* ``numpy`` (always available): brute-force inner-product search. Zero extra
  dependencies, so tests and CI never need native wheels.
* ``faiss`` (optional): a FAISS ``IndexFlatIP`` index. Enabled when ``faiss`` is
  importable and ``USE_FAISS`` is truthy. Demonstrates the industry-standard
  vector-search library while keeping the NumPy path as a guaranteed fallback.

Vectors are assumed to be L2-normalized (see :mod:`embeddings`), so inner product
equals cosine similarity.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


def _want_faiss() -> bool:
    flag = os.getenv("USE_FAISS", "auto").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return _faiss_available()  # "auto": use faiss when present


class VectorStore:
    def __init__(self, dim: int, use_faiss: Optional[bool] = None):
        self.dim = int(dim)
        self.metadatas: List[Dict] = []
        self._use_faiss = _want_faiss() if use_faiss is None else use_faiss
        if self._use_faiss and not _faiss_available():
            self._use_faiss = False

        self._vectors: Optional[np.ndarray] = None  # numpy backend
        self._index = None  # faiss backend
        if self._use_faiss:
            import faiss

            self._index = faiss.IndexFlatIP(self.dim)

    @property
    def backend(self) -> str:
        return "faiss" if self._use_faiss else "numpy"

    def __len__(self) -> int:
        return len(self.metadatas)

    def add(self, vectors: np.ndarray, metadatas: List[Dict]) -> None:
        """Append ``(n, dim)`` vectors and their per-row metadata to the store."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of shape (n, {self.dim}), got {vectors.shape}"
            )
        if len(metadatas) != vectors.shape[0]:
            raise ValueError("vectors and metadatas length mismatch")

        self.metadatas.extend(metadatas)
        if self._use_faiss:
            self._index.add(vectors)
        else:
            self._vectors = (
                vectors
                if self._vectors is None
                else np.vstack([self._vectors, vectors])
            )

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        """Return the top-``k`` ``(score, metadata)`` pairs by cosine similarity."""
        if len(self) == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        k = min(k, len(self))

        if self._use_faiss:
            scores, idxs = self._index.search(query, k)
            scores, idxs = scores[0], idxs[0]
        else:
            sims = self._vectors @ query[0]
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs]

        results: List[Tuple[float, Dict]] = []
        for score, idx in zip(scores, idxs, strict=True):
            if idx < 0:
                continue
            results.append((float(score), self.metadatas[int(idx)]))
        return results

    # -- persistence -----------------------------------------------------------
    def save(self, path_prefix: str) -> None:
        """Persist to ``<prefix>.meta.json`` (+ ``.npy`` or ``.faiss``)."""
        with open(f"{path_prefix}.meta.json", "w", encoding="utf-8") as fh:
            json.dump(
                {"dim": self.dim, "backend": self.backend, "metadatas": self.metadatas},
                fh,
            )
        if self._use_faiss:
            import faiss

            faiss.write_index(self._index, f"{path_prefix}.faiss")
        else:
            np.save(
                f"{path_prefix}.npy",
                self._vectors
                if self._vectors is not None
                else np.zeros((0, self.dim), dtype=np.float32),
            )

    @classmethod
    def load(cls, path_prefix: str) -> "VectorStore":
        with open(f"{path_prefix}.meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        use_faiss = meta["backend"] == "faiss"
        store = cls(dim=meta["dim"], use_faiss=use_faiss)
        store.metadatas = meta["metadatas"]
        if use_faiss:
            import faiss

            store._index = faiss.read_index(f"{path_prefix}.faiss")
        else:
            store._vectors = np.load(f"{path_prefix}.npy").astype(np.float32)
        return store
