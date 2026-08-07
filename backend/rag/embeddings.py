"""Pluggable text-embedding adapter.

Design goals
------------
* Work fully offline with zero heavy dependencies so the whole RAG pipeline is
  runnable and unit-testable anywhere (CI, laptops, containers) without an API
  key or a multi-gigabyte model download.
* Still expose production-grade semantic backends (sentence-transformers and
  OpenAI) behind the same interface, selectable via an environment variable.

Backends
--------
* ``auto`` (default): pick ``sbert`` when sentence-transformers is installed and
  loads cleanly, else fall back to ``hashing``. Installing the optional ``ml``
  extra therefore upgrades retrieval quality with no configuration change.
* ``hashing``: a deterministic, NumPy-only *feature hashing* embedder
  (a.k.a. the "hashing trick"). It maps word unigrams and bigrams into a fixed
  dimensional space with signed hashing to limit collisions. It is not as strong
  as a neural encoder, but it is a legitimate IR technique, is fully offline, and
  produces stable vectors that make retrieval + tests deterministic.
* ``sbert``: sentence-transformers (``all-MiniLM-L6-v2`` by default). Real neural
  sentence embeddings; recommended for production. Enable with
  ``EMBED_BACKEND=sbert`` (requires ``pip install -r requirements-ml.txt``).
* ``openai``: OpenAI ``text-embedding-3-small``. Enable with
  ``EMBED_BACKEND=openai`` and a valid ``OPENAI_API_KEY``.

All backends return L2-normalized, fixed-dimensional ``float32`` vectors, so the
downstream vector store can treat inner product as cosine similarity.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_HASHING_DIM = 256


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    """Process-independent hash (Python's built-in ``hash`` is salted per run)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def _sbert_available() -> bool:
    """True when the optional sentence-transformers backend can be imported."""
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


class Embedder:
    """Uniform embedding interface across backends.

    Parameters
    ----------
    backend:
        One of ``auto``, ``hashing``, ``sbert``, ``openai``. Defaults to the
        ``EMBED_BACKEND`` environment variable, or ``auto`` when unset.
    dim:
        Dimensionality for the hashing backend (ignored by neural backends, which
        have a fixed native dimension).
    """

    def __init__(self, backend: str | None = None, dim: int = DEFAULT_HASHING_DIM):
        requested = (backend or os.getenv("EMBED_BACKEND", "auto")).lower()
        self._hashing_dim = dim
        self._sbert_model = None
        self._openai_client = None

        # "auto" (the default) prefers the neural sbert encoder when it is both
        # installed and able to load, otherwise falls back to the always-available
        # offline hashing embedder — so installing the optional `ml` extra
        # upgrades retrieval quality with no configuration change.
        if requested == "auto":
            if _sbert_available():
                try:
                    self._init_sbert()
                    self.backend = "sbert"
                    return
                except Exception:
                    self._sbert_model = None  # download/load failed -> fall back
            self.backend = "hashing"
            return

        self.backend = requested
        if self.backend == "sbert":
            self._init_sbert()
        elif self.backend == "openai":
            self._init_openai()
        elif self.backend != "hashing":
            raise ValueError(f"Unknown EMBED_BACKEND: {self.backend!r}")

    # -- backend initialisation ------------------------------------------------
    def _init_sbert(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
        self._sbert_model = SentenceTransformer(model_name)
        # Newer sentence-transformers renamed this method; support both names.
        get_dim = getattr(
            self._sbert_model,
            "get_embedding_dimension",
            self._sbert_model.get_sentence_embedding_dimension,
        )
        self._dim = int(get_dim())

    def _init_openai(self) -> None:
        from openai import OpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("EMBED_BACKEND=openai requires OPENAI_API_KEY")
        self._openai_client = OpenAI(api_key=api_key)
        self._openai_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self._dim = 1536

    # -- public API ------------------------------------------------------------
    @property
    def dim(self) -> int:
        if self.backend == "hashing":
            return self._hashing_dim
        return self._dim

    @property
    def name(self) -> str:
        if self.backend == "sbert":
            return f"sbert:{os.getenv('SBERT_MODEL', 'all-MiniLM-L6-v2')}"
        if self.backend == "openai":
            return f"openai:{self._openai_model}"
        return f"hashing:{self._hashing_dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` matrix of L2-normalized float32 vectors."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.backend == "hashing":
            vectors = np.vstack([self._embed_hashing(t) for t in texts])
        elif self.backend == "sbert":
            vectors = np.asarray(
                self._sbert_model.encode(texts, normalize_embeddings=False),
                dtype=np.float32,
            )
        else:  # openai
            vectors = self._embed_openai(texts)

        return _l2_normalize(vectors.astype(np.float32))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    # -- hashing backend -------------------------------------------------------
    def _embed_hashing(self, text: str) -> np.ndarray:
        dim = self._hashing_dim
        vec = np.zeros(dim, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vec

        features = list(tokens)
        features.extend(
            f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)
        )  # add bigrams for a little word-order signal (lengths differ by one)

        for feature in features:
            h = _stable_hash(feature)
            idx = h % dim
            sign = 1.0 if (h >> 8) & 1 else -1.0  # signed hashing reduces collisions
            vec[idx] += sign
        return vec

    # -- openai backend --------------------------------------------------------
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        resp = self._openai_client.embeddings.create(
            model=self._openai_model, input=texts
        )
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so inner product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D vectors (assumes finite values)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# A process-wide default embedder is convenient and keeps the (potentially heavy)
# neural model loaded once. Callers may still construct their own.
_default_embedder: Embedder | None = None


def get_default_embedder() -> Embedder:
    """Return the lazily-created, process-wide shared embedder."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder
