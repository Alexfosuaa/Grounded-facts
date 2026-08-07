import numpy as np

from backend.rag.embeddings import Embedder, cosine_similarity


def test_shape_and_normalization(embedder):
    vecs = embedder.embed(["hello world", "another sentence here"])
    assert vecs.shape == (2, embedder.dim)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_determinism(embedder):
    a = embedder.embed_one("the quick brown fox")
    b = embedder.embed_one("the quick brown fox")
    assert np.array_equal(a, b)


def test_empty_input(embedder):
    vecs = embedder.embed([])
    assert vecs.shape == (0, embedder.dim)


def test_semantic_ordering(embedder):
    # A near-paraphrase should be more similar than an unrelated sentence.
    anchor = embedder.embed_one("Paris is the capital city of France")
    similar = embedder.embed_one("The capital of France is Paris")
    different = embedder.embed_one("Whales are large marine mammals")
    assert cosine_similarity(anchor, similar) > cosine_similarity(anchor, different)


def test_unknown_backend_raises():
    import pytest

    with pytest.raises(ValueError):
        Embedder(backend="does-not-exist")
