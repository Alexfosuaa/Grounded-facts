import numpy as np
import pytest

from backend.rag.vectorstore import VectorStore


DOCS = [
    "The Eiffel Tower is located in Paris, France.",
    "Photosynthesis occurs in the chloroplasts of plant cells.",
    "Paris is the capital of France.",
]


def _build(embedder, use_faiss):
    store = VectorStore(dim=embedder.dim, use_faiss=use_faiss)
    store.add(embedder.embed(DOCS), [{"id": i, "text": d} for i, d in enumerate(DOCS)])
    return store


@pytest.mark.parametrize("use_faiss", [False, True])
def test_retrieval_returns_nearest(embedder, use_faiss):
    store = _build(embedder, use_faiss)
    q = embedder.embed_one("Where is the Eiffel Tower?")
    results = store.search(q, k=2)
    assert len(results) == 2
    top_ids = {meta["id"] for _, meta in results}
    assert top_ids & {0, 2}  # a Paris/Eiffel document should surface


def test_dimension_mismatch_raises(embedder):
    store = VectorStore(dim=embedder.dim, use_faiss=False)
    with pytest.raises(ValueError):
        store.add(np.zeros((1, embedder.dim + 1), dtype=np.float32), [{"id": 0}])


def test_empty_store_returns_empty(embedder):
    store = VectorStore(dim=embedder.dim, use_faiss=False)
    assert store.search(embedder.embed_one("anything"), k=3) == []


def test_save_and_load_roundtrip(embedder, tmp_path):
    store = _build(embedder, use_faiss=False)
    prefix = str(tmp_path / "index")
    store.save(prefix)
    loaded = VectorStore.load(prefix)
    assert len(loaded) == len(store)
    q = embedder.embed_one("capital of France")
    assert loaded.search(q, k=1)[0][1]["id"] in {0, 2}
