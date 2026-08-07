from backend.rag.chunking import chunk_text, split_sentences
import pytest


def test_short_text_single_chunk():
    assert chunk_text("hello world", chunk_words=120) == ["hello world"]


def test_windows_overlap():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(words, chunk_words=100, overlap=20)
    assert len(chunks) > 1
    # each chunk within the size bound
    assert all(len(c.split()) <= 100 for c in chunks)
    # consecutive chunks share the overlap
    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_empty_text():
    assert chunk_text("") == []


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_words=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_words=0)


def test_split_sentences():
    assert split_sentences("A first one. A second one! A third?") == [
        "A first one.",
        "A second one!",
        "A third?",
    ]
    assert split_sentences("") == []
