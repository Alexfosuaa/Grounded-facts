"""Shared pytest fixtures. All tests run fully offline (hashing embedder,
injected sources) so the suite never touches the network or an API key.
"""

import os

import pytest

# Ensure the offline, deterministic configuration for the whole suite.
os.environ.setdefault("EMBED_BACKEND", "hashing")
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("USE_LLM", None)
# Disable the on-disk article cache in tests so dummy fixtures never leak into
# (or get served from) the real cache directory.
os.environ["WIKI_CACHE_DIR"] = ""

from backend.rag.embeddings import Embedder  # noqa: E402
from backend.services import db  # noqa: E402


@pytest.fixture
def embedder():
    return Embedder(backend="hashing", dim=256)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the db module at an isolated temporary database."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    db.init_db()
    return db


@pytest.fixture
def sample_sources():
    return [
        {
            "title": "Photosynthesis",
            "url": "https://en.wikipedia.org/wiki/Photosynthesis",
            "text": (
                "Photosynthesis is a process used by plants and other organisms to "
                "convert light energy into chemical energy. It takes place in the "
                "chloroplasts of plant cells. Chlorophyll absorbs light in the blue "
                "and red parts of the spectrum. The process releases oxygen as a "
                "byproduct. Carbon dioxide and water are the primary raw materials. "
                "Photosynthesis is essential for life on Earth because it produces "
                "the oxygen and organic compounds that most organisms depend on."
            ),
        }
    ]
