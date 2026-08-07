"""Ingestion: turn a topic into a searchable vector index.

Pipeline:  fetch multiple sources -> chunk -> embed -> index.

Sources can be injected directly (used by tests and the evaluation harness) so
the whole pipeline runs offline without touching the network.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from backend.rag import fetcher
from backend.rag.chunking import chunk_text
from backend.rag.embeddings import Embedder, get_default_embedder
from backend.rag.vectorstore import VectorStore

# Process-local LRU cache so repeated previews/sends for the same topic don't
# refetch and re-embed. Keyed by (topic, explicit-title, embedder name). It is
# bounded so a stream of distinct topics from public endpoints can't grow the
# process's memory without limit (least-recently-used entries are evicted).
_INDEX_CACHE: "OrderedDict[Tuple[str, str, str], VectorStore]" = OrderedDict()
_INDEX_CACHE_MAX = 128


def _cache_put(key: Tuple[str, str, str], store: VectorStore) -> None:
    """Insert/refresh a cache entry, evicting the oldest beyond the size cap."""
    _INDEX_CACHE[key] = store
    _INDEX_CACHE.move_to_end(key)
    while len(_INDEX_CACHE) > _INDEX_CACHE_MAX:
        _INDEX_CACHE.popitem(last=False)


def build_index(
    topic: str,
    sources: Optional[List[Dict]] = None,
    embedder: Optional[Embedder] = None,
    chunk_words: int = 120,
    overlap: int = 30,
    max_pages: int = 3,
    title: Optional[str] = None,
) -> VectorStore:
    """Build a :class:`VectorStore` for ``topic``.

    Parameters
    ----------
    sources:
        Optional pre-fetched ``[{"title", "url", "text"}, ...]``. When ``None``
        they are fetched from Wikipedia.
    title:
        Optional explicit Wikipedia page to lock onto (from disambiguation).
    """
    embedder = embedder or get_default_embedder()
    if sources is None:
        sources = fetcher.fetch_sources(topic, max_pages=max_pages, title=title)

    chunk_texts: List[str] = []
    metadatas: List[Dict] = []
    order = 0  # running position in the corpus, so facts can later be re-sorted
    for src in sources:
        # `source_order` restarts per article, so the first chunk of *each*
        # source (source_order == 0) is that article's lead/definition. `order`
        # stays global so facts across sources can still be sorted into corpus
        # order. Retrieval uses source_order to force in every article's lead.
        for source_order, chunk in enumerate(
            chunk_text(src.get("text", ""), chunk_words, overlap)
        ):
            chunk_texts.append(chunk)
            metadatas.append(
                {
                    "text": chunk,
                    "source_title": src.get("title", topic),
                    "source_url": src.get("url", ""),
                    "topic": topic,
                    "order": order,
                    "source_order": source_order,
                }
            )
            order += 1

    store = VectorStore(dim=embedder.dim)
    if chunk_texts:
        vectors = embedder.embed(chunk_texts)
        store.add(vectors, metadatas)
    return store


def get_index(
    topic: str,
    sources: Optional[List[Dict]] = None,
    embedder: Optional[Embedder] = None,
    force_rebuild: bool = False,
    title: Optional[str] = None,
) -> VectorStore:
    """Return a cached index for ``topic``, building it on first use.

    Injected ``sources`` always trigger a fresh build (they bypass the cache) so
    deterministic offline callers are never served stale network data. The cache
    key includes any explicit ``title`` so different disambiguation choices for
    the same topic don't collide.
    """
    embedder = embedder or get_default_embedder()
    key = (topic.lower().strip(), (title or "").lower().strip(), embedder.name)

    if sources is not None or force_rebuild:
        # Injected sources (custom URLs, QA multi-article fetches, tests) bypass
        # the cache for BOTH read and write. Caching them under the
        # (topic, title, embedder) key would let a later *default* request for
        # the same topic be served the injected content — cache poisoning — and
        # would also serve stale injected data. Always build fresh, never store.
        return build_index(topic, sources=sources, embedder=embedder, title=title)

    if key not in _INDEX_CACHE:
        store = build_index(topic, embedder=embedder, title=title)
        # Never cache an empty index: a transient fetch failure would otherwise
        # be "sticky" and starve every later request for this topic. Leaving it
        # uncached means the next request simply retries the fetch.
        if len(store) > 0:
            _cache_put(key, store)
        return store
    _INDEX_CACHE.move_to_end(key)  # mark as recently used
    return _INDEX_CACHE[key]


def clear_cache() -> None:
    _INDEX_CACHE.clear()
