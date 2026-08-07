"""RAG pipeline.

Data flows through these modules in order:

    fetcher -> chunking -> embeddings -> vectorstore   (ingest builds the index)
    ingest  -> curator                                 (retrieve + ground + guard)
"""
