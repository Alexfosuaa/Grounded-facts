"""Backend package for the Grounded Facts service.

Layout:
* ``backend.rag``      – the retrieval-augmented-generation pipeline.
* ``backend.services`` – application services (DB, dedup, email, worker).
* ``backend.api``      – the FastAPI app that also serves the web frontend.
"""
