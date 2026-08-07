"""FastAPI application.

Responsibilities:
* Expose the RAG curation pipeline as a small REST API under ``/api``.
* Serve the static web frontend (``/frontend``) at the site root.

Run locally with::

    uvicorn backend.api:app --reload

The UI, the delivery worker, and any external client all talk to the same core
through this one boundary.
"""

from __future__ import annotations

# Load .env before importing modules that read configuration at import time
# (e.g. services.db resolves DB_PATH). No-op if python-dotenv or .env is absent.
from dotenv import load_dotenv

load_dotenv()

import datetime
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.rag import curator, fetcher
from backend.rag.embeddings import get_default_embedder
from backend.rag.vectorstore import VectorStore
from backend.services import db, emailer, worker


def _require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Optional shared-secret gate for management/delivery routes.

    Off by default so the app runs open as a local single-user demo. If
    ``ADMIN_TOKEN`` is set (e.g. on a public deployment), the subscriber list,
    delete, and manual delivery routes require a matching ``X-Admin-Token``
    header — closing the door on anonymous enumeration/deletion/mail-triggering.
    """
    expected = os.getenv("ADMIN_TOKEN")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="admin token required")


def _mask_email(email: str) -> str:
    """Partially redact an email so the list endpoint never leaks full addresses."""
    local, _, domain = email.partition("@")
    if not domain:
        return "***"
    shown = local[0] if local else ""
    return f"{shown}***@{domain}"


def _resolve_custom_sources(source_url: str | None) -> list[dict] | None:
    """Turn an optional user-supplied URL into injected grounding sources.

    Returns ``None`` when no URL is given, so callers fall back to the default
    Wikipedia path. When a URL *is* given but can't be fetched safely, we raise
    400 rather than silently falling back — the user asked for that source.
    """
    if not source_url or not source_url.strip():
        return None
    src = fetcher.fetch_url_source(source_url.strip())
    if not src:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read that source URL. Use a public http(s) article "
                "link, or leave it blank to use Wikipedia."
            ),
        )
    return [src]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the database schema exists before serving any request.
    db.init_db()
    yield


app = FastAPI(title="Grounded Facts API", version="1.0.0", lifespan=lifespan)
# All JSON endpoints live under /api so the site root can serve the frontend.
router = APIRouter(prefix="/api")


# --- request/response schemas ----------------------------------------------
class SubscribeRequest(BaseModel):
    email: EmailStr
    topic: str = Field(min_length=1, max_length=200)
    cadence: str = Field(default="daily")
    # How many grounded facts each digest should aim to include. Kept to a small
    # 2–5 range so a scheduled email stays focused (default 3).
    max_facts: int = Field(default=3, ge=2, le=5)

    @field_validator("topic")
    @classmethod
    def _clean_topic(cls, v: str) -> str:
        # Strip surrounding space and reject control characters (e.g. CR/LF),
        # which would otherwise flow into the email Subject header and could be
        # used for header injection or to crash message construction.
        v = v.strip()
        if not v:
            raise ValueError("topic is required")
        if any(ord(c) < 32 for c in v):
            raise ValueError("topic contains invalid control characters")
        return v


class Fact(BaseModel):
    fact: str
    source_title: str
    source_url: str
    grounding_score: float
    method: str


class PreviewResponse(BaseModel):
    topic: str
    facts: List[Fact]


class Candidate(BaseModel):
    title: str
    description: str


class DisambiguateResponse(BaseModel):
    topic: str
    ambiguous: bool
    candidates: List[Candidate]


class Chunk(BaseModel):
    score: float
    text: str
    source_title: str
    source_url: str


class RetrieveResponse(BaseModel):
    topic: str
    hits: List[Chunk]


class DigestResponse(BaseModel):
    topic: str
    subject: str
    body: str
    facts: List[Fact]
    dry_run: bool


class Citation(BaseModel):
    source_title: str
    source_url: str
    snippet: str
    score: float


class AnswerAlternative(BaseModel):
    """A runner-up answer sentence surfaced by the "try another answer" control."""

    answer: str
    confidence: float
    source_title: str
    source_url: str


class AnswerResponse(BaseModel):
    question: str
    answer: str | None
    grounded: bool
    confidence: float
    source_title: str
    source_url: str
    citations: List[Citation]
    alternatives: List[AnswerAlternative] = []


class InfoResponse(BaseModel):
    embedder: str
    embedding_dim: int
    vector_backend: str
    generation_mode: str
    grounding_threshold: float
    email_dry_run: bool


# --- routes ----------------------------------------------------------------
@router.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Report which pipeline backends are active (surfaced in the UI header)."""
    emb = get_default_embedder()
    return InfoResponse(
        embedder=emb.name,
        embedding_dim=emb.dim,
        vector_backend=VectorStore(dim=emb.dim).backend,
        generation_mode="llm" if curator._llm_enabled() else "extractive",
        grounding_threshold=curator._grounding_threshold(),
        email_dry_run=emailer.is_dry_run(),
    )


@router.get("/disambiguate", response_model=DisambiguateResponse)
def disambiguate(topic: str, limit: int = 6) -> DisambiguateResponse:
    """Return candidate Wikipedia pages so the UI can ask a follow-up question on
    broad/ambiguous topics.

    ``ambiguous`` is ``True`` when the bare topic has no single primary article
    (it resolves to a disambiguation page), which is the UI's cue to *prompt* the
    user to pick a sense rather than just offering the candidates as related
    reading.
    """
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic is required")
    limit = max(1, min(limit, 10))  # bound the upstream fan-out
    candidates = fetcher.fetch_candidates(topic, limit=limit)
    return DisambiguateResponse(
        topic=topic,
        ambiguous=fetcher.is_ambiguous(topic),
        candidates=candidates,
    )


@router.get("/preview", response_model=PreviewResponse)
def preview(
    topic: str,
    max_facts: int = 3,
    title: str | None = None,
    source_url: str | None = None,
) -> PreviewResponse:
    """Curate grounded facts for a topic (``title`` pins a disambiguated page).

    When ``source_url`` is supplied the facts are grounded on that page instead
    of Wikipedia; otherwise Wikipedia is used (the default).
    """
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic is required")
    max_facts = max(1, min(max_facts, 50))  # clamp to a sane range
    sources = _resolve_custom_sources(source_url)
    facts = curator.curate_facts(
        topic, max_facts=max_facts, title=title, sources=sources
    )
    return PreviewResponse(topic=topic, facts=facts)


@router.get("/retrieve", response_model=RetrieveResponse)
def retrieve(topic: str, k: int = 5, title: str | None = None) -> RetrieveResponse:
    """Expose the raw retrieved chunks so the UI can show *why* a fact was kept."""
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic is required")
    k = max(1, min(k, 50))  # avoid k<=0 (backend assert -> 500) and huge scans
    return RetrieveResponse(topic=topic, hits=curator.retrieve(topic, k=k, title=title))


@router.get("/digest", response_model=DigestResponse)
def digest(topic: str, max_facts: int = 3, title: str | None = None) -> DigestResponse:
    """Compose the digest email that *would* be sent for a topic — with no side
    effects (unlike a real delivery pass, it does not mark facts as sent).

    This lets the UI show subscribers exactly what their digest looks like even
    in demo/dry-run mode, where email is logged rather than delivered. The digest
    curates on the subscriber's behalf across several related articles (see
    ``curator.DIGEST_MAX_SOURCES``) so facts aren't confined to a single page.
    """
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic is required")
    max_facts = max(1, min(max_facts, 50))
    facts = curator.curate_facts(
        topic,
        max_facts=max_facts,
        title=title,
        max_sources=curator.DIGEST_MAX_SOURCES,
    )
    subject = f"Your facts about {topic}"
    body = worker.format_email(topic, facts) if facts else "No new facts right now."
    return DigestResponse(
        topic=topic,
        subject=subject,
        body=body,
        facts=facts,
        dry_run=emailer.is_dry_run(),
    )


@router.get("/ask", response_model=AnswerResponse)
def ask(
    question: str,
    title: str | None = None,
    source_url: str | None = None,
) -> AnswerResponse:
    """Answer a free-form question from grounded sources, or abstain.

    Unlike ``/preview`` (which lists facts about a topic), this retrieves the
    passages most relevant to the *question* and returns the best-supported
    sentence as the answer — with citations. When nothing clears the grounding
    threshold it abstains (``grounded=False``) rather than guessing. ``title``
    pins a specific Wikipedia page when the question is ambiguous, and
    ``source_url`` grounds the answer on a user-supplied page instead of
    Wikipedia (the default).
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    sources = _resolve_custom_sources(source_url)
    result = curator.answer_question(question, title=title, sources=sources)
    return AnswerResponse(**result)


@router.post("/subscribe")
def subscribe(req: SubscribeRequest) -> dict:
    """Create a recurring digest subscription for a topic."""
    if req.cadence not in ("hourly", "daily", "weekly"):
        raise HTTPException(status_code=400, detail="invalid cadence")
    db.add_subscription(
        req.email,
        req.topic,
        req.cadence,
        datetime.datetime.utcnow(),
        max_facts=req.max_facts,
    )
    return {
        "status": "subscribed",
        "email": req.email,
        "topic": req.topic,
        "max_facts": req.max_facts,
    }


@router.get("/subscriptions")
def list_subscriptions(_: None = Depends(_require_admin)) -> dict:
    """List all subscriptions (emails are masked so the list can't be harvested)."""
    subs = []
    for s in db.list_subscriptions():
        s = dict(s)
        if s.get("email"):
            s["email"] = _mask_email(s["email"])
        subs.append(s)
    return {"subscriptions": subs}


@router.delete("/subscriptions/{sub_id}")
def unsubscribe(sub_id: int, _: None = Depends(_require_admin)) -> dict:
    """Remove a subscription by id."""
    db.delete_subscription(sub_id)
    return {"status": "deleted", "id": sub_id}


@router.post("/run-due")
def run_due(_: None = Depends(_require_admin)) -> dict:
    """Trigger a single delivery pass (normally driven by the worker)."""
    return worker.run_once()


app.include_router(router)

# Mount the built Svelte SPA last so the /api routes above take precedence.
# The bundle is produced by `npm run build` in ../frontend and emitted to
# frontend/dist. If it hasn't been built yet the mount is skipped, and the API
# still works headless (useful in CI and for pure-backend tests).
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIR.exists():
    app.mount(
        "/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend"
    )
