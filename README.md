# 🧭 Grounded Facts — a mini RAG system with a hallucination guard

Grounded Facts turns any topic into a set of **source-grounded, cited facts** and
can email them on a schedule. It is a compact but complete
**Retrieval-Augmented Generation (RAG)** pipeline: it fetches sources, chunks and
embeds them into a vector index, retrieves the most relevant passages, generates
candidate facts, and then runs a **hallucination guard** that drops any claim not
supported by the retrieved evidence.

The whole thing runs **fully offline by default** — no API key, no network at
inference time, no GPU — because the default embedder is a deterministic
feature-hashing model. Swap in a neural embedder or an LLM with a single
environment variable when you want higher quality.

---

## Why this project is interesting

- **Real RAG, not an LLM wrapper.** Retrieval, chunking, embeddings, a vector
  store, grounded generation, citations, and semantic dedup are all implemented
  here rather than delegated to a hosted "chat with your docs" API.
- **A hallucination guard with a measurable effect.** Every candidate fact is
  embedded and compared against the retrieved chunks; unsupported claims are
  rejected. The [evaluation harness](#evaluation) quantifies how well it works.
- **Grounded question answering that knows when to abstain.** An extractive
  QA endpoint (`/api/ask`) retrieves, re-ranks candidate sentences (with a
  lead/definition boost and a negation penalty), and **abstains with no answer**
  when nothing clears the grounding threshold — the QA counterpart of the guard.
- **Offline-first and deterministic.** The default hashing embedder makes the
  pipeline and the whole test suite reproducible with zero external dependencies,
  which is exactly what you want for CI and for someone reviewing the repo.
- **Pluggable backends.** Embedder (`hashing` / `sbert` / `openai`), vector store
  (NumPy / FAISS), and generation (extractive / LLM) are all swappable via env
  vars behind stable interfaces.
- **Productionish shape.** Clean REST API, a decoupled delivery worker, a Svelte
  single-page frontend, a pytest suite, an evaluation harness, and a multi-stage
  Docker build.

---

## Architecture

```
                            ┌──────────────────────────┐
   Browser ───────────────▶ │  Svelte SPA (frontend/)   │
                            └────────────┬─────────────┘
                                         │  /api/*  (JSON)
                            ┌────────────▼─────────────┐
                            │   FastAPI app (api.py)    │
                            │  preview · ask · digest · │
                            │  retrieve · subscribe ·   │
                            │  run-due                  │
                            └───┬───────────────┬───────┘
                                │               │
                 ┌──────────────▼───┐     ┌─────▼──────────────┐
                 │  RAG core (rag/) │     │ Services (services/)│
                 │                  │     │                     │
                 │ fetch → chunk →  │     │  db (SQLite)        │
                 │ embed → index →  │     │  dedup (spaced rep) │
                 │ retrieve →       │     │  emailer (SMTP/dry) │
                 │ generate → GUARD │     │  worker (scheduler) │
                 └──────────────────┘     └─────────────────────┘
```

The **delivery worker** is a standalone process that shares the SQLite database
with the API. It periodically curates fresh, de-duplicated facts for each due
subscription and (dry-run by default) emails them.

---

## Repository structure

```
backend/
  api.py                 FastAPI app: REST API under /api + serves the built SPA
  rag/                   The RAG pipeline
    fetcher.py           Fetch Wikipedia sources (with disambiguation handling)
    chunking.py          Split text into overlapping word-window chunks
    embeddings.py        Pluggable embedder: hashing (default) / sbert / openai
    vectorstore.py       Vector index: NumPy always, optional FAISS IndexFlatIP
    ingest.py            fetch → chunk → embed → index (with a process cache)
    curator.py           retrieve → generate → hallucination guard → dedup
  services/
    db.py                SQLite: subscriptions + sent-facts (for dedup)
    dedup.py             Spaced-repetition semantic dedup of already-sent facts
    emailer.py           SMTP delivery with a demo-friendly dry-run mode
    worker.py            Standalone delivery scheduler (run_once / run_forever)
frontend/                Svelte 5 + Vite single-page app (consumes /api/*)
  src/
    App.svelte           App shell: tabs (Explore · Ask · Subscribe · Subscriptions)
    app.css              Design tokens + all component styling (light/dark)
    config.js            Frontend config (API base URL)
    lib/api.js           fetch wrapper for the JSON API
    lib/history.svelte.js  Local search history with named folders (runes store)
    lib/theme.svelte.js    Light/dark theme store (persisted)
    lib/components/
      CuratePanel.svelte      Explore: grounded facts for a topic (1–50)
      AskPanel.svelte         Ask: extractive QA with citations + abstain state
      FactCard.svelte         A single grounded fact with its citation
      SubscribePanel.svelte   Create a subscription
      SubscriptionsPanel.svelte  List subs + preview the email digest (demo mode)
      DidYouMean.svelte       Disambiguation suggestions
      HistoryList.svelte      Recent searches grouped into named folders
      PipelineBadges.svelte   Live "which backends are active" badges
      BackToTop.svelte        Floating scroll-to-top button
      Tabs.svelte             Accessible tab bar
eval/eval.py             Offline evaluation harness (retrieval + guard metrics)
tests/                   Offline pytest suite (no network, no API key)
Dockerfile               Multi-stage: build the SPA, then the Python backend
docker-compose.yml       api + worker services sharing a DB volume
```

---

## How the RAG pipeline works

1. **Fetch** (`rag/fetcher.py`) — pull a few Wikipedia pages for the topic,
   resolving disambiguation pages to concrete articles.
2. **Chunk** (`rag/chunking.py`) — split each source into overlapping word
   windows so retrieval can return focused passages.
3. **Embed** (`rag/embeddings.py`) — turn chunks into L2-normalized dense
   vectors. The default `hashing` backend uses signed feature hashing over
   unigrams+bigrams (numpy-only, deterministic). `sbert` and `openai` backends
   are available for higher quality.
4. **Index + retrieve** (`rag/vectorstore.py`) — cosine similarity search over
   the chunk vectors (FAISS `IndexFlatIP` when available, NumPy otherwise).
5. **Generate** (`rag/curator.py`) — with an OpenAI key, an LLM is constrained to
   the retrieved context and asked for JSON facts with source indices. Without a
   key, a deterministic extractive fallback selects the most on-topic sentences.
6. **Hallucination guard** (`rag/curator.py`) — every candidate fact is embedded
   and scored against the retrieved evidence; facts below the grounding threshold
   are dropped, and the best-matching chunk is attached as a citation.
7. **Dedup** (`rag/curator.py`, `services/dedup.py`) — near-duplicate facts are
   removed, and facts already emailed to a subscriber are suppressed on future
   sends.

---

## Question answering (the "Ask" tab)

Beyond curating a list of facts, `GET /api/ask?question=` answers a natural
question **extractively** — it never generates free text. It resolves the most
relevant article, retrieves passages, splits them into candidate sentences, and
scores each against the question.

Two design choices keep answers honest with only a lexical embedder:

- a **negation filter** — for an affirmative question, sentences that negate
  (e.g. *"…does **not** produce oxygen"*) are dropped from contention, because
  they share the question's keywords yet make poor answers; the filter falls
  back to negated sentences only if every candidate is negated;
- the answer is simply the **highest-scoring remaining sentence**, so the
  reported `confidence` is that sentence's own similarity and never an inflated
  or borrowed score.

QA compares a whole question against a single sentence, which shares fewer exact
tokens than a sentence does with its broad topic, so it uses a slightly lower
grounding bar than fact extraction (`QA_GROUNDING_THRESHOLD`, default `0.15`). If
nothing clears that bar the endpoint **abstains** (`grounded: false`, `answer:
null`) instead of guessing. Every answer ships with the source title/URL and a
short list of **citations** (the passages it drew from), so the UI can always
show *why*.

> **Honest by design.** Because answers are extracted verbatim, a correct answer
> phrased differently from the question (e.g. *"releases oxygen"* for *"what does
> photosynthesis produce?"*) can still read as **Low** confidence. That is the
> lexical embedder being cautious, not a wrong answer — the app would rather
> under-claim than overstate.

## Demo mode (email without SMTP)

With no SMTP configured the emailer runs in **dry-run** mode: digests are composed
but not sent. `GET /api/info` exposes this as `email_dry_run`, the UI shows a
"demo mode" banner, and `GET /api/digest?topic=` lets you **preview the exact
email** (subject + body) a subscription would receive — so the feature is fully
demonstrable offline.

---

## Running locally

### 1. Backend (API + pipeline)

```powershell
# From the repo root. Python 3.10+.
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -e .                       # installs the backend package + deps

uvicorn backend.api:app --reload --port 8000
```

- API docs (Swagger UI): <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/api/health>

### 2. Frontend (Svelte SPA)

**Development** (hot reload, proxies `/api` to the backend on port 8000):

```powershell
cd frontend
npm install
npm run dev            # http://localhost:5173
```

**Production build** (emits `frontend/dist`, which the backend serves at `/`):

```powershell
cd frontend
npm run build
```

Then open the app at the backend root: <http://127.0.0.1:8000/>.

### 3. Delivery worker (optional)

```powershell
python -m backend.services.worker      # polls for due subscriptions
```

---

## Running with Docker

The multi-stage build compiles the Svelte app and bakes it into the Python image;
`docker-compose` runs the API and the worker sharing one database volume.

```bash
docker compose up --build
# API + UI on http://localhost:8000
```

---

## Configuration

Every setting is optional. With no configuration the app runs fully offline.
Copy `.env.example` to `.env` to change any of them.

| Variable              | Default            | Purpose                                              |
| --------------------- | ------------------ | ---------------------------------------------------- |
| `EMBED_BACKEND`       | `hashing`          | `hashing` (offline) · `sbert` · `openai`             |
| `GROUNDING_THRESHOLD` | `0.20`             | Min cosine similarity for a fact to pass the guard   |
| `QA_GROUNDING_THRESHOLD` | `0.15`          | Min similarity for an **Ask** answer (see note below) |
| `USE_FAISS`           | `auto`             | `auto` · `faiss` · `numpy`                           |
| `OPENAI_API_KEY`      | *(unset)*          | Enables grounded LLM generation                      |
| `OPENAI_MODEL`        | `gpt-4o-mini`      | Model used when an API key is set                    |
| `USE_LLM`             | `auto`             | Set `0` to force the offline extractive fallback     |
| `EMAIL_DRY_RUN`       | `1` (when no SMTP) | Log emails instead of sending                        |
| `SMTP_HOST` / `PORT` / `USER` / `PASS` / `FROM_EMAIL` | *(unset)* | Real email delivery |
| `ADMIN_TOKEN`         | *(unset)*          | If set, subscription **management** routes require `X-Admin-Token` |
| `DB_PATH`             | `subscriptions.db` | SQLite location (shared by API + worker in Docker)   |
| `SCHED_POLL_INTERVAL` | `60`               | Worker poll interval, seconds                        |

> **Note on the grounding thresholds.** `0.20` is tuned for the lexical hashing
> embedder; neural backends produce a different score distribution and generally
> warrant ~`0.35`. Question answering uses the lower `QA_GROUNDING_THRESHOLD`
> (`0.15`) because a question and a single answer sentence share fewer exact
> tokens than a sentence shares with its broad topic.

---

## API

All endpoints are under `/api`.

| Method + path                   | Description                                        |
| ------------------------------- | -------------------------------------------------- |
| `GET /api/health`               | Liveness check                                     |
| `GET /api/info`                 | Active backends (embedder, vectors, generation) + email dry-run flag |
| `GET /api/preview?topic=&max_facts=` | Grounded, cited facts for a topic (1–50)      |
| `GET /api/ask?question=`        | Extractive **question answering** with citations; abstains when nothing is grounded |
| `GET /api/digest?topic=&max_facts=` | Preview the email digest for a topic (subject + body) |
| `GET /api/retrieve?topic=&k=`   | Raw retrieved chunks (the "why" behind a fact)     |
| `POST /api/subscribe`           | Create a subscription (`email`, `topic`, `cadence`)|
| `GET /api/subscriptions`        | List subscriptions *(admin)* — emails are masked   |
| `DELETE /api/subscriptions/{id}`| Delete a subscription *(admin)*                    |
| `POST /api/run-due`             | Trigger one delivery pass for due subscriptions *(admin)* |

Routes marked *(admin)* require an `X-Admin-Token` header **only when**
`ADMIN_TOKEN` is set; unset leaves them open for local demoing. The list route
masks addresses (`a***@example.com`) so it never echoes back a full email.

> **Known limitation — recurring digests are effectively one-shot.** Curation is
> deterministic and each digest dedupes against what a subscriber has already
> seen, so after the first send a topic yields `no_new_facts` until its source
> article changes. This keeps the demo honest (no fabricated "new" facts) rather
> than simulating a live feed.

---

## Testing

The suite is fully offline (hashing embedder, injected sources, dry-run email):

```powershell
pip install -e ".[dev]"
pytest -q
```

Covers chunking, embeddings, the vector store, retrieval + the hallucination
guard, semantic dedup, the database layer, and the API endpoints.

---

## Evaluation

`eval/eval.py` measures the pipeline against small, hand-labeled seed corpora,
fully offline and deterministically:

```powershell
python -m eval.eval
```

It reports **retrieval hit-rate@k** and the **hallucination-guard** quality
(precision / recall / F1) plus the mean grounding score for supported vs.
unsupported facts. Representative results with the default `hashing` embedder:

```
retrieval hit-rate@5         : 100%
guard precision              : 67%
guard recall                 : 80%
guard F1                     : 0.73
mean grounding (supported)   : 0.354
mean grounding (unsupported) : 0.178
separation margin            : 0.177
```

The corpora deliberately include **hard negatives** — false claims that reuse the
topic's subject (e.g. *"Alan Turing discovered penicillin"*). A purely lexical
embedder cannot fully separate those, which is exactly why the embedder is
pluggable: switching to `EMBED_BACKEND=sbert` widens the separation margin and
raises guard precision.

---

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn, NumPy, FAISS, SQLite, OpenAI SDK
  (optional), sentence-transformers (optional).
- **Frontend:** Svelte 5 + Vite (vanilla, no UI framework bloat).
- **Tooling:** pytest, Docker (multi-stage), docker-compose.

---

## License

Released under the [MIT License](LICENSE).

---

## Resume-oriented highlights

- Designed and implemented an end-to-end **RAG pipeline** (fetch → chunk → embed
  → vector search → grounded generation → citation) with a **hallucination guard**
  that rejects unsupported claims, plus an **offline evaluation harness** that
  quantifies retrieval hit-rate and guard precision/recall.
- Added **grounded extractive question answering** with citations and an
  **abstain path** — a custom sentence re-ranker (lead-definition boost + negation
  penalty) on top of retrieval, so weak matches are surfaced honestly rather than
  hallucinated.
- Built the system **offline-first and deterministic** via a numpy-only
  feature-hashing embedder, with pluggable `sbert`/`openai` backends behind a
  stable interface — enabling a fully reproducible test + eval suite.
- Shipped a clean **FastAPI** service, a decoupled **delivery worker**, and a
  polished **Svelte 5 + Vite** single-page frontend (light/dark theming,
  customizable fact count, foldered search history, digest preview, back-to-top),
  all containerized with a **multi-stage Docker build** and orchestrated via
  **docker-compose**.
