# Deploying Grounded Facts

The app is a single container: a multi-stage `Dockerfile` builds the Svelte SPA
and bakes it into the FastAPI image, which serves **both** the API (`/api/*`) and
the UI (`/`). It binds the `$PORT` the host provides (falling back to `8000`) and
exposes a health check at `/api/health`.

By default it runs **fully offline** — the deterministic hashing embedder means
no API key, no model download, and a small enough image for a free 512 MB tier.

---

## Option A — Render (one-click, durable) ✅ recommended

A [`render.yaml`](./render.yaml) blueprint is included, so a public URL is a few
clicks away. The button pre-loads this repo — just sign in with GitHub and click
**Apply**:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Alexfosuaa/Grounded-facts)

Or do it manually:

1. Push this repo to GitHub (already done).
2. Go to <https://render.com> → **New ▸ Blueprint** and connect the repo.
3. Render reads `render.yaml`, builds the Dockerfile, and deploys a **free** web
   service. Click **Apply** and wait for the first build (a few minutes).
4. Your app is live at `https://grounded-facts-XXXX.onrender.com`.

Notes:
- Free services **spin down after ~15 min idle**; the next request cold-starts in
  under a minute.
- SQLite is on the container's ephemeral disk, so subscriptions reset on
  redeploy/spin-down. To persist them, add a paid Render **disk** mounted at
  `/data` and set `DB_PATH=/data/subscriptions.db`.
- To enable grounded **LLM** facts, add an `OPENAI_API_KEY` env var in the
  dashboard (kept out of the repo).

---

## Option B — Any Docker host (Fly.io, Railway, a VM)

The image needs no host-specific config; it already honors `$PORT`.

```bash
docker build -t grounded-facts .
docker run -p 8000:8000 -e PORT=8000 grounded-facts
# open http://localhost:8000
```

- **Fly.io:** `fly launch` (it detects the Dockerfile), then `fly deploy`.
- **Railway:** New Project ▸ Deploy from Repo — it builds the Dockerfile
  automatically.

Add a persistent volume mounted at `/data` (and `DB_PATH=/data/subscriptions.db`)
if you want subscriptions to survive restarts.

---

## Option C — Instant temporary link (no account, for a quick share)

To share a **locally running** server without deploying, expose it with a
Cloudflare Quick Tunnel (no login, no account):

```bash
# 1) run the app
uvicorn backend.api:app --host 0.0.0.0 --port 8000
# 2) in another shell, tunnel it — prints a https://<random>.trycloudflare.com URL
cloudflared tunnel --url http://localhost:8000
```

The link stays up only while both processes run on your machine, so it's great
for a demo but not a real deployment — use Option A for anything lasting.

---

## Configuration

Every setting is optional (see the table in [`README.md`](./README.md#configuration)).
The ones that matter most for a public deploy:

| Variable         | Why it matters in production                                  |
| ---------------- | ------------------------------------------------------------- |
| `ADMIN_TOKEN`    | **Set this.** Gates the subscriber list/delete/run-due routes |
| `OPENAI_API_KEY` | Enables grounded LLM generation (otherwise extractive)        |
| `EMBED_BACKEND`  | `hashing` (default, light) · `sbert` (needs the `ml` extra)   |
| `SMTP_*`         | Real email delivery (otherwise dry-run/logged)                |
| `DB_PATH`        | Point at a mounted volume to persist subscriptions            |
