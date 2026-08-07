# syntax=docker/dockerfile:1

# ---- Stage 1: build the Svelte frontend into static files ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
# Install deps first (better layer caching), then build.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend that also serves the built frontend ----
FROM python:3.11-slim AS backend
WORKDIR /app

# Install Python deps via the editable install so `backend.*` is importable and
# api.py can resolve ../frontend/dist relative to the source tree.
COPY pyproject.toml requirements.txt ./
COPY backend/ ./backend/
COPY eval/ ./eval/
RUN pip install --no-cache-dir -e .

# Bring in the compiled frontend from stage 1.
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Store the SQLite database on a mounted volume by default.
ENV DB_PATH=/data/subscriptions.db
ENV EMAIL_DRY_RUN=1
VOLUME ["/data"]

EXPOSE 8000
# Shell form so hosts that inject a $PORT (Render/Railway/Fly) are honored;
# falls back to 8000 for local `docker run` and docker-compose.
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
