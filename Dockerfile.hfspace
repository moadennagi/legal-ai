# Monolithic image for HuggingFace Spaces (Docker SDK).
# Bundles: Postgres 16 + pgvector, Ollama (bge-m3 only), FastAPI, Streamlit.
# Heavy ingestion deps (docling, unstructured) are excluded to keep the image
# under ~3 GB. See requirements-hfspace.txt.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEBIAN_FRONTEND=noninteractive \
    PGDATA=/var/lib/postgresql/data \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/root/.ollama/models

# ─── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg lsb-release \
        build-essential libpq-dev \
        supervisor procps zstd \
    && echo "deb http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/postgresql.gpg \
    && apt-get update && apt-get install -y --no-install-recommends \
        postgresql-16 postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*

# ─── Ollama ─────────────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ─── Python deps (cached layer before code) ─────────────────────────────────
WORKDIR /app
COPY requirements-hfspace.txt /app/requirements-hfspace.txt
RUN pip install --no-cache-dir -r /app/requirements-hfspace.txt

# ─── Pre-pull bge-m3 at build time (avoids ~3 min cold start at boot) ───────
RUN ollama serve & \
    SERVER_PID=$! ; \
    for i in $(seq 1 30); do \
        curl -sf http://localhost:11434/api/tags > /dev/null && break ; \
        sleep 1 ; \
    done ; \
    ollama pull bge-m3 ; \
    kill $SERVER_PID ; \
    wait $SERVER_PID 2>/dev/null || true

# ─── Application code ───────────────────────────────────────────────────────
COPY src /app/src
COPY frontend/streamlit_app.py /app/frontend/streamlit_app.py
COPY evals/summary.json /app/evals/summary.json
COPY sql /app/sql
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ─── Init Postgres data dir + permissions ───────────────────────────────────
# HuggingFace Spaces runs as user 1000 (no root). Postgres needs to own PGDATA.
RUN mkdir -p ${PGDATA} /var/run/postgresql /root/.ollama \
    && chown -R postgres:postgres ${PGDATA} /var/run/postgresql \
    && chmod 700 ${PGDATA}

# Streamlit listens on the port HuggingFace exposes.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:7860/_stcore/health || exit 1

CMD ["/app/entrypoint.sh"]
