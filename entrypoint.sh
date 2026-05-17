#!/usr/bin/env bash
# Orchestrate Postgres + Ollama + FastAPI + Streamlit in a single container.
# Intended for HuggingFace Space (Docker SDK).

set -euo pipefail

PGUSER="${POSTGRES_USER:-postgres}"
PGDB="${POSTGRES_DB:-legal_ai}"
PGPW="${POSTGRES_PASSWORD:-demo}"
PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PG_BIN="/usr/lib/postgresql/16/bin"

export DATABASE_URL="${DATABASE_URL:-postgresql://${PGUSER}@localhost:5432/${PGDB}}"
export LLM_PROVIDER="${LLM_PROVIDER:-openrouter}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export EMBEDING_MODEL="${EMBEDING_MODEL:-bge-m3}"
export API_URL="${API_URL:-http://localhost:8000}"

log() { echo "[entrypoint] $*"; }

# ─── 1. Postgres : init + start ─────────────────────────────────────────────
if [ ! -s "${PGDATA}/PG_VERSION" ]; then
    log "Initialising Postgres cluster in ${PGDATA}"
    # Use trust auth — no password needed for local connections inside the container.
    su postgres -c "${PG_BIN}/initdb -D ${PGDATA} --auth=trust --username=${PGUSER}"
    echo "listen_addresses='localhost'" >> "${PGDATA}/postgresql.conf"
fi

log "Starting Postgres"
su postgres -c "${PG_BIN}/pg_ctl -D ${PGDATA} -l /tmp/postgres.log -w start"

# Create DB + extension on first boot
if ! su postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${PGDB}'\"" | grep -q 1; then
    log "Creating database '${PGDB}' + pgvector extension"
    su postgres -c "createdb ${PGDB}"
    su postgres -c "psql -d ${PGDB} -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
fi

# Apply schema migrations (dependency order: init first, then numbered, then fix)
log "Applying schema migrations from /app/sql/"
PSQL_CMD="psql --no-psqlrc -d ${PGDB}"
for sql_file in \
    /app/sql/init.sql \
    $(ls /app/sql/[0-9]*.sql 2>/dev/null | sort) \
    /app/sql/fix_rename_content_column.sql; do
    [ -e "$sql_file" ] || continue
    log "  → $(basename "$sql_file")"
    su postgres -c "${PSQL_CMD} -f $sql_file" \
        > /dev/null 2>&1 || log "    (skipped or already applied)"
done

# Restore the seed dump if present and DB is empty
if [ -f /app/sql/seed_chunks.sql.gz ]; then
    CHUNK_COUNT=$(su postgres -c "psql --no-psqlrc -d ${PGDB} -tAc 'SELECT count(*) FROM document_chunks'" \
        2>/dev/null || echo "0")
    if [ "${CHUNK_COUNT}" = "0" ]; then
        log "Restoring seed dump (this may take a few minutes)"
        gunzip -c /app/sql/seed_chunks.sql.gz | su postgres -c "psql --no-psqlrc -d ${PGDB}" || true
        FINAL=$(su postgres -c "psql --no-psqlrc -d ${PGDB} -tAc 'SELECT count(*) FROM document_chunks'" \
            2>/dev/null || echo "0")
        log "Seed restored : ${FINAL} chunks"
    else
        log "Database already contains ${CHUNK_COUNT} chunks → skipping restore"
    fi
else
    log "⚠️  /app/sql/seed_chunks.sql.gz not found — DB will be empty"
fi

# ─── 2. Ollama : start in background ────────────────────────────────────────
log "Starting Ollama"
nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready
for i in $(seq 1 30); do
    curl -sf http://localhost:11434/api/tags > /dev/null && break
    sleep 1
done
log "Ollama ready (PID ${OLLAMA_PID})"
# Point the Python client to localhost — OLLAMA_HOST was set to 0.0.0.0:11434
# in the Dockerfile so that ollama serve binds to all interfaces; we override
# it here so OllamaLLMClientAdapter connects to the right address.
export OLLAMA_HOST="http://localhost:11434"

# ─── 3. FastAPI : start in background ───────────────────────────────────────
log "Starting FastAPI on :8000"
nohup uvicorn legal_ai.api.main:app --host 0.0.0.0 --port 8000 \
    > /tmp/api.log 2>&1 &
API_PID=$!

# Wait for API health
for i in $(seq 1 60); do
    curl -sf http://localhost:8000/health > /dev/null && break
    sleep 1
done
log "FastAPI ready (PID ${API_PID})"

# ─── 4. Streamlit : foreground on :7860 ─────────────────────────────────────
log "Starting Streamlit on :7860"
exec streamlit run /app/frontend/streamlit_app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
