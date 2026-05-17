#!/usr/bin/env bash
# Dump le sous-ensemble de données nécessaire à la démo HuggingFace Space.
#
# Produit : sql/seed_chunks.sql.gz (33 MB, versionnage direct sans Git LFS)
#
# Pré-requis : Docker Desktop actif, container postgres en cours d'exécution
#
# Usage :
#   ./scripts/dump_for_demo.sh [CONTAINER_NAME]   (défaut : legal_ai_postgres)

set -euo pipefail

cd "$(dirname "$0")/.."

CONTAINER="${1:-legal_ai_postgres}"
PGUSER="${POSTGRES_USER:-postgres}"
PGDB="${POSTGRES_DB:-legal_ai}"
OUT_FILE="sql/seed_chunks.sql"

# Cherche le container postgres (le nom peut varier)
if ! docker inspect "${CONTAINER}" > /dev/null 2>&1; then
    # Fallback : chercher n'importe quel container postgres actif
    CONTAINER=$(docker ps --filter "ancestor=postgres" --filter "ancestor=pgvector/pgvector:pg17" \
        --format "{{.Names}}" 2>/dev/null | head -1)
    if [[ -z "${CONTAINER}" ]]; then
        CONTAINER=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -i postgres | head -1)
    fi
    if [[ -z "${CONTAINER}" ]]; then
        echo "❌ Aucun container postgres trouvé. Lance: docker-compose up -d postgres"
        exit 1
    fi
fi

echo "📦 Container postgres détecté : ${CONTAINER}"

# Vérifier que la DB est joignable
if ! docker exec "${CONTAINER}" psql -U "${PGUSER}" -d "${PGDB}" -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ DB '${PGDB}' inaccessible dans le container ${CONTAINER}"
    exit 1
fi

echo "📊 État du corpus :"
docker exec "${CONTAINER}" psql -U "${PGUSER}" -d "${PGDB}" -c \
    "SELECT (SELECT count(*) FROM sources) AS sources, (SELECT count(*) FROM documents) AS documents, (SELECT count(*) FROM document_chunks) AS chunks;"

echo "📦 Génération du dump..."
mkdir -p sql
docker exec "${CONTAINER}" pg_dump -U "${PGUSER}" -d "${PGDB}" \
    --no-owner --no-privileges --data-only \
    --table=sources --table=documents --table=document_chunks \
    > "${OUT_FILE}"

gzip -f "${OUT_FILE}"

SIZE_MB=$(du -m "${OUT_FILE}.gz" | cut -f1)
echo "✅ Dump créé : ${OUT_FILE}.gz (${SIZE_MB} MB)"

if [[ "${SIZE_MB}" -gt 100 ]]; then
    echo ""
    echo "⚠️  Dump > 100 MB → configure Git LFS avant de commit :"
    echo "    git lfs install"
    echo "    git lfs track 'sql/seed_chunks.sql.gz'"
    echo "    git add .gitattributes ${OUT_FILE}.gz"
fi
