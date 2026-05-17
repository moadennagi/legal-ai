---
title: Legal AI — RAG Démo
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Démo RAG souverain sur le Bulletin Officiel marocain
---

# Legal AI — démo HuggingFace Space

Démo accompagnant le mémoire de master de **Moad Ennagi** (v8_final).
Pipeline RAG complet sur le Bulletin Officiel du Royaume du Maroc :

- **Embedding** : `bge-m3` (Ollama, CPU local au container)
- **Retrieval** : pgvector (IVFFLAT, 10 probes) + cross-encoder reranking
- **Génération** : Mistral 7B via [Together AI](https://together.ai)
- **UI** : Streamlit avec 3 onglets (démo, résultats RAGAS, méthodologie)

## Architecture

Un seul container regroupe :

| Service       | Port  | Rôle                                 |
|---------------|-------|--------------------------------------|
| Streamlit     | 7860  | UI (exposée publiquement par HF)     |
| FastAPI       | 8000  | API OpenAI-compatible (interne)      |
| Ollama        | 11434 | Embeddings bge-m3 (interne)          |
| Postgres 16   | 5432  | pgvector + données chargées au boot  |

## Configuration des secrets HF

Dans **Settings → Variables and secrets** du Space :

| Clé                | Type   | Valeur                                       |
|--------------------|--------|----------------------------------------------|
| `TOGETHER_API_KEY` | secret | Ta clé Together AI                           |
| `LLM_PROVIDER`     | var    | `together` (ou `groq` en backup)             |
| `GROQ_API_KEY`     | secret | (optionnel, backup)                          |
| `POSTGRES_PASSWORD`| secret | mot de passe libre (utilisé en interne)      |

## Build local pour tester avant de pousser

```bash
docker build -f Dockerfile.hfspace -t legal-ai-demo .
docker run -p 7860:7860 \
  -e TOGETHER_API_KEY=sk-... \
  -e LLM_PROVIDER=together \
  legal-ai-demo
```

Ouvre <http://localhost:7860>.

## Données pré-chargées

Le dump SQL `sql/seed_chunks.sql.gz` (versionné dans le repo) contient un
sous-ensemble du corpus pré-embedé. Il est restauré automatiquement au premier
boot. Pour le régénérer :

```bash
./scripts/dump_for_demo.sh
```

## Limites assumées

- **Latence** : ~10-30 sec/requête sur CPU (embedding bge-m3 + retrieval +
  rerank + appel Together AI). Acceptable pour une démo de soutenance.
- **Cold start** : la première requête après un long inactif peut prendre
  jusqu'à 60 sec (chargement du cross-encoder en mémoire).
- **Pas de persistance** : si le Space redémarre, la DB est restaurée depuis
  le dump versionné (pas de perte de données mais pas d'écriture utilisateur).
