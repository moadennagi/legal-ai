# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (development) — project uses uv as package manager
uv pip install -e .
# or: pip install -e .

# Linting
ruff check

# Type checking
mypy src/

# Tests
pytest
pytest --cov                              # with coverage
pytest tests/test_crawler.py              # single file
pytest -k test_function_name             # single test by name
```

## Architecture Overview

This is a **document ingestion and RAG pipeline** for Moroccan legal documents (Bulletin Officiel from sgg.gov.ma). The pipeline runs in four sequential phases:

### Phase 1: Ingestion (`pipeline/ingestion.py`)
- `SGGCrawler` scrapes the Moroccan government site and returns `TargetPayload` records
- `DataIngesion` (intentional class name — note the typo) stores these as `Target` rows, then downloads PDFs concurrently (semaphore-limited to `SEMAPHORE` env var, default 10)
- Downloaded PDFs land in `FILE_PATH` (env var); file paths are stored on `Document` records

### Phase 2: Text Extraction (`pipeline/processing.py`)
- `DocumentProcessing` uses **Docling** to convert PDFs → Markdown (OCR disabled)
- Extracted Markdown is stored in `Document.text_content`

### Phase 3: Embedding (`pipeline/embedding.py`)
- `DocumentEmbedding` first runs `fix_heading_hierarchy()` (`crawlers/sgg_heading_rules.py`) on the extracted Markdown to enforce Moroccan legal document hierarchy: DAHIR → Loi → Décret → Chapitre → Article (Articles become bold paragraphs, not headings)
- Then chunks with `MarkdownHeaderTextSplitter` (6 header levels: division/instrument/partie/titre/chapitre/section) followed by `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
- Creates 768-dim vectors via Ollama (`nomic-embed-text`), stored in `document_chunks` with pgvector

### Phase 4: RAG (`pipeline/rag.py`)
- `RAG` embeds a query, retrieves top-10 chunks by cosine distance (≤ threshold, default 0.3), deduplicates preferring recent docs (official_date DESC), augments a French-language prompt, and calls `dolphin-llama3`
- Each chunk is formatted with a breadcrumb: `[instrument > partie > titre > chapitre > section]`

### Data Layer
- **ORM**: SQLAlchemy 2.x (`models/document.py`); session management via `get_session()` context manager in `database.py` (auto-commit/rollback)
- **Pydantic DTOs**: `models/schemas.py` — `TargetPayload`, `TaskPayload`, `SourcePayload` decouple transport objects from ORM models
- **Repositories** (`repositories/`): thin data-access wrappers — `document.py`, `source.py`, `task.py`, `target.py`
- **pgvector**: `document_chunks.embedding` column (768-dim), IVFFLAT index (100 lists) with cosine ops
- **Timestamps**: all `created_at`/`updated_at` fields are Unix epoch integers, not datetime objects

### Key Relationships
```
sources → tasks, targets, documents
documents → document_chunks (1:N)
targets → documents (1:1)
tasks → tasks (self-join for task hierarchy)
```
Unique constraints: `documents(source_id, number)`, `targets(source_id, number)`, `document_chunks(document_id, chunk_index)`.

### Interfaces (`interfaces.py`)
- `CrawlerInterface`: async `crawl_and_return_targets(task_id)` → `list[TargetPayload]`
- `DownloaderInterface`: async `download_document(url, http_session)` → `bytes`

New crawlers must implement `CrawlerInterface`; new downloaders must implement `DownloaderInterface`.

## Infrastructure

**PostgreSQL with pgvector** is required:

```bash
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -e POSTGRES_DB=legal_ai \
  postgres
# Then install pgvector extension inside the container
```

Apply migrations in order from the `sql/` directory. Migration filenames are prefixed with a Unix timestamp.

**Ollama** must be running locally with the models used:
- `nomic-embed-text` — embeddings (768-dim)
- `dolphin-llama3` — answer generation

## Configuration

Copy `env.example` to `.env`:

```
DATABASE_URL=postgresql://postgres:mysecretpassword@localhost:5432/legal_ai
FILE_PATH=/path/to/data      # where PDFs are saved
SEMAPHORE=10                 # concurrent download limit
```

Settings are loaded via `settings.py` (Pydantic BaseSettings).

## Linting Configuration

Ruff is configured in `pyproject.toml`: 100-char line length, Python 3.10+ target rules. Mypy uses Python 3.10, strict `warn_return_any`. Runtime requires Python 3.12+.
