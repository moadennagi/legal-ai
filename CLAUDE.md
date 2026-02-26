# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (development) — project uses uv as package manager
uv pip install -e .

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
- `DocumentProcessing` uses **Docling** via `DoclingDocumentConverterAdapter` (`adapters.py`) to convert PDFs → Markdown (OCR disabled)
- Extracted Markdown is stored in `Document.text_content`

### Phase 3: Embedding (`pipeline/embedding.py`)
- `DocumentEmbedding` calls `BODocumentSplitter.split_document()` which internally runs `_fix_heading_hierarchy()` to enforce the Moroccan legal hierarchy, then chunks with `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` (chunk_size=1500, overlap=300)
- Creates vectors via Ollama, stored in `document_chunks` with pgvector

### Phase 4: RAG (`pipeline/rag.py`)
- `RAG` uses **HyDE** (Hypothetical Document Embedding): embeds the user query AND a generated hypothetical answer, retrieves chunks for both, then cross-encoder **reranks** all candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` (via HuggingFace `transformers` + `torch`)
- Returns top-k chunks formatted with breadcrumbs: `[instrument > partie > titre > chapitre > section]`
- IVFFLAT probes set to 10 per query for better recall

### Phase 5: Conversation (`pipeline/conversation.py`)
- `ConversationManager` wraps `RAG` and maintains a sliding-window history
- When token count (approximated as `len(message) // 4`) exceeds 2000, it compresses the history by asking the LLM to summarize and keeping the last 4 messages

### Splitter (`splitters/bo_splitter.py`)
`BODocumentSplitter` implements `DocumentSplitterInterface` and contains the full heading normalization logic:
- **Keyword headings**: fixed legal vocabulary (DAHIR, Loi, Décret, Chapitre, etc.) → level assigned by `_KEYWORD_RULES`
- **Free-text headings**: inferred one level below the last known keyword heading, capped at H6
- **Articles**: converted to `**bold**` paragraphs (not headings)
- `_filter_chunks()` drops empty, very short (<50 chars), SOMMAIRE, and table-like chunks

### Interfaces (`interfaces.py`)
All major components program to interfaces:
- `CrawlerInterface` / `DownloaderInterface` — ingestion
- `DocumentConverterInterface` — PDF → Markdown
- `DocumentSplitterInterface` — chunking + breadcrumb formatting
- `LLMClientInterface` — `embeddings()` (async) + `chat()` (sync)
- `RAGInterface` — `ask(user_query, similarity_threshold, history)`
- `EmbeddingServiceInterface` — `split_and_insert_embeddings()`

Concrete implementations: `OllamaLLMClientAdapter` and `DoclingDocumentConverterAdapter` in `adapters.py`.

### Data Layer
- **ORM**: SQLAlchemy 2.x (`models/document.py`); session management via `get_session()` context manager in `database.py` (auto-commit/rollback)
- **Pydantic DTOs**: `models/schemas.py` — `TargetPayload`, `TaskPayload`, `SourcePayload` decouple transport objects from ORM models
- **Repositories** (`repositories/`): thin data-access wrappers — `document.py`, `source.py`, `task.py`, `target.py`
- **pgvector**: `document_chunks.embedding` column, IVFFLAT index (100 lists) with cosine ops
- **Timestamps**: all `created_at`/`updated_at` fields are Unix epoch integers, not datetime objects

### Key Relationships
```
sources → tasks, targets, documents
documents → document_chunks (1:N)
targets → documents (1:1)
tasks → tasks (self-join for task hierarchy)
```
Unique constraints: `documents(source_id, number)`, `targets(source_id, number)`, `document_chunks(document_id, chunk_index)`.

## Configuration

Copy `env.example` to `.env`. Key settings (with defaults from `settings.py`):

```
DATABASE_URL=postgresql://postgres:mysecretpassword@localhost:5432/legal_ai
FILE_PATH=/path/to/data           # where PDFs are saved
SEMAPHORE=10                      # concurrent download limit
EMBEDING_MODEL=bge-m3             # note: intentional typo in settings key
GENERATION_MODEL=qwen2.5:7b
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Infrastructure

**PostgreSQL with pgvector** is required. Apply migrations in order from the `sql/` directory (filenames are Unix-timestamp-prefixed).

**Ollama** must be running locally with the models configured in `.env` (defaults: `bge-m3` for embeddings, `qwen2.5:7b` for generation).

**HuggingFace** reranking model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is downloaded automatically on first `RAG` instantiation.

## Linting Configuration

Ruff is configured in `pyproject.toml`: 100-char line length, Python 3.10+ target rules. Mypy uses Python 3.10, strict `warn_return_any`. Runtime requires Python 3.12+.
