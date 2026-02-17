# Legal AI

Document ingestion pipeline for Moroccan legal documents (Bulletin Officiel). Downloads PDFs for RAG applications.

## Setup

```bash
# Install
pip install -e .

# Start Postgres
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -e POSTGRES_DB=legal_ai postgres

# Run migrations
psql -U postgres -d legal_ai < [init.sql](http://_vscodecontentref_/0)
```

## Usage
```python
from legal_ai.pipeline.ingestion import DataIngesion
from legal_ai.crawlers.sgg_crawler import SGGCrawler

# Crawl document list
ingestion = DataIngesion()
await ingestion.crawl_and_insert_targets(SGGCrawler())

# Download PDFs
await ingestion.download_target_contents()
```