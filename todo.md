### 1. Ingestion
#### 1.1 Data collection
- [ ] crawling: data collection is done on 2 steps, first the crawler, its constructed given a source, the logic of the
crawler, specially the parsing is heavily dependent on the source, so the crawler is a SourceCrawler, the crawler should
expose one public method which returns the crawled data (Targets)
- [ ] insert targets into the database.
- [ ] read targets from the database and download them, store them into to database
#### 1.2 Splitting and embedding
- [ ] design the database models regarding splitting and embedding
- [ ] split documents into chunks and store them into the database
