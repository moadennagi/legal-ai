if __name__ == "__main__":
    import asyncio
    from legal_ai.pipeline.ingestion import DataIngesion
    from legal_ai.logging_config import setup_logging
    from legal_ai.crawlers.sgg_crawler import SGGCrawler
    from legal_ai.pipeline.processing import DocumentProcessing
    from legal_ai.database import get_session
    from sqlalchemy import select
    from legal_ai.models.document import Document
    from docling.document_converter import DocumentConverter

    setup_logging()

    data_ingestion = DataIngesion()
    document_processing = DocumentProcessing()

    async def main():
        crawler = SGGCrawler()
        await data_ingestion.crawl_and_insert_targets(crawler=crawler)
        await data_ingestion.download_target_contents()

    def process():
        document_processing.extract_text_from_documents()

    def read_documents():
        # read one document, check the hierarchy
        with get_session() as session:
            stmt = select(Document).where(Document.number == "7462")
            doc = session.execute(stmt).scalar_one()
            output = f"/tmp/{doc.number}_review.md"
            with open(output, "w") as f:
                f.write(doc.text_content)

    # read_documents()
    process()
