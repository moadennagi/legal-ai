if __name__ == "__main__":
    from legal_ai.pipeline.ingestion import DataIngesion
    from legal_ai.logging_config import setup_logging
    from legal_ai.crawlers.sgg_crawler import SGGCrawler
    from legal_ai.pipeline.processing import DocumentProcessing
    from legal_ai.database import get_session
    from sqlalchemy import select
    from legal_ai.models.document import Document, DocumentChunk
    from legal_ai.pipeline.embedding import DocumentEmbedding
    from legal_ai.repositories.document import DocumentRepository
    from legal_ai.pipeline.rag import RAG

    setup_logging()

    data_ingestion = DataIngesion()
    document_repository = DocumentRepository()
    document_processing = DocumentProcessing()
    document_embedding = DocumentEmbedding(embedding_model="nomic-embed-text")
    rag = RAG(
        embedding_model="nomic-embed-text",
        generation_model="dolphin-llama3",
    )

    async def crawl_and_insert_targets():
        crawler = SGGCrawler()
        await data_ingestion.crawl_and_insert_targets(crawler=crawler)

    async def download_and_insert_documents():
        await data_ingestion.download_target_contents()

    def extract_text_from_documents():
        document_processing.extract_text_from_documents()

    def insert_document_chunks():
        # read one document, check the hierarchy
        with get_session() as session:
            stmt = (
                select(Document)
                .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.id.is_(None))
            )
            documents = session.execute(stmt).scalars().all()
            document_embedding.split_and_insert_document_chunks(documents=documents)

    def query_with_rag(query: str):
        answer = rag.ask(user_query=query, similarity_threshold=0.3)
        return answer

    extract_text_from_documents()
    # insert_document_chunks()
    # process()
