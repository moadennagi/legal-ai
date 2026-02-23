if __name__ == "__main__":
    import asyncio
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
    from legal_ai.adapters import DoclingDocumentConverterAdapter

    setup_logging()

    data_ingestion = DataIngesion()
    document_repository = DocumentRepository()
    document_converter = DoclingDocumentConverterAdapter()
    document_processing = DocumentProcessing(document_converter=document_converter)
    document_embedding = DocumentEmbedding(embedding_model="bge-m3")
    rag = RAG(
        embedding_model="bge-m3",
        generation_model="qwen2.5:7b",
    )

    async def ingest():
        crawler = SGGCrawler()
        await data_ingestion.crawl_and_insert_targets(crawler=crawler)
        await data_ingestion.download_target_contents()

    def extract_text_from_documents():
        document_processing.extract_text_from_documents()

    async def split_and_embed_chunks():
        # read one document, check the hierarchy
        with get_session() as session:
            stmt = (
                select(Document)
                .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.id.is_(None))
            )
            documents = session.execute(stmt).scalars().all()
            await document_embedding.split_and_insert_document_chunks(documents=documents)

    def query_with_rag(query: str):
        answer = rag.ask(user_query=query, similarity_threshold=0.3)
        return answer

    # extract_text_from_documents()
    # asyncio.run(insert_document_chunks())
    # process()
    while True:
        q = input("Type a question: ")
        res = query_with_rag(q)
        if q == "exit":
            break
        print(res["answer"])
