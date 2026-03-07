if __name__ == "__main__":
    import asyncio
    import sys
    from legal_ai.pipeline.ingestion import DataIngesion
    from legal_ai.logging_config import setup_logging
    from legal_ai.crawlers.sgg_crawler import SGGCrawler
    from legal_ai.pipeline.rag import RAG
    from legal_ai.database import get_session
    from sqlalchemy import select
    from legal_ai.models.document import Document, DocumentChunk
    from legal_ai.pipeline.embedding import DocumentEmbedding
    from legal_ai.repositories.document import DocumentRepository
    from legal_ai.adapters import DoclingDocumentConverterAdapter, OllamaLLMClientAdapter
    from legal_ai.splitters.moroccan_bo_splitter import MoroccanBulettinOfficielSplitter
    from legal_ai.settings import settings
    from legal_ai.pipeline.conversation import ConversationManager

    setup_logging()

    document_repository = DocumentRepository()
    document_converter = DoclingDocumentConverterAdapter()
    data_ingestion = DataIngesion(document_converter=document_converter)
    ollama_client = OllamaLLMClientAdapter()
    bo_document_splitter = MoroccanBulettinOfficielSplitter()
    rag_client = RAG(
        generation_model=settings.generation_model,
        embedding_model=settings.embeding_model,
        llm_client=ollama_client,
        document_splitter=bo_document_splitter,
    )
    conversation_manager = ConversationManager(llm_client=ollama_client, rag=rag_client)
    document_embedding = DocumentEmbedding(
        embedding_model=settings.embeding_model,
        llm_client=ollama_client,
        document_splitters={4: bo_document_splitter},
    )

    async def ingest():
        crawler = SGGCrawler()
        await data_ingestion.crawl_and_insert_targets(crawler=crawler)
        await data_ingestion.download_target_contents()

    def extract_text_from_documents():
        data_ingestion.extract_text_from_documents_without_content()

    async def split_and_embed_chunks():
        # read one document, check the hierarchy
        with get_session() as session:
            stmt = (
                select(Document)
                .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.id.is_(None))
            )
            documents = session.execute(stmt).scalars().all()
            await document_embedding.split_and_insert_embeddings(documents=documents)

    async def ask():
        sys.stdin.reconfigure(encoding="utf-8")
        print(f"Model = {settings.generation_model}")
        while True:
            q = input("Type a question: ")
            res = await conversation_manager.ask(query=q, similarity_threshold=0.3)
            if q == "exit":
                break
            print(res["answer"])
            for source in res["sources"]:
                print(source["instrument"])

    # extract_text_from_documents()
    # asyncio.run(insert_document_chunks())
    # process()

    asyncio.run(ask())

    async def test_document_index_table():
        doc_number = "7480"
        stmt = select(Document).where(Document.number == doc_number)
        with get_session() as session:
            res = session.execute(stmt).fetchone()
            if not res:
                raise ValueError("Document not found")
            doc = res[0]
            data_ingestion.extract_text_from_documents([doc])
            await document_embedding.split_and_insert_embeddings(documents=[doc])

    # asyncio.run(test_document_index_table())

    # from unstructured.partition.pdf import partition_pdf

    # filename = "/home/moadennagi/projects/legal-ai/data/7462.pdf"
    # elements = partition_pdf(filename, language="french")
    # print(elements)

    async def debug_reranking():
        x = 0.3
        query = "quel est le statut de l'agence des medicaments ?"
        query_embedding = await rag_client._embed_query(query=query)
        similar_chunks = rag_client._retrieve_similar_chunks(
            query_embedding=query_embedding, similarity_threshold=x
        )
        hyde_chunks = await rag_client.hyde(query=query, similarity_threshold=x)

        all_chunks = hyde_chunks + similar_chunks
        res = rag_client.rerank(query=query, chunks=all_chunks)

    # asyncio.run(debug_reranking())
