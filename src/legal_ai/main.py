if __name__ == "__main__":
    import asyncio
    import sys
    import ollama
    from sqlalchemy import text
    from legal_ai.pipeline.ingestion import DataIngesion
    from legal_ai.logging_config import setup_logging
    from legal_ai.crawlers.sgg_crawler import SGGCrawler
    from legal_ai.pipeline.rag import RAG
    from legal_ai.database import get_session
    from sqlalchemy import select
    from sqlalchemy.orm import load_only
    from legal_ai.models.document import Document, DocumentChunk
    from legal_ai.pipeline.embedding import DocumentEmbedding
    from legal_ai.repositories.document import DocumentRepository
    from legal_ai.adapters import DoclingDocumentConverterAdapter, OllamaLLMClientAdapter
    from legal_ai.splitters.moroccan_bo_splitter import MoroccanBulettinOfficielSplitter
    from legal_ai.splitters.generic_splitter import GenericSplitter
    from legal_ai.settings import settings
    from legal_ai.pipeline.conversation import ConversationManager

    setup_logging()

    document_repository = DocumentRepository()
    document_converter = DoclingDocumentConverterAdapter()
    data_ingestion = DataIngesion(document_converter=document_converter)
    ollama_client = OllamaLLMClientAdapter()
    bo_document_splitter = MoroccanBulettinOfficielSplitter()
    generic_splitter = GenericSplitter()
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
        generation_model=settings.generation_model,
    )

    async def ingest():
        crawler = SGGCrawler()
        await data_ingestion.crawl_and_insert_targets(crawler=crawler)
        await data_ingestion.download_target_contents()

    async def text_extraction():
        stmt = select(Document)
        with get_session() as session:
            res = session.execute(stmt).scalars().all()
            docs = [document_repository.get_document_schema_from_document(doc) for doc in res]
            if not res:
                raise ValueError("Document not found")
            data_ingestion.extract_text_from_documents(documents=docs)

    async def embedding(document_number: str):
        stmt = select(Document)
        if document_number:
            stmt = (
                select(Document)
                .where(Document.number == document_number)
                .options(
                    load_only(
                        Document.id,
                        Document.number,
                        Document.source_id,
                        Document.text_content,
                        Document.file_path,
                        Document.url,
                    )
                )
            )

        with get_session() as session:
            res = session.execute(stmt).scalars().all()
            docs = [doc for doc in res]
            if not res:
                raise ValueError("Document not found")
            session.expunge_all()
        await document_embedding.split_and_insert_embeddings(documents=docs)

    async def split_and_embed_chunks():
        # read one document, check the hierarchy
        with get_session() as session:
            stmt = (
                select(Document)
                .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.id.is_(None))
            )
            documents = session.execute(stmt).scalars().all()
            docs = [doc for doc in documents]
            await document_embedding.split_and_insert_embeddings(documents=docs)

    async def ask():
        sys.stdin.reconfigure(encoding="utf-8")
        print(f"Model = {settings.generation_model}")
        while True:
            q = input("Type a question: ")
            res = await conversation_manager.ask(query=q, similarity_threshold=0.5)
            if q == "exit":
                break
            print(res["answer"])

    async def debug_history():
        q1 = "quel est le statut de l'agence des medicaments ?"
        answer = await conversation_manager.ask(query=q1, similarity_threshold=0.3)
        print(answer)
        q2 = "ou est son siège ?"
        answer = await conversation_manager.ask(query=q2, similarity_threshold=0.3)
        print(answer)

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

    async def check_distance():
        client = ollama.AsyncClient()
        query = "équivalence de diplômes étrangers"
        q_emb = (await client.embeddings(model="bge-m3", prompt=query))["embedding"]

        with get_session() as session:
            stmt = text("""
                SELECT dc.content, 
                (dc.embedding <=> :query_embedding)
                AS distance FROM document_chunks dc
                WHERE document_id = (SELECT id FROM documents WHERE number = '7462')
                ORDER BY distance ASC;""")
            rows = (
                session.execute(
                    stmt,
                    {"query_embedding": str(q_emb)},
                )
                .mappings()
                .all()
            )
            for row in rows:
                print(row["distance"])

    # asyncio.run(text_extraction_and_embedding())
    # asyncio.run(text_extraction())
    # asyncio.run(embedding("7462"))
    asyncio.run(ask())
