import asyncio
import sys
from pathlib import Path
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
from legal_ai.adapters import (
    DoclingDocumentConverterAdapter,
    OllamaLLMClientAdapter,
    OpenAILLMClientAdapter,
)
from legal_ai.splitters.moroccan_bo_splitter import MoroccanBulettinOfficielSplitter
from legal_ai.splitters.generic_splitter import GenericSplitter
from legal_ai.settings import settings
from legal_ai.pipeline.conversation import ConversationManager
from legal_ai.evaluation.utils import export_ragas_seed_csv
from legal_ai.evaluation.qa_generator import QASyntheticGenerator
from legal_ai.evaluation.ragas import RagasEvaluation
from legal_ai.evaluation.utils import load_eval_dataset
from legal_ai.utils import sanitize_model_name
import json

if __name__ == "__main__":
    setup_logging()

    ollama_client = OllamaLLMClientAdapter()
    bo_document_splitter = MoroccanBulettinOfficielSplitter()
    document_repository = DocumentRepository()
    document_converter = DoclingDocumentConverterAdapter()
    data_ingestion = DataIngesion(document_converter=document_converter)

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

    async def create_document_chunks_with_embeddings():
        with get_session() as session:
            stmt = (
                select(Document)
                .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.id.is_(None))
            )
            documents = session.execute(stmt).scalars().all()
            await document_embedding.split_and_insert_embeddings(documents=documents)

    async def run_evaluation(
        rag: RAG,
        embedding_model: str,
        generation_model: str,
        hyde: bool,
        rerank: bool,
        contextualize_query: bool,
    ):
        judge_model = "gpt-4o-mini"
        similarity_threshold = 0.5
        ragas_evaluation = RagasEvaluation(llm_as_judge=judge_model)
        rows = load_eval_dataset("evals/ragas_eval_dataset.csv")
        result = await ragas_evaluation.evaluate_response(
            rag=rag,
            rows=rows,
            similarity_threshold=similarity_threshold,
            hyde=hyde,
            rerank=rerank,
            contextualize_query=contextualize_query,
        )
        df = result.to_pandas()

        df["judge_model"] = judge_model
        df["embedding_model"] = embedding_model
        df["generation_model"] = generation_model
        df["hyde"] = hyde
        df["rerank"] = rerank
        df["contextualize_query"] = contextualize_query
        df["similarity_threshold"] = similarity_threshold

        emb_tag = sanitize_model_name(embedding_model)
        gen_tag = sanitize_model_name(generation_model)
        config_tag = f"emb={emb_tag}_gen={gen_tag}_hyde={hyde}_rerank={rerank}_contextualize={contextualize_query}_threshold={similarity_threshold}"

        Path("evals/results").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"evals/results/results_{config_tag}.csv", index=False)

        summary = {
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "generation_model": generation_model,
            "hyde": hyde,
            "rerank": rerank,
            "contextualize_query": contextualize_query,
            "similarity_threshold": similarity_threshold,
            "scores": df[
                ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            ]
            .mean()
            .to_dict(),
        }
        with open("evals/summary.json", "a") as f:
            f.write(json.dumps(summary) + "\n")

    if len(sys.argv) > 1 and sys.argv[1] == "export-ragas-csv":
        output_path = (
            sys.argv[2]
            if len(sys.argv) > 2
            else "src/legal_ai/evaluation/dataset/ragas_seed_dataset.csv"
        )
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
        rows = export_ragas_seed_csv(
            output_path=output_path,
            source_id=4,
            limit=limit,
            overwrite=True,
        )
        print(f"Exported {rows} rows to {output_path}")
        raise SystemExit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "generate-ragas-dataset":
        # Usage: python -m legal_ai.main generate-ragas-dataset <output_csv> [--sample N]
        # Requires: evaluation/qa_generator.py (write-manually)

        openai_client = OpenAILLMClientAdapter(api_key=settings.openai_api_key.get_secret_value())
        input_csv = "evals/ragas_seed_dataset.csv"
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "evals/ragas_eval_dataset.csv"
        sample_size = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--sample" else 60
        generator = QASyntheticGenerator(
            llm_client=openai_client,
            model="gpt-4o-mini",
        )
        count = asyncio.run(
            generator.generate(input_csv=input_csv, output_csv=output_csv, limit=sample_size)
        )
        print(f"Generated {count} QA pairs → {output_csv}")
        raise SystemExit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "run-ragas-eval":
        ablation_configs = [
            {"hyde": False, "rerank": False, "contextualize_query": False},
            {"hyde": False, "rerank": True, "contextualize_query": False},
            {"hyde": True, "rerank": False, "contextualize_query": False},
            {"hyde": True, "rerank": True, "contextualize_query": False},
        ]

        model_configs = [
            {"embedding_model": "bge-m3", "generation_model": "qwen2.5:7b"},
            # {"embedding_model": "multilingual-e5-large", "generation_model": "qwen2.5:7b"},
            {"embedding_model": "bge-m3", "generation_model": "mistral:7b"},
            {"embedding_model": "bge-m3", "generation_model": "gemma2:9b"},
        ]

        async def run_all():
            for model_cfg in model_configs:
                rag = RAG(
                    generation_model=model_cfg["generation_model"],
                    embedding_model=model_cfg["embedding_model"],
                    llm_client=ollama_client,
                    document_splitter=bo_document_splitter,
                )
                for ablation_cfg in ablation_configs:
                    print(f"\n=== Running: {model_cfg} | {ablation_cfg} ===")
                    await run_evaluation(rag=rag, **model_cfg, **ablation_cfg)

        asyncio.run(run_all())
        raise SystemExit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "create-embeddings":
        asyncio.run(create_document_chunks_with_embeddings())

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

    async def embedding(document_number: str | None = None):
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
            # select only documents without chunks
            documents = session.execute(stmt).scalars().all()
            if not documents:
                raise ValueError("Document not found")
            await document_embedding.split_and_insert_embeddings(documents=documents)

    async def ask():
        sys.stdin.reconfigure(encoding="utf-8")
        print(f"Model = {settings.generation_model}")
        while True:
            q = input("Type a question: ")
            res = await conversation_manager.ask(query=q, similarity_threshold=0.5)
            if q == "exit":
                break
            print(res.answer)

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
        rag_client.rerank(query=query, chunks=all_chunks)

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

    asyncio.run(embedding())
