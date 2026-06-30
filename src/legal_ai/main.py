import asyncio
import sys
from datetime import datetime
from pathlib import Path
import os
import ollama
from sqlalchemy import text
from legal_ai.pipeline.ingestion import DataIngesion
from legal_ai.logging_config import setup_logging
from legal_ai.crawlers.sgg_crawler import SGGCrawler
from legal_ai.pipeline.rag import RAG
from legal_ai.database import get_session
from sqlalchemy import select
from sqlalchemy.orm import load_only
from legal_ai.interfaces import CrawlerInterface
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
from legal_ai.crawlers.local_files_crawler import LocalFilesCrawler
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
        document_splitters={4: bo_document_splitter, 8: generic_splitter},
        generation_model=settings.generation_model,
    )
    sgg_crawler = SGGCrawler(
        name="sgg",
        url="https://www.sgg.gov.ma/BulletinOfficiel.aspx",
        base_url="https://www.sgg.gov.ma/",
        api_url="https://www.sgg.gov.ma/DesktopModules/MVC/TableListBO/BO/AjaxMethod",
    )

    current_file_path = Path(__file__).resolve()
    frat_crawler = LocalFilesCrawler(
        file_path=Path("/home/moadennagi/projects/legal-ai/data/frat"), name="frat"
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
        results_dir: Path = Path("evals/results"),
        eval_csv: str = "evals/ragas_eval_dataset.csv",
    ):
        judge_model = "gpt-4o-mini"
        similarity_threshold = 0.5
        ragas_evaluation = RagasEvaluation(llm_as_judge=judge_model)
        rows = load_eval_dataset(eval_csv)
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

        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / f"results_{config_tag}.csv", index=False)

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
        # Usage: python -m legal_ai.main export-ragas-csv --source N [--output path] [--limit N]
        args = sys.argv[2:]
        source_id = int(args[args.index("--source") + 1]) if "--source" in args else None
        limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
        if "--output" in args:
            output_path = args[args.index("--output") + 1]
        elif source_id is not None:
            output_path = f"evals/ragas_seed_source_{source_id}.csv"
        else:
            output_path = "evals/ragas_seed_dataset.csv"
        rows = export_ragas_seed_csv(
            output_path=output_path,
            source_id=source_id,
            limit=limit,
            overwrite=True,
        )
        print(f"Exported {rows} rows to {output_path}")
        raise SystemExit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "generate-ragas-dataset":
        # Usage: python -m legal_ai.main generate-ragas-dataset --input path [--output path] [--sample N]
        args = sys.argv[2:]
        input_csv = (
            args[args.index("--input") + 1] if "--input" in args else "evals/ragas_seed_dataset.csv"
        )
        sample_size = int(args[args.index("--sample") + 1]) if "--sample" in args else 60
        if "--output" in args:
            output_csv = args[args.index("--output") + 1]
        else:
            stem = Path(input_csv).stem.replace("seed", "eval")
            output_csv = f"evals/{stem}.csv"
        openai_client = OpenAILLMClientAdapter(api_key=settings.openai_api_key.get_secret_value())
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
        # Usage: python -m legal_ai.main run-ragas-eval [--input path]
        args = sys.argv[2:]
        eval_csv = (
            args[args.index("--input") + 1] if "--input" in args else "evals/ragas_eval_dataset.csv"
        )

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

        # un dossier par lancement, horodaté au démarrage du run
        run_dir = Path("evals/results") / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        print(f"Résultats de ce lancement → {run_dir}")

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
                    await run_evaluation(
                        rag=rag, results_dir=run_dir, eval_csv=eval_csv, **model_cfg, **ablation_cfg
                    )

        asyncio.run(run_all())
        raise SystemExit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "create-embeddings":
        asyncio.run(create_document_chunks_with_embeddings())

    async def ingest(crawler: CrawlerInterface, source_id: int):
        await data_ingestion.crawl_and_insert_targets(crawler=crawler)
        with get_session() as session:
            targets = data_ingestion.collect_targets(session=session, source_id=source_id)
            await data_ingestion.download_target_contents(
                targets=targets, data_dir="/home/moadennagi/projects/legal-ai/data/frat"
            )

    async def text_extraction(source_id: int):
        stmt = select(Document).where(Document.source_id == source_id)
        with get_session() as session:
            res = session.execute(stmt).scalars().all()
            docs = [document_repository.get_document_schema_from_document(doc) for doc in res]
            if not res:
                raise ValueError("Document not found")
            data_ingestion.extract_text_from_documents(documents=docs)

    async def embedding(source_id: int, document_number: str | None = None):
        stmt = select(Document).where(Document.source_id == source_id)
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

    # asyncio.run(ingest(crawler=frat_crawler, source_id=8))
    # asyncio.run(text_extraction(source_id=8))
    asyncio.run(embedding(source_id=8))
