from typing import Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import re
from time import perf_counter
import torch
from sqlalchemy import text
from legal_ai.database import get_session
from legal_ai.interfaces import (
    LLMClientInterface,
    DocumentSplitterInterface,
    RAGInterface,
    ChunkResult,
)
from legal_ai.settings import settings

logger = logging.getLogger(__name__)


class RAG(RAGInterface):
    # receive user query, and look for similar chunks (top k)
    # if chunks have the same similarity or close enough prefer chunks coming
    # form recent document
    # once we have similar chunks, we add the chunks to the query
    # then we call the llm model with the augmented query

    def __init__(
        self,
        generation_model: str,
        embedding_model: str,
        llm_client: LLMClientInterface,
        document_splitter: DocumentSplitterInterface,
        top_k: int = 6,
    ) -> None:
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.llm_client = llm_client
        self.document_splitter = document_splitter
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=settings.reranking_model
        )
        self.reranking_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=settings.reranking_model, num_labels=1
        )

    async def _embed_query(self, query: str) -> list[float]:
        """Return user query embedding

        Args:
            query (str): the user query

        Returns:
            list[float]: user query embedding
        """
        embedding = await self.llm_client.embeddings(model=self.embedding_model, prompt=query)
        return embedding

    def _generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical answer to the query using the LLM.

        This implements HyDE (Hypothetical Document Embedding) to bridge
        vocabulary gaps between the query and relevant documents.
        """
        prompt = (
            "Tu es un expert en droit marocain. "
            "Génère une réponse hypothétique et plausible à la question suivante, "
            "comme si tu citais un texte juridique officiel du Bulletin Officiel. "
            "Réponds directement sans préambule, en une ou deux phrases.\n\n"
            f"Question : {query}"
        )
        response = self.llm_client.chat(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response

    def _retrieve_similar_chunks(
        self, query_embedding: list[float], similarity_threshold: float
    ) -> list[dict[str, Any]]:
        """Retrieve chunks from the database for a single embedding vector."""
        stmt = text("""
            SELECT
                dc.id,
                dc.content,
                dc.chunk_index,
                dc.metadata,
                dc.document_id,
                d.official_date,
                d.number AS document_number,
                (dc.embedding <=> :query_embedding) AS distance
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.embedding IS NOT NULL
              AND (dc.embedding <=> :query_embedding) <= :similarity_threshold
            ORDER BY distance ASC, d.official_date DESC
            LIMIT :top_k
        """)

        with get_session() as session:
            # Increase IVFFLAT probes for better recall (default is 1,
            # which only searches 1 of 100 index lists ≈ 1% of chunks)
            session.execute(text("SET ivfflat.probes = 30"))
            rows = (
                session.execute(
                    stmt,
                    {
                        "query_embedding": str(query_embedding),
                        "similarity_threshold": similarity_threshold,
                        "top_k": self.top_k,
                    },
                )
                .mappings()
                .all()
            )
            if not rows:
                logger.info("No chunks within threshold=%.2f. ", similarity_threshold)
            return [dict(row) for row in rows]

    async def hyde(self, query: str, similarity_threshold: float) -> list[dict[str, Any]]:
        """
        Hyde: generate (LLM) a hypothethical answer to the query, retrieve similar
        chunks to the query and returns them.

        Args:
            query (str): user query
            similarity_threshold (float): similaarity threshold

        Returns:
            list[dict[str, Any]]: list of chunks similar to the hypothethical answer
        """
        hypothetical_answer = self._generate_hypothetical_answer(query)
        hyde_embedding = await self._embed_query(hypothetical_answer)
        hyde_chunks = self._retrieve_similar_chunks(hyde_embedding, similarity_threshold)
        return hyde_chunks

    async def retrieve_similar_chunks(
        self, query: str, similarity_threshold: float, hyde: bool = True
    ) -> list[dict[str, Any]]:
        """
        Retrieve similar chunks, perform HyDE and return top_k
        chunks from most recent documents.

        Args:
            query (str): the user query
            similarity_threshold (float): the similarity threshold

        Returns:
            list[dict[str, Any]]: list for dicts representing chunks
        """
        query_embedding = await self._embed_query(query)
        hyde_chunks: list[dict[str, Any]] = []

        if hyde:
            a = perf_counter()
            # HyDE: generate a hypothetical answer and embed it
            hyde_chunks = await self.hyde(query=query, similarity_threshold=similarity_threshold)
            logger.info(f"Hyde in {perf_counter() - a}")

        # TODO: remove all per counter
        a = perf_counter()

        # Retrieve chunks for both embeddings
        query_chunks = self._retrieve_similar_chunks(query_embedding, similarity_threshold)
        logger.info(f"Similar chunks in {perf_counter() - a}")
        # Reranking
        a = perf_counter()
        chunks = self.rerank(query=query, chunks=query_chunks + hyde_chunks)
        logger.info(f"Reranking in {perf_counter() - a}")

        return chunks[: self.top_k]

    def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Rerank chunks using cross encoder.

        Args:
            query (str): user query
            chunks (list[dict[str, Any]]): chunks
        """
        if not chunks:
            return chunks

        input_pairs = [[query, doc["content"]] for doc in chunks]
        inputs = self.tokenizer(
            input_pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = self.reranking_model(**inputs)
            scores = outputs.logits.squeeze(-1).tolist()
        scored_documents = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        documents = [doc[1] for doc in scored_documents]
        return documents

    def _augment_query(self, user_query: str, chunks: list[str]) -> tuple[str, str]:
        """Build the augmented prompt with context and user question."""
        context = "\n\n---\n\n".join(chunks)

        system = (
            "Tu es un assistant juridique spécialisé en droit marocain. "
            "Réponds uniquement en te basant sur les extraits fournis. "
            "Si la réponse n'est pas dans les extraits, dis-le explicitement sans inventer. "
            "Cite toujours les sources précises : instrument juridique et numéro d'article."
        )
        user = (
            f"Extraits du Bulletin Officiel :\n\n{context}\n\n"
            "---\n\n"
            f"Question : {user_query}\n\n"
            "Réponds de manière structurée :\n"
            "1. Réponse directe à la question\n"
            "2. Sources citées (instrument > article)\n"
            "Si tu n'es pas certain à 100% que la réponse figure dans les extraits ci-dessus, commence ta réponse par : "
            "'Cette information n'est pas disponible dans les documents fournis."
        )
        return system, user

    async def ask(
        self,
        user_query: str,
        similarity_threshold: float,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Find similar chunks (cosine similarity), generate hypothetical answer and find similar
        chunks (HyDE), rerank chunks, augment the user query with chunks and generate and answer.

        Args:
            user_query (str): user query
            similarity_threshold (float): threshold for cosine similarity
            history (list[dict[str, str]]): conversation history

        Returns:
            dict[str, Any]: answer dictionary
        """
        formatted_chunks: list[str] = []
        # retrieve similar chunks to the user query from vector database
        # generate a hypothetical answer and retrieve similar chunks
        # rerank chunks
        chunks = await self.retrieve_similar_chunks(
            query=user_query, similarity_threshold=similarity_threshold
        )
        if not chunks:
            return {
                "answer": "Aucun document pertinent trouvé pour cette question.",
                "sources": [],
            }
        formatted_chunks: list[str] = []
        # add metadata to chunks: metadata contain the chunk instrument (loi, decret ...)
        for chunk in chunks:
            chunk_result = ChunkResult(
                id=chunk["id"], page_content=chunk["content"], metadata=chunk["metadata"]
            )
            formatted_chunk = self.document_splitter.construct_enriched_content(chunk_result)
            formatted_chunks.append(formatted_chunk)

        # augment user query by adding chunks
        system, user = self._augment_query(user_query=user_query, chunks=formatted_chunks)
        messages = [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": user},
        ]
        # TODO: remove counter
        a = perf_counter()
        answer = self.llm_client.chat(
            model=self.generation_model,
            messages=messages,
        )
        logger.info(f"LLM response in {perf_counter() - a}")
        sources: list[dict[str, str | int | None]] = []
        for c in chunks:
            meta = c.get("metadata") or {}
            sources.append(
                {
                    "document_number": c.get("document_number"),
                    "official_date": str(c.get("official_date", "")),
                    "instrument": re.sub(r"^#+\s*", "", meta.get("instrument", "")),
                    "distance": round(c.get("distance", 0), 4),
                }
            )
        return {"answer": answer, "sources": sources}
