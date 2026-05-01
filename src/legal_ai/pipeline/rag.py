from typing import Any, Coroutine
import asyncio
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
from legal_ai.models.schemas import ResponseWithContext
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
        super().__init__(
            generation_model=generation_model,
            embedding_model=embedding_model,
            llm_client=llm_client,
            document_splitter=document_splitter,
            top_k=top_k,
        )
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

    async def _generate_hypothetical_answer(self, query: str) -> str:
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
        response = await self.llm_client.chat(
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
        query_embedding: list[float] = []
        query_embedding = await self._embed_query(query)

        # Retrieve chunks for both embeddings
        chunks = self._retrieve_similar_chunks(query_embedding, similarity_threshold)

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
            "2. Sources citées\n"
            "Si la réponse n'est pas dans les extraits, dis uniquement : 'Information non disponible."
        )
        return system, user

    async def _contextualize_question(self, user_query: str, history: list[dict[str, str]]) -> str:
        """
        Contextualize user query according to conversation's history.
        """
        if not history:
            return user_query

        history_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
        prompt = (
            "Étant donné l'historique de conversation ci-dessous et une question de suivi, "
            "reformule la question de suivi en une question autonome et complète, "
            "compréhensible sans l'historique. "
            "Si la question est déjà autonome, retourne-la telle quelle. "
            "Réponds UNIQUEMENT avec la question reformulée, sans explication.\n\n"
            f"Historique :\n{history_text}\n\n"
            f"Question de suivi : {user_query}\n\n"
            "Question reformulée :"
        )
        message = {"role": "user", "content": prompt}
        response = await self.llm_client.chat(model=self.generation_model, messages=[message])
        return response

    def _get_unique_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return unique chunks by chunk.id

        Args:
            chunks (list[dict[str, Any]]): a list of chunks

        Returns:
            list[dict[str, Any]]: list of unique chunks
        """
        seen: set[str] = set()
        unique_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk["id"] in seen:
                continue
            unique_chunks.append(chunk)
            seen.add(chunk["id"])
        return unique_chunks

    async def ask(
        self,
        user_query: str,
        similarity_threshold: float,
        history: list[dict[str, str]] | None = None,
        hyde: bool = False,
        rerank: bool = True,
        contextualize_query: bool = True,
    ) -> ResponseWithContext:
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
        embedding_tasks: list[Coroutine[None, None, list[dict[str, Any]]]] = []

        hypothetical_answer = None
        chunks_a, chunks_b = [], []
        if not history:
            history = []

        contextualized_question = user_query
        if contextualize_query:
            # contextualize question according to conversation's history
            contextualized_question = await self._contextualize_question(
                user_query=user_query, history=history
            )

        if hyde:
            hypothetical_answer = await self._generate_hypothetical_answer(contextualized_question)

        # retrieve similar chunks to the user query from vector database
        # generate a hypothetical answer and retrieve similar chunks
        # rerank chunks
        embedding_tasks.append(
            self.retrieve_similar_chunks(
                query=contextualized_question, similarity_threshold=similarity_threshold
            )
        )
        if hypothetical_answer:
            embedding_tasks.append(
                self.retrieve_similar_chunks(
                    query=hypothetical_answer, similarity_threshold=similarity_threshold
                )
            )

            chunks_a, chunks_b = await asyncio.gather(*embedding_tasks)
        else:
            (chunks_a,) = await asyncio.gather(*embedding_tasks)

        chunks = chunks_a + chunks_b

        if not chunks:
            res = ResponseWithContext(
                answer="Aucun document pertinent trouvé pour cette question.",
                sources=[],
                context=[],
            )
            return res

        # chunk a and b could potentially have duplicates
        unique_chunks = self._get_unique_chunks(chunks=chunks)

        # Reranking
        if rerank:
            a = perf_counter()
            chunks = self.rerank(query=contextualized_question, chunks=unique_chunks)
            chunks = chunks[: self.top_k]
            logger.info(f"Reranking in {perf_counter() - a}")

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
        answer = await self.llm_client.chat(
            model=self.generation_model,
            messages=messages,
        )
        logger.info(f"LLM response in {perf_counter() - a}")
        sources: list[dict[str, str | None]] = []
        for c in chunks:
            sources.append(
                {
                    "document_number": c.get("document_number"),
                    "official_date": str(c.get("official_date", "")),
                }
            )
        # TODO: make a dataclass in models.schemas
        response_with_context = ResponseWithContext(answer=answer, context=chunks, sources=sources)
        return response_with_context
