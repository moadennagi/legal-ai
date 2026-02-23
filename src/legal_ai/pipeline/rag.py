from typing import Any
import re
from sqlalchemy import text
from legal_ai.database import get_session
from legal_ai.interfaces import LLMClientInterface


class RAG:
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
        top_k: int = 6,
    ) -> None:
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.llm_client = llm_client

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
            session.execute(text("SET ivfflat.probes = 10"))
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
            return [dict(row) for row in rows]

    async def retrieve_similar_chunks(
        self, query: str, similarity_threshold: float
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

        # HyDE: generate a hypothetical answer and embed it
        hypothetical_answer = self._generate_hypothetical_answer(query)
        hyde_embedding = await self._embed_query(hypothetical_answer)

        # Retrieve chunks for both embeddings
        query_chunks = self._retrieve_similar_chunks(query_embedding, similarity_threshold)
        hyde_chunks = self._retrieve_similar_chunks(hyde_embedding, similarity_threshold)

        # Merge and deduplicate: keep minimum distance per chunk id
        seen: dict[int, dict[str, Any]] = {}
        for chunk in query_chunks + hyde_chunks:
            chunk_id = chunk["id"]
            if chunk_id not in seen or chunk["distance"] < seen[chunk_id]["distance"]:
                seen[chunk_id] = chunk

        # Sort by distance, then by recency
        def sort_key(c: dict[str, Any]) -> tuple[float, int]:
            d = c.get("official_date")
            return (c["distance"], -(d.toordinal() if d else 0))

        merged = sorted(seen.values(), key=sort_key)

        return merged[: self.top_k]

    def format_chunk_for_prompt(self, chunk: dict[str, Any]) -> str:
        """Turn a chunk + its metadata into a context block for the LLM

        Args:
            chunk (dict[str, Any]): chunk

        Returns:
            str: formatted chunk
        """
        meta: dict[str, Any] = chunk.get("metadata") or {}

        # Build a breadcrumb from whatever metadata keys exist
        breadcrumb_keys = ["instrument", "partie", "titre", "chapitre", "section"]
        parts: list[str] = []
        for key in breadcrumb_keys:
            if key in meta:
                # Strip the markdown heading markers (##, ###, etc.)
                clean = re.sub(r"^#+\s*", "", meta[key])
                parts.append(clean)

        header = " > ".join(parts) if parts else "Source inconnue"
        return f"[{header}]\n{chunk['content']}"

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
            "Si l'information est absente des extraits, réponds : "
            '"Cette information n\'est pas disponible dans les documents fournis."'
        )
        return system, user

    async def ask(self, user_query: str, similarity_threshold: float) -> dict[str, Any]:
        formatted_chunks: list[str] = []
        chunks = await self.retrieve_similar_chunks(user_query, similarity_threshold)
        if not chunks:
            return {
                "answer": "Aucun document pertinent trouvé pour cette question.",
                "sources": [],
            }
        formatted_chunks: list[str] = []
        for chunk in chunks:
            formatted_chunk = self.format_chunk_for_prompt(chunk)
            formatted_chunks.append(formatted_chunk)

        system, user = self._augment_query(user_query=user_query, chunks=formatted_chunks)
        answer = self.llm_client.chat(
            model=self.generation_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
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
