from typing import Any
import ollama
import re
from sqlalchemy import text
from legal_ai.models.document import DocumentChunk
from legal_ai.database import get_session


class RAG:
    # receive user query, and look for similar chunks (top k)
    # if chunks have the same similarity or close enough prefer chunks coming
    # form recent document
    # once we have similar chunks, we add the chunks to the query
    # then we call the llm model with the augmented query

    def __init__(self, generation_model: str, embedding_model: str) -> None:
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.epsilon = 0.15

    def _embed_user_query(self, query: str) -> tuple[str, list[float]]:
        """Return user query embedding

        Args:
            query (str): the user query

        Returns:
            tuple[str, list[float]]: user query embedding
        """
        embedding = ollama.embeddings(model=self.embedding_model, prompt=query)
        return embedding["embedding"]

    def retrieve_similar_chunks(
        self, query: str, similarity_threshold: float
    ) -> list[dict[str, Any]]:
        # first embed the user query
        query_embedding = self._embed_user_query(query)
        # query the databse for cosine similarity
        # if chunks have the similarity or close enough
        # (using epsilon, the difference of their scores is smaller that epsilon)
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
            rows = (
                session.execute(
                    stmt,
                    {
                        "query_embedding": str(query_embedding),
                        "similarity_threshold": similarity_threshold,
                        "top_k": 10,
                    },
                )
                .mappings()
                .all()
            )

            # Convert to plain dicts so they're usable outside the session
            return [dict(row) for row in rows]

    def format_chunk_for_prompt(self, chunk: dict[str, Any]) -> str:
        """Turn a chunk + its metadata into a context block for the LLM."""
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

    def augment_query(self, user_query: str, chunks: list[str]) -> str:
        """Build the augmented prompt with context and user question."""
        context = "\n\n---\n\n".join(chunks)

        prompt = f"""Basé sur les extraits suivants du Bulletin Officiel, réponds à la question. Cite les sources (loi, décret, arrêté, chapitre, article).

            {context}

            ---

            Question : {user_query}
        """
        return prompt

    def get_answer_from_llm(self, prompt: str) -> str:
        """Query the LLM with the augmented prompt."""
        response = ollama.chat(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def ask(self, user_query: str, similarity_threshold: float) -> dict[str, Any]:
        formatted_chunks: list[str] = []
        chunks = self.retrieve_similar_chunks(user_query, similarity_threshold)
        if not chunks:
            return {
                "answer": "Aucun document pertinent trouvé pour cette question.",
                "sources": [],
            }
        formatted_chunks: list[str] = []
        for chunk in chunks:
            formatted_chunk = self.format_chunk_for_prompt(chunk)
            formatted_chunks.append(formatted_chunk)

        prompt = self.augment_query(user_query=user_query, chunks=formatted_chunks)
        answer = self.get_answer_from_llm(prompt=prompt)
        sources = []
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
