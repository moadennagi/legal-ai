from typing import Any
import ollama
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
    ) -> list[DocumentChunk]:
        # first embed the user query
        user_query_embedding = self._embed_user_query(query)
        # query the databse for cosine similarity
        # if chunks have the similarity or close enough
        # (using epsilon, the difference of their scores is smaller that epsilon)
        with get_session() as session:
            stmt = text("""
                SELECT *, embedding <=> :query_embedding AS distance
                FROM document_chunks
                WHERE embedding is NOT NULL
                AND distance <= :similarity_threshold
                ORDER BY distance ASC, created_at DESC
                LIMIT :top_k
            """)
            res = (
                session.execute(
                    stmt,
                    {
                        "query_embedding": user_query_embedding,
                        "top_k": 5,
                        "similarity_threshold": similarity_threshold,
                    },
                )
                .scalars()
                .all()
            )
        # return similar chunks
        return res

    def augment_query(self, query: str, chunks: list[DocumentChunk]) -> str:
        # in the format of context
        pass

    def get_answer_from_llm(self, query: str) -> str:
        # query llm with query and return the answer
        pass

    def ask(self, query: str, similarity_threshold: float) -> dict[str, Any]:
        chunks = self.retrieve_similar_chunks(query, similarity_threshold)
        prompt = self.augment_query(query=query, chunks=chunks)
        answer = self.get_answer_from_llm(query=prompt)
        return answer
