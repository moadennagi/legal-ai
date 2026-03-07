import asyncio
from asyncio import Semaphore
from legal_ai.utils import run_with_semaphore
from typing import Any, Coroutine
from legal_ai.models.document import Document
from legal_ai.models.document import DocumentChunk
from sqlalchemy.dialects.postgresql import insert
from legal_ai.repositories.document import DocumentChunkRepository
from legal_ai.database import get_session
from legal_ai.interfaces import (
    EmbeddingServiceInterface,
    LLMClientInterface,
    DocumentSplitterInterface,
    ChunkResult,
)


class DocumentEmbedding(EmbeddingServiceInterface):
    def __init__(
        self,
        embedding_model: str,
        llm_client: LLMClientInterface,
        document_splitters: dict[int, DocumentSplitterInterface],
    ) -> None:
        super().__init__(llm_client)
        self.document_chunk_repository = DocumentChunkRepository()
        self.embedding_model = embedding_model
        self.document_splitters = document_splitters
        self.semaphore = Semaphore(100)

    def _get_splitter(self, document: Document):
        """
        Return the right Document splitter according to document source_id.

        Args:
            document (Document): an instance of Document
        """
        return self.document_splitters.get(document.source_id)

    async def embded_chunks(self, chunks: list[ChunkResult]) -> list[list[float]]:
        """
        Get embeddings to given chunks.

        Args:
            chunks (list[DocumentPartInterface]): langchain Document

        Returns:
            list[DocumentChunk]: list of embeddings
        """
        if not self.document_splitter:
            raise ValueError()

        embedding_tasks: list[Coroutine[None, None, list[float]]] = []
        # gather all embedding tasks for a single document
        for _, chunk in enumerate(chunks):
            # build an enriched chunk; strip surrogates that Docling may produce
            # from malformed PDF text (e.g. \udcc3 from mis-decoded UTF-8 bytes)
            enriched_chunk = self.document_splitter.construct_enriched_content(chunk)
            enriched_chunk = enriched_chunk.encode("utf-8", errors="surrogateescape").decode(
                "utf-8"
            )
            embedding_task = self.llm_client.embeddings(
                model=self.embedding_model, prompt=enriched_chunk
            )
            embedding_tasks.append(run_with_semaphore(self.semaphore, embedding_task))

        results = await asyncio.gather(*embedding_tasks)
        return results

    def construct_document_chunks(
        self, document_id: int, chunks: list[ChunkResult], embeddings: list[list[float]]
    ) -> list[DocumentChunk]:
        """
        Given a list of embedding and a list of chunks, construct and isntance of
        DocumentChunk

        Args:
            document_id (int): Document id
            chunks (list[DocumentPartInterface]): list of langchain documents
            embeddings (list[list[float]]): list of embeddings (list of float)

        Returns:
            list[DocumentChunk]: _description_
        """
        # loop over embedding and construct DocumentChunk
        document_chunks: list[DocumentChunk] = []
        for i, embedding in enumerate(embeddings):
            document_chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunks[i].page_content,
                    embedding=embedding,
                    chunk_metadata=chunks[i].metadata,
                )
            )
        return document_chunks

    async def split_and_insert_embeddings(self, documents: list[Document]):
        """
        Split each document into chunks, get their embeddings and insert them
        into the database.

        Args:
            documents (list[Document]): a list of Document instances
        """
        self.document_splitter = self._get_splitter(documents[0])
        if not self.document_splitter:
            raise ValueError("Could not set document_splitter")

        document_chunks_dict_data: list[dict[str, Any]] = []
        with get_session() as session:
            for document in documents:
                chunks = self.document_splitter.split_document(document)
                embeddings = await self.embded_chunks(chunks)

                document_chunks = self.construct_document_chunks(
                    document_id=document.id, chunks=chunks, embeddings=embeddings
                )
                document_chunks_dict_data = [
                    self.document_chunk_repository.get_dict_data(doc_chunk)
                    for doc_chunk in document_chunks
                ]
                if not document_chunks_dict_data:
                    continue
                stmt = insert(DocumentChunk).values(document_chunks_dict_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["document_id", "chunk_index"],
                    set_={
                        "content": stmt.excluded.content,
                        "embedding": stmt.excluded.embedding,
                        "token_count": stmt.excluded.token_count,
                        "updated_at": stmt.excluded.updated_at,
                        "metadata": stmt.excluded.metadata,
                    },
                )
                session.execute(stmt)
                session.commit()
