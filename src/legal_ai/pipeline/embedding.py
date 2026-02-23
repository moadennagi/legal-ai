import ollama
import re
import asyncio
from asyncio import Semaphore
from legal_ai.utils import run_with_semaphore
from typing import Any, Coroutine
from legal_ai.models.document import Document
from legal_ai.models.document import DocumentChunk
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from legal_ai.crawlers.sgg_heading_rules import fix_heading_hierarchy, HEADERS_TO_SPLIT_ON
from sqlalchemy.dialects.postgresql import insert
from legal_ai.repositories.document import DocumentChunkRepository
from legal_ai.database import get_session


class DocumentEmbedding:
    def __init__(
        self, embedding_model: str, chunk_size: int = 1500, chunk_overlap: int = 300
    ) -> None:
        self.document_chunk_repository = DocumentChunkRepository()
        self.embedding_model = embedding_model
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.ollama_async_client = ollama.AsyncClient()
        self.semaphore = Semaphore(100)

    def _construct_enriched_content(self, chunk: LangchainDocument) -> str:
        """
        Construct a string reprsenting the chunk with its context.
        (for contextual embedding)

        Args:
            chunk (dict[str, Any]): a chunk as created by langchain text splitter
        """
        metadata_keys = ["instrument", "partie", "titre", "chapitre", "section"]
        breadcrumbs: list[str] = []
        if not chunk.metadata:
            return chunk.page_content

        for key in metadata_keys:
            if key not in chunk.metadata:
                continue
            clean = re.sub(r"^#+\s*", "", chunk.metadata.get(key))
            breadcrumbs.append(clean)
        header = " > ".join(breadcrumbs)
        return f"[{header}]\n{chunk.page_content}"

    def _is_table_like(self, text: str) -> bool:
        """Check if content is mostly table formatting (pipe-delimited rows)."""
        lines = text.strip().splitlines()
        if not lines:
            return True
        pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
        return pipe_lines / len(lines) > 0.5

    def _filter_chunks(self, chunks: list[LangchainDocument]) -> list[LangchainDocument]:
        """
        Filter out bad chunks: empty, too short, table-of-contents,
        and table-like formatting artifacts.

        Args:
            chunks (list[LangchainDocument]): a list of langchain documents

        Returns:
            list[LangchainDocument]: of list good langchain documents
        """
        good_chunks: list[LangchainDocument] = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            if not content:
                continue

            # skip table of content
            # TODO: this is tighetly coupled to BO
            if chunk.metadata.get("division", "").strip().upper() == "SOMMAIRE":
                continue

            # skip very short chunks (headers, separators, etc.)
            if len(content) < 50:
                continue
            # skip table-like formatting artifacts
            if self._is_table_like(content):
                continue
            good_chunks.append(chunk)
        return good_chunks

    def _split_document(self, document: Document) -> list[LangchainDocument]:
        """
        Split a document's markdown content into chunks.

        1. MarkdownHeaderTextSplitter splits on headings → semantic sections
           with metadata (division, instrument, partie, titre, chapitre, section).
        2. RecursiveCharacterTextSplitter sub-splits long sections to fit
           the embedding model's context window.

        If a document lacks some heading levels (e.g. no PARTIE or TITRE),
        those metadata keys simply won't appear — no errors, no empty splits.
        """
        if not document.text_content:
            return []
        fixed_markdown = fix_heading_hierarchy(document.text_content, articles_as_bold=False)
        md_sections = self.md_splitter.split_text(fixed_markdown)
        chunks = self.text_splitter.split_documents(md_sections)
        chunks = self._filter_chunks(chunks)
        return chunks

    async def _embded_chunks(self, chunks: list[LangchainDocument]) -> list[list[float]]:
        """
        Get embeddings to given chunks.

        Args:
            chunks (list[LangchainDocument]): langchain Document

        Returns:
            list[DocumentChunk]: list of embeddings
        """
        embedding_tasks: list[Coroutine[None, None, list[float]]] = []
        # gather all embedding tasks for a single document
        for _, chunk in enumerate(chunks):
            # build an enriched chunk
            enriched_chunk = self._construct_enriched_content(chunk)
            embedding_task = self.get_embedding(chunk=enriched_chunk)
            embedding_tasks.append(run_with_semaphore(self.semaphore, embedding_task))

        results = await asyncio.gather(*embedding_tasks)
        return results

    def _construct_document_chunks(
        self, document_id: int, chunks: list[LangchainDocument], embeddings: list[list[float]]
    ) -> list[DocumentChunk]:
        """
        Given a list of embedding and a list of chunks, construct and isntance of
        DocumentChunk

        Args:
            document_id (int): Document id
            chunks (list[LangchainDocument]): list of langchain documents
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

    async def get_embedding(self, chunk: str) -> list[float]:
        """
        Get embedding.

        Args:
            chunk (str):string representing a chunk of from a document

        Returns:
            list[float]: embedding for the given chunk
        """
        embedding = await self.ollama_async_client.embeddings(
            model=self.embedding_model, prompt=chunk
        )
        return embedding["embedding"]

    async def split_and_insert_document_chunks(self, documents: list[Document]):
        """
        Split each document into chunks, get their embeddings and insert them
        into the database.

        Args:
            documents (list[Document]): a list of Document instances
        """
        document_chunks_dict_data: list[dict[str, Any]] = []
        with get_session() as session:
            for document in documents:
                chunks = self._split_document(document)
                embeddings = await self._embded_chunks(chunks)

                document_chunks = self._construct_document_chunks(
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
