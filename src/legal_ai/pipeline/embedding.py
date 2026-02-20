import ollama
from typing import Any
from legal_ai.models.document import Document, DocumentChunk
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from legal_ai.crawlers.sgg_heading_rules import fix_heading_hierarchy
from sqlalchemy.dialects.postgresql import insert
from legal_ai.repositories.document import DocumentChunkRepository
from legal_ai.database import get_session

headers_to_split_on = [
    ("#", "division"),  # DAHIR, TEXTES GENERAUX, TEXTES PARTICULIERS
    ("##", "instrument"),  # Dahir n°, Loi n°, Décret n°, Arrêté…
    ("###", "partie"),  # PREMIÈRE PARTIE…
    ("####", "titre"),  # TITRE PREMIER…
    ("#####", "chapitre"),  # Chapitre premier…
    ("######", "section"),  # Dénomination et objet, Missions…
]


class DocumentEmbedding:
    def __init__(
        self, embedding_model: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> None:
        self.document_chunk_repository = DocumentChunkRepository()
        self.embedding_model = embedding_model
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_document_into_chunks(self, document: Document) -> list[DocumentChunk]:
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

        document_chunks: list[DocumentChunk] = []
        for i, chunk in enumerate(chunks):
            if not chunk.page_content.strip():
                continue

            # skip table of content
            if chunk.metadata.get("division", "").strip().upper() == "SOMMAIRE":
                continue

            embedding = self.get_embedding_for_chunk(chunk=chunk.page_content)
            document_chunks.append(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk.page_content,
                    embedding=embedding,
                    chunk_metadata=chunk.metadata,
                )
            )
        return document_chunks

    def get_embedding_for_chunk(self, chunk: str) -> list[float]:
        """
        Get embedding for chunk.

        Args:
            chunk (str):string representing a chunk of from a document

        Returns:
            list[float]: embedding for the given chunk
        """
        embedding = ollama.embeddings(model=self.embedding_model, prompt=chunk)
        return embedding["embedding"]

    def split_and_insert_document_chunks(self, documents: list[Document]):
        """
        Split each document into chunks, get their embeddings and insert them
        into the database.

        Args:
            documents (list[Document]): a list of Document instances
        """
        document_chunks_dict_data: list[dict[str, Any]] = []
        for document in documents:
            with get_session() as session:
                document_chunks = self.split_document_into_chunks(document)
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
