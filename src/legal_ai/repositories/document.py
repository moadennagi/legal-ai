from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from legal_ai.models.document import Document, DocumentChunk
from legal_ai.models.schemas import TargetSchema, DocumentSchema


class DocumentRepository:
    def get_document_schema_from_document(self, document: Document) -> DocumentSchema:
        """
        Return a DocumentSchema instance constructed from Document.

        Args:
            document (Document): Document instance
        """
        document_schema = DocumentSchema(
            id=document.id, number=document.number, file_path=document.file_path
        )
        return document_schema

    def construct_document_from_target_payload(self, target: TargetSchema) -> Document:
        """
        Constructs an instance of Document from TargetPayload

        Args:
            target (TargetSchema): TargetSchema instance

        Returns:
            Document: Document instance
        """
        document = Document(
            number=target.number,
            url=target.url,
            official_date=target.official_date,
            source_id=target.source_id,
            created_at=int(datetime.now(tz=timezone.utc).timestamp()),
        )
        return document

    def get_dict_data(self, object: Document) -> dict[str, Any]:
        """Return a dict of document (object) data"""
        columns_to_exclude = {
            "id",
        }
        data = {
            c.name: getattr(object, c.name, None)
            for c in Document.__table__.columns
            if c.name not in columns_to_exclude
        }
        return data

    def insert_documents(self, session: Session, documents: list[Document]) -> int:
        """Given a list of documents bulk insert into the database"""
        documents_to_insert: list[dict[str, Any]] = []
        for document in documents:
            data_to_insert = self.get_dict_data(document)
            documents_to_insert.append(data_to_insert)
        insert_stmt = insert(Document).values(documents_to_insert)
        insert_stmt = insert_stmt.on_conflict_do_nothing(
            index_elements=["source_id", "number"],
        )
        res = session.execute(insert_stmt)
        session.flush()
        return res.rowcount

    def insert_single_document(self, session: Session, document: Document) -> int:
        """Insert a single document and return the id"""
        data_to_insert = self.get_dict_data(document)
        insert_stmt = insert(Document).values(data_to_insert)
        insert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["source_id", "number"],
            set_={"official_date": insert_stmt.excluded.official_date},
        ).returning(Document.id)
        res = session.execute(insert_stmt)
        row = res.fetchone()
        if row:
            return row[0]

        stmt = select(Document).where(
            Document.number == document.number, Document.source_id == document.source_id
        )
        row = session.scalar(stmt)
        if not row:
            raise ValueError()
        return row.id

    def collect_documents_without_content(self, session: Session) -> list[Document]:
        """Return a list of documents without text content"""
        stmt = select(Document).where(Document.text_content.is_(None))
        res = session.execute(stmt).scalars().all()
        return res

    def update_document_content(self, session: Session, document_id: int, content: str):
        stmt = update(Document).where(Document.id == document_id).values({"text_content": content})
        session.execute(stmt)
        session.flush()

    def insert_documents_bulk(
        self, session: Session, documents: list[Document | BaseException]
    ) -> int:
        """Given a list of documents insert them into the database"""
        good_documents: list[Document] = []
        errors: list[BaseException] = []
        for document in documents:
            if isinstance(document, BaseException):
                errors.append(document)
            else:
                good_documents.append(document)
        res = self.insert_documents(documents=good_documents, session=session)
        return res


class DocumentChunkRepository:
    def get_dict_data(self, document_chunk: DocumentChunk) -> dict[str, Any]:
        """_summary_

        Args:
            document_chunk (DocumentChunk): _description_

        Returns:
            dict[str, Any]: _description_
        """
        return dict(
            document_id=document_chunk.document_id,
            content=document_chunk.content,
            embedding=document_chunk.embedding,
            created_at=document_chunk.created_at,
            updated_at=document_chunk.updated_at,
            token_count=document_chunk.token_count,
            chunk_index=document_chunk.chunk_index,
            chunk_metadata=document_chunk.chunk_metadata,
        )
