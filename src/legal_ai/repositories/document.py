from sqlalchemy.orm import Session
from legal_ai.models.document import Target, Document
from typing import Any
from legal_ai.models.schemas import TargetPayload
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import update, select
from legal_ai.repositories.source import SourceRepository
from datetime import datetime, timezone


class TargetRepository:
    def __init__(self) -> None:
        self.source_store = SourceRepository()

    def create_insertion_data(self, object: TargetPayload) -> dict[str, Any]:
        """Return a dict of TargetPayload (object) data"""
        columns_to_exclude = {
            "id",
        }
        data = {
            c.name: getattr(object, c.name, None)
            for c in Target.__table__.columns
            if c.name not in columns_to_exclude
        }
        return data

    def construct_target_payload_from_target(self, target: Target) -> TargetPayload:
        target_payload = TargetPayload(
            row_id=target.id,
            number=target.number,
            url=target.url,
            source_id=target.source_id,
            task_id=target.task_id,
            created_at=target.created_at,
            updated_at=target.updated_at,
            official_date=target.official_date,
        )
        return target_payload

    def set_source_id(self, session: Session, target: TargetPayload) -> TargetPayload:
        source = self.source_store.get_or_create_source(
            session=session, source_name=target.source.name, source_url=target.source.url
        )
        target.source_id = source.id
        return target

    def insert_targets(self, session: Session, targets_payload: list[TargetPayload]) -> int:
        """Insert a list of Targets and return the count"""
        # transform basemodel to ORM
        targets_dicts: list[dict[str, Any]] = []
        for target_payload in targets_payload:
            self.set_source_id(session=session, target=target_payload)
            targets_dicts.append(self.create_insertion_data(target_payload))

        if targets_dicts:
            stmt = insert(Target).values(targets_dicts)
            stmt = stmt.on_conflict_do_update(
                index_elements=["source_id", "number"],
                set_={
                    "task_id": stmt.excluded.task_id,
                    "url": stmt.excluded.url,
                    "official_date": stmt.excluded.official_date,
                    "updated_at": int(datetime.now(tz=timezone.utc).timestamp()),
                },
            )
            res = session.execute(stmt)
            return res.rowcount

    def update_target_document_id(self, session: Session, target_id: int, document_id: int):
        """Update target having target_id with the inserted document.id"""
        stmt = update(Target).where(Target.id == target_id).values({"document_id": document_id})
        session.execute(stmt)


class DocumentRepository:
    def construct_document_from_target_payload(self, target: TargetPayload) -> Document:
        """Constructs an instance of Document from TargetPayload"""
        document = Document(
            number=target.number,
            url=target.url,
            official_date=target.official_date,
            source_id=target.source_id,
            created_at=int(datetime.now(tz=timezone.utc).timestamp()),
        )
        return document

    def create_insertion_data(self, object: Document) -> dict[str, Any]:
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
            data_to_insert = self.create_insertion_data(document)
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
        data_to_insert = self.create_insertion_data(document)
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
