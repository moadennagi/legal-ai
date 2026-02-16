from sqlalchemy.orm import Session
from legal_ai.models.document import Target, Document
from typing import Any
from legal_ai.models.schemas import TargetPayload
from sqlalchemy.dialects.postgresql import insert
from legal_ai.repositories.source import SourceRepository
from datetime import datetime, timezone


class TargetRespository:
    def __init__(self) -> None:
        self.source_store = SourceRepository()

    def create_insertion_data(self, object: TargetPayload | Document) -> dict[str, Any]:
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
        target_payload = TargetPayload(number=target.number, url=target.url)
        data = target.__dict__
        for k, value in data.items():
            if k not in TargetPayload.__dict__.keys():
                continue
            setattr(target_payload, k, value)
        return target_payload

    def set_source_id(self, session: Session, target: TargetPayload) -> TargetPayload:
        source = self.source_store.get_or_create_source(
            session=session, source_name=target.source.name, source_url=target.source.url
        )
        target.source_id = source.id
        return target

    def insert_targets(self, session: Session, targets_payload: list[TargetPayload]) -> int:
        """Insert a list of Targets and return the count
        handles integrity error.
        """
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
                    "updated_at": int(datetime.now(tz=timezone.utc).timestamp()),
                },
            )
            session.execute(stmt)


class DocumentRepository:
    def construct_document_from_target_payload(self, target: TargetPayload) -> Document:
        document = Document(number=target.number, url=target.url)
        data = target.__dict__
        for k, value in data.items():
            if k not in Document.__dict__.keys():
                continue
            if k == "created_at":
                value = int(datetime.now(tz=timezone.utc).timestamp())
            setattr(document, k, value)
        return document

    def create_insertion_data(self, object: TargetPayload | Document) -> dict[str, Any]:
        columns_to_exclude = {
            "id",
        }
        data = {
            c.name: getattr(object, c.name, None)
            for c in Target.__table__.columns
            if c.name not in columns_to_exclude
        }
        return data

    def insert_documents(self, session: Session, documents: list[Document]) -> list[Document]:
        stmt = insert(Document).values(documents)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["source_id", "number"],
        )
        session.execute(stmt)
