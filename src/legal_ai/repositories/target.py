from datetime import datetime, timezone
from typing import Any

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from legal_ai.models.document import Target
from legal_ai.models.schemas import TargetPayload
from legal_ai.repositories.source import SourceRepository


class TargetRepository:
    def __init__(self) -> None:
        self.source_store = SourceRepository()

    def get_dict_data(self, object: TargetPayload) -> dict[str, Any]:
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
            targets_dicts.append(self.get_dict_data(target_payload))

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
