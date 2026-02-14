from sqlalchemy.orm import Session
from legal_ai.models.document import Target
from typing import Any
from legal_ai.models.schemas import TargetPayload
from sqlalchemy.dialects.postgresql import insert


class TargetStore:
    def map_target_payload_to_target(self, target_payload: TargetPayload) -> Target:
        """Construct an instance of Target from TargetPayload"""
        target = Target(
            number=target_payload.number,
            url=target_payload.url,
            created_at=target_payload.created_at,
            task_id=target_payload.task_id,
            source_id=target_payload.source_id,
        )
        return target

    def insert_targets(self, session: Session, targets_payload: list[TargetPayload]) -> int:
        """Insert a list of Targets and return the count
        handles integrity error.
        """
        # transform basemodel to ORM
        targets: list[Target] = []
        for target_payload in targets_payload:
            targets.append(self.map_target_payload_to_target(target_payload))

        # insert targets into the database
        targets_dicts: list[dict[str, Any]] = []
        for target in targets:
            targets_dicts.append(
                {c.name: getattr(target, c.name) for c in Target.__table__.columns}
            )

        if targets_dicts:
            stmt = insert(Target).values(targets_dicts)
            stmt = stmt.on_conflict_do_nothing(index_elements=["source_id", "number"])
            session.execute(stmt)
