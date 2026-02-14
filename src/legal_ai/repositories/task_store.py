from sqlalchemy.orm import Session
from legal_ai.models.document import Task
from legal_ai.models.enums import TaskStatus, TaskType
from datetime import datetime, timezone

class TaskStore:
    def create_a_crawling_task(self, session: Session, source_id: int) -> Task:
        """Create a crawling task and return its id"""
        task = Task(
            type=TaskType.crawling,
            status=TaskStatus.in_progress,
            created_at=datetime.now(tz=timezone.utc).timestamp(),
            source_id=source_id
        )
        session.add(task)
        session.flush()
        return task
