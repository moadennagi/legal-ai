from sqlalchemy.orm import Session
from legal_ai.models.document import Task, TaskStatus, TaskType
from datetime import datetime, timezone
from sqlalchemy import select
from legal_ai.models.schemas import TaskSchema


class TaskRepository:
    def create_a_crawling_task(self, session: Session, source_id: int) -> Task:
        """Create a crawling task and return its id"""
        task = Task(
            type=TaskType.crawling,
            status=TaskStatus.in_progress,
            created_at=datetime.now(tz=timezone.utc).timestamp(),
            source_id=source_id,
        )
        session.add(task)
        session.flush()
        return task

    def get_tasks(self, session: Session) -> list[Task]:
        stmt = select(Task)
        tasks = session.execute(statement=stmt).scalars().all()
        return tasks

    def create_downloading_task(self, task_payload: TaskSchema):
        task = Task(
            parent_id=task_payload.id,
            status=TaskStatus.in_progress,
            source_id=task_payload.source_id,
        )
        task.created_at = int(datetime.now(tz=timezone.utc).timestamp())
        return task
