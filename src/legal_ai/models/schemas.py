from pydantic import BaseModel, Field
from datetime import datetime, timezone
from legal_ai.models.document import TaskStatus


class SourcePayload(BaseModel):
    name: str
    url: str
    id: int | None = None


class TaskPayload(BaseModel):
    id: int | None = None
    source_id: int
    status: TaskStatus


class TargetPayload(BaseModel):
    number: str
    url: str
    content: str | None = None
    task_id: int | None = None
    source_id: int | None = None
    document_id: int | None = None
    claimed_at: int | None = None
    updated_at: int | None = None
    created_at: int = Field(default_factory=lambda: int(datetime.now(tz=timezone.utc).timestamp()))
    # TODO: add validation rules (URL scheme, number format) if needed
    source: SourcePayload | None = None
