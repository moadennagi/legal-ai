from pydantic import BaseModel, Field
from datetime import datetime, timezone


class TargetPayload(BaseModel):
    number: str
    url: str
    content: str | None = None
    task_id: int | None = None
    source_id: int | None = None
    document_id: int
    claimed_at: int | None = None
    created_at: float = Field(default_factory=lambda: datetime.now(tz=timezone.utc).timestamp())
    # TODO: add validation rules (URL scheme, number format) if needed


class SourcePayload(BaseModel):
    name: str
    url: str
    id: int | None = None


class TaskPayload(BaseModel):
    id: int | None = None
