from pydantic import BaseModel, Field
from datetime import datetime, timezone, date
from legal_ai.models.document import TaskStatus
from dataclasses import dataclass
from typing import Any


class SourceSchema(BaseModel):
    name: str
    url: str
    id: int | None = None


class TaskSchema(BaseModel):
    id: int | None = None
    source_id: int
    status: TaskStatus


class TargetSchema(BaseModel):
    row_id: int | None = None
    number: str
    url: str
    task_id: int | None = None
    source_id: int | None = None
    document_id: int | None = None
    official_date: date
    claimed_at: int | None = None
    updated_at: int | None = None
    created_at: int = Field(default_factory=lambda: int(datetime.now(tz=timezone.utc).timestamp()))
    # TODO: add validation rules (URL scheme, number format) if needed
    source: SourceSchema | None = None


class DocumentSchema(BaseModel):
    id: int
    number: str
    text_content: str | None = None
    file_path: str


@dataclass
class SingleTurnSample:
    user_input: str
    response: str
    retrieved_contexts: list[dict[str, Any]]
