from sqlalchemy import ForeignKey, UniqueConstraint
from datetime import date
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from enum import Enum


class TaskStatus(Enum):
    succeeded = "succeeded"
    failed = "failed"
    in_progress = "in_progress"


class TaskType(Enum):
    crawling = "crawling"
    download = "download"


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    url: Mapped[str]

    tasks: Mapped[list["Task"]] = relationship(back_populates="source")
    targets: Mapped[list["Target"]] = relationship(back_populates="source")
    documents: Mapped[list["Document"]] = relationship(back_populates="source")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str]
    number: Mapped[str]
    file_path: Mapped[str]

    text_content: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[int]
    updated_at: Mapped[int | None] = mapped_column(nullable=True)

    official_date: Mapped[date]

    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))

    source: Mapped["Source"] = relationship(back_populates="documents")
    target: Mapped["Target"] = relationship(
        back_populates="document", single_parent=True, uselist=False
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")

    __table_args__ = (UniqueConstraint("number", "source_id"),)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)

    content: Mapped[str]
    chunk_index: Mapped[int]
    embedding: Mapped[Vector | None] = mapped_column(Vector(768), nullable=True)

    chunk_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    created_at: Mapped[int]
    updated_at: Mapped[int | None] = mapped_column(nullable=True)

    token_count: Mapped[int | None] = mapped_column(nullable=True)
    start_char: Mapped[int | None] = mapped_column(nullable=True)
    end_char: Mapped[int | None] = mapped_column(nullable=True)

    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)


class Target(Base):
    __tablename__ = "targets"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str]
    number: Mapped[str]

    created_at: Mapped[int]
    updated_at: Mapped[int | None] = mapped_column(nullable=True)
    claimed_at: Mapped[int | None] = mapped_column(nullable=True)

    official_date: Mapped[date]

    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"))
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))

    document: Mapped["Document"] = relationship(back_populates="target", single_parent=True)
    task: Mapped["Task"] = relationship(back_populates="targets")
    source: Mapped["Source"] = relationship(back_populates="targets")

    __table_args__ = (UniqueConstraint("source_id", "number"),)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[TaskStatus]
    created_at: Mapped[int]
    updated_at: Mapped[int | None] = mapped_column(nullable=True)

    type: Mapped[TaskType]

    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    source: Mapped["Source"] = relationship(back_populates="tasks")

    parent_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    parent: Mapped["Task"] = relationship(back_populates="children", remote_side=[id])
    children: Mapped[list["Task"]] = relationship(back_populates="parent")

    targets: Mapped[list["Target"]] = relationship(back_populates="task")
