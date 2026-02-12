from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from morocco_legal_ai.models.enums import TaskStatus, TaskType


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
    content: Mapped[str]

    created_at: Mapped[int]
    updated_at: Mapped[int] | None = mapped_column(nullable=True)

    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))

    target: Mapped["Target"] = relationship(back_populates="document", single_parent=True)

    __table_args__ = (UniqueConstraint("number", "source_id"),)


class Target(Base):
    __tablename__ = "targets"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str]
    number: Mapped[str]

    created_at: Mapped[int]
    updated_at: Mapped[int] | None = mapped_column(nullable=True)
    claimed_at: Mapped[int] | None = mapped_column(nullable=True)

    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"))
    document_id: Mapped[int] | None = mapped_column(ForeignKey("documents.id"), nullable=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))

    document: Mapped["Document"] = relationship(back_populates="target", single_parent=True)
    task: Mapped["Task"] = relationship(back_populates="target")
    source: Mapped["Source"] = relationship(back_populates="target")

    __table_args__ = (UniqueConstraint("number", "source_id"),)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[TaskStatus]
    created_at: Mapped[int]
    updated_at: Mapped[int]

    type: Mapped[TaskType]

    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))

    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"))
    task: Mapped["Task"] = relationship(back_populates="task")

    targets: Mapped[list["Target"]] = relationship(back_populates="task")
