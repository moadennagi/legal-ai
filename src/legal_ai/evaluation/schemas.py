from dataclasses import dataclass


@dataclass
class EvaluationDatasetRow:
    question: str
    ground_truth: str
    chunk_index: int | None = None
    context: str | None = None
    source_doc: str | None = None
    source_article: str | None = None
    official_date: str | None = None
