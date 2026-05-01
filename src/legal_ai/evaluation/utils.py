import csv
import re
from pathlib import Path

from sqlalchemy import select

from legal_ai.database import get_session
from legal_ai.models.document import Document, DocumentChunk
from legal_ai.models.schemas import EvaluationDatasetRow


CSV_HEADERS = [
    "context",
    "question",
    "ground_truth",
    "source_doc",
    "source_article",
    "chunk_index",
    "official_date",
]

ARTICLE_PATTERN = re.compile(r"\bArticle\s+([0-9]+(?:[-./][0-9]+)?)", re.IGNORECASE)
NOISE_MARKERS = (
    "ABONNEMENT",
    "IMPRIMERIE OFFICIELLE",
    "SOMMAIRE",
)


def load_eval_dataset(csv_path: str) -> list[EvaluationDatasetRow]:
    """Load a completed RAGAS evaluation CSV (question + ground_truth filled)."""
    rows: list[EvaluationDatasetRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["question"].strip() or not row["ground_truth"].strip():
                continue
            rows.append(
                EvaluationDatasetRow(
                    question=row["question"].strip(),
                    ground_truth=row["ground_truth"].strip(),
                )
            )
    return rows


def _normalize_context(raw: str) -> str:
    # Keep line breaks for readability while removing empty/noisy lines.
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_source_article(context: str) -> str:
    match = ARTICLE_PATTERN.search(context)
    if not match:
        return ""
    return match.group(1)


def _is_noise_chunk(context: str) -> bool:
    head = context[:600].upper()
    return any(marker in head for marker in NOISE_MARKERS)


def export_ragas_seed_csv(
    output_path: str,
    source_id: int | None = None,
    min_chars: int = 250,
    max_chars: int = 3000,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    """Export chunk contexts into a CSV staging file for manual QA authoring."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} already exists. Set overwrite=True or choose another output path."
        )

    stmt = (
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.content.is_not(None))
        .order_by(Document.official_date.desc(), Document.number.desc(), DocumentChunk.chunk_index)
    )

    if source_id is not None:
        stmt = stmt.where(Document.source_id == source_id)

    rows_written = 0
    with get_session() as session, output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for chunk, document in session.execute(stmt):
            context = _normalize_context(chunk.content)
            if len(context) < min_chars:
                continue
            if _is_noise_chunk(context):
                continue

            if len(context) > max_chars:
                context = context[:max_chars].rstrip() + "\n[...]"

            writer.writerow(
                {
                    "context": context,
                    "question": "",
                    "ground_truth": "",
                    "source_doc": document.number,
                    "source_article": _extract_source_article(context),
                    "chunk_index": chunk.chunk_index,
                    "official_date": document.official_date.isoformat(),
                }
            )
            rows_written += 1

            if limit is not None and rows_written >= limit:
                break

    return rows_written
