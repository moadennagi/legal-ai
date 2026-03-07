from unittest.mock import AsyncMock, Mock, MagicMock, patch
from datetime import date
import pytest

from legal_ai.pipeline.ingestion import DataIngesion
from legal_ai.interfaces import CrawlerInterface, DocumentConverterInterface
from legal_ai.models.document import TaskStatus
from legal_ai.models.schemas import TargetSchema, SourceSchema, DocumentSchema


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_converter():
    return Mock(spec=DocumentConverterInterface)


@pytest.fixture
def ingestion(mock_converter):
    sut = DataIngesion(mock_converter)
    # Replace all repositories with mocks so no DB is needed
    sut.source_repository = Mock()
    sut.task_repository = Mock()
    sut.target_repository = Mock()
    sut.document_repository = Mock()
    return sut


def make_target_schema(number="1234", source_id=1):
    return TargetSchema(
        number=number,
        url=f"https://example.com/{number}.pdf",
        official_date=date(2024, 1, 1),
        source_id=source_id,
        source=SourceSchema(name="sgg", url="https://sgg.gov.ma"),
    )


# ── _collect_targets ──────────────────────────────────────────────────────────


@patch("legal_ai.pipeline.ingestion.get_session")
def test_collect_targets_returns_target_schemas(mock_get_session, ingestion):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_task = Mock()
    mock_task.id = 1
    ingestion.task_repository.get_tasks = Mock(return_value=[mock_task])

    mock_raw_target = Mock()
    mock_session.execute.return_value.scalars.return_value.all.return_value = [mock_raw_target]

    expected_schema = make_target_schema()
    ingestion.target_repository.construct_target_payload_from_target = Mock(
        return_value=expected_schema
    )

    result = ingestion._collect_targets(mock_session)

    assert len(result) == 1
    assert result[0] is expected_schema
    ingestion.target_repository.construct_target_payload_from_target.assert_called_once_with(
        mock_raw_target
    )


@patch("legal_ai.pipeline.ingestion.get_session")
def test_collect_targets_returns_empty_list_when_no_tasks(mock_get_session, ingestion):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    ingestion.task_repository.get_tasks = Mock(return_value=[])
    mock_session.execute.return_value.scalars.return_value.all.return_value = []

    result = ingestion._collect_targets(mock_session)
    assert result == []


# ── crawl_and_insert_targets ──────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("legal_ai.pipeline.ingestion.get_session")
async def test_crawl_and_insert_targets_happy_path(mock_get_session, ingestion):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_source = Mock()
    mock_source.id = 1
    ingestion.source_repository.get_or_create_source = Mock(return_value=mock_source)

    mock_task = Mock()
    mock_task.id = 10
    ingestion.task_repository.create_a_crawling_task = Mock(return_value=mock_task)
    ingestion.target_repository.insert_targets = Mock(return_value=2)

    mock_crawler = AsyncMock(spec=CrawlerInterface)
    mock_crawler.name = "sgg"
    mock_crawler.url = "https://sgg.gov.ma"
    mock_crawler.crawl_and_return_targets = AsyncMock(
        return_value=[make_target_schema("1"), make_target_schema("2")]
    )

    await ingestion.crawl_and_insert_targets(mock_crawler)

    mock_crawler.crawl_and_return_targets.assert_awaited_once_with(mock_task.id)
    ingestion.target_repository.insert_targets.assert_called_once()
    assert mock_task.status == TaskStatus.succeeded


@pytest.mark.asyncio
@patch("legal_ai.pipeline.ingestion.get_session")
async def test_crawl_and_insert_targets_marks_task_failed_on_crawler_error(
    mock_get_session, ingestion
):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_source = Mock()
    mock_source.id = 1
    ingestion.source_repository.get_or_create_source = Mock(return_value=mock_source)

    mock_task = Mock()
    mock_task.id = 10
    ingestion.task_repository.create_a_crawling_task = Mock(return_value=mock_task)

    mock_crawler = AsyncMock(spec=CrawlerInterface)
    mock_crawler.name = "sgg"
    mock_crawler.url = "https://sgg.gov.ma"
    mock_crawler.crawl_and_return_targets = AsyncMock(side_effect=Exception("network error"))

    with pytest.raises(Exception, match="network error"):
        await ingestion.crawl_and_insert_targets(mock_crawler)

    assert mock_task.status == TaskStatus.failed
    mock_session.add.assert_called_once_with(mock_task)
    mock_session.flush.assert_called_once()


# ── extract_text_from_documents ───────────────────────────────────────────────


@patch("legal_ai.pipeline.ingestion.get_session")
def test_extract_text_from_documents_calls_converter_and_updates(
    mock_get_session, ingestion, mock_converter
):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_converter.convert = Mock(return_value="extracted text")

    docs = [DocumentSchema(id=1, number="1234", file_path="/path/doc.pdf")]
    ingestion.extract_text_from_documents(docs)

    mock_converter.convert.assert_called_once_with(file_path="/path/doc.pdf")
    ingestion.document_repository.update_document_content.assert_called_once_with(
        mock_session, 1, "extracted text"
    )


@patch("legal_ai.pipeline.ingestion.get_session")
def test_extract_text_from_documents_continues_on_converter_error(
    mock_get_session, ingestion, mock_converter
):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_converter.convert = Mock(side_effect=Exception("OCR failed"))

    docs = [
        DocumentSchema(id=1, number="0001", file_path="/a.pdf"),
        DocumentSchema(id=2, number="0002", file_path="/b.pdf"),
    ]

    # Must not raise
    ingestion.extract_text_from_documents(docs)

    ingestion.document_repository.update_document_content.assert_not_called()


@patch("legal_ai.pipeline.ingestion.get_session")
def test_extract_text_from_documents_processes_all_docs(
    mock_get_session, ingestion, mock_converter
):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_converter.convert = Mock(return_value="text")

    docs = [DocumentSchema(id=i, number=str(i), file_path=f"/{i}.pdf") for i in range(1, 4)]
    ingestion.extract_text_from_documents(docs)

    assert mock_converter.convert.call_count == 3
    assert ingestion.document_repository.update_document_content.call_count == 3
