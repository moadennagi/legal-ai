from datetime import date
from unittest.mock import AsyncMock, Mock, MagicMock, patch
import pytest

from legal_ai.processors import DocumentProcessor
from legal_ai.models.document import Document
from legal_ai.models.schemas import TargetSchema, SourceSchema
from legal_ai.repositories.document import DocumentRepository
from legal_ai.repositories.target import TargetRepository
from legal_ai.interfaces import DownloaderInterface


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def processor():
    return DocumentProcessor()


def make_target(number="1234", row_id=1, source_id=1):
    return TargetSchema(
        row_id=row_id,
        number=number,
        url=f"https://example.com/{number}.pdf",
        official_date=date(2024, 1, 1),
        source_id=source_id,
        source=SourceSchema(name="sgg", url="https://sgg.gov.ma"),
    )


# ── _get_file_path ────────────────────────────────────────────────────────────


def test_get_file_path_returns_correct_path(processor):
    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = "/data/pdfs"
        result = processor._get_file_path("1234")
    assert result == "/data/pdfs/1234.pdf"


def test_get_file_path_appends_pdf_extension(processor):
    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = "/tmp"
        result = processor._get_file_path("BO-5678")
    assert result.endswith(".pdf")
    assert "BO-5678" in result


# ── target_file_exists ────────────────────────────────────────────────────────


def test_target_file_exists_returns_true(processor, tmp_path):
    pdf = tmp_path / "1234.pdf"
    pdf.write_bytes(b"fake")

    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = str(tmp_path)
        assert processor.target_file_exists("1234") is True


def test_target_file_exists_returns_false(processor, tmp_path):
    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = str(tmp_path)
        assert processor.target_file_exists("9999") is False


# ── write_document_to_path ────────────────────────────────────────────────────


def test_write_document_to_path_creates_file(processor, tmp_path):
    content = b"%PDF-1.4 content"
    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = str(tmp_path)
        path = processor.write_document_to_path(content, "1234")

    assert (tmp_path / "1234.pdf").read_bytes() == content
    assert path == str(tmp_path / "1234.pdf")


def test_write_document_to_path_returns_file_path(processor, tmp_path):
    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = str(tmp_path)
        result = processor.write_document_to_path(b"data", "5678")

    assert result.endswith("5678.pdf")


# ── read_document_file_content ────────────────────────────────────────────────


def test_read_document_file_content_returns_bytes(processor, tmp_path):
    expected = b"PDF data here"
    (tmp_path / "1234.pdf").write_bytes(expected)

    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = str(tmp_path)
        result = processor.read_document_file_content("1234")

    assert result == expected


# ── download_target_content_and_insert_document ───────────────────────────────


@pytest.mark.asyncio
@patch("legal_ai.processors.get_session")
async def test_download_skips_download_when_file_exists(mock_get_session, processor, tmp_path):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_downloader = AsyncMock(spec=DownloaderInterface)
    mock_doc_repo = Mock(spec=DocumentRepository)
    mock_target_repo = Mock(spec=TargetRepository)
    mock_http_session = AsyncMock()

    mock_document = Mock(spec=Document)
    mock_doc_repo.construct_document_from_target_payload = Mock(return_value=mock_document)
    mock_doc_repo.insert_single_document = Mock(return_value=99)

    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = str(tmp_path)
        # Create the file so it appears to already exist
        (tmp_path / "1234.pdf").write_bytes(b"existing")

        await processor.download_target_content_and_insert_document(
            target=make_target(number="1234"),
            downloader=mock_downloader,
            document_repository=mock_doc_repo,
            target_repository=mock_target_repo,
            http_session=mock_http_session,
        )

    mock_downloader.download_document.assert_not_awaited()


@pytest.mark.asyncio
@patch("legal_ai.processors.get_session")
async def test_download_downloads_when_file_missing(mock_get_session, processor, tmp_path):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    pdf_content = b"%PDF content"
    mock_downloader = AsyncMock(spec=DownloaderInterface)
    mock_downloader.download_document = AsyncMock(return_value=pdf_content)

    mock_doc_repo = Mock(spec=DocumentRepository)
    mock_target_repo = Mock(spec=TargetRepository)
    mock_http_session = AsyncMock()

    mock_document = Mock(spec=Document)
    mock_doc_repo.construct_document_from_target_payload = Mock(return_value=mock_document)
    mock_doc_repo.insert_single_document = Mock(return_value=42)

    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = str(tmp_path)

        await processor.download_target_content_and_insert_document(
            target=make_target(number="9999"),
            downloader=mock_downloader,
            document_repository=mock_doc_repo,
            target_repository=mock_target_repo,
            http_session=mock_http_session,
        )

    mock_downloader.download_document.assert_awaited_once()
    assert (tmp_path / "9999.pdf").read_bytes() == pdf_content


@pytest.mark.asyncio
@patch("legal_ai.processors.get_session")
async def test_download_inserts_document_and_updates_target(mock_get_session, processor, tmp_path):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_downloader = AsyncMock(spec=DownloaderInterface)
    mock_downloader.download_document = AsyncMock(return_value=b"data")

    mock_doc_repo = Mock(spec=DocumentRepository)
    mock_target_repo = Mock(spec=TargetRepository)
    mock_http_session = AsyncMock()

    mock_document = Mock(spec=Document)
    mock_doc_repo.construct_document_from_target_payload = Mock(return_value=mock_document)
    mock_doc_repo.insert_single_document = Mock(return_value=55)

    target = make_target(number="0001", row_id=7)

    with patch("legal_ai.processors.settings") as mock_settings:
        mock_settings.file_path = str(tmp_path)

        result = await processor.download_target_content_and_insert_document(
            target=target,
            downloader=mock_downloader,
            document_repository=mock_doc_repo,
            target_repository=mock_target_repo,
            http_session=mock_http_session,
        )

    mock_doc_repo.insert_single_document.assert_called_once_with(
        session=mock_session, document=mock_document
    )
    mock_target_repo.update_target_document_id.assert_called_once_with(
        session=mock_session, target_id=7, document_id=55
    )
    assert result is mock_document
