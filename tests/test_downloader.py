from unittest.mock import AsyncMock, Mock
import pytest
from legal_ai.downloader import Downloader


@pytest.fixture
def downloader():
    return Downloader()


@pytest.fixture
def mock_session():
    session = AsyncMock()
    return session


# ── download_document ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_download_document_returns_bytes(downloader, mock_session):
    expected = b"%PDF-1.4 fake content"
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.content.read = AsyncMock(return_value=expected)
    mock_session.get = AsyncMock(return_value=mock_response)

    result = await downloader.download_document("https://example.com/doc.pdf", mock_session)

    assert result == expected
    mock_session.get.assert_awaited_once_with("https://example.com/doc.pdf")


@pytest.mark.asyncio
async def test_download_document_raises_on_http_error(downloader, mock_session):
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock(side_effect=Exception("HTTP 404"))
    mock_session.get = AsyncMock(return_value=mock_response)

    with pytest.raises(Exception, match="HTTP 404"):
        await downloader.download_document("https://example.com/missing.pdf", mock_session)


@pytest.mark.asyncio
async def test_download_document_returns_empty_bytes(downloader, mock_session):
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.content.read = AsyncMock(return_value=b"")
    mock_session.get = AsyncMock(return_value=mock_response)

    result = await downloader.download_document("https://example.com/empty.pdf", mock_session)

    assert result == b""


@pytest.mark.asyncio
async def test_download_document_calls_raise_for_status(downloader, mock_session):
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.content.read = AsyncMock(return_value=b"data")
    mock_session.get = AsyncMock(return_value=mock_response)

    await downloader.download_document("https://example.com/doc.pdf", mock_session)

    mock_response.raise_for_status.assert_called_once()
