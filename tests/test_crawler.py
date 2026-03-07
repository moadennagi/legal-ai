from unittest.mock import patch, AsyncMock, Mock
import pytest
from legal_ai.crawlers.sgg_crawler import SGGCrawler
from legal_ai.models.schemas import SourceSchema


@pytest.fixture
def crawler():
    sut = SGGCrawler()
    return sut


# ── source property ──────────────────────────────────────────────────────────


def test_source_returns_correct_schema(crawler):
    source = crawler.source
    assert isinstance(source, SourceSchema)
    assert source.name == "sgg"
    assert source.url == "https://www.sgg.gov.ma/BulletinOfficiel.aspx"


# ── _get_page_content ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("legal_ai.crawlers.sgg_crawler.aiohttp.ClientSession")
async def test_get_page_content(mock_session_class, crawler):
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.content.read = AsyncMock(return_value="dummy content")
    mock_response.raise_for_status = Mock()
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_session_class.return_value = mock_session

    res = await crawler._get_page_content()
    assert res == "dummy content"


@pytest.mark.asyncio
@patch("legal_ai.crawlers.sgg_crawler.aiohttp.ClientSession")
async def test_get_page_content_raises_on_http_error(mock_session_class, crawler):
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock(side_effect=Exception("HTTP 500"))
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_session_class.return_value = mock_session

    with pytest.raises(Exception, match="HTTP 500"):
        await crawler._get_page_content()


# ── _extract_verification_token ───────────────────────────────────────────────


def test_extract_verification_token_returns_token(crawler):
    html = b'<html><body><input name="__RequestVerificationToken" value="abc123" /></body></html>'
    token = crawler._extract_verification_token(html)
    assert token == "abc123"


def test_extract_verification_token_raises_when_missing(crawler):
    html = b"<html><body><p>No token here</p></body></html>"
    with pytest.raises(ValueError):
        crawler._extract_verification_token(html)


# ── crawl_and_return_targets ─────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("legal_ai.crawlers.sgg_crawler.aiohttp.ClientSession")
async def test_crawl_and_return_targets_returns_list(mock_session_class, crawler):
    html_with_token = (
        b'<html><body><input name="__RequestVerificationToken" value="tok" /></body></html>'
    )
    api_payload = [
        {"BoUrl": "/BO/2024/1234.pdf", "BoNum": "1234", "BoDate": "/Date(1682982000000)/"},
        {"BoUrl": "/BO/2024/5678.pdf", "BoNum": "5678", "BoDate": "/Date(1685574000000)/"},
    ]

    mock_page_response = AsyncMock()
    mock_page_response.content.read = AsyncMock(return_value=html_with_token)
    mock_page_response.raise_for_status = Mock()

    mock_api_response = AsyncMock()
    mock_api_response.raise_for_status = Mock()
    mock_api_response.json = AsyncMock(return_value=api_payload)

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(side_effect=[mock_page_response, mock_api_response])
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_session_class.return_value = mock_session

    targets = await crawler.crawl_and_return_targets(task_id=1)

    assert len(targets) == 2
    assert targets[0].number == "1234"
    assert targets[1].number == "5678"
    assert targets[0].url.startswith("https://www.sgg.gov.ma/")
    assert targets[0].task_id == 1
    assert targets[0].source.name == "sgg"


@pytest.mark.asyncio
@patch("legal_ai.crawlers.sgg_crawler.aiohttp.ClientSession")
async def test_crawl_and_return_targets_raises_when_no_page_content(mock_session_class, crawler):
    mock_page_response = AsyncMock()
    mock_page_response.content.read = AsyncMock(return_value=None)
    mock_page_response.raise_for_status = Mock()

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_page_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_session_class.return_value = mock_session

    with pytest.raises(ValueError):
        await crawler.crawl_and_return_targets(task_id=1)


@pytest.mark.asyncio
@patch("legal_ai.crawlers.sgg_crawler.aiohttp.ClientSession")
async def test_crawl_and_return_targets_empty_api_response(mock_session_class, crawler):
    html_with_token = (
        b'<html><body><input name="__RequestVerificationToken" value="tok" /></body></html>'
    )

    mock_page_response = AsyncMock()
    mock_page_response.content.read = AsyncMock(return_value=html_with_token)
    mock_page_response.raise_for_status = Mock()

    mock_api_response = AsyncMock()
    mock_api_response.raise_for_status = Mock()
    mock_api_response.json = AsyncMock(return_value=[])

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(side_effect=[mock_page_response, mock_api_response])
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_session_class.return_value = mock_session

    targets = await crawler.crawl_and_return_targets(task_id=1)
    assert targets == []
