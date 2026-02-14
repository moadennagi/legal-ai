from bs4 import BeautifulSoup
import aiohttp
from urllib.parse import urljoin
from typing import Any
from legal_ai.models.schemas import TargetPayload, SourcePayload
from legal_ai.constants import API_URL, BASE_URL

class Crawler:
    async def _get_page_content(self, url: str) -> bytes | None:
        """Get page content and return a response."""
        async with aiohttp.ClientSession() as session:
            response = await session.get(url)
            response.raise_for_status()
            html = await response.content.read()
            return html

    def _extract_verification_token(self, page_content: bytes) -> str:
        """
        Construct a beautiful soup instance, parse and return
        the verification token.
        """
        soup = BeautifulSoup(page_content, "html.parser")
        token = soup.select_one("input[name=__RequestVerificationToken]")
        if not token:
            raise ValueError()
        return str(token.attrs["value"])

    async def crawl_and_return_targets(self, source: SourcePayload) -> list[TargetPayload]:
        """
        Get page content, parse target info and return a list of Target
        instances.
        """
        targets: list[TargetPayload] = []
        page_content = await self._get_page_content(source.url)
        if not page_content:
            raise ValueError()
        token = self._extract_verification_token(page_content)
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "ModuleId": "2873",
            "TabId": "775",
            "RequestVerificationToken": token,
        }
        json: list[dict[str, Any]] = []
        async with aiohttp.ClientSession() as session:
            response = await session.get(API_URL, headers=headers)
            response.raise_for_status()
            json = await response.json()

        # parse the json
        for obj in json:
            url = urljoin(BASE_URL, obj["BoUrl"])
            target = TargetPayload(url=url, number=obj["BoNum"])
            targets.append(target)
        return targets
