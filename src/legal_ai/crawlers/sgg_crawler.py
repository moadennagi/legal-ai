from bs4 import BeautifulSoup
import aiohttp
from urllib.parse import urljoin
from typing import Any
from legal_ai.models.schemas import TargetSchema, SourceSchema
from legal_ai.interfaces import CrawlerInterface
from legal_ai.utils import parse_ms_json_date


class SGGCrawler(CrawlerInterface):
    name = "sgg"
    url = "https://www.sgg.gov.ma/BulletinOfficiel.aspx"
    base_url = "https://www.sgg.gov.ma/"
    api_url = "https://www.sgg.gov.ma/DesktopModules/MVC/TableListBO/BO/AjaxMethod"

    @property
    def source(self):
        return SourceSchema(name=self.name, url=self.url)

    async def _get_page_content(self) -> bytes | None:
        """Get page content and return a response."""
        async with aiohttp.ClientSession() as session:
            response = await session.get(self.url)
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

    async def crawl_and_return_targets(self, task_id: int) -> list[TargetSchema]:
        """
        Get page content, parse target info and return a list of Target
        instances.
        """
        targets: list[TargetSchema] = []
        page_content = await self._get_page_content()
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
            response = await session.get(self.api_url, headers=headers)
            response.raise_for_status()
            json = await response.json()

        # parse the json
        for obj in json:
            url = urljoin(self.base_url, obj["BoUrl"])
            official_date = parse_ms_json_date(obj["BoDate"])
            target = TargetSchema(
                url=url,
                number=obj["BoNum"],
                source=self.source,
                task_id=task_id,
                official_date=official_date,
            )
            targets.append(target)
        return targets
