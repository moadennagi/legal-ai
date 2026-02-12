from bs4 import BeautifulSoup
import aiohttp
from typing import Any
from morocco_legal_ai.models import Target
from morocco_legal_ai.interfaces.crawler import CrawlerInterface

URL = "https://www.sgg.gov.ma/BulletinOfficiel.aspx"
API_URL = "https://www.sgg.gov.ma/DesktopModules/MVC/TableListBO/BO/AjaxMethod"
BASE_URL = "https://www.sgg.gov.ma/"


class Crawler(CrawlerInterface):
    async def _get_page_content(self, url: str) -> str | None:
        """Get page content and return a response."""
        async with aiohttp.ClientSession() as session:
            response = await session.get(url)
            response.raise_for_status()
            html = await response.content.read()
            return html

    def _extract_verification_token(self, page_content: str):
        """
        Construct a beautiful soup instance, parse and return
        the verification token.
        """
        soup = BeautifulSoup(page_content, "html.parser")
        token = soup.select_one("input[name=__RequestVerificationToken]")
        if not token:
            raise ValueError()
        return token.attrs["value"]

    async def crawl_and_return_targets(self, url):
        """
        Get page content, parse target info and return a list of Target
        instances.
        """
        targets: list[Target] = []
        page_content = await self._get_page_content(url)
        token = self._extract_verification_token(page_content)
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "ModuleId": "2873",
            "TabId": "775",
            "RequestVerificationToken": token,
        }
        json: list[dict[str, Any]] = {}
        async with aiohttp.ClientSession() as session:
            response = await session.get(API_URL, headers=headers)
            response.raise_for_status()
            json = await response.json()

        # parse the json
        for obj in json:
            target = Target(url=obj["BoUrl"], number=obj["BoNum"])
            targets.append(target)
        return targets


if __name__ == "__main__":
    import asyncio
    import json

    crawler = Crawler()

    async def main():
        targets = await crawler.crawl_and_return_targets(URL)
        with open("targets.json", "w") as fp:
            data = [{"bo_num": target.number, "bo_url": target.url} for target in targets]
            json.dump(data, fp)

    asyncio.run(main())
