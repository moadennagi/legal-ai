import aiohttp

from legal_ai.interfaces import DownloaderInterface


class Downloader(DownloaderInterface):
    async def download_document(self, url: str, http_session: aiohttp.ClientSession) -> bytes:
        """Given a target, download its content (url), the content is a pdf"""
        res = await http_session.get(url)
        res.raise_for_status()
        content = await res.content.read()
        return content
