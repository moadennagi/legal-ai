from abc import ABC, abstractmethod
import aiohttp
from legal_ai.models.schemas import TargetPayload


class CrawlerInterface(ABC):
    url: str
    name: str

    @abstractmethod
    async def crawl_and_return_targets(self, task_id: int) -> list[TargetPayload]:
        pass


class DownloaderInterface(ABC):
    @abstractmethod
    async def download_document(self, url: str, http_session: aiohttp.ClientSession) -> bytes:
        pass
