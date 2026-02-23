from abc import ABC, abstractmethod
import aiohttp
from legal_ai.models.schemas import TargetPayload
from typing import Any


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


class ConvertedDocument(ABC):
    @abstractmethod
    def export_to_markdown(self) -> str:
        pass

    @abstractmethod
    def export_to_dict(self) -> dict[str, Any]:
        pass


class ConversionResultInterface(ABC):
    @property
    @abstractmethod
    def document(self) -> ConvertedDocument:
        pass


class DocumentConverterInterface(ABC):
    @abstractmethod
    def convert(self, file_path: str) -> ConversionResultInterface:
        pass
