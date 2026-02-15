from abc import ABC, abstractmethod
from legal_ai.models.schemas import TargetPayload


class CrawlerInterface(ABC):
    url: str
    name: str

    @abstractmethod
    async def crawl_and_return_targets() -> list[TargetPayload]:
        pass
