from abc import ABC, abstractmethod
from legal_ai.models.schemas import TargetPayload, SourcePayload

class CrawlerInterface(ABC):

    @abstractmethod
    async def crawl_and_return_targets(self, source: SourcePayload) -> list[TargetPayload]:
        pass
