from bs4 import BeautifulSoup

from abc import ABC, abstractmethod
from morocco_legal_ai.models import Target


class CrawlerInterface(ABC):
    @abstractmethod
    async def crawl_and_return_targets(self, url: str) -> list[Target]:
        """
        Crawl the url qnd return a list of Document

        :param self: Description
        :param url: Description
        :type url: str
        """
        pass
