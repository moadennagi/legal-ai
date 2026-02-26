from abc import ABC, abstractmethod
import aiohttp
from legal_ai.models.schemas import TargetSchema
from typing import Any
from legal_ai.models.document import Document
from dataclasses import dataclass, field


class CrawlerInterface(ABC):
    url: str
    name: str

    @abstractmethod
    async def crawl_and_return_targets(self, task_id: int) -> list[TargetSchema]:
        pass


class DownloaderInterface(ABC):
    @abstractmethod
    async def download_document(self, url: str, http_session: aiohttp.ClientSession) -> bytes:
        pass


class DocumentConverterInterface(ABC):
    @abstractmethod
    def convert(self, file_path: str) -> str:
        pass


@dataclass
class ChunkResult:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


class DocumentSplitterInterface(ABC):
    @abstractmethod
    def split_document(self, document: Document) -> list[ChunkResult]:
        pass

    @abstractmethod
    def construct_enriched_content(self, chunk: ChunkResult) -> str:
        pass


class LLMClientInterface(ABC):
    @abstractmethod
    async def embeddings(
        self,
        model: str,
        prompt: str,
    ) -> list[float]:
        pass

    @abstractmethod
    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        pass


class EmbeddingServiceInterface(ABC):
    def __init__(self, llm_client: LLMClientInterface) -> None:
        self.llm_client = llm_client

    @abstractmethod
    async def split_and_insert_embeddings(self, documents: list[Document]) -> None:
        pass


class RAGInterface(ABC):
    @abstractmethod
    async def ask(
        self,
        user_query: str,
        similarity_threshold: float,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        pass
