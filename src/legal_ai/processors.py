import os
import aiohttp
from legal_ai.interfaces import DownloaderInterface
from legal_ai.repositories.document import DocumentRepository
from legal_ai.models.document import Target, Document
from legal_ai.models.schemas import TargetPayload
from datetime import datetime, timezone


class DocumentProcessor:
    def write_document_to_path(self, destination_path: str, content: bytes, number: str):
        file_path = os.path.join(destination_path, number)
        with open(f"{file_path}.pdf", "wb") as fp:
            fp.write(content)
        return f"{file_path}.pdf"

    async def process_target(
        self,
        target: TargetPayload,
        downloader: DownloaderInterface,
        target_repository: DocumentRepository,
        http_session: aiohttp.ClientSession
    ) -> Document:
        """Download taregt content"""
        target.claimed_at = int(datetime.now(tz=timezone.utc).timestamp())
        content = await downloader.download_document(url=target.url, http_session=http_session)
        document = target_repository.construct_document_from_target_payload(target)
        file_path = self.write_document_to_path(
            "/home/moadennagi/projects/legal-ai/data", content, target.number
        )
        document.file_path = file_path
        return document
