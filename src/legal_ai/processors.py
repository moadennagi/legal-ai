import os
import aiohttp
import logging
from pathlib import Path
from legal_ai.interfaces import DownloaderInterface
from legal_ai.repositories.document import DocumentRepository
from legal_ai.models.document import Document
from legal_ai.models.schemas import TargetPayload
from datetime import datetime, timezone
from legal_ai.settings import settings
from legal_ai.database import get_session

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def write_document_to_path(self, content: bytes, number: str) -> str:
        """Write pdf content to path"""
        file_path = self._get_file_path(number)
        with open(f"{file_path}.pdf", "wb") as fp:
            fp.write(content)
        return f"{file_path}.pdf"

    def read_document_file_content(self, number: str) -> bytes:
        """Read document file content"""
        file_path = self._get_file_path(number)
        with open(file_path, "rb") as fp:
            data = fp.read()
        return data

    def target_file_exists(self, number: str) -> bool:
        """Return whether the pdf file exists for the target in the destination folder"""
        file_path = Path(self._get_file_path(number))
        exists = file_path.is_file()
        if exists:
            logger.info(f"Found file for document {file_path}")
        return exists

    def _get_file_path(self, number: str) -> str:
        """Return file path as FILE_PATH joined with number"""
        file_path = os.path.join(settings.file_path, number)
        return f"{file_path}.pdf"

    async def download_target_content(
        self,
        target: TargetPayload,
        downloader: DownloaderInterface,
        document_repository: DocumentRepository,
        http_session: aiohttp.ClientSession,
        overwrite_downloaded_file: bool = False,
    ) -> Document:
        """Download target content"""
        target.claimed_at = int(datetime.now(tz=timezone.utc).timestamp())
        document = document_repository.construct_document_from_target_payload(target)
        file_exists = self.target_file_exists(number=target.number)
        # only download files if they do not exist
        if not file_exists or overwrite_downloaded_file is True:
            logger.info(f"Downloading file for document {target.number}")
            content = await downloader.download_document(url=target.url, http_session=http_session)
            self.write_document_to_path(content, target.number)

        document.file_path = self._get_file_path(target.number)
        with get_session() as session:
            document_repository.insert_single_document(session=session, document=document)
        return document
