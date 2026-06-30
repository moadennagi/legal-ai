import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import aiohttp
import io
from legal_ai.database import get_session
from legal_ai.interfaces import DownloaderInterface
from legal_ai.models.document import Document
from legal_ai.models.schemas import TargetSchema
from legal_ai.repositories.document import DocumentRepository
from legal_ai.repositories.target import TargetRepository
from legal_ai.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def write_document_to_path(self, content: bytes, number: str) -> str:
        """Write pdf content to path"""
        file_path = self._create_file_path(number, content)
        with open(file_path, "wb") as fp:
            fp.write(content)
        return str(file_path)

    def read_document_file_content(self, number: str, data_dir: str) -> bytes | None:
        """Read document file content"""
        file_path = self._get_existing_file_path(number, data_dir)
        if not file_path:
            return None
        with open(file_path, "rb") as fp:
            data = fp.read()
        return data

    def target_file_exists(self, number: str, data_dir: str = settings.file_path) -> bool | None:
        """Return whether the pdf file exists for the target in the destination folder"""
        existing_path = self._get_existing_file_path(number, data_dir)
        if not existing_path:
            return None
        file_path = Path(existing_path)
        exists = file_path.is_file()
        if exists:
            logger.info(f"Found file for document {file_path}")
        return exists

    def _get_existing_file_path(
        self, number: str, data_dir: str = settings.file_path
    ) -> Path | None:
        """Return file path as FILE_PATH joined with number"""
        res = Path(data_dir).glob(f"{number}*")
        data: dict[str, Path] = {}
        for obj in res:
            _, extension = os.path.splitext(obj)
            target_file = Path(f"{os.path.join(data_dir, number)}")
            if obj != target_file:
                continue
            data[extension] = obj
        if ".pdf" in data:
            return data[".pdf"]
        if not len(data.values()):
            return None
        return list(data.values())[0]

    def _create_file_path(self, number: str, content: bytes):
        """Create file"""
        file_path = os.path.join(settings.file_path, number)
        extension = self._get_content_type(content)
        if not extension:
            raise ValueError("Content type could be inferred")

        return Path(f"{file_path}.{extension}")

    def _get_content_type(self, content: bytes) -> str | None:
        """Return the type of the content: pdf or docx"""
        if content.startswith(b"%PDF"):
            return "pdf"
        elif content.startswith(b"PK\x03\x04"):
            try:
                with zipfile.ZipFile(io.BytesIO(content), "r") as z:
                    z.testzip()
                    if "word/" in z.namelist():
                        return "docx"
                    else:
                        return None
            except zipfile.BadZipFile:
                return None

        return None

    async def download_target_content_and_insert_document(
        self,
        target: TargetSchema,
        downloader: DownloaderInterface,
        document_repository: DocumentRepository,
        target_repository: TargetRepository,
        http_session: aiohttp.ClientSession,
        overwrite_downloaded_file: bool = False,
        data_dir: str = settings.file_path,
    ) -> Document:
        """Download target content"""
        target.claimed_at = int(datetime.now(tz=timezone.utc).timestamp())
        document = document_repository.construct_document_from_target_payload(target)
        # this should only check if the file exists, currently this does a lot of things: 
        # 
        file_exists = self.target_file_exists(number=target.number, data_dir=data_dir)

        # only download files if they do not exist
        if not file_exists or overwrite_downloaded_file is True:
            logger.info(f"Downloading file for document {target.number}")
            content = await downloader.download_document(url=target.url, http_session=http_session)
            self.write_document_to_path(content, target.number)

        document.file_path = str(self._get_existing_file_path(target.number, data_dir))
        with get_session() as session:
            document_id = document_repository.insert_single_document(
                session=session, document=document
            )
            assert target.row_id
            target_repository.update_target_document_id(
                session=session, target_id=target.row_id, document_id=document_id
            )
        return document
