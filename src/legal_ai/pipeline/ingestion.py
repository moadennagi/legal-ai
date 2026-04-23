import asyncio
import gc
import logging
from typing import Coroutine
from tqdm import tqdm
import aiohttp
from legal_ai.processors import DocumentProcessor
from sqlalchemy import select
from sqlalchemy.orm import Session

from legal_ai.database import get_session
from legal_ai.interfaces import CrawlerInterface
from legal_ai.models.document import Document, Target, TaskStatus
from legal_ai.models.schemas import TargetSchema, DocumentSchema
from legal_ai.repositories.document import DocumentRepository
from legal_ai.repositories.target import TargetRepository
from legal_ai.repositories.source import SourceRepository
from legal_ai.repositories.task import TaskRepository
from legal_ai.utils import run_with_semaphore
from legal_ai.downloader import Downloader
from legal_ai.settings import settings
from legal_ai.interfaces import DocumentConverterInterface

logger = logging.getLogger(__name__)


class DataIngesion:
    def __init__(self, document_converter: DocumentConverterInterface) -> None:
        self.target_repository = TargetRepository()
        self.document_repository = DocumentRepository()
        self.task_repository = TaskRepository()
        self.source_repository = SourceRepository()
        self.document_processor = DocumentProcessor()
        self.downloader = Downloader()
        self.document_repository = DocumentRepository()
        self.document_converter = document_converter

    def _collect_targets(self, session: Session) -> list[TargetSchema]:
        """Return a list of TargetPayload instances

        Returns:
            list[TargetPayload]: TargetPayload instances
        """
        # TODO: decide what tasks I zhould take, do I create a downloading task for every task ?
        tasks = self.task_repository.get_tasks(session)
        task_ids: list[int] = [task.id for task in tasks]
        logger.info(f"Collected {len(task_ids)} tasks from the database")
        # only get the targets for which the documents have no
        stmt = select(Target).where(Target.task_id.in_(task_ids))
        targets = session.execute(stmt).scalars().all()
        logger.info(f"Found {len(targets)} targets")
        targets_payload: list[TargetSchema] = [
            self.target_repository.construct_target_payload_from_target(target)
            for target in targets
        ]
        return targets_payload

    async def crawl_and_insert_targets(
        self,
        crawler: CrawlerInterface,
    ):
        """Run the given crawler and insert targets"""
        with get_session() as session:
            # create or get the source
            source = self.source_repository.get_or_create_source(
                session, source_name=crawler.name, source_url=crawler.url
            )
            # create a crawling task for the source
            task = self.task_repository.create_a_crawling_task(session, source.id)
            # crawl the source
            try:
                targets_payload = await crawler.crawl_and_return_targets(task.id)
                logger.info(f"Crawled {source.name}, found {len(targets_payload)}")
                # mark the task as finished
            except Exception:
                task.status = TaskStatus.failed
                session.add(task)
                session.flush()
                raise

            # insert targets
            res = self.target_repository.insert_targets(session, targets_payload)
            logger.info(f"Inserted {res} targets")
            task.status = TaskStatus.succeeded

    async def download_target_contents(self) -> list[Document | BaseException]:
        """Download documents for every crawling task without a download task"""
        # TODO: create a download task
        coroutines: list[Coroutine[None, None, Document]] = []
        timeout = aiohttp.ClientTimeout()
        sem = asyncio.Semaphore(settings.semaphore)

        with get_session() as session:
            targets_payload = self._collect_targets(session=session)

        async with aiohttp.ClientSession(timeout=timeout) as http_session:
            for target in targets_payload:
                coroutine = self.document_processor.download_target_content_and_insert_document(
                    http_session=http_session,
                    target=target,
                    downloader=self.downloader,
                    document_repository=self.document_repository,
                    target_repository=self.target_repository,
                )
                coroutines.append(run_with_semaphore(sem, coroutine))
            logger.info(f"Processing {len(coroutines)}")
            documents = await asyncio.gather(*coroutines, return_exceptions=True)
            return documents

    def extract_text_from_documents(self, documents: list[DocumentSchema]):
        """Extract text from documents and update them in the database

        Args:
            documents (list[Document]): list of documents
        """
        for document in tqdm(documents, desc="Extracting text", unit="doc"):
            with get_session() as session:
                doc_id = document.id
                try:
                    content = self.document_converter.convert(file_path=document.file_path)
                    self.document_repository.update_document_content(session, doc_id, content)
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to extract text from document {doc_id}: {e}")
                    continue
                finally:
                    gc.collect()

    def extract_text_from_documents_without_content(self):
        """
        Extract text from documents without text_content.
        """
        document_schemas: list[DocumentSchema] = []
        with get_session() as session:
            stmt = select(Document).where(Document.text_content.is_(None))
            res = session.execute(stmt).scalars().all()
            for row in res:
                document_schemas.append(
                    DocumentSchema(id=row.id, number=row.number, file_path=row.file_path)
                )
        self.extract_text_from_documents(documents=document_schemas)
