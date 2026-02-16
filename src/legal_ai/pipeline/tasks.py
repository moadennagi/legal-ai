from legal_ai.interfaces import CrawlerInterface, DownloaderInterface
from legal_ai.repositories.source import SourceRepository
from legal_ai.repositories.task import TaskRepository
from legal_ai.repositories.document import DocumentRepository, TargetRespository
from legal_ai.models.document import TaskStatus, Target, Document
from legal_ai.models.schemas import TargetPayload
from legal_ai.database import get_session
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from typing import Coroutine
from processors import DocumentProcessor
import asyncio
from typing import Any
import aiohttp
import logging

logger = logging.basicConfig(level=logging.DEBUG)


async def crawl_and_insert_targets(
    crawler: CrawlerInterface,
    source_store: SourceRepository,
    task_store: TaskRepository,
    target_store: TargetRespository,
):
    """Run the given crawler and insert targets"""
    with get_session() as session:
        # create or get the source
        source = source_store.get_or_create_source(
            session, source_name=crawler.name, source_url=crawler.url
        )
        # create a crawling task for the source
        task = task_store.create_a_crawling_task(session, source.id)
        # crawl the source
        try:
            targets_payload = await crawler.crawl_and_return_targets(task.id)
            # mark the task as finished
        except Exception:
            task.status = TaskStatus.failed
            session.add(task)
            session.flush()
            raise

        # insert targets
        target_store.insert_targets(session, targets_payload)
        task.status = TaskStatus.succeeded
        session.flush()


def get_targets(
    task_store: TaskRepository,
    target_repository: TargetRespository,
) -> list[TargetPayload]:
    """Get all target from recent tasks"""
    # retreive targets for tasks (recent maybe ?)
    # download documents (pdf content)
    # insert documents (instances of Document)
    with get_session() as session:
        tasks = task_store.get_tasks(session)
        task_ids: list[int] = [task.id for task in tasks]
        # only get the targets for which the documents have no
        stmt = select(Target).where(Target.task_id.in_(task_ids))
        targets = session.execute(stmt).scalars().all()
        targets_payload: list[TargetPayload] = [
            target_repository.construct_target_payload_from_target(target) for target in targets
        ]
    return targets_payload


async def download_with_semaphore(
    sem: asyncio.Semaphore, coroutine: Coroutine[None, None, Document]
):
    """Download with semaphore"""
    async with sem:
        return await coroutine


async def download_target_contents_and_insert_documents(
    downloader: DownloaderInterface,
    task_store: TaskRepository,
    target_repository: TargetRespository,
    document_repository: DocumentRepository,
    document_processor: DocumentProcessor,
):
    """Download documents for every crawling task without a download task"""
    coroutines: list[Coroutine[None, None, Document]] = []
    # documents: list[Document] = []
    targets_payload = get_targets(task_store=task_store, target_repository=target_repository)
    timeout = aiohttp.ClientTimeout(60.0 * 5)
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession(timeout=timeout) as http_session:
        for target in targets_payload:
            coroutine = document_processor.process_target(
                http_session=http_session,
                target=target,
                downloader=downloader,
                target_repository=document_repository,
            )
            coroutines.append(download_with_semaphore(sem, coroutine))
            # document = await document_processor.process_target(
            #     http_session=http_session,
            #     target=target,
            #     downloader=downloader,
            #     target_repository=target_repository,
            # )
            # documents.append(document)
        documents = await asyncio.gather(*coroutines, return_exceptions=True)
    documents_to_insert: list[dict[str, Any]] = []
    with get_session() as session:
        for document in documents:
            if isinstance(document, BaseException):
                continue
            else:
                data_to_insert = target_repository.create_insertion_data(document)
                documents_to_insert.append(data_to_insert)
        insert_stmt = insert(Document).values(documents_to_insert)
        insert_stmt = insert_stmt.on_conflict_do_nothing(
            index_elements=["source_id", "number"],
        )
        session.execute(insert_stmt)
