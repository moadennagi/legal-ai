from legal_ai.interfaces.crawler import CrawlerInterface
from legal_ai.repositories.source_store import SourceStore
from legal_ai.repositories.task_store import TaskStore
from legal_ai.repositories.target_store import TargetStore
from legal_ai.models.document import TaskStatus
from legal_ai.models.schemas import SourcePayload
from legal_ai.database import get_session


async def crawl_and_insert_targets(
    crawler: CrawlerInterface,
    source_store: SourceStore,
    task_store: TaskStore,
    target_store: TargetStore,
):
    with get_session() as session:
        # create or get the source
        source = source_store.get_or_create_source(
            session, source_name=crawler.name, source_url=crawler.url
        )
        source_data = SourcePayload(name=source.name, url=source.url)
        # create a crawling task for the source
        task = task_store.create_a_crawling_task(session, source.id)
        # crawl the source
        try:
            targets_payload = await crawler.crawl_and_return_targets(source_data)
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
