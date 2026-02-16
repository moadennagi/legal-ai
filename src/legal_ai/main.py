if __name__ == "__main__":
    import asyncio
    from legal_ai.pipeline.tasks import (
        crawl_and_insert_targets,
        download_target_contents_and_insert_documents,
    )
    from repositories import TaskRepository, SourceRepository, DocumentRepository, TargetRespository
    from crawlers.sgg_crawler import SGGCrawler
    from downloader import Downloader
    from processors import DocumentProcessor

    source_store = SourceRepository()
    target_store = TargetRespository()
    document_repository = DocumentRepository()
    task_store = TaskRepository()
    downloader = Downloader()
    document_processor = DocumentProcessor()

    # asyncio.run(
    #     crawl_and_insert_targets(
    #         task_store=task_store,
    #         source_store=source_store,
    #         target_store=target_store,
    #         crawler=SGGCrawler(),
    #     )
    # )

    asyncio.run(
        download_target_contents_and_insert_documents(
            downloader=downloader,
            task_store=task_store,
            target_repository=target_store,
            document_processor=document_processor,
            document_repository=document_repository,
        )
    )
