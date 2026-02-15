if __name__ == "__main__":
    import asyncio
    from legal_ai.pipeline.tasks.crawl import crawl_and_insert_targets
    from repositories import TaskStore, SourceStore, TargetStore
    from crawlers.sgg_crawler import SGGCrawler

    source_store = SourceStore()
    target_store = TargetStore()
    task_store = TaskStore()

    asyncio.run(
        crawl_and_insert_targets(
            task_store=task_store,
            source_store=source_store,
            target_store=target_store,
            crawler=SGGCrawler(),
        )
    )
