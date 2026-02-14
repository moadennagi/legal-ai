
if __name__ == "__main__":
    import asyncio
    from legal_ai.pipeline.tasks.crawl import crawl_and_insert_targets
    from repositories import TaskStore, SourceStore, TargetStore
    from legal_ai.database import get_session

    source_store = SourceStore()
    target_store = TargetStore()
    task_store = TaskStore()
    
    asyncio.run(
        crawl_and_insert_targets(task_store=task_store, source_store=source_store, target_store=task_store, session=get_session)
    )
