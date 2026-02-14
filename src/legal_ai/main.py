
if __name__ == "__main__":
    import asyncio
    from legal_ai.pipeline.tasks.crawl import crawl_and_insert_targets
    from storage import TaskStore, SourceStore, TargetStore

    source_store = SourceStore()
    
    asyncio.run(crawl_and_insert_targets())
