if __name__ == "__main__":
    import asyncio
    from legal_ai.pipeline.ingestion import DataIngesion
    from legal_ai.logging_config import setup_logging
    from legal_ai.crawlers.sgg_crawler import SGGCrawler

    setup_logging()

    data_ingestion = DataIngesion()

    async def main():
        # crawler = SGGCrawler()
        # await data_ingestion.crawl_and_insert_targets(crawler=crawler)
        await data_ingestion.download_target_contents()

    asyncio.run(main())
