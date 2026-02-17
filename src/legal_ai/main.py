if __name__ == "__main__":
    import asyncio
    from legal_ai.pipeline.ingestion import DataIngesion
    from legal_ai.logging_config import setup_logging

    setup_logging()

    data_ingestion = DataIngesion()

    async def main():
        await data_ingestion.download_target_contents()

    asyncio.run(main())
