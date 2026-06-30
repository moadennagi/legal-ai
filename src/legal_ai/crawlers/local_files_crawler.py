from legal_ai.interfaces import CrawlerInterface
from legal_ai.models.schemas import TargetSchema
from pathlib import Path
import os
from datetime import date
from legal_ai.models.schemas import SourceSchema


class LocalFilesCrawler(CrawlerInterface):
    name = "local_files_crawler"

    def __init__(self, file_path: Path, name: str) -> None:
        self.file_path = file_path
        self.name = name
        self.url = str(file_path)

    @property
    def source(self):
        return SourceSchema(name=self.name, url=self.url)

    def _collect_files_from_file_path(self, file_path: Path) -> list[Path]:
        """Collect all files recursively"""
        # this is a simpliste recursive function
        if not os.path.isdir(file_path):
            return [file_path]
        res: list[Path] = []
        for entry in Path(file_path).iterdir():
            paths = self._collect_files_from_file_path(entry)
            res.extend(paths)
        return res

    async def crawl_and_return_targets(self, task_id: int) -> list[TargetSchema]:
        """Loop over file paths and construct a list of TargetSchema"""
        target_schema_list: list[TargetSchema] = []
        file_paths = self._collect_files_from_file_path(self.file_path)
        today = date.today()
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            obj = TargetSchema(
                number=str(file_name),
                url=str(file_path),
                official_date=today,
                source=self.source,
                task_id=task_id,
            )
            target_schema_list.append(obj)
        return target_schema_list


if __name__ == "__main__":
    import asyncio

    crawler = LocalFilesCrawler(
        file_path=Path("/home/moadennagi/projects/legal-ai/data/frat"), name="frat"
    )
    asyncio.run(crawler.crawl_and_return_targets(1))
