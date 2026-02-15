import asyncio
import aiohttp
from legal_ai.models.schemas import TargetPayload
from typing import Coroutine
from datetime import datetime, timezone


class Downloader:
    async def _download_document(self, target: TargetPayload) -> TargetPayload:
        target.claimed_at = datetime.now(tz=timezone.utc())
        async with aiohttp.ClientSession() as session:
            res = await session.get(target.url)
            res.raise_for_status()
            content = await res.content.read()
            target.content = content
        return target

    async def download_documents(self, targets: list[TargetPayload]) -> list[TargetPayload]:
        tasks: list[Coroutine[None, None, TargetPayload]] = []
        for target in targets:
            tasks.append(self._download_document(target))
        res = await asyncio.gather(*tasks)
        return res
