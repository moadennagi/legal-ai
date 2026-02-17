from asyncio import Semaphore
from typing import Any, Coroutine


async def run_with_semaphore(sem: Semaphore, coroutine: Coroutine[None, None, Any]):
    """Download with semaphore"""
    async with sem:
        return await coroutine
