from asyncio import Semaphore
from typing import Any, Coroutine
from datetime import datetime, date
import re


async def run_with_semaphore(sem: Semaphore, coroutine: Coroutine[None, None, Any]):
    """Download with semaphore"""
    async with sem:
        return await coroutine


def parse_ms_json_date(date_string: str) -> date | None:
    """Parse Microsoft JSON date format like '/Date(1682982000000)/'"""
    date_value = re.search(r"\d+", date_string)
    if not date_value:
        return None
    timestamp_ms = int(date_value.group())
    return datetime.fromtimestamp(timestamp_ms / 1000).date()
