from enum import Enum


class TaskStatus(Enum):
    succeeded = "succeeded"
    failed = "failed"
    in_progress = "in_progress"


class TaskType(Enum):
    crawling = "crawling"
    download = "download"
