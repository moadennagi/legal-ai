from typing import Any

from sqlalchemy.types import TypeEngine

class Vector(TypeEngine[Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
