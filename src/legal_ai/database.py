from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from contextlib import contextmanager
from typing import Generator, Any
from legal_ai.settings import settings

engine = create_engine(settings.database_url)


@contextmanager
def get_session() -> Generator[Session, Any, None]:
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
