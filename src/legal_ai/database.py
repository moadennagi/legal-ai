from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from contextlib import contextmanager
from typing import Generator, Any

engine = create_engine("postgresql://postgres:mysecretpassword@0.0.0.0:5432/legal_ai", echo=True)

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
