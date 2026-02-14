from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session
from contextlib import contextmanager

engine = create_engine("sqlite+pysqlite:///database.db", echo=True)

@contextmanager
def get_session(engine: Engine):
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
