from sqlalchemy import create_engine

engine = create_engine("sqlite+pysqlite:///database.db", echo=True)
