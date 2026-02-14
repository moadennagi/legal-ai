from sqlalchemy.orm import Session
from sqlalchemy import select
from legal_ai.models.document import Source

class SourceStore:
    def get_or_create_source(self, session: Session, source_name: str, source_url: str) -> Source:
        """Should retrieve or create a new source id"""
        source = Source(name=source_name, url=source_url)
         # insert source in database only if it does not exist
        query = select(Source).where(Source.name == source_name)
        res = session.scalar(query)
        if res:
            source = res
        else:
            session.add(source)
            session.flush()
        return source
