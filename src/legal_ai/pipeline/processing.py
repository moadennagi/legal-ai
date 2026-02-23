from legal_ai.database import get_session
from legal_ai.repositories.document import DocumentRepository


import logging
from typing import Any
from legal_ai.interfaces import DocumentConverterInterface

logger = logging.getLogger(__name__)


class DocumentProcessing:
    def __init__(self, document_converter: DocumentConverterInterface):
        self.document_repository = DocumentRepository()
        self.document_converter = document_converter

    def _extract_markdown(self, file_path: str) -> str:
        """Using docling to extract text from document file content"""
        result = self.document_converter.convert(file_path)
        doc = result.document
        mark_down_content = doc.export_to_markdown()
        return mark_down_content

    def _extract_text_as_dict(self, file_path: str) -> dict[str, Any]:
        """Using docling to extract text qs dict from document file content"""
        result = self.document_converter.convert(file_path)
        doc = result.document
        mark_down_content = doc.export_to_dict()
        return mark_down_content

    def extract_text_from_documents(self):
        """Extract text from documents and update them in the database"""
        with get_session() as session:
            documents = self.document_repository.collect_documents_without_content(session)
            doc_ids = [(doc.id, doc.file_path) for doc in documents]
        for doc_id, file_path in doc_ids:
            with get_session() as session:
                try:
                    content = self._extract_markdown(file_path=file_path)
                    self.document_repository.update_document_content(session, doc_id, content)
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to extract text from document {doc_id}: {e}")
                    continue
