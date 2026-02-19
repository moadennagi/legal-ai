from legal_ai.database import get_session
from legal_ai.models.document import Document
from legal_ai.repositories.document import DocumentRepository

from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
import logging
from typing import Any


logger = logging.getLogger(__name__)


class DocumentProcessing:
    def __init__(self):
        options = PdfPipelineOptions()
        options.do_ocr = False
        self.document_repository = DocumentRepository()
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def extract_text_as_markdown(self, document: Document) -> str:
        """Using docling to extract text from document file content"""
        source = document.file_path
        doc = self.converter.convert(source).document
        mark_down_content = doc.export_to_markdown()
        return mark_down_content

    def extract_text_as_dict(self, document: Document) -> dict[str, Any]:
        """Using docling to extract text qs dict from document file content"""
        source = document.file_path
        doc = self.converter.convert(source).document
        mark_down_content = doc.export_to_dict()
        return mark_down_content

    def extract_text_from_documents(self):
        """Extract text from documents and update them in the database"""
        with get_session() as session:
            documents = self.document_repository.collect_documents_without_content(session)
            for document in documents:
                try:
                    content = self.extract_text_as_markdown(document)
                    document.text_content = content
                    session.add(document)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to extract text from document {document.id}: {e}")
                    continue
