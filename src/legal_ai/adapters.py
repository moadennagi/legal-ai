from legal_ai.interfaces import ConversionResultInterface, DocumentConverterInterface
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions


class DoclingDocumentConverterAdapter(DocumentConverterInterface):
    def __init__(self):
        options = PdfPipelineOptions()
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def convert(self, file_path: str) -> ConversionResultInterface:
        doc = self.converter.convert(file_path)
        return doc
