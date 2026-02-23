from legal_ai.interfaces import (
    ConversionResultInterface,
    DocumentConverterInterface,
    LLMClientInterface,
)
import ollama
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


class OllamaLLMClientAdapter(LLMClientInterface):
    def __init__(self) -> None:
        self.async_client = ollama.AsyncClient()

    async def embeddings(self, model: str, prompt: str) -> list[float]:
        response = await self.async_client.embeddings(model=model, prompt=prompt)
        embedding = response["embedding"]
        return embedding

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]
