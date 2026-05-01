from legal_ai.interfaces import (
    DocumentConverterInterface,
    LLMClientInterface,
)
import ollama
from openai import AsyncOpenAI
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.labels import DocItemLabel
from legal_ai.settings import settings


class DoclingDocumentConverterAdapter(DocumentConverterInterface):
    _EXPORT_LABELS = set(DocItemLabel) - {DocItemLabel.DOCUMENT_INDEX}

    def __init__(self):
        options = PdfPipelineOptions()
        options.do_table_structure = False
        options.do_ocr = False
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def convert(self, file_path: str) -> str:
        doc = self.converter.convert(file_path)
        result = doc.document.export_to_markdown(labels=self._EXPORT_LABELS)
        return result


class OpenAILLMClientAdapter(LLMClientInterface):
    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def embeddings(self, model: str, prompt: str) -> list[float]:
        raise NotImplementedError("OpenAILLMClientAdapter does not support embeddings")

    async def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(model=model, messages=messages)  # type: ignore[arg-type]
        return response.choices[0].message.content or ""


class OllamaLLMClientAdapter(LLMClientInterface):
    def __init__(self) -> None:
        self.async_client = ollama.AsyncClient(host=settings.ollama_host)

    async def embeddings(self, model: str, prompt: str) -> list[float]:
        response = await self.async_client.embeddings(model=model, prompt=prompt)
        embedding = response["embedding"]
        return embedding

    async def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        response = await self.async_client.chat(model=model, messages=messages)
        return response["message"]["content"]
