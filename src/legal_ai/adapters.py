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


class OpenAICompatibleAdapter(LLMClientInterface):
    """Adapter for any OpenAI-compatible endpoint (Together AI, Groq, vLLM, etc.).

    Used for the public online demo where running Ollama in cloud requires GPU.
    Does NOT support embeddings — those still require a local model or a dedicated
    embedding provider.
    """

    def __init__(self, api_key: str, base_url: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embeddings(self, model: str, prompt: str) -> list[float]:
        raise NotImplementedError(
            "OpenAICompatibleAdapter does not support embeddings. "
            "Use OllamaLLMClientAdapter for embeddings or extend this adapter "
            "with a provider-specific embedding endpoint."
        )

    async def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""


def build_chat_client() -> LLMClientInterface:
    """Build the chat client according to LLM_PROVIDER setting."""
    provider = settings.llm_provider
    if provider == "ollama":
        return OllamaLLMClientAdapter()
    if provider == "together":
        return OpenAICompatibleAdapter(
            api_key=settings.together_api_key.get_secret_value(),
            base_url=settings.together_base_url,
        )
    if provider == "groq":
        return OpenAICompatibleAdapter(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url,
        )
    if provider == "openai":
        return OpenAILLMClientAdapter(api_key=settings.openai_api_key.get_secret_value())
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def get_chat_model() -> str:
    """Return the chat model name according to the active provider."""
    provider = settings.llm_provider
    if provider == "together":
        return settings.together_model
    if provider == "groq":
        return settings.groq_model
    return settings.generation_model
