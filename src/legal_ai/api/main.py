from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from typing import Any

from legal_ai.adapters import OllamaLLMClientAdapter, build_chat_client, get_chat_model
from legal_ai.api.schemas import ChatRequest
from legal_ai.pipeline.conversation import ConversationManager
from legal_ai.pipeline.rag import RAG
from legal_ai.settings import settings
from legal_ai.splitters.moroccan_bo_splitter import MoroccanBulettinOfficielSplitter

__version__ = "0.1.0"

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Legal AI",
    description="Sovereign RAG pipeline for French legal documents.",
    version=__version__,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Embeddings always run on Ollama locally — only the chat completion is routed
# through the cloud provider when LLM_PROVIDER != "ollama".
embedding_client = OllamaLLMClientAdapter()
chat_client = build_chat_client()
chat_model = get_chat_model()

bo_document_splitter = MoroccanBulettinOfficielSplitter()
rag_client = RAG(
    generation_model=chat_model,
    embedding_model=settings.embeding_model,
    llm_client=chat_client if settings.llm_provider == "ollama" else embedding_client,
    document_splitter=bo_document_splitter,
)
# When the chat provider differs from the embedding provider, swap the chat call
# manually inside the conversation manager.
conversation_manager = ConversationManager(llm_client=chat_client, rag=rag_client)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": settings.llm_provider,
        "chat_model": chat_model,
        "embedding_model": settings.embeding_model,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "legal-ai-rag",
                "object": "model",
                "owned_by": "local",
                "chat_model": chat_model,
                "embedding_model": settings.embeding_model,
            }
        ],
    }


@app.post("/v1/chat/completions")
@limiter.limit("10/minute")
async def chat_completions(request: Request, req: ChatRequest) -> JSONResponse:
    user_query = req.messages[-1].content

    conversation_manager.history = [
        {"role": m.role, "content": m.content} for m in req.messages[:-1]
    ]
    res = await conversation_manager.ask(user_query, similarity_threshold=0.55)
    message: dict[str, str] = {"role": "assistant", "content": res.answer}
    return JSONResponse(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": chat_model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
        }
    )
