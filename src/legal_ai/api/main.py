from fastapi import FastAPI
from typing import Any
from legal_ai.splitters.moroccan_bo_splitter import MoroccanBulettinOfficielSplitter
from legal_ai.settings import settings
from legal_ai.adapters import OllamaLLMClientAdapter
from legal_ai.pipeline.rag import RAG
from legal_ai.pipeline.conversation import ConversationManager
from legal_ai.api.schemas import ChatRequest


app = FastAPI()


ollama_client = OllamaLLMClientAdapter()
bo_document_splitter = MoroccanBulettinOfficielSplitter()
rag_client = RAG(
    generation_model=settings.generation_model,
    embedding_model=settings.embeding_model,
    llm_client=ollama_client,
    document_splitter=bo_document_splitter,
)
conversation_manager = ConversationManager(llm_client=ollama_client, rag=rag_client)


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": "legal-ai-rag", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest) -> dict[str, Any]:
    user_query = req.messages[-1].content

    conversation_manager.history = [
        {"role": m.role, "content": m.content} for m in req.messages[:-1]
    ]
    res = await conversation_manager.ask(user_query, similarity_threshold=0.55)
    message: dict[str, str] = {"role": "assistant", "content": res["answer"]}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
    }
