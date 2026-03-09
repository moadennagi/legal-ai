import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from legal_ai.api.main import app

MOCK_RAG_RESPONSE = {
    "answer": "L'Agence Marocaine des Médicaments est un établissement public.",
    "sources": [{"instrument": "Dahir n° 1-21-98"}],
}


@pytest.fixture
def test_client():
    client = TestClient(app)
    return client


def test_list_models_should_retrun_a_list_of_models(test_client):
    response = test_client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "legal-ai-rag", "object": "model", "owned_by": "local"}],
    }


def test_chat_should_return_chat_completion(test_client):
    with patch("legal_ai.api.main.conversation_manager") as mock_conversation_manager:
        mock_conversation_manager.ask = AsyncMock(return_value=MOCK_RAG_RESPONSE)

        req = {
            "model": "",
            "messages": [
                {
                    "role": "user",
                    "content": "Quel est le statut de l'Agence Marocaine des Médicaments ?",
                }
            ],
        }
        response = test_client.post("/v1/chat/completions", json=req)
        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data and data["choices"] is not None
