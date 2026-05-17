import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from legal_ai.api.main import app


@pytest.fixture
def test_client():
    return TestClient(app)


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "llm_provider" in data


def test_list_models_returns_a_list_of_models(test_client):
    response = test_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "legal-ai-rag"
    assert data["data"][0]["object"] == "model"


def test_chat_returns_chat_completion(test_client):
    mock_response = SimpleNamespace(
        answer="L'Agence Marocaine des Médicaments est un établissement public.",
        sources=[{"instrument": "Dahir n° 1-21-98"}],
    )

    with patch("legal_ai.api.main.conversation_manager") as mock_conversation_manager:
        mock_conversation_manager.ask = AsyncMock(return_value=mock_response)

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
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data and len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
