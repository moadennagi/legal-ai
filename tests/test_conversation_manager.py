import pytest
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
from legal_ai.pipeline.conversation import ConversationManager


def test_compress_should_keep_last_four_messages():
    llm_client = Mock()
    llm_client.chat = Mock(return_value="resume")
    rag = Mock()
    sut = ConversationManager(llm_client=llm_client, rag=rag)
    sut.history = [
        {"role": "user", "content": "foo1"},
        {"role": "user", "content": "foo2"},
        {"role": "user", "content": "foo3"},
        {"role": "user", "content": "foo4"},
        {"role": "user", "content": "foo5"},
        {"role": "user", "content": "foo6"},
    ]
    sut._compress()
    assert sut.history == [
        {"role": "system", "content": "resume"},
        {"role": "user", "content": "foo3"},
        {"role": "user", "content": "foo4"},
        {"role": "user", "content": "foo5"},
        {"role": "user", "content": "foo6"},
    ]


@pytest.mark.asyncio
async def test_ask_should_update_history_with_question_and_answer():
    q = "user question"
    llm_client = Mock()
    llm_client.chat = Mock(return_value="resume")

    rag = Mock()
    rag_response = SimpleNamespace(
        answer="answer",
        sources=[{"instrument": "loi"}],
    )
    rag.ask = AsyncMock(return_value=rag_response)
    sut = ConversationManager(llm_client=llm_client, rag=rag)

    await sut.ask(query=q, similarity_threshold=0.3)
    assert sut.history == [
        {"role": "user", "content": "user question"},
        {"role": "assistant", "content": "answer"},
    ]
