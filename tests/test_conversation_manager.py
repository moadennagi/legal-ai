import pytest
from legal_ai.pipeline.conversation import ConversationManager
from unittest.mock import Mock, AsyncMock
from legal_ai.interfaces import LLMClientInterface, RAGInterface


def test_compress_should_keep_last_for_messages():
    llm_client = Mock(spec=LLMClientInterface)
    rag = Mock(spec=RAGInterface)
    llm_client.chat.return_value = "resume"
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
    llm_client = Mock(spec=LLMClientInterface)

    rag = Mock(spec=RAGInterface)
    rag.ask = AsyncMock(return_value={"answer": "answer", "sources": [{"instrument": "loi"}]})
    llm_client.chat.return_value = "resume"
    sut = ConversationManager(llm_client=llm_client, rag=rag)

    await sut.ask(query=q, similarity_threshold=0.3)
    assert sut.history == [
        {"role": "user", "content": "user question"},
        {"role": "assistant", "content": "answer"},
    ]
