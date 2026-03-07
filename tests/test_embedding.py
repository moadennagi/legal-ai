from unittest.mock import AsyncMock, Mock, MagicMock, patch
import pytest
from legal_ai.pipeline.embedding import DocumentEmbedding
from legal_ai.interfaces import LLMClientInterface, DocumentSplitterInterface, ChunkResult
from legal_ai.models.document import Document, DocumentChunk


# ── helpers ───────────────────────────────────────────────────────────────────


def make_embedding_service(splitters=None, model="bge-m3"):
    mock_llm = AsyncMock(spec=LLMClientInterface)
    return DocumentEmbedding(model, mock_llm, splitters or {}), mock_llm


def make_document(source_id=1, doc_id=10):
    doc = Mock(spec=Document)
    doc.source_id = source_id
    doc.id = doc_id
    return doc


# ── _get_splitter ─────────────────────────────────────────────────────────────


def test_get_splitter_returns_correct_splitter():
    mock_splitter = Mock(spec=DocumentSplitterInterface)
    sut, _ = make_embedding_service(splitters={1: mock_splitter})

    result = sut._get_splitter(make_document(source_id=1))
    assert result is mock_splitter


def test_get_splitter_returns_none_for_unknown_source():
    mock_splitter = Mock(spec=DocumentSplitterInterface)
    sut, _ = make_embedding_service(splitters={1: mock_splitter})

    result = sut._get_splitter(make_document(source_id=99))
    assert result is None


# ── _construct_document_chunks ────────────────────────────────────────────────


def test_construct_document_chunks_returns_correct_count():
    sut, _ = make_embedding_service()
    chunks = [ChunkResult(page_content="a"), ChunkResult(page_content="b")]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = sut._construct_document_chunks(42, chunks, embeddings)
    assert len(result) == 2


def test_construct_document_chunks_maps_data_correctly():
    sut, _ = make_embedding_service()
    chunks = [ChunkResult(page_content="hello", metadata={"key": "val"})]
    embeddings = [[0.1, 0.2, 0.3]]

    result = sut._construct_document_chunks(7, chunks, embeddings)
    chunk = result[0]

    assert isinstance(chunk, DocumentChunk)
    assert chunk.document_id == 7
    assert chunk.chunk_index == 0
    assert chunk.content == "hello"
    assert chunk.embedding == [0.1, 0.2, 0.3]
    assert chunk.chunk_metadata == {"key": "val"}


def test_construct_document_chunks_indices_are_sequential():
    sut, _ = make_embedding_service()
    chunks = [ChunkResult(page_content=f"c{i}") for i in range(4)]
    embeddings = [[float(i)] for i in range(4)]

    result = sut._construct_document_chunks(1, chunks, embeddings)
    indices = [c.chunk_index for c in result]
    assert indices == [0, 1, 2, 3]


# ── _embded_chunks ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_embded_chunks_calls_llm_for_each_chunk():
    mock_splitter = Mock(spec=DocumentSplitterInterface)
    mock_splitter.construct_enriched_content = Mock(side_effect=lambda c: c.page_content)

    sut, mock_llm = make_embedding_service(splitters={1: mock_splitter})
    sut.document_splitter = mock_splitter
    mock_llm.embeddings = AsyncMock(return_value=[0.1, 0.2])

    chunks = [ChunkResult(page_content="chunk1"), ChunkResult(page_content="chunk2")]
    result = await sut._embded_chunks(chunks)

    assert len(result) == 2
    assert mock_llm.embeddings.await_count == 2


@pytest.mark.asyncio
async def test_embded_chunks_returns_embeddings_in_order():
    mock_splitter = Mock(spec=DocumentSplitterInterface)
    mock_splitter.construct_enriched_content = Mock(side_effect=lambda c: c.page_content)

    sut, mock_llm = make_embedding_service()
    sut.document_splitter = mock_splitter

    embeddings = [[float(i)] * 3 for i in range(3)]
    mock_llm.embeddings = AsyncMock(side_effect=embeddings)

    chunks = [ChunkResult(page_content=f"chunk{i}") for i in range(3)]
    result = await sut._embded_chunks(chunks)

    assert result == embeddings


# ── split_and_insert_embeddings ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_split_and_insert_embeddings_raises_when_no_splitter():
    sut, _ = make_embedding_service(splitters={})
    doc = make_document(source_id=99)

    with pytest.raises(ValueError, match="Could not set document_splitter"):
        await sut.split_and_insert_embeddings([doc])


@pytest.mark.asyncio
@patch("legal_ai.pipeline.embedding.get_session")
async def test_split_and_insert_embeddings_calls_split_and_execute(mock_get_session):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_splitter = Mock(spec=DocumentSplitterInterface)
    mock_splitter.construct_enriched_content = Mock(return_value="enriched")
    mock_splitter.split_document = Mock(return_value=[ChunkResult(page_content="text")])

    sut, mock_llm = make_embedding_service(splitters={1: mock_splitter})
    mock_llm.embeddings = AsyncMock(return_value=[0.0] * 1024)

    doc = make_document(source_id=1, doc_id=5)
    await sut.split_and_insert_embeddings([doc])

    mock_splitter.split_document.assert_called_once_with(doc)
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
@patch("legal_ai.pipeline.embedding.get_session")
async def test_split_and_insert_embeddings_skips_empty_chunks(mock_get_session):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_get_session.return_value.__exit__ = Mock(return_value=False)

    mock_splitter = Mock(spec=DocumentSplitterInterface)
    mock_splitter.split_document = Mock(return_value=[])

    sut, mock_llm = make_embedding_service(splitters={1: mock_splitter})
    mock_llm.embeddings = AsyncMock(return_value=[])

    doc = make_document(source_id=1)
    await sut.split_and_insert_embeddings([doc])

    mock_session.execute.assert_not_called()
