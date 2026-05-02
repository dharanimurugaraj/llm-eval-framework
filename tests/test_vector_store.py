"""Tests for vector store and RAG pipeline (mocked — no live Qdrant/OpenAI)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

from backend.rag_pipeline import RAGPipeline
from ingestion.vector_store import VectorStoreManager


@patch("ingestion.vector_store.EmbeddingManager")
@patch("ingestion.vector_store.QdrantClient")
def test_vector_store_manager_init(
    mock_client_cls: MagicMock,
    mock_embed_cls: MagicMock,
) -> None:
    """VectorStoreManager should wire client + embedder and set collection name."""
    mock_embed = MagicMock()
    mock_embed.get_model_info.return_value = {
        "dimensions": 1536,
        "model_name": "text-embedding-3-small",
    }
    mock_embed_cls.return_value = mock_embed
    mock_client_cls.return_value = MagicMock()

    mgr = VectorStoreManager(
        collection_name="my_collection",
        embedding_model="openai",
    )
    assert mgr.collection_name == "my_collection"
    mock_client_cls.assert_called_once()
    mock_embed_cls.assert_called_once_with(model_name="openai")


@patch("ingestion.vector_store.EmbeddingManager")
@patch("ingestion.vector_store.QdrantClient")
def test_search_returns_formatted_results(
    mock_client_cls: MagicMock,
    mock_embed_cls: MagicMock,
) -> None:
    """search() should return plain dicts with expected keys."""
    mock_embed = MagicMock()
    mock_embed.get_model_info.return_value = {"dimensions": 4, "model_name": "mock"}
    mock_embed.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_embed_cls.return_value = mock_embed

    mock_hit = MagicMock()
    mock_hit.score = 0.92
    mock_hit.payload = {
        "text": "hello world",
        "source": "a.pdf",
        "page": 2,
        "chunk_strategy": "recursive",
        "chunk_index": 0,
    }

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.search.return_value = [mock_hit]
    mock_client_cls.return_value = mock_client

    mgr = VectorStoreManager(collection_name="c1", embedding_model="openai")
    out: list[dict[str, Any]] = mgr.search("q", top_k=3)
    assert len(out) == 1
    row = out[0]
    assert row["text"] == "hello world"
    assert "score" in row
    assert row["source"] == "a.pdf"
    assert row["page"] == 2


def test_format_context() -> None:
    """format_context should number chunks and include source lines."""
    mock_vs = MagicMock()
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.vector_store = mock_vs
    pipeline.model = "gpt-4o-mini"
    pipeline.top_k = 5
    pipeline.llm = MagicMock()

    chunks: list[dict[str, Any]] = [
        {"text": "Alpha content", "source": "one.pdf", "page": 1},
        {"text": "Beta content", "source": "two.pdf", "page": 2},
    ]
    ctx: str = pipeline.format_context(chunks)
    assert "[Chunk 1" in ctx
    assert "[Chunk 2" in ctx
    assert "Alpha content" in ctx
    assert "Beta content" in ctx


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-for-unit-tests"}, clear=False)
@patch.object(RAGPipeline, "retrieve")
@patch.object(RAGPipeline, "generate", return_value="mock answer")
def test_rag_pipeline_run_returns_required_keys(
    mock_generate: MagicMock,
    mock_retrieve: MagicMock,
) -> None:
    """run() should expose all keys needed for downstream RAGAS wiring."""
    mock_vs = MagicMock()
    mock_retrieve.return_value = [{"text": "ctx", "source": "s", "page": 1}]
    pipeline = RAGPipeline(vector_store=mock_vs, model="gpt-4o-mini", top_k=3)
    out: dict[str, Any] = pipeline.run("test query")
    for key in (
        "question",
        "answer",
        "contexts",
        "source_chunks",
        "model",
        "top_k",
    ):
        assert key in out
    assert out["question"] == "test query"
    assert out["answer"] == "mock answer"
    assert out["contexts"] == ["ctx"]
    assert out["top_k"] == 3
