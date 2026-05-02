"""Tests for document loading and chunking pipeline."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ingestion.chunker import (
    FixedSizeChunker,
    RecursiveChunker,
    get_chunker,
)
from ingestion.loader import DocumentLoader


def test_fixed_size_chunker_produces_chunks() -> None:
    """Fixed-size splitter should yield multiple chunks and required metadata."""
    text: str = "word " * 600  # ~3000 chars
    doc = Document(page_content=text, metadata={"source": "big.txt"})
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
    chunks: list[Document] = chunker.chunk([doc])
    assert len(chunks) > 1
    assert all(c.metadata.get("chunk_strategy") == "fixed_size" for c in chunks)
    assert all("chunk_index" in c.metadata for c in chunks)


def test_recursive_chunker_produces_chunks() -> None:
    """Recursive splitter should yield multiple chunks and strategy metadata."""
    text: str = "word " * 600
    doc = Document(page_content=text, metadata={"source": "big.txt"})
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks: list[Document] = chunker.chunk([doc])
    assert len(chunks) > 1
    assert all(c.metadata.get("chunk_strategy") == "recursive" for c in chunks)


def test_chunkers_produce_different_results() -> None:
    """Same document: both strategies produce chunks; counts need not match."""
    paragraphs: list[str] = [f"Section {i}: " + ("detail. " * 12) for i in range(80)]
    text: str = "\n\n".join(paragraphs)
    doc = Document(page_content=text, metadata={"source": "structured.txt"})
    fixed = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
    recursive = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    fixed_chunks = fixed.chunk([doc])
    recursive_chunks = recursive.chunk([doc])
    print(f"fixed_size chunks: {len(fixed_chunks)}")
    print(f"recursive chunks: {len(recursive_chunks)}")
    assert len(fixed_chunks) > 1
    assert len(recursive_chunks) > 1
    # Deliberately do not assert equal counts — boundary logic differs by design.


def test_get_chunker_factory() -> None:
    """Factory returns concrete types and validates strategy names."""
    assert isinstance(get_chunker("fixed_size"), FixedSizeChunker)
    assert isinstance(get_chunker("recursive"), RecursiveChunker)
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("invalid")


def test_document_stats(sample_documents: list[Document]) -> None:
    """Stats helper should aggregate counts and average length."""
    loader = DocumentLoader(data_dir="data/raw")
    stats = loader.get_document_stats(sample_documents)
    assert stats["total_documents"] == 3
    assert stats["total_pages"] == 3
    assert stats["avg_page_length"] > 0
    assert len(stats["sources"]) == 3
