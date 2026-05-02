"""Pytest fixtures shared across all test files."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_document() -> Document:
    """Single document with realistic financial text for testing."""
    return Document(
        page_content="""
        Q3 2024 Financial Results
        Revenue increased 23% year-over-year to $4.2 billion.
        Operating margin expanded 150 basis points to 18.3%.
        The company raised full-year guidance citing strong demand.
        Free cash flow generation remained robust at $890 million.
        Management expects continued growth in cloud services segment.
        """
        * 10,
        metadata={"source": "test_report.pdf", "page": 1},
    )


@pytest.fixture
def sample_documents(sample_document: Document) -> list[Document]:
    """List of 3 documents for testing batch operations."""
    docs: list[Document] = []
    for i in range(3):
        doc = Document(
            page_content=sample_document.page_content,
            metadata={"source": f"test_report_{i}.pdf", "page": i},
        )
        docs.append(doc)
    return docs


@pytest.fixture
def sample_eval_dataset() -> dict[str, list]:
    """Minimal eval-shaped dict for eval pipeline tests."""
    return {
        "questions": ["What is RAG?"],
        "contexts": [["RAG combines retrieval and generation."]],
        "answers": ["RAG is Retrieval Augmented Generation."],
    }
