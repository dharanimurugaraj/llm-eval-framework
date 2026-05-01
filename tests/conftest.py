"""Pytest fixtures shared across all test files."""

import pytest


@pytest.fixture
def sample_documents() -> list[object]:
    class Document:
        def __init__(self, page_content: str, metadata: dict) -> None:
            self.page_content = page_content
            self.metadata = metadata

    return [Document("Fake document content", {"source": "fake.pdf"})]


@pytest.fixture
def sample_eval_dataset() -> dict:
    return {
        "questions": ["What is RAG?"],
        "contexts": [["RAG combines retrieval and generation."]],
        "answers": ["RAG is Retrieval Augmented Generation."],
    }
