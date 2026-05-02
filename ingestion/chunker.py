"""Chunking strategies for RAG ingestion (Phase 1).

Provides three strategies—fixed-size, recursive, and semantic—so later
experiments can compare which yields better RAGAS retrieval scores.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Load .env for OpenAI-backed semantic chunking (never hardcode keys).
load_dotenv()


class FixedSizeChunker:
    """Splits documents by character count with overlap.

    WHY: Simplest strategy. Fast, predictable, but ignores sentence
    and paragraph boundaries. Chunks may cut mid-sentence.

    WHEN IT BREAKS: Technical documents where cutting mid-formula
    or mid-table destroys meaning.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Build a :class:`~langchain_text_splitters.CharacterTextSplitter`.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters shared between adjacent chunks (keeps
                boundary context).
        """
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        # Empty separator => split purely by character length (not by "\n\n" blocks).
        self._splitter: CharacterTextSplitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="",
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split all input documents and tag each chunk with strategy metadata.

        Args:
            documents: Full pages or segments to split.

        Returns:
            Chunk-level documents with ``chunk_strategy``, sizes, and
            ``chunk_index`` (0-based within each source document).
        """
        out: list[Document] = []
        for doc in documents:
            splits: list[Document] = self._splitter.split_documents([doc])
            for i, ch in enumerate(splits):
                meta: dict[str, str | int] = {
                    **ch.metadata,
                    "chunk_strategy": "fixed_size",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunk_index": i,
                }
                ch.metadata = meta
                out.append(ch)
        return out


class RecursiveChunker:
    """Splits by trying a hierarchy of separators: paragraphs → sentences
    → words → characters.

    WHY: More intelligent than fixed-size. Tries to keep paragraphs
    together, falls back to sentences, then words. Respects natural
    language boundaries much better.

    THIS IS THE DEFAULT RECOMMENDED STRATEGY for most RAG use cases.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Build a :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`.

        Args:
            chunk_size: Target maximum chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        separators: list[str] = ["\n\n", "\n", ". ", " ", ""]
        self._splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents recursively and annotate chunk metadata."""
        out: list[Document] = []
        for doc in documents:
            splits: list[Document] = self._splitter.split_documents([doc])
            for i, ch in enumerate(splits):
                meta: dict[str, str | int] = {
                    **ch.metadata,
                    "chunk_strategy": "recursive",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunk_index": i,
                }
                ch.metadata = meta
                out.append(ch)
        return out


class SemanticChunker:
    """Groups sentences by semantic similarity using embeddings.

    WHY: Most intelligent strategy. Instead of splitting by character
    count, it detects where the MEANING shifts. Sentences about the
    same topic stay together even if they span many paragraphs.

    TRADEOFF: Slower (requires embedding each sentence) and costs
    money (uses OpenAI embeddings API). But produces highest quality
    chunks for complex documents.

    Note:
        Calling :meth:`chunk` issues embedding API calls (roughly on the
        order of ~$0.0001 per small document; scales with sentence count).
    """

    def __init__(self, breakpoint_threshold: float = 0.8) -> None:
        """Configure OpenAI embeddings and LangChain's semantic splitter.

        ``OPENAI_API_KEY`` is read from the environment (via ``python-dotenv``).

        Args:
            breakpoint_threshold: Interpreted as a percentile cutoff in
                ``[0, 1]`` and mapped to LangChain's ``breakpoint_threshold_amount``
                (e.g. ``0.8`` → ``80`` percentile) to decide when to start a new chunk.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file for semantic chunking."
            )
        self.breakpoint_threshold: float = breakpoint_threshold
        embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Map user-facing 0–1 threshold to percentile amount expected by LangChain.
        amount: float = max(1.0, min(99.0, float(breakpoint_threshold) * 100.0))
        self._splitter: LangChainSemanticChunker = LangChainSemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=amount,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split using semantic boundaries; tags ``chunk_strategy`` as ``semantic``."""
        out: list[Document] = []
        for doc in documents:
            splits: list[Document] = self._splitter.split_documents([doc])
            for i, ch in enumerate(splits):
                meta: dict[str, str | int | float] = {
                    **ch.metadata,
                    "chunk_strategy": "semantic",
                    "breakpoint_threshold": self.breakpoint_threshold,
                    "chunk_index": i,
                }
                ch.metadata = meta
                out.append(ch)
        return out


def get_chunker(
    strategy: str,
    **kwargs: int | float,
) -> FixedSizeChunker | RecursiveChunker | SemanticChunker:
    """Factory — returns the right chunker by strategy name.

    Usage::

        chunker = get_chunker("recursive", chunk_size=500)

    Args:
        strategy: One of ``"fixed_size"``, ``"recursive"``, ``"semantic"``.
        **kwargs: Passed to the chunker constructor (e.g. ``chunk_size``).

    Returns:
        A concrete chunker instance.

    Raises:
        ValueError: If ``strategy`` is not recognized.
    """
    key: str = strategy.strip().lower()
    if key == "fixed_size":
        return FixedSizeChunker(
            chunk_size=int(kwargs.get("chunk_size", 1000)),
            chunk_overlap=int(kwargs.get("chunk_overlap", 200)),
        )
    if key == "recursive":
        return RecursiveChunker(
            chunk_size=int(kwargs.get("chunk_size", 1000)),
            chunk_overlap=int(kwargs.get("chunk_overlap", 200)),
        )
    if key == "semantic":
        return SemanticChunker(
            breakpoint_threshold=float(kwargs.get("breakpoint_threshold", 0.8)),
        )
    raise ValueError(
        f"Unknown chunking strategy: {strategy!r}. "
        "Use 'fixed_size', 'recursive', or 'semantic'."
    )


if __name__ == "__main__":
    # Demo: compare fixed vs recursive on synthetic long text (skip semantic to avoid API cost).
    lorem: str = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 80
    )
    demo_doc: Document = Document(page_content=lorem, metadata={"source": "demo.txt"})

    for name, chunker_obj in (
        ("fixed_size", FixedSizeChunker(chunk_size=400, chunk_overlap=40)),
        ("recursive", RecursiveChunker(chunk_size=400, chunk_overlap=40)),
    ):
        chunks: list[Document] = chunker_obj.chunk([demo_doc])
        avg_len: float = (
            sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0.0
        )
        print(f"{name}: chunks={len(chunks)}, avg_chunk_length={avg_len:.1f} chars")
