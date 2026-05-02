"""End-to-end RAG: retrieve from Qdrant, format context, generate with OpenAI.

This module is the system under evaluation — RAGAS scores reflect how well
retrieval and generation perform together.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ingestion.vector_store import VectorStoreManager

load_dotenv()


class RAGPipeline:
    """Complete RAG pipeline: retrieve context + generate answer.

    WHY THIS EXISTS: This is the system we are evaluating. RAGAS
    measures the quality of exactly this pipeline — how well it
    retrieves relevant context and how faithfully the LLM uses
    that context to answer questions.

    This class is intentionally simple and transparent so every
    step can be observed and evaluated.

    Flow:

    Query → :meth:`~ingestion.vector_store.VectorStoreManager.search`
    → context chunks → format prompt with context → OpenAI LLM → answer
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
    ) -> None:
        """Wire retrieval client and chat model (keys from environment only).

        ``OPENAI_API_KEY`` must be set in ``.env`` for the LLM.

        Args:
            vector_store: Phase 2 vector store (Qdrant + embeddings).
            model: OpenAI chat model id.
            top_k: Number of chunks to retrieve per query.

        Raises:
            ValueError: If ``OPENAI_API_KEY`` is missing.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env for the RAG pipeline."
            )
        self.vector_store: VectorStoreManager = vector_store
        self.model: str = model
        self.top_k: int = top_k
        # temperature=0 for deterministic answers — important for reproducible evaluation.
        self.llm: ChatOpenAI = ChatOpenAI(model=model, temperature=0)
        print(f"RAG Pipeline ready — model: {model}, top_k: {top_k}")

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a query.

        Separated from :meth:`generate` so we can evaluate retrieval
        independently from generation.

        Args:
            query: Natural-language question.

        Returns:
            Chunk dicts from :meth:`VectorStoreManager.search`.
        """
        return self.vector_store.search(query, top_k=self.top_k)

    def format_context(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a single context string.

        WHY FORMAT MATTERS: How we present context to the LLM affects
        answer quality. We number chunks and include source info so the
        LLM can reference them and we can trace faithfulness.

        Args:
            chunks: Dicts with at least ``text``, ``source``, ``page``.

        Returns:
            Single string suitable for the user prompt.
        """
        parts: list[str] = []
        for i, ch in enumerate(chunks, start=1):
            src: str = str(ch.get("source", "unknown"))
            page: int = int(ch.get("page", 0) or 0)
            text: str = str(ch.get("text", ""))
            block: str = (
                f"[Chunk {i} - {src}, page {page}]\n{text}\n---"
            )
            parts.append(block)
        return "\n".join(parts)

    def generate(self, query: str, context: str) -> str:
        """Generate an answer using retrieved context.

        WHY THIS PROMPT STRUCTURE: We explicitly instruct the LLM to:

        1. Only use the provided context (enables faithfulness measurement)
        2. Say "I don't know" if context is insufficient (reduces hallucination)
        3. Be concise (reduces noise in evaluation)

        Args:
            query: User question.
            context: Formatted context from :meth:`format_context`.

        Returns:
            Model answer text.
        """
        system_text: str = (
            "You are a helpful assistant that answers questions based strictly "
            "on the provided context. If the answer cannot be found in the "
            "context, say 'I don't have enough information to answer this.' "
            "Do not use any knowledge outside the provided context."
        )
        user_text: str = (
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        response = self.llm.invoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]
        )
        content: str | list[str | dict[Any, Any]] = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def run(self, query: str) -> dict[str, Any]:
        """Run full RAG: retrieve + generate.

        WHY RETURN ALL THIS: RAGAS needs the question, answer, contexts,
        and ground truth to compute metrics. This dict holds the fields
        we will align with RAGAS inputs (ground truth added elsewhere).

        Args:
            query: User question.

        Returns:
            Dict with ``question``, ``answer``, ``contexts``, ``source_chunks``,
            ``model``, ``top_k``.
        """
        chunks: list[dict[str, Any]] = self.retrieve(query)
        context: str = self.format_context(chunks)
        answer: str = self.generate(query, context)
        contexts: list[str] = [str(c.get("text", "")) for c in chunks]
        return {
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "source_chunks": chunks,
            "model": self.model,
            "top_k": self.top_k,
        }


def _ensure_project_root_on_path() -> None:
    """Allow ``python backend/rag_pipeline.py`` to import ``ingestion``."""
    from pathlib import Path
    import sys

    root: Path = Path(__file__).resolve().parents[1]
    root_s: str = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


if __name__ == "__main__":
    _ensure_project_root_on_path()

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env to run the RAG demo.")
        raise SystemExit(1)

    vs = VectorStoreManager(embedding_model="openai")
    stats: dict[str, Any] = vs.get_collection_stats()
    if stats.get("total_vectors", 0) == 0 or stats.get("status") == "not_found":
        print(
            "No vectors in Qdrant yet. Run ingestion first, e.g.:\n"
            "  python ingestion/pipeline.py\n"
            f"Collection: {vs.collection_name!r}, stats: {stats}"
        )
        raise SystemExit(1)

    pipeline = RAGPipeline(vector_store=vs, model="gpt-4o-mini", top_k=5)
    queries: list[str] = [
        "What was the revenue growth?",
        "What is the operating margin?",
        "What is the free cash flow?",
    ]

    for q in queries:
        print("\n" + "=" * 60)
        print("Question:", q)
        chunks: list[dict[str, Any]] = pipeline.retrieve(q)
        print("Retrieved chunks:")
        for i, c in enumerate(chunks, start=1):
            print(
                f"  {i}. score={c.get('score', 0):.4f} "
                f"source={c.get('source')} page={c.get('page')}"
            )
        result: dict[str, Any] = pipeline.run(q)
        print("Answer:", result["answer"])
