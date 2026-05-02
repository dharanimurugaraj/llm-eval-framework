"""Qdrant vector store for Phase 2 — embed chunks and retrieve by similarity.

This module isolates all Qdrant client usage so the rest of the codebase
depends on a small, testable surface (store, search, stats).
"""

from __future__ import annotations

import math
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingestion.embedder import EmbeddingManager

load_dotenv()


def _qdrant_api_key() -> str | None:
    """Return API key only if set and not a placeholder from `.env.example`."""
    raw: str | None = os.getenv("QDRANT_API_KEY")
    if raw is None or raw.strip() == "":
        return None
    if raw.startswith("your_"):
        return None
    return raw


class VectorStoreManager:
    """Manages Qdrant vector store operations.

    WHY THIS EXISTS: Qdrant stores document chunks as vectors so we
    can do semantic search at query time. This class abstracts all
    Qdrant operations so the rest of the codebase never directly
    touches the Qdrant client.

    Responsibilities:

    - Create and manage Qdrant collections
    - Store embedded chunks with full metadata
    - Retrieve similar chunks for a given query
    - Delete and recreate collections for fresh experiments
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model: str = "openai",
    ) -> None:
        """Connect to Qdrant and wire up the configured embedding backend.

        Reads ``QDRANT_URL``, ``QDRANT_API_KEY``, and ``QDRANT_COLLECTION_NAME``
        from the environment (via ``python-dotenv``). ``collection_name``
        overrides the env default when provided.

        Args:
            collection_name: Optional Qdrant collection name.
            embedding_model: ``"openai"`` or ``"bge"`` — passed to
                :class:`~ingestion.embedder.EmbeddingManager`.
        """
        url: str = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
        env_collection: str = os.getenv(
            "QDRANT_COLLECTION_NAME", "llm_eval_docs"
        ).strip()
        self.collection_name: str = (
            collection_name.strip() if collection_name else env_collection
        )
        api_key: str | None = _qdrant_api_key()
        self.client: QdrantClient = QdrantClient(url=url, api_key=api_key)
        self.embedder: EmbeddingManager = EmbeddingManager(model_name=embedding_model)
        print(f"Connected to Qdrant at {url}")
        print(f"Using collection: {self.collection_name}")

    def create_collection(self, recreate: bool = False) -> None:
        """Create a Qdrant collection for storing document embeddings.

        WHY ``recreate`` parameter: During experiments we often want a fresh
        collection (``recreate=True``) to avoid mixing chunks from different
        chunking strategies. Always recreate between experiments.

        Vector size is determined by the embedding model:

        - OpenAI ``text-embedding-3-small``: 1536 dimensions
        - BGE-M3: 1024 dimensions

        Args:
            recreate: If ``True``, drop the collection if it exists, then
                create a new one. If ``False``, create only when missing.
        """
        info: dict[str, str | int] = self.embedder.get_model_info()
        dims: int = int(info["dimensions"])

        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dims,
                    # Cosine measures angle between vectors — standard for text similarity.
                    distance=Distance.COSINE,
                ),
            )

        print(f"Collection {self.collection_name} created with {dims} dimensions")
        print("Distance metric: COSINE")

    def store_chunks(self, chunks: list[Document]) -> int:
        """Embed and store document chunks in Qdrant.

        WHY we store metadata: At retrieval time we need to know which
        document and page each chunk came from, what chunking strategy
        produced it, and what its position was. This metadata is what
        lets us trace retrieved chunks back to source documents during
        evaluation.

        Args:
            chunks: LangChain documents produced by Phase 1 chunkers.

        Returns:
            Total number of points written.
        """
        if not chunks:
            return 0

        texts: list[str] = [doc.page_content for doc in chunks]
        vectors: list[list[float]] = self.embedder.embed_documents(texts)

        points: list[PointStruct] = []
        for i, (doc, vector) in enumerate(zip(chunks, vectors, strict=True)):
            page_val: Any = doc.metadata.get("page", 0)
            page_int: int = int(page_val) if isinstance(page_val, (int, float)) else 0
            cidx_val: Any = doc.metadata.get("chunk_index", i)
            chunk_index: int = (
                int(cidx_val) if isinstance(cidx_val, (int, float)) else i
            )
            csize_val: Any = doc.metadata.get("chunk_size", 0)
            chunk_size_meta: int = (
                int(csize_val) if isinstance(csize_val, (int, float)) else 0
            )
            payload: dict[str, str | int] = {
                "text": doc.page_content,
                "source": str(doc.metadata.get("source", "unknown")),
                "page": page_int,
                "chunk_index": chunk_index,
                "chunk_strategy": str(
                    doc.metadata.get("chunk_strategy", "unknown")
                ),
                "chunk_size": chunk_size_meta,
            }
            points.append(
                PointStruct(id=i, vector=vector, payload=payload),
            )

        batch_size: int = 100
        # Batching prevents timeout on large document sets.
        total_batches: int = max(1, math.ceil(len(points) / batch_size))
        for b in range(total_batches):
            batch: list[PointStruct] = points[b * batch_size : (b + 1) * batch_size]
            print(f"Storing batch {b + 1}/{total_batches}...")
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for chunks most similar to the query.

        WHY ``top_k=5``: In RAG, we typically retrieve 3–5 chunks. Too few
        misses relevant context. Too many adds noise that confuses the LLM.
        Five is a safe default; we can tune this in experiments.

        Returns:
            Plain dicts (not Qdrant types) so callers stay decoupled from
            Qdrant. Each dict includes ``text``, ``score``, ``source``,
            ``page``, ``chunk_strategy``, ``chunk_index``.
        """
        query_embedding: list[float] = self.embedder.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        out: list[dict[str, Any]] = []
        for hit in results:
            payload: dict[str, Any] = dict(hit.payload or {})
            text: str = str(payload.get("text", ""))
            score: float = float(hit.score)
            # Qdrant cosine scores are similarity-like (higher = closer).
            clamped: float = max(0.0, min(1.0, score))
            out.append(
                {
                    "text": text,
                    "score": clamped,
                    "source": str(payload.get("source", "unknown")),
                    "page": int(payload.get("page", 0) or 0),
                    "chunk_strategy": str(
                        payload.get("chunk_strategy", "unknown")
                    ),
                    "chunk_index": int(payload.get("chunk_index", 0) or 0),
                }
            )
        return out

    def get_collection_stats(self) -> dict[str, Any]:
        """Return stats about the current collection."""
        info: dict[str, str | int] = self.embedder.get_model_info()
        dims: int = int(info["dimensions"])
        model_label: str = str(info["model_name"])

        if not self.client.collection_exists(self.collection_name):
            return {
                "collection_name": self.collection_name,
                "total_vectors": 0,
                "embedding_model": model_label,
                "vector_dimensions": dims,
                "status": "not_found",
            }

        coll = self.client.get_collection(collection_name=self.collection_name)
        points_count: int = int(coll.points_count or 0)
        status: str = str(coll.status) if coll.status is not None else "unknown"

        return {
            "collection_name": self.collection_name,
            "total_vectors": points_count,
            "embedding_model": model_label,
            "vector_dimensions": dims,
            "status": status,
        }

    def delete_collection(self) -> None:
        """Delete the current collection (cleanup between experiments)."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        print(f"Collection {self.collection_name} deleted")


if __name__ == "__main__":
    from pathlib import Path
    import sys

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from ingestion.chunker import RecursiveChunker

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env to run the vector store demo.")
        raise SystemExit(1)

    mgr = VectorStoreManager(embedding_model="openai")
    mgr.create_collection(recreate=True)

    sample_docs: list[Document] = [
        Document(
            page_content=(
                "Q3 2024 revenue grew 23% year-over-year to $4.2 billion. "
                "Cloud segment revenue increased 45%."
            ),
            metadata={"source": "q3_report.pdf", "page": 1},
        ),
        Document(
            page_content=(
                "Operating margin reached 18.3%. Free cash flow was $890 million. "
                "Management raised full-year guidance."
            ),
            metadata={"source": "q3_report.pdf", "page": 2},
        ),
        Document(
            page_content=(
                "Capital expenditures were disciplined. Share repurchases totaled "
                "$200 million in the quarter."
            ),
            metadata={"source": "q3_report.pdf", "page": 3},
        ),
    ]

    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
    chunks: list[Document] = chunker.chunk(sample_docs)
    stored: int = mgr.store_chunks(chunks)
    print(f"Stored {stored} chunks.")

    query: str = "revenue growth"
    hits: list[dict[str, Any]] = mgr.search(query, top_k=3)
    print(f"\nTop 3 results for query: {query!r}")
    for i, h in enumerate(hits, start=1):
        print(f"  {i}. score={h['score']:.4f} source={h['source']} page={h['page']}")
        preview: str = h["text"][:120].replace("\n", " ")
        print(f"     {preview}...")

    stats: dict[str, Any] = mgr.get_collection_stats()
    print("\nCollection stats:", stats)
