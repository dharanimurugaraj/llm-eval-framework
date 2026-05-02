"""Embedding helpers for vector ingestion (Phase 1).

Wraps OpenAI and local HuggingFace embedding models so experiments can
compare retrieval quality across providers.
"""

from __future__ import annotations

import logging
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding models for converting text to vectors.

    WHY THIS EXISTS: Different embedding models produce different
    quality embeddings for different domains. A core experiment in
    this framework is comparing OpenAI embeddings vs BGE-M3 to see
    which produces better retrieval scores.

    Supports:

    - ``openai``: ``text-embedding-3-small`` (API-based, costs money, very good).
    - ``bge``: ``BAAI/bge-m3`` (local, free, strong multilingual performance).
    """

    def __init__(self, model_name: str = "openai") -> None:
        """Initialize the backing LangChain embeddings implementation.

        API keys are never hardcoded; ``OPENAI_API_KEY`` is read from the
        environment (after ``load_dotenv()``).

        Args:
            model_name: ``"openai"`` or ``"bge"``.

        Raises:
            ValueError: If ``model_name`` is unknown or OpenAI is selected without a key.
        """
        key: str = model_name.strip().lower()
        self.model_name: str = key

        if key == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to your .env for OpenAI embeddings."
                )
            self.embeddings: OpenAIEmbeddings | HuggingFaceEmbeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
            )
        elif key == "bge":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": False},
            )
        else:
            raise ValueError(
                f"Unknown model_name: {model_name!r}. Use 'openai' or 'bge'."
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings (e.g. chunks) in one batch-friendly call.

        Args:
            texts: Plain strings to embed.

        Returns:
            One embedding vector (list of floats) per input string, same order.
        """
        logger.info("Embedding %d documents with %s...", len(texts), self.model_name)
        vectors: list[list[float]] = self.embeddings.embed_documents(texts)
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string (used at retrieval time).

        Args:
            query: User or system query text.

        Returns:
            One embedding vector as a list of floats.
        """
        vec = self.embeddings.embed_query(query)
        return list(map(float, vec))

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Cosine similarity between two equal-length vectors.

        Math: ``cos(theta) = (a · b) / (||a|| * ||b||)`` — dot product divided
        by the product of L2 norms. Result in ``[-1, 1]`` (1 = same direction).

        Args:
            vec1: First embedding.
            vec2: Second embedding (same dimension as ``vec1``).

        Returns:
            Cosine similarity as a scalar float.

        Raises:
            ValueError: If shapes differ or a vector has zero norm.
        """
        a: np.ndarray = np.asarray(vec1, dtype=np.float64)
        b: np.ndarray = np.asarray(vec2, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same length for cosine similarity.")
        denom: float = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            raise ValueError("Cannot compute cosine similarity for a zero vector.")
        return float(np.dot(a, b) / denom)

    def get_model_info(self) -> dict[str, str | int]:
        """Return static model metadata for logging or UI."""
        if self.model_name == "openai":
            return {
                "model_name": "text-embedding-3-small",
                "provider": "openai",
                "dimensions": 1536,
                "cost_per_token": "~$0.00002/1K tokens",
            }
        return {
            "model_name": "BAAI/bge-m3",
            "provider": "huggingface",
            "dimensions": 1024,
            "cost_per_token": "free (local)",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env to run the embedding demo.")
        raise SystemExit(0)

    mgr = EmbeddingManager(model_name="openai")
    s1 = "The company reported strong revenue growth."
    s2 = "Financial results exceeded analyst expectations."
    v1, v2 = mgr.embed_documents([s1, s2])
    sim: float = mgr.cosine_similarity(v1, v2)
    print(f"Cosine similarity: {sim:.4f}")
    print(
        "High similarity (~0.85+) is expected because both sentences discuss "
        "strong financial performance in similar language."
    )
