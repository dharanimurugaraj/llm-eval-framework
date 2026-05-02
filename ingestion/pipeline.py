"""Orchestrates Phase 1 + Phase 2: load → chunk → embed → Qdrant.

Single entry point for ingestion experiments (chunking and embedding knobs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document

from ingestion.chunker import get_chunker
from ingestion.loader import DocumentLoader
from ingestion.vector_store import VectorStoreManager

load_dotenv()


class IngestionPipeline:
    """Orchestrates the full ingestion flow: Load PDFs → Chunk → Embed → Store.

    WHY THIS EXISTS: Instead of manually calling loader, chunker,
    and vector store separately, this class runs the full pipeline
    in one call. It also logs experiment config so we know exactly
    what settings produced which results.

    This is the entry point for running ingestion experiments:

    - Change ``chunking_strategy`` to compare chunk quality
    - Change ``embedding_model`` to compare embedding quality
    - ``recreate_collection=True`` ensures a clean slate between experiments
    """

    def __init__(
        self,
        chunking_strategy: str = "recursive",
        embedding_model: str = "openai",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str | None = None,
    ) -> None:
        """Build loader, chunker, and vector store from experiment parameters.

        Args:
            chunking_strategy: ``fixed_size``, ``recursive``, or ``semantic``.
            embedding_model: ``openai`` or ``bge`` for :class:`EmbeddingManager`.
            chunk_size: Passed to fixed/recursive chunkers (semantic ignores size).
            chunk_overlap: Overlap for fixed/recursive chunkers.
            collection_name: Optional Qdrant collection override.
        """
        self.chunking_strategy: str = chunking_strategy
        self.embedding_model: str = embedding_model
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

        self.loader: DocumentLoader = DocumentLoader()
        self.chunker = get_chunker(
            chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store: VectorStoreManager = VectorStoreManager(
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        self.collection_name: str = self.vector_store.collection_name

        print(
            "Ingestion Pipeline Config:\n"
            f"  Chunking Strategy: {chunking_strategy}\n"
            f"  Embedding Model: {embedding_model}\n"
            f"  Chunk Size: {chunk_size}\n"
            f"  Chunk Overlap: {chunk_overlap}\n"
            f"  Collection: {self.collection_name}"
        )

    def run(self, recreate_collection: bool = True) -> dict[str, Any]:
        """Run load → chunk → Qdrant collection → store.

        Args:
            recreate_collection: When ``True``, drop and recreate the Qdrant
                collection before upsert (recommended between experiments).

        Returns:
            Metadata dict for logging (e.g. to W&B).
        """
        print("Step 1/4: Loading documents...")
        documents: list[Document] = self.loader.load_all_pdfs()

        print(f"Step 2/4: Chunking with {self.chunking_strategy}...")
        chunks: list[Document] = self.chunker.chunk(documents)

        print("Step 3/4: Preparing Qdrant collection...")
        self.vector_store.create_collection(recreate=recreate_collection)

        print(f"Step 4/4: Storing {len(chunks)} chunks in Qdrant...")
        self.vector_store.store_chunks(chunks)

        lengths: list[int] = [len(c.page_content or "") for c in chunks]
        avg_chunk: float = (
            sum(lengths) / float(len(lengths)) if lengths else 0.0
        )

        return {
            "chunking_strategy": self.chunking_strategy,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "avg_chunk_length": avg_chunk,
            "collection_name": self.collection_name,
        }


def _ensure_project_root_on_path() -> None:
    """Allow ``python ingestion/pipeline.py`` to resolve ``ingestion`` imports."""
    import sys

    root: Path = Path(__file__).resolve().parents[1]
    root_s: str = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


if __name__ == "__main__":
    _ensure_project_root_on_path()

    raw_dir: Path = Path("data/raw").resolve()
    pdfs: list[Path] = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(
            "No PDFs found in data/raw/\n"
            "Add any PDF to data/raw/ and run again.\n"
            "Example: copy any PDF into data/raw/ and re-run this script."
        )
        raise SystemExit(1)

    pipe = IngestionPipeline(
        chunking_strategy="recursive",
        embedding_model="openai",
    )
    meta: dict[str, Any] = pipe.run(recreate_collection=True)
    print("\nIngestion summary:")
    for key, val in meta.items():
        print(f"  {key}: {val}")
