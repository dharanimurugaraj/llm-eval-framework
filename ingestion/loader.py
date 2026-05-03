"""PDF document loading for the ingestion pipeline.

Loads PDFs from ``data/raw/`` (or a custom directory), attaches rich metadata
to each page-level ``Document``, and exposes helpers for batch loading and
aggregate statistics. Used by Phase 1 before chunking and embedding.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads PDF files from disk into LangChain ``Document`` objects (one per page).

    Each loaded page carries metadata for traceability (source file, path,
    page count, load timestamp). Use :meth:`load_all_pdfs` to ingest every
    ``.pdf`` in the configured directory.
    """

    def __init__(self, data_dir: str = "data/raw") -> None:
        """Resolve ``data_dir`` to an absolute :class:`pathlib.Path` and log PDF count.

        Args:
            data_dir: Directory to scan for ``*.pdf`` files (default ``data/raw``).
        """
        self.data_dir: Path = Path(data_dir).resolve()
        
        # Fallback to sample_data/ if data/raw/ is empty
        # This handles Streamlit Cloud deployment where
        # data/raw/ is gitignored
        if not self.data_dir.exists() or not list(
            self.data_dir.glob("*.pdf")
        ):
            fallback = Path("sample_data").resolve()
            if fallback.exists() and list(fallback.glob("*.pdf")):
                print(f"data/raw/ empty — using sample_data/ instead")
                self.data_dir = fallback

        pdf_paths: list[Path] = sorted(self.data_dir.glob("*.pdf"))
        logger.info(
            "DocumentLoader initialized: data_dir=%s, pdf_count=%d",
            self.data_dir,
            len(pdf_paths),
        )
        print(
            f"[DocumentLoader] data_dir={self.data_dir} — "
            f"found {len(pdf_paths)} PDF file(s)."
        )

    def load_pdf(self, file_path: str) -> list[Document]:
        """Load a single PDF into one :class:`~langchain_core.documents.Document` per page.

        Uses LangChain's :class:`~langchain_community.document_loaders.PyPDFLoader`.
        Enriches each page's metadata with ``source``, ``file_path``, ``page_count``,
        ``loaded_at``, and ``page`` (1-based page index).

        Args:
            file_path: Absolute or relative path to a ``.pdf`` file.

        Returns:
            List of page documents, in order.

        Raises:
            FileNotFoundError: If the path does not exist or is not a file.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            msg = f"PDF not found or not a file: {path}"
            raise FileNotFoundError(msg)

        loader = PyPDFLoader(str(path))
        pages: list[Document] = loader.load()
        filename: str = path.name
        loaded_at: str = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        page_count: int = len(pages)

        for i, doc in enumerate(pages, start=1):
            merged: dict[str, str | int] = {
                **doc.metadata,
                "source": filename,
                "file_path": str(path),
                "page_count": page_count,
                "loaded_at": loaded_at,
                "page": i,
            }
            doc.metadata = merged

        return pages

    def load_all_pdfs(self) -> list[Document]:
        """Load every ``*.pdf`` under :attr:`data_dir` and concatenate all pages.

        Prints progress lines of the form
        ``Loading: filename.pdf (X of Y)``.

        Returns:
            All page documents from all files, in filename order then page order.

        Raises:
            ValueError: If the directory contains no PDF files.
        """
        pdf_paths: list[Path] = sorted(self.data_dir.glob("*.pdf"))
        if not pdf_paths:
            raise ValueError(
                f"No PDF files found in {self.data_dir}. "
                "Add one or more .pdf files to this directory and try again."
            )

        all_docs: list[Document] = []
        total: int = len(pdf_paths)
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            print(f"Loading: {pdf_path.name} ({idx} of {total})")
            all_docs.extend(self.load_pdf(str(pdf_path)))
        return all_docs

    def get_document_stats(self, documents: list[Document]) -> dict[str, float | int | list[str]]:
        """Summarize a list of page-level documents for logging or dashboards.

        Args:
            documents: Typically the output of :meth:`load_pdf` or :meth:`load_all_pdfs`.

        Returns:
            Dictionary with:

            - ``total_documents``: number of unique source filenames.
            - ``total_pages``: total number of ``Document`` rows (pages).
            - ``sources``: sorted unique ``source`` metadata values.
            - ``avg_page_length``: mean character length of ``page_content``.
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_pages": 0,
                "sources": [],
                "avg_page_length": 0.0,
            }

        sources_list: list[str] = []
        for d in documents:
            src = d.metadata.get("source")
            if isinstance(src, str):
                sources_list.append(src)
        unique_sources: list[str] = sorted(set(sources_list))
        total_pages: int = len(documents)
        lengths: list[int] = [len(d.page_content or "") for d in documents]
        avg_len: float = sum(lengths) / float(total_pages) if total_pages else 0.0

        return {
            "total_documents": len(unique_sources),
            "total_pages": total_pages,
            "sources": unique_sources,
            "avg_page_length": avg_len,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DocumentLoader()
    try:
        docs = loader.load_all_pdfs()
        stats = loader.get_document_stats(docs)
        print("Stats:", stats)
    except ValueError as e:
        print(f"No PDFs to load: {e}")
        print("Place .pdf files under data/raw/ and run this module again.")
