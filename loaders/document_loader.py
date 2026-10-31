"""Load PDF and text documents from a directory or individual file paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.schema import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(source: str) -> List[Document]:
    """
    Load documents from a file path or directory.

    Args:
        source: Path to a single file or a directory.

    Returns:
        List of LangChain Document objects with metadata.
    """
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if path.is_file():
        return _load_single_file(path)

    # Directory – load all supported file types
    docs: List[Document] = []
    for ext in SUPPORTED_EXTENSIONS:
        for fpath in path.rglob(f"*{ext}"):
            try:
                docs.extend(_load_single_file(fpath))
            except Exception as exc:
                print(f"[DocumentLoader] Skipping {fpath}: {exc}")
    return docs


def _load_single_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    # Enrich metadata
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("filename", path.name)
    return docs
