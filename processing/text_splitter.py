"""Recursive character text splitter with sensible defaults for research papers."""
from __future__ import annotations

from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks suitable for embedding.

    Args:
        docs: Raw LangChain Document objects.

    Returns:
        List of chunked Document objects with inherited metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Tag each chunk with its index within its source file
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[src] = idx + 1

    return chunks
