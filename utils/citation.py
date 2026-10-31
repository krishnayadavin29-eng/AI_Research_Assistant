"""Format source documents into readable citation strings."""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.schema import Document


def format_citations(docs: List[Document]) -> str:
    """
    Build a numbered citation block from retrieved source documents.

    Args:
        docs: The retrieved Document objects (after reranking).

    Returns:
        Formatted multi-line string with one citation per line.
    """
    if not docs:
        return "_No sources retrieved._"

    lines: List[str] = []
    seen: set[str] = set()

    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "Unknown")
        filename = Path(src).name
        page = doc.metadata.get("page")
        chunk = doc.metadata.get("chunk_index")

        parts = [f"**[{i}]** `{filename}`"]
        if page is not None:
            parts.append(f"page {page + 1}")
        if chunk is not None:
            parts.append(f"chunk {chunk}")

        citation = " · ".join(parts)
        if citation not in seen:
            seen.add(citation)
            lines.append(citation)

    return "\n".join(lines)


def build_context_block(docs: List[Document]) -> str:
    """
    Concatenate retrieved document chunks into a single context string for the LLM prompt.
    Each chunk is labelled with its source filename.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        src = Path(doc.metadata.get("source", "Unknown")).name
        parts.append(f"[Source {i}: {src}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)
