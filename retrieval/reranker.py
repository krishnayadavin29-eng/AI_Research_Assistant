"""Cross-encoder reranker using sentence-transformers."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain.schema import Document
from sentence_transformers import CrossEncoder

import config

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(query: str, docs: List[Document], top_k: int = config.TOP_K_FINAL) -> List[Document]:
    """
    Rerank documents using a cross-encoder model.

    Args:
        query:  The user query.
        docs:   Candidate documents to rerank.
        top_k:  Number of top documents to return.

    Returns:
        Reranked list of documents.
    """
    if not docs:
        return []

    encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = encoder.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
