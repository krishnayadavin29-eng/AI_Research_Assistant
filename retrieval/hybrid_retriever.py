"""Hybrid retriever: fuse BM25 and vector search results via Reciprocal Rank Fusion."""
from __future__ import annotations

from typing import List, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

import config
from retrieval.bm25_search import BM25Retriever


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    def __init__(self, vector_store: FAISS, bm25_retriever: BM25Retriever) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25_retriever

    def retrieve(self, query: str, k: int = config.TOP_K_FINAL) -> List[Document]:
        """Retrieve and fuse results from both retrieval methods."""
        vector_hits = self.vector_store.similarity_search(query, k=config.TOP_K_VECTOR)
        bm25_hits = self.bm25.search(query, k=config.TOP_K_BM25)

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(vector_hits):
            uid = self._uid(doc)
            scores[uid] = scores.get(uid, 0.0) + _rrf_score(rank) * config.HYBRID_ALPHA
            doc_map[uid] = doc

        for rank, doc in enumerate(bm25_hits):
            uid = self._uid(doc)
            scores[uid] = scores.get(uid, 0.0) + _rrf_score(rank) * (1 - config.HYBRID_ALPHA)
            doc_map[uid] = doc

        ranked = sorted(scores.keys(), key=lambda uid: scores[uid], reverse=True)
        return [doc_map[uid] for uid in ranked[:k]]

    @staticmethod
    def _uid(doc: Document) -> str:
        src = doc.metadata.get("source", "")
        idx = doc.metadata.get("chunk_index", 0)
        return f"{src}::{idx}"
