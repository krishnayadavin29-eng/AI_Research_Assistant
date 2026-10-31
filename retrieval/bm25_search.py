"""BM25 keyword retriever over document chunks."""
from __future__ import annotations

from typing import List

from langchain.schema import Document
from rank_bm25 import BM25Okapi

import config


class BM25Retriever:
    def __init__(self, docs: List[Document]) -> None:
        self.docs = docs
        tokenized = [self._tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    # ── public ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = config.TOP_K_BM25) -> List[Document]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked_indices[:k]]

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()
