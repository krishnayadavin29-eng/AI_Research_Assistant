"""
Main RAG pipeline.

Responsibilities:
  - Ingest documents (load → split → embed → index)
  - Answer questions   (rewrite → hybrid retrieve → rerank → generate)
  - Summarise corpus
"""
from __future__ import annotations

from typing import List, Optional

from langchain.schema import Document

import config
from loaders.document_loader import load_documents
from processing.text_splitter import split_documents
from retrieval.vector_store import build_vector_store, load_vector_store, vector_store_exists
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import rerank
from retrieval.rewrite_query import rewrite_query
from chains.qa_chain import answer_question
from chains.summary_chain import summarise_documents
from memory.chat_memory import ChatMemory
from utils.citation import format_citations


class ResearchPipeline:
    def __init__(self) -> None:
        self._chunks: List[Document] = []
        self._hybrid: Optional[HybridRetriever] = None
        self.memory = ChatMemory()
        self._ready = False

    # ── ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, source: str) -> int:
        """
        Load, chunk, and index documents from a path or directory.

        Returns:
            Number of chunks indexed.
        """
        raw_docs = load_documents(source)
        self._chunks = split_documents(raw_docs)
        vector_store = build_vector_store(self._chunks)
        bm25 = BM25Retriever(self._chunks)
        self._hybrid = HybridRetriever(vector_store, bm25)
        self._ready = True
        return len(self._chunks)

    def load_existing_index(self) -> bool:
        """
        Load an already-built FAISS index from disk.

        Returns:
            True if the index was found and loaded successfully.
        """
        if not vector_store_exists():
            return False
        try:
            vector_store = load_vector_store()
            # BM25 needs chunks in memory; rebuild from the vector store's docstore
            self._chunks = list(vector_store.docstore._dict.values())
            bm25 = BM25Retriever(self._chunks)
            self._hybrid = HybridRetriever(vector_store, bm25)
            self._ready = True
            return True
        except Exception as exc:
            print(f"[Pipeline] Failed to load index: {exc}")
            return False

    # ── querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a single question.

        Returns:
            dict with keys: answer, citations, sources (List[Document])
        """
        self._require_ready()

        # 1. Query rewriting
        queries = rewrite_query(question)

        # 2. Hybrid retrieval across all query variants
        all_docs: List[Document] = []
        seen_ids: set[str] = set()
        for q in queries:
            for doc in self._hybrid.retrieve(q, k=config.TOP_K_FINAL * 2):
                uid = f"{doc.metadata.get('source')}::{doc.metadata.get('chunk_index')}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    all_docs.append(doc)

        # 3. Rerank combined candidates
        final_docs = rerank(question, all_docs, top_k=config.TOP_K_FINAL)

        # 4. Generate answer with memory
        self.memory.add_user(question)
        answer = answer_question(question, final_docs, self.memory)
        self.memory.add_assistant(answer)

        return {
            "answer": answer,
            "citations": format_citations(final_docs),
            "sources": final_docs,
        }

    def summarise(self) -> str:
        """Return a structured summary of all indexed documents."""
        self._require_ready()
        return summarise_documents(self._chunks)

    def reset_memory(self) -> None:
        self.memory.clear()

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Pipeline is not initialised. Call ingest() or load_existing_index() first.")
