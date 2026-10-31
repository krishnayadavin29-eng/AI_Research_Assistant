"""FAISS-backed vector store: build, persist, and load."""
from __future__ import annotations

import os
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

import config
from processing.embeddings import get_embedding_model


def build_vector_store(chunks: List[Document]) -> FAISS:
    """Build a FAISS index from document chunks and persist it to disk."""
    embeddings = get_embedding_model()
    store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
    store.save_local(config.VECTOR_STORE_PATH)
    return store


def load_vector_store() -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embedding_model()
    return FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def vector_store_exists() -> bool:
    """Return True if a saved vector store exists on disk."""
    return os.path.isdir(config.VECTOR_STORE_PATH) and bool(
        os.listdir(config.VECTOR_STORE_PATH)
    )


def similarity_search(store: FAISS, query: str, k: int = config.TOP_K_VECTOR) -> List[Document]:
    return store.similarity_search(query, k=k)
