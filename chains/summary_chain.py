"""Summarisation chain: produce a concise summary of all indexed documents."""
from __future__ import annotations

from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, Document

import config

_SYSTEM_PROMPT = """You are an expert research summariser.
Given a set of document excerpts, produce a structured summary with:
1. A one-paragraph executive overview.
2. Key themes and findings (bullet list).
3. Notable methodologies mentioned.
4. Suggested follow-up questions a researcher might ask.
Be factual and concise."""


def summarise_documents(docs: List[Document]) -> str:
    """
    Summarise a collection of document chunks.

    Args:
        docs: Document chunks to summarise.

    Returns:
        A structured markdown summary string.
    """
    from utils.citation import build_context_block

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.3,
        max_output_tokens=config.LLM_MAX_TOKENS,
    )

    context = build_context_block(docs[:20])  # cap to stay within context window

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Document excerpts:\n\n{context}"),
    ]

    response = llm.invoke(messages)
    return response.content
