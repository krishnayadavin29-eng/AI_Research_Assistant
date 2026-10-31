"""Query rewriting: expand or reformulate a user query for better retrieval."""
from __future__ import annotations

import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

import config

_SYSTEM_PROMPT = """You are a query expansion assistant for a scientific literature search engine.
Given a user question, output 3 alternative search queries that capture different aspects of the intent.
Respond with ONLY a JSON array of strings and nothing else.
Example: ["query 1", "query 2", "query 3"]"""


def rewrite_query(query: str) -> list[str]:
    """
    Generate alternative search queries using Gemini.

    Returns the original query plus up to 3 rewritten variants.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.5,
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip().strip("```json").strip("```").strip()
        variants: list[str] = json.loads(raw)
        return [query] + [v for v in variants if v != query]

    except Exception:
        # Graceful fallback: return original query unchanged
        return [query]
