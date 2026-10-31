"""Question-answering chain with retrieved context and conversation history."""
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

import config
from memory.chat_memory import ChatMemory
from utils.citation import build_context_block

_SYSTEM_PROMPT = """You are an expert AI research assistant specialising in scientific literature.
Answer the user's question based ONLY on the provided context passages.
If the context does not contain enough information, say so clearly rather than guessing.
Always ground your answer in the context; reference source numbers like [Source 1] when relevant.
Be precise, concise, and cite specific passages when possible."""


def answer_question(query: str, docs: list, memory: ChatMemory) -> str:
    """
    Generate an answer grounded in retrieved document chunks.

    Args:
        query:  The user's current question.
        docs:   Reranked Document objects to use as context.
        memory: Conversation history buffer.

    Returns:
        The assistant's answer as a string.
    """
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=config.LLM_MAX_TOKENS,
    )

    context = build_context_block(docs)

    # Build message history
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    for msg in memory.get_history():
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    # Add current question with context
    messages.append(
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    )

    response = llm.invoke(messages)
    return response.content
