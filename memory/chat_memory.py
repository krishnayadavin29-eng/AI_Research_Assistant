"""Conversation memory: store and retrieve chat history."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Message:
    role: str      # "user" or "assistant"
    content: str


class ChatMemory:
    """
    Simple in-process conversation buffer.

    Keeps the last `max_turns` exchanges so the prompt window stays manageable.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._history: List[Message] = []
        self.max_turns = max_turns

    # ── public ────────────────────────────────────────────────────────────────

    def add_user(self, text: str) -> None:
        self._history.append(Message(role="user", content=text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._history.append(Message(role="assistant", content=text))
        self._trim()

    def get_history(self) -> List[Message]:
        return list(self._history)

    def to_lc_messages(self) -> List[dict]:
        """Return history as LangChain-style message dicts."""
        return [{"role": m.role, "content": m.content} for m in self._history]

    def clear(self) -> None:
        self._history.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _trim(self) -> None:
        # Keep system message if present, trim oldest pairs otherwise
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
