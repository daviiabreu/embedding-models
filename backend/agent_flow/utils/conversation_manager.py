"""
Conversation management utilities.

This module provides utilities for managing conversation history:
- Automatic summarization of long conversations
- Context window optimization
- History pruning strategies
"""

import os
from dataclasses import dataclass, field
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationMessage:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def token_estimate(self) -> int:
        """Estimate token count for this message."""
        # Rough estimate: ~4 chars per token
        return len(self.content) // 4 + 10  # +10 for role/formatting overhead


@dataclass
class ConversationSummary:
    """Summary of a conversation segment."""

    summary: str
    message_count: int
    start_index: int
    end_index: int
    topics: list[str] = field(default_factory=list)


class ConversationManager:
    """
    Manages conversation history with automatic summarization.

    Features:
    - Automatic summarization when history exceeds threshold
    - Multiple pruning strategies
    - Token budget management
    - Topic tracking

    Examples:
        >>> manager = ConversationManager(max_messages=20)
        >>> manager.add_message("user", "What courses does Inteli offer?")
        >>> manager.add_message("assistant", "Inteli offers Computer Science...")
        >>> history = manager.get_history_for_llm(max_tokens=4000)
    """

    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 8000,
        summarize_threshold: int = 15,
        keep_recent: int = 5,
    ):
        """
        Initialize the conversation manager.

        Args:
            max_messages: Maximum messages before pruning
            max_tokens: Maximum token budget for history
            summarize_threshold: Trigger summarization after this many messages
            keep_recent: Always keep this many recent messages
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self.keep_recent = keep_recent

        self.messages: list[ConversationMessage] = []
        self.summaries: list[ConversationSummary] = []
        self._llm_model = None

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
            timestamp: Optional timestamp
            metadata: Optional metadata
        """
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        self.messages.append(message)

        # Check if we need to summarize
        if len(self.messages) >= self.summarize_threshold:
            self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        """Summarize older messages if threshold is exceeded."""
        if len(self.messages) < self.summarize_threshold:
            return

        # Keep recent messages, summarize the rest
        messages_to_summarize = self.messages[: -self.keep_recent]
        recent_messages = self.messages[-self.keep_recent :]

        if len(messages_to_summarize) < 3:
            return  # Not enough to summarize

        # Generate summary
        summary = self._generate_summary(messages_to_summarize)

        if summary:
            self.summaries.append(summary)
            self.messages = recent_messages

            logger.info(
                "conversation_summarized",
                messages_summarized=summary.message_count,
                summary_length=len(summary.summary),
                remaining_messages=len(self.messages),
            )

    def _generate_summary(
        self, messages: list[ConversationMessage]
    ) -> ConversationSummary | None:
        """Generate a summary of messages using LLM or heuristics."""
        if not messages:
            return None

        # Try LLM-based summarization if available
        if os.getenv("GOOGLE_API_KEY"):
            try:
                return self._llm_summarize(messages)
            except Exception as e:
                logger.warning(f"LLM summarization failed, using heuristic: {e}")

        # Fallback to heuristic summarization
        return self._heuristic_summarize(messages)

    def _llm_summarize(
        self, messages: list[ConversationMessage]
    ) -> ConversationSummary:
        """Use LLM to generate a conversation summary."""
        import google.generativeai as genai

        if self._llm_model is None:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self._llm_model = genai.GenerativeModel(
                os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
            )

        # Format messages for summarization
        messages_text = "\n".join([f"{m.role.upper()}: {m.content}" for m in messages])

        prompt = f"""Resuma a seguinte conversa de forma concisa, mantendo os pontos principais:

{messages_text}

Forneça um resumo de 2-3 frases que capture:
1. Os tópicos principais discutidos
2. Informações importantes mencionadas
3. Qualquer decisão ou conclusão

Responda apenas com o resumo, sem explicações adicionais."""

        response = self._llm_model.generate_content(prompt)
        summary_text = response.text.strip()

        # Extract topics (simple heuristic)
        topics = self._extract_topics(messages)

        return ConversationSummary(
            summary=summary_text,
            message_count=len(messages),
            start_index=0,
            end_index=len(messages) - 1,
            topics=topics,
        )

    def _heuristic_summarize(
        self, messages: list[ConversationMessage]
    ) -> ConversationSummary:
        """Generate a simple heuristic-based summary."""
        # Extract first user question and last assistant answer
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]

        parts = []
        if user_messages:
            first_q = user_messages[0].content[:100]
            parts.append(f"Usuario perguntou sobre: {first_q}...")
        if assistant_messages:
            last_a = assistant_messages[-1].content[:100]
            parts.append(f"Ultima resposta: {last_a}...")

        summary_text = " ".join(parts) if parts else "Conversa anterior."

        return ConversationSummary(
            summary=summary_text,
            message_count=len(messages),
            start_index=0,
            end_index=len(messages) - 1,
            topics=self._extract_topics(messages),
        )

    def _extract_topics(self, messages: list[ConversationMessage]) -> list[str]:
        """Extract topics from messages using keyword matching."""
        topic_keywords = {
            "cursos": ["curso", "graduacao", "ciencia", "engenharia", "computacao"],
            "admissao": ["vestibular", "inscricao", "processo seletivo", "admissao"],
            "bolsas": ["bolsa", "bolsas", "financiamento", "desconto"],
            "campus": ["campus", "localizacao", "estrutura", "laboratorio"],
            "pessoas": ["fundador", "ceo", "professor", "equipe"],
        }

        all_content = " ".join(m.content.lower() for m in messages)
        found_topics = []

        for topic, keywords in topic_keywords.items():
            if any(kw in all_content for kw in keywords):
                found_topics.append(topic)

        return found_topics

    def get_history_for_llm(
        self, max_tokens: int | None = None
    ) -> list[dict[str, str]]:
        """
        Get conversation history formatted for LLM consumption.

        Args:
            max_tokens: Maximum token budget (uses default if None)

        Returns:
            List of message dicts with "role" and "content"
        """
        budget = max_tokens or self.max_tokens
        result = []
        current_tokens = 0

        # First add summaries
        for summary in self.summaries:
            summary_msg = {
                "role": "system",
                "content": f"[Resumo da conversa anterior: {summary.summary}]",
            }
            tokens = len(summary.summary) // 4 + 20
            if current_tokens + tokens <= budget:
                result.append(summary_msg)
                current_tokens += tokens

        # Then add recent messages (in reverse to prioritize most recent)
        for message in reversed(self.messages):
            msg_dict = {"role": message.role, "content": message.content}
            tokens = message.token_estimate()
            if current_tokens + tokens <= budget:
                result.insert(
                    len([r for r in result if r["role"] == "system"]), msg_dict
                )
                current_tokens += tokens
            else:
                break

        return result

    def get_full_history(self) -> list[dict[str, Any]]:
        """Get complete conversation history including summaries."""
        return {
            "summaries": [
                {
                    "summary": s.summary,
                    "message_count": s.message_count,
                    "topics": s.topics,
                }
                for s in self.summaries
            ],
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "total_messages": sum(s.message_count for s in self.summaries)
            + len(self.messages),
        }

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages.clear()
        self.summaries.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get conversation statistics."""
        return {
            "active_messages": len(self.messages),
            "summaries": len(self.summaries),
            "total_messages_processed": sum(s.message_count for s in self.summaries)
            + len(self.messages),
            "estimated_tokens": sum(m.token_estimate() for m in self.messages)
            + sum(len(s.summary) // 4 for s in self.summaries),
            "topics_discussed": list(
                set(topic for s in self.summaries for topic in s.topics)
            ),
        }


# Singleton instance for shared conversation management
_default_manager: ConversationManager | None = None


def get_conversation_manager() -> ConversationManager:
    """Get the default conversation manager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConversationManager()
    return _default_manager


__all__ = [
    "ConversationManager",
    "ConversationMessage",
    "ConversationSummary",
    "get_conversation_manager",
]
