"""Sliding window history manager. O(1) growth instead of O(n)."""

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nightshift.compression.summarizer import Summarizer
    from nightshift.history.knowledge import KnowledgeGraph

log = logging.getLogger(__name__)


class SlidingWindowManager:
    """Replaces unbounded conversation history with bounded, compressed history.

    Three tiers:
    1. Active window: last N turns in full detail
    2. Compressed archive: older turns summarized locally
    3. Knowledge graph: facts extracted and persisted
    """

    def __init__(self, config: Any, summarizer: Summarizer | None = None) -> None:
        self.window_size = config.window_size
        self._archive: list[str] = []  # compressed summaries
        self._active: list[dict[str, str]] = []  # recent messages in full
        self._summarizer = summarizer
        self._kg: KnowledgeGraph | None = None

    def apply(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Apply sliding window to message history."""
        if len(messages) <= self.window_size * 2:
            return messages

        # System message always preserved
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= self.window_size * 2:
            return messages

        # Split: old messages to compress, recent to keep
        cutoff = len(non_system) - (self.window_size * 2)
        old = non_system[:cutoff]
        recent = non_system[cutoff:]

        # Compress old messages
        summary = self._compress_messages(old)
        self._archive.append(summary)

        # Rebuild: system + archive summary + recent
        archive_msg = {
            "role": "system",
            "content": f"[Compressed history from {len(old)} previous messages:\n{summary}]",
        }

        return system_msgs + [archive_msg] + recent

    def ingest_response(self, response: dict[str, Any]) -> None:
        """Extract factual claims from API response and store in knowledge graph."""
        if self._kg is None:
            return

        content = response.get("content", "")
        if not content or len(content) < 50:
            return

        facts = self._extract_facts(content)
        if facts:
            model = response.get("model", "unknown")
            self._kg.add(
                facts,
                metadata=[{"source": f"api:{model}"}] * len(facts),
            )
            log.debug(f"Ingested {len(facts)} facts from API response")

    def set_summarizer(self, summarizer: Summarizer) -> None:
        """Inject summarizer after construction (for deferred model loading)."""
        self._summarizer = summarizer

    def set_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """Inject knowledge graph for response fact extraction."""
        self._kg = kg

    @staticmethod
    def _extract_facts(text: str) -> list[str]:
        """Extract factual statements from response text.

        Heuristic extraction: numbered items, bullet points, and
        sentences containing factual indicators.
        """
        facts: list[str] = []

        # Split into lines for structured extraction
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Numbered items (1. ..., 1) ..., etc.)
            if re.match(r"^\d+[\.\)]\s+", line):
                fact = re.sub(r"^\d+[\.\)]\s+", "", line).strip()
                if len(fact) > 20:
                    facts.append(fact)
                continue

            # Bullet points (-, *, >)
            if re.match(r"^[-*>]\s+", line):
                fact = re.sub(r"^[-*>]\s+", "", line).strip()
                if len(fact) > 20:
                    facts.append(fact)
                continue

        # If no structured facts found, extract sentences with factual indicators
        if not facts:
            indicators = [
                "found that", "shows that", "demonstrates", "indicates",
                "suggests", "reveals", "confirmed", "discovered",
                "achieved", "outperforms", "results in", "leads to",
                "is defined as", "consists of", "refers to",
            ]
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 30 or len(sent) > 500:
                    continue
                if any(ind in sent.lower() for ind in indicators):
                    facts.append(sent)

        # Cap at 20 facts per response to avoid noise
        return facts[:20]

    def _compress_messages(self, messages: list[dict[str, str]]) -> str:
        """Compress a batch of messages into a summary.

        Uses T5-small when available for abstractive summaries.
        Falls back to extractive (first N chars) when no model loaded.
        """
        if self._summarizer is not None:
            # Concatenate messages into a single block, then summarize
            block = "\n".join(
                f"[{msg.get('role', '?')}]: {msg.get('content', '')}"
                for msg in messages
                if msg.get("content")
            )
            # T5-small has 512 token input limit; chunk if needed
            if len(block) > 1800:
                chunks = [block[i:i + 1800] for i in range(0, len(block), 1800)]
                summaries = self._summarizer.summarize_chunks(chunks, max_length=80)
                return "\n".join(summaries)
            return self._summarizer.summarize(block, max_length=120)

        # Fallback: extractive (first 100 chars per message)
        lines = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                lines.append(f"[{msg.get('role', '?')}]: {content[:100]}...")
        return "\n".join(lines[-20:])
