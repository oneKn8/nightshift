"""Sliding window history manager. O(1) growth instead of O(n)."""

from __future__ import annotations

from typing import Any


class SlidingWindowManager:
    """Replaces unbounded conversation history with bounded, compressed history.

    Three tiers:
    1. Active window: last N turns in full detail
    2. Compressed archive: older turns summarized locally
    3. Knowledge graph: facts extracted and persisted
    """

    def __init__(self, config: Any) -> None:
        self.window_size = config.window_size
        self._archive: list[str] = []  # compressed summaries
        self._active: list[dict[str, str]] = []  # recent messages in full

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
        """Ingest API response for knowledge graph extraction."""
        # TODO: extract facts from response, store in persistent KG
        pass

    def _compress_messages(self, messages: list[dict[str, str]]) -> str:
        """Compress a batch of messages into a summary."""
        # TODO: use T5-small or Gemma 270M for local summarization
        # For now, extract key lines
        lines = []
        for msg in messages:
            content = msg.get("content", "")
            # Take first 100 chars of each message as crude summary
            if content:
                lines.append(f"[{msg.get('role', '?')}]: {content[:100]}...")
        return "\n".join(lines[-20:])  # keep last 20 summaries
