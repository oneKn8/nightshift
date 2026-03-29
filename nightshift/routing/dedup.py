"""Content deduplication. Never re-send what the API has already seen."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class SentContent:
    content_hash: str
    call_id: int
    token_count: int


class ContentDedup:
    """Tracks content sent to the API. Replaces duplicates with references."""

    def __init__(self) -> None:
        self._cache: dict[str, SentContent] = {}
        self._call_counter = 0

    def process(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Strip duplicate content, replace with compact references."""
        self._call_counter += 1
        result = []

        for msg in messages:
            content = msg.get("content", "")
            if len(content) < 200:
                # Too short to bother deduplicating
                result.append(msg)
                continue

            h = self._hash(content)
            if h in self._cache:
                ref = self._cache[h]
                result.append({
                    **msg,
                    "content": (
                        f"[Previously provided in call #{ref.call_id}, "
                        f"{ref.token_count} tokens, unchanged. "
                        f"Refer to that context.]"
                    ),
                })
            else:
                token_est = len(content) // 4
                self._cache[h] = SentContent(
                    content_hash=h,
                    call_id=self._call_counter,
                    token_count=token_est,
                )
                result.append(msg)

        return result

    def clear(self) -> None:
        """Reset cache (e.g., between sessions)."""
        self._cache.clear()
        self._call_counter = 0

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
