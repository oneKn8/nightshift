"""Shared utilities."""
from __future__ import annotations

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(content: str | list[dict[str, str]]) -> int:
    """Count tokens using tiktoken cl100k_base encoding.

    Accepts a string or a list of message dicts.
    """
    if isinstance(content, list):
        total = 0
        for msg in content:
            total += count_tokens(msg.get("content", ""))
            total += 4  # role + formatting overhead per message
        return total
    return len(_encoder.encode(content))
