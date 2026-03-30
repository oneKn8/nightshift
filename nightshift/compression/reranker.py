"""Reranking stage using MiniLM cross-encoder or length-based fallback."""
from __future__ import annotations

from typing import Any


class Reranker:
    """Rank text chunks by relevance to query. Uses cross-encoder when available."""

    def __init__(self, use_model: bool = True, model: Any = None) -> None:
        self._use_model = use_model and model is not None
        self._model = model

    def rank(
        self, chunks: list[str], query: str | None = None, top_k: int = 10
    ) -> list[str]:
        if not chunks:
            return []
        if len(chunks) <= top_k:
            return chunks
        if self._use_model and query:
            return self._model_rank(chunks, query, top_k)
        return self._fallback_rank(chunks, query, top_k)

    def _model_rank(self, chunks: list[str], query: str, top_k: int) -> list[str]:
        """Use cross-encoder to score and rank."""
        pairs = [(query, chunk) for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

    def _fallback_rank(
        self, chunks: list[str], query: str | None, top_k: int
    ) -> list[str]:
        """Fallback: keyword overlap scoring or just first K."""
        if not query:
            return chunks[:top_k]
        query_words = set(query.lower().split())
        scored = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            scored.append((overlap, chunk))
        scored.sort(reverse=True)
        return [chunk for _, chunk in scored[:top_k]]
