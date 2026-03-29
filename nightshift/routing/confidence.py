"""Confidence-gated router. Skip API when local models can handle it."""

from __future__ import annotations

from typing import Any


class ConfidenceGate:
    """Estimates whether a task can be handled locally without API call.

    Decision matrix:
    - EXTRACTION (NER, parsing): always local
    - RETRIEVAL (search, rerank): always local
    - GENERATION (simple): try local, fallback to API
    - GENERATION (complex): route to API
    - EVALUATION (clear cases): local; ambiguous: API
    """

    def __init__(self, config: Any) -> None:
        self.threshold = config.confidence_threshold
        self._local_hits = 0
        self._api_routes = 0

    def try_local(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        """Attempt local handling. Returns result if confident, None if API needed."""
        task_type = self._classify_task(messages)
        confidence = self._estimate_confidence(messages, task_type)

        if confidence >= self.threshold:
            result = self._handle_locally(messages, task_type)
            if result is not None:
                self._local_hits += 1
                return result

        self._api_routes += 1
        return None

    def stats(self) -> dict[str, int]:
        return {"local_hits": self._local_hits, "api_routes": self._api_routes}

    def _classify_task(self, messages: list[dict[str, str]]) -> str:
        """Classify the task type from message content."""
        # TODO: use ModernBERT classifier or heuristic rules
        last_content = messages[-1].get("content", "").lower()

        if any(kw in last_content for kw in ["extract", "find entities", "parse", "identify"]):
            return "extraction"
        if any(kw in last_content for kw in ["search", "find similar", "retrieve", "rank"]):
            return "retrieval"
        if any(kw in last_content for kw in ["is this novel", "evaluate", "judge", "score"]):
            return "evaluation"
        return "generation"

    def _estimate_confidence(self, messages: list[dict[str, str]], task_type: str) -> float:
        """Estimate confidence that local models can handle this."""
        if task_type in ("extraction", "retrieval"):
            return 0.95  # always handle locally
        if task_type == "evaluation":
            # TODO: check embedding distance to known items
            return 0.5  # ambiguous by default
        # generation
        content_len = sum(len(m.get("content", "")) for m in messages)
        if content_len < 500:
            return 0.7  # short generation, local might handle
        return 0.3  # complex generation, need API

    def _handle_locally(
        self, messages: list[dict[str, str]], task_type: str
    ) -> dict[str, Any] | None:
        """Handle task with local models."""
        # TODO: route to appropriate local model (GLiNER, MiniLM, Gemma 270M)
        return None
