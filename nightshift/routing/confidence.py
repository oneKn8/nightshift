"""Confidence-gated router. Skip API when local models can handle it."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nightshift.compression.summarizer import Summarizer
    from nightshift.compression.reranker import Reranker
    from nightshift.history.knowledge import KnowledgeGraph

log = logging.getLogger(__name__)


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
        self._summarizer: Summarizer | None = None
        self._reranker: Reranker | None = None
        self._kg: KnowledgeGraph | None = None

    def set_resources(
        self,
        summarizer: Summarizer | None = None,
        reranker: Reranker | None = None,
        kg: KnowledgeGraph | None = None,
    ) -> None:
        """Inject local models and knowledge graph for local handling."""
        if summarizer is not None:
            self._summarizer = summarizer
        if reranker is not None:
            self._reranker = reranker
        if kg is not None:
            self._kg = kg

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
        last_content = messages[-1].get("content", "").lower()

        if any(kw in last_content for kw in [
            "extract", "find entities", "parse", "identify",
            "list the", "pull out", "what are the key",
        ]):
            return "extraction"
        if any(kw in last_content for kw in [
            "search", "find similar", "retrieve", "rank",
            "look up", "what do we know about", "recall",
        ]):
            return "retrieval"
        if any(kw in last_content for kw in [
            "is this novel", "evaluate", "judge", "score",
            "compare", "how does this relate",
        ]):
            return "evaluation"
        return "generation"

    def _estimate_confidence(self, messages: list[dict[str, str]], task_type: str) -> float:
        """Estimate confidence that local models can handle this."""
        if task_type in ("extraction", "retrieval"):
            return 0.95  # always handle locally
        if task_type == "evaluation":
            # Higher confidence if we have knowledge graph data to compare against
            if self._kg is not None and self._kg.count() > 0:
                return 0.85
            return 0.5  # ambiguous without knowledge base
        # generation
        content_len = sum(len(m.get("content", "")) for m in messages)
        if content_len < 500:
            return 0.7  # short generation, local might handle
        return 0.3  # complex generation, need API

    def _handle_locally(
        self, messages: list[dict[str, str]], task_type: str
    ) -> dict[str, Any] | None:
        """Handle task with local models. Returns result dict or None if unable."""
        last_content = messages[-1].get("content", "")

        if task_type == "extraction":
            return self._handle_extraction(last_content)
        if task_type == "retrieval":
            return self._handle_retrieval(last_content)
        if task_type == "evaluation":
            return self._handle_evaluation(last_content)
        return None  # generation tasks go to API

    def _handle_extraction(self, content: str) -> dict[str, Any] | None:
        """Extract key information using local summarizer."""
        if self._summarizer is None:
            return None

        # Use extractive summarization to pull key sentences
        summary = self._summarizer.summarize(content, max_length=200)
        if not summary:
            return None

        log.info("Confidence gate: handled extraction locally")
        return {
            "content": summary,
            "model": "local",
            "input_tokens": len(content) // 4,
            "output_tokens": len(summary) // 4,
            "_local_handled": True,
            "_task_type": "extraction",
        }

    def _handle_retrieval(self, query: str) -> dict[str, Any] | None:
        """Retrieve relevant facts from knowledge graph."""
        if self._kg is None or self._kg.count() == 0:
            return None

        results = self._kg.query(query, top_k=10)
        if not results:
            return None

        # Filter by distance threshold
        relevant = [r for r in results if r["distance"] < 0.7]
        if not relevant:
            return None

        # Rerank if available
        facts = [r["fact"] for r in relevant]
        if self._reranker is not None and len(facts) > 1:
            facts = self._reranker.rank(facts, query=query, top_k=5)

        response = "\n".join(f"- {f}" for f in facts)
        log.info(f"Confidence gate: retrieved {len(facts)} facts locally")
        return {
            "content": response,
            "model": "local",
            "input_tokens": len(query) // 4,
            "output_tokens": len(response) // 4,
            "_local_handled": True,
            "_task_type": "retrieval",
        }

    def _handle_evaluation(self, content: str) -> dict[str, Any] | None:
        """Evaluate against known knowledge graph facts."""
        if self._kg is None or self._kg.count() == 0:
            return None

        # Find most similar existing facts
        results = self._kg.query(content, top_k=5)
        if not results:
            return None

        # Check if any existing fact is very close (not novel)
        closest = results[0]
        similar_facts = [r for r in results if r["distance"] < 0.3]

        if similar_facts:
            matches = "\n".join(f"- {r['fact']} (distance: {r['distance']:.3f})" for r in similar_facts)
            response = f"Found {len(similar_facts)} closely matching existing facts:\n{matches}"
        else:
            response = (
                f"No close matches found. Nearest fact is at distance {closest['distance']:.3f}:\n"
                f"- {closest['fact']}"
            )

        log.info(f"Confidence gate: evaluated locally against {len(results)} facts")
        return {
            "content": response,
            "model": "local",
            "input_tokens": len(content) // 4,
            "output_tokens": len(response) // 4,
            "_local_handled": True,
            "_task_type": "evaluation",
        }
