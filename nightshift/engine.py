"""Core NightShift engine. Intercepts LLM calls, optimizes token spend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nightshift.compression.pipeline import CompressionPipeline
from nightshift.routing.confidence import ConfidenceGate
from nightshift.routing.dedup import ContentDedup
from nightshift.history.window import SlidingWindowManager
from nightshift.economics.tracker import TokenTracker
from nightshift.economics.bandit import BudgetBandit


@dataclass
class NightShiftConfig:
    api_budget: float = 10.0  # USD hard cap
    window_size: int = 5  # active history turns
    compress_threshold: int = 2000  # tokens above which compression kicks in
    confidence_threshold: float = 0.8  # gate threshold for local handling
    knowledge_db: str = "./nightshift_kb"  # persistent knowledge path
    local_models: str = "auto"  # "auto" downloads on first run, or path to models
    duration: str | None = None  # "overnight", "48h", or None for single run


class NightShift:
    """The agent runtime that makes autonomous AI research 10x cheaper.

    Sits between your agent and the LLM API. Intercepts calls, compresses
    input, deduplicates content, gates by confidence, manages history,
    and allocates budget optimally.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = NightShiftConfig(**kwargs)
        self.compression = CompressionPipeline(self.config)
        self.gate = ConfidenceGate(self.config)
        self.dedup = ContentDedup()
        self.history = SlidingWindowManager(self.config)
        self.tracker = TokenTracker(budget=self.config.api_budget)
        self.bandit = BudgetBandit()

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-sonnet-4-20250514",
        compress: bool = True,
        gate: bool = True,
        **api_kwargs: Any,
    ) -> dict[str, Any]:
        """Drop-in replacement for LLM API calls with automatic optimization.

        1. Compress long content locally (free)
        2. Deduplicate previously-sent content
        3. Manage conversation history (sliding window)
        4. Gate: handle locally if confident enough
        5. Schedule: check budget, allocate optimally
        6. Dispatch to API with minimal tokens
        7. Track economics
        """
        # Step 1: Compress
        if compress:
            messages = self.compression.process(messages)

        # Step 2: Deduplicate
        messages = self.dedup.process(messages)

        # Step 3: Manage history
        messages = self.history.apply(messages)

        # Step 4: Confidence gate
        if gate:
            local_result = self.gate.try_local(messages)
            if local_result is not None:
                self.tracker.record_local(messages)
                return local_result

        # Step 5: Budget check
        if not self.tracker.can_afford(messages, model):
            return self._best_effort_local(messages)

        # Step 6: Dispatch to API
        response = self._dispatch(messages, model, **api_kwargs)

        # Step 7: Track
        self.tracker.record_api(messages, response, model)
        self.bandit.update(response)
        self.history.ingest_response(response)

        return response

    def wrap(self, agent_fn: Any, budget: str = "$10", duration: str = "overnight") -> Any:
        """Wrap an entire agent function. All LLM calls inside are optimized."""
        # TODO: implement agent wrapping with monkey-patched LLM clients
        raise NotImplementedError("Agent wrapping coming in v0.2")

    def report(self) -> dict[str, Any]:
        """Return token economics report for current session."""
        return self.tracker.report()

    def _dispatch(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Send optimized messages to the actual LLM API."""
        # TODO: implement multi-provider dispatch (OpenAI, Anthropic, local)
        raise NotImplementedError("API dispatch coming in Phase 1")

    def _best_effort_local(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Budget exhausted. Return best local result."""
        # TODO: synthesize from knowledge graph
        raise NotImplementedError("Best-effort local coming in Phase 2")
