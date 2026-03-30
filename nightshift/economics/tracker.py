"""Token economics tracker. Know exactly where every dollar goes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from nightshift.utils import count_tokens


# Approximate pricing per 1M tokens (input) as of March 2026
MODEL_PRICING: dict[str, float] = {
    "gpt-5.4-mini": 0.75,
    "gpt-5.4": 2.00,
    "gpt-4.1-mini": 0.40,
    "gpt-4.1-nano": 0.10,
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "claude-sonnet-4-20250514": 3.0,
    "claude-opus-4-20250514": 15.0,
    "claude-haiku-4-5-20251001": 0.80,
    "gemini-2.0-flash": 0.0,
    "gemini-2.5-pro": 1.25,
    "deepseek-chat": 0.27,
    "local": 0.0,
}


@dataclass
class CallRecord:
    call_id: int
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    task_type: str
    was_local: bool
    # Post-hoc value metrics (populated later)
    new_facts: int = 0
    confidence_delta: float = 0.0

    @property
    def cost_per_insight(self) -> float:
        if self.new_facts == 0:
            return float("inf")
        return self.cost_usd / self.new_facts


class TokenTracker:
    """Tracks every API call's cost and value. Enforces budget."""

    def __init__(self, budget: float = 10.0) -> None:
        self.budget = budget
        self.spent = 0.0
        self.records: list[CallRecord] = []
        self._call_counter = 0
        self._local_saves = 0
        self._local_tokens_saved = 0

    def can_afford(self, messages: list[dict[str, str]], model: str) -> bool:
        """Check if we can afford this API call within budget."""
        estimated_tokens = count_tokens(messages)
        estimated_cost = self._estimate_cost(estimated_tokens, model)
        return (self.spent + estimated_cost) <= self.budget

    def record_api(
        self,
        messages: list[dict[str, str]],
        response: dict[str, Any],
        model: str,
    ) -> CallRecord:
        """Record an API call with its costs."""
        self._call_counter += 1
        input_tokens = count_tokens(messages)
        output_tokens = len(str(response)) // 4
        cost = self._estimate_cost(input_tokens, model)
        self.spent += cost

        record = CallRecord(
            call_id=self._call_counter,
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_type="api",
            was_local=False,
        )
        self.records.append(record)
        return record

    def record_local(self, messages: list[dict[str, str]]) -> None:
        """Record a call that was handled locally (API skipped)."""
        self._local_saves += 1
        tokens_saved = count_tokens(messages)
        self._local_tokens_saved += tokens_saved

    def report(self) -> dict[str, Any]:
        """Generate token economics report."""
        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        return {
            "budget_usd": self.budget,
            "spent_usd": round(self.spent, 4),
            "remaining_usd": round(self.budget - self.spent, 4),
            "api_calls": len(self.records),
            "local_handles": self._local_saves,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "tokens_saved_by_local": self._local_tokens_saved,
            "avg_cost_per_call": round(self.spent / max(len(self.records), 1), 4),
            "records": self.records,
        }

    def _estimate_cost(self, input_tokens: int, model: str) -> float:
        price_per_m = MODEL_PRICING.get(model, 3.0)  # default to $3/M
        return (input_tokens / 1_000_000) * price_per_m
