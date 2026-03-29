"""Base agent class for NightShift-powered agents."""

from __future__ import annotations

from typing import Any

from nightshift.engine import NightShift


class BaseAgent:
    """Base class for agents that run on the NightShift runtime.

    Subclass this to build agents (research, PDF analysis, CTO office, etc.)
    that automatically get token optimization.
    """

    def __init__(self, engine: NightShift | None = None, **engine_kwargs: Any) -> None:
        self.engine = engine or NightShift(**engine_kwargs)

    def run(self, task: str) -> dict[str, Any]:
        """Execute the agent's main loop."""
        raise NotImplementedError

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Make an LLM call through the NightShift engine."""
        return self.engine.complete(messages, **kwargs)

    def report(self) -> dict[str, Any]:
        """Return token economics report."""
        return self.engine.report()
