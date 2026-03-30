"""Core NightShift engine. Intercepts LLM calls, optimizes token spend."""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from nightshift.compression.pipeline import CompressionPipeline
from nightshift.dispatch import Dispatcher
from nightshift.routing.confidence import ConfidenceGate
from nightshift.routing.dedup import ContentDedup
from nightshift.history.window import SlidingWindowManager
from nightshift.history.knowledge import KnowledgeGraph
from nightshift.economics.tracker import TokenTracker
from nightshift.economics.bandit import BudgetBandit

log = logging.getLogger(__name__)


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
        self.dispatcher = Dispatcher()
        self.kg = KnowledgeGraph(path=self.config.knowledge_db)
        self._history_model_wired = False

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

        # Wire local models into components on first use
        if not self._history_model_wired:
            summarizer = self.compression._ensure_summarizer()
            reranker = self.compression._ensure_reranker()
            self.history.set_summarizer(summarizer)
            self.history.set_knowledge_graph(self.kg)
            self.gate.set_resources(summarizer=summarizer, reranker=reranker, kg=self.kg)
            self._history_model_wired = True

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

    def wrap(self, agent_fn: Callable[..., Any], budget: str = "$10", duration: str = "overnight") -> Callable[..., Any]:
        """Wrap an entire agent function. All LLM calls inside are optimized.

        Monkey-patches openai and anthropic SDK create methods to route through
        NightShift's complete() pipeline. Restores originals after the function returns.

        Usage:
            engine = NightShift(api_budget=5.0)
            result = engine.wrap(my_agent_fn, budget="$5")("research topic")
        """
        # Parse budget
        budget_usd = float(budget.replace("$", "").strip()) if isinstance(budget, str) else float(budget)
        self.tracker.budget = budget_usd

        engine = self

        @functools.wraps(agent_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with engine._patch_sdk_clients():
                return agent_fn(*args, **kwargs)

        return wrapper

    @contextmanager
    def _patch_sdk_clients(self) -> Any:
        """Context manager that patches OpenAI and Anthropic SDK create methods."""
        patches: list[tuple[Any, str, Any]] = []

        # Patch OpenAI
        try:
            import openai.resources.chat.completions as oai_mod
            original_oai = oai_mod.Completions.create
            oai_mod.Completions.create = self._make_openai_interceptor(original_oai)
            patches.append((oai_mod.Completions, "create", original_oai))
            log.info("Patched openai.chat.completions.create")
        except (ImportError, AttributeError):
            pass

        # Patch Anthropic
        try:
            import anthropic.resources.messages as anth_mod
            original_anth = anth_mod.Messages.create
            anth_mod.Messages.create = self._make_anthropic_interceptor(original_anth)
            patches.append((anth_mod.Messages, "create", original_anth))
            log.info("Patched anthropic.messages.create")
        except (ImportError, AttributeError):
            pass

        try:
            yield
        finally:
            # Restore originals
            for cls, attr, original in patches:
                setattr(cls, attr, original)
            log.info(f"Restored {len(patches)} SDK patches")

    def _make_openai_interceptor(self, original: Any) -> Any:
        """Create an OpenAI-compatible interceptor that routes through NightShift."""
        engine = self

        def interceptor(self_oai: Any, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            # Convert to simple message format
            simple_msgs = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ]
            result = engine.complete(simple_msgs, model=model, **kwargs)

            # Build OpenAI-compatible response object
            from types import SimpleNamespace
            choice = SimpleNamespace(
                message=SimpleNamespace(content=result["content"], role="assistant"),
                index=0,
                finish_reason="stop",
            )
            usage = SimpleNamespace(
                prompt_tokens=result.get("input_tokens", 0),
                completion_tokens=result.get("output_tokens", 0),
                total_tokens=result.get("input_tokens", 0) + result.get("output_tokens", 0),
            )
            return SimpleNamespace(
                id="nightshift-wrapped",
                choices=[choice],
                model=result.get("model", model),
                usage=usage,
            )

        return interceptor

    def _make_anthropic_interceptor(self, original: Any) -> Any:
        """Create an Anthropic-compatible interceptor that routes through NightShift."""
        engine = self

        def interceptor(self_anth: Any, *, model: str, messages: list[dict[str, Any]], max_tokens: int = 4096, **kwargs: Any) -> Any:
            # Handle Anthropic system parameter
            system = kwargs.pop("system", None)
            simple_msgs: list[dict[str, str]] = []
            if system:
                simple_msgs.append({"role": "system", "content": system})
            for m in messages:
                simple_msgs.append(
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                )
            result = engine.complete(simple_msgs, model=model, max_tokens=max_tokens, **kwargs)

            # Build Anthropic-compatible response object
            from types import SimpleNamespace
            content_block = SimpleNamespace(type="text", text=result["content"])
            usage = SimpleNamespace(
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
            )
            return SimpleNamespace(
                id="nightshift-wrapped",
                content=[content_block],
                model=result.get("model", model),
                usage=usage,
                stop_reason="end_turn",
                type="message",
                role="assistant",
            )

        return interceptor

    def report(self) -> dict[str, Any]:
        """Return token economics report for current session."""
        return self.tracker.report()

    def _dispatch(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Send optimized messages to the actual LLM API."""
        result = self.dispatcher.dispatch_sync(messages, model, **kwargs)
        return {
            "content": result.content,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

    def _best_effort_local(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Budget exhausted. Synthesize best answer from knowledge graph."""
        query = messages[-1].get("content", "") if messages else ""
        results = self.kg.query(query, top_k=10)

        if not results:
            log.warning("Budget exhausted and no knowledge graph entries available")
            return {
                "content": "[Budget exhausted. No cached knowledge available for this query.]",
                "model": "local",
                "input_tokens": 0,
                "output_tokens": 0,
                "_budget_exhausted": True,
            }

        # Summarize top facts into a coherent response
        facts = [r["fact"] for r in results if r["distance"] < 0.8]
        if not facts:
            facts = [results[0]["fact"]]

        summarizer = self.compression._ensure_summarizer()
        combined = "\n".join(f"- {f}" for f in facts)

        # Use T5-small to compress facts into a summary if available
        summary = summarizer.summarize(combined, max_length=150)

        log.info(f"Budget exhausted. Returning local synthesis from {len(facts)} facts")
        self.tracker.record_local(messages)
        return {
            "content": f"[Local synthesis from {len(facts)} cached facts]\n{summary}",
            "model": "local",
            "input_tokens": 0,
            "output_tokens": len(summary) // 4,
            "_budget_exhausted": True,
        }
