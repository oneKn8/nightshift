"""Tests for engine.wrap() SDK monkey-patching."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nightshift.engine import NightShift


def _mock_dispatch(messages, model, **kwargs):
    return {
        "content": "Wrapped response",
        "model": model,
        "input_tokens": 10,
        "output_tokens": 5,
    }


class TestEngineWrap:
    def test_wrap_returns_callable(self):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch

        def my_agent(topic):
            return f"researched: {topic}"

        wrapped = engine.wrap(my_agent, budget="$5")
        assert callable(wrapped)

    def test_wrap_sets_budget(self):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch

        def my_agent():
            return "done"

        engine.wrap(my_agent, budget="$3.50")
        assert engine.tracker.budget == 3.50

    def test_wrap_preserves_function_name(self):
        engine = NightShift(api_budget=10.0)

        def my_special_agent():
            pass

        wrapped = engine.wrap(my_special_agent)
        assert wrapped.__name__ == "my_special_agent"

    def test_wrap_executes_function(self):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch

        def my_agent(x, y):
            return x + y

        wrapped = engine.wrap(my_agent, budget="$5")
        result = wrapped(3, 4)
        assert result == 7

    def test_wrap_passes_kwargs(self):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch

        def my_agent(topic, depth="shallow"):
            return f"{topic}:{depth}"

        wrapped = engine.wrap(my_agent, budget="$5")
        result = wrapped("AI", depth="deep")
        assert result == "AI:deep"

    def test_openai_interceptor_format(self):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch

        interceptor = engine._make_openai_interceptor(None)
        result = interceptor(
            None,
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert result.choices[0].message.content == "Wrapped response"
        assert result.choices[0].message.role == "assistant"
        assert result.model == "gpt-5.4-mini"
        assert result.usage.prompt_tokens >= 0

    def test_anthropic_interceptor_format(self):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch

        interceptor = engine._make_anthropic_interceptor(None)
        result = interceptor(
            None,
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
        )

        assert result.content[0].text == "Wrapped response"
        assert result.content[0].type == "text"
        assert result.role == "assistant"
        assert result.stop_reason == "end_turn"

    def test_anthropic_interceptor_handles_system(self):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch

        interceptor = engine._make_anthropic_interceptor(None)
        result = interceptor(
            None,
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            system="You are a helpful assistant.",
        )

        assert result.content[0].text == "Wrapped response"

    def test_wrap_budget_parsing(self):
        engine = NightShift(api_budget=100.0)
        engine.wrap(lambda: None, budget="$0.50")
        assert engine.tracker.budget == 0.50

        engine.wrap(lambda: None, budget="25")
        assert engine.tracker.budget == 25.0
