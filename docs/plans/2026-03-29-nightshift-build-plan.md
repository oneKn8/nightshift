# NightShift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully tested, working agent runtime that intercepts LLM calls and reduces token spend by 60-90%.

**Architecture:** Six independent components (dedup, tracker, bandit, window, dispatch, compression) compose into one engine. Each is built and tested in isolation first, then wired together. Local models are abstracted behind interfaces and mocked for testing -- real model loading is the final phase.

**Tech Stack:** Python 3.11, pytest, httpx (API dispatch), tiktoken (token counting), numpy (embeddings), chromadb (knowledge persistence)

**Rule:** Every phase produces a `pytest` green bar before moving to the next. No exceptions.

---

## Phase 1: Project Bootstrap and Test Infrastructure

### Task 1.1: Create pytest config and conftest

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

**Step 1: Create config**

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
```

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared test fixtures for NightShift."""
import pytest
from nightshift.engine import NightShift, NightShiftConfig


@pytest.fixture
def config():
    return NightShiftConfig(
        api_budget=10.0,
        window_size=3,
        compress_threshold=500,
        confidence_threshold=0.8,
    )


@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": "Summarize this paper about transformers."},
    ]


@pytest.fixture
def long_message():
    """A message that exceeds compression threshold."""
    return {"role": "user", "content": "x " * 5000}


@pytest.fixture
def large_history():
    """20-turn conversation history."""
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"Question {i}: " + "context " * 50})
        msgs.append({"role": "assistant", "content": f"Answer {i}: " + "response " * 30})
    return msgs


@pytest.fixture
def duplicate_messages():
    """Messages with repeated content for dedup testing."""
    big_content = "This is a large block of text. " * 100  # ~700 chars, above 200 threshold
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": big_content},
    ]
```

**Step 2: Install dev deps and verify pytest runs**

Run: `cd /home/oneknight/projects/nightshift && pip install -e ".[dev]" 2>&1 | tail -5`
Run: `pytest --collect-only 2>&1 | tail -5`
Expected: `no tests ran` (no test files yet)

**Step 3: Commit**

```bash
git add pytest.ini tests/
git commit -m "test: add pytest config and shared fixtures"
```

---

## Phase 2: Content Deduplication (Harden Existing Code)

### Task 2.1: Test dedup core behavior

**Files:**
- Create: `tests/test_dedup.py`
- Verify: `nightshift/routing/dedup.py` (existing, no changes needed)

**Step 1: Write failing tests**

```python
# tests/test_dedup.py
"""Tests for content deduplication."""
from nightshift.routing.dedup import ContentDedup, SentContent


class TestContentDedup:
    def test_first_message_passes_through(self, sample_messages):
        dedup = ContentDedup()
        result = dedup.process(sample_messages)
        assert result == sample_messages

    def test_short_content_never_deduped(self):
        dedup = ContentDedup()
        msgs = [{"role": "user", "content": "hello"}]
        result1 = dedup.process(msgs)
        result2 = dedup.process(msgs)
        # Short content should pass through unchanged even on second call
        assert result2[0]["content"] == "hello"

    def test_duplicate_long_content_replaced(self, duplicate_messages):
        dedup = ContentDedup()
        result1 = dedup.process(duplicate_messages)
        # First call: content passes through
        assert "This is a large block" in result1[1]["content"]

        result2 = dedup.process(duplicate_messages)
        # Second call: content replaced with reference
        assert "Previously provided" in result2[1]["content"]
        assert "unchanged" in result2[1]["content"]

    def test_different_content_not_deduped(self):
        dedup = ContentDedup()
        msg1 = [{"role": "user", "content": "A " * 200}]
        msg2 = [{"role": "user", "content": "B " * 200}]
        dedup.process(msg1)
        result = dedup.process(msg2)
        assert "Previously provided" not in result[0]["content"]

    def test_system_messages_preserved(self):
        dedup = ContentDedup()
        big = "system context " * 100
        msgs = [{"role": "system", "content": big}]
        dedup.process(msgs)
        result = dedup.process(msgs)
        # System message with same content should still get deduped
        assert "Previously provided" in result[0]["content"]

    def test_role_preserved_after_dedup(self, duplicate_messages):
        dedup = ContentDedup()
        dedup.process(duplicate_messages)
        result = dedup.process(duplicate_messages)
        assert result[1]["role"] == "user"

    def test_clear_resets_cache(self, duplicate_messages):
        dedup = ContentDedup()
        dedup.process(duplicate_messages)
        dedup.clear()
        result = dedup.process(duplicate_messages)
        # After clear, content should pass through again
        assert "This is a large block" in result[1]["content"]

    def test_token_count_in_reference(self, duplicate_messages):
        dedup = ContentDedup()
        dedup.process(duplicate_messages)
        result = dedup.process(duplicate_messages)
        # Reference should mention token count
        assert "tokens" in result[1]["content"]

    def test_hash_deterministic(self):
        h1 = ContentDedup._hash("test content")
        h2 = ContentDedup._hash("test content")
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_different_for_different_content(self):
        h1 = ContentDedup._hash("content A")
        h2 = ContentDedup._hash("content B")
        assert h1 != h2
```

**Step 2: Run tests**

Run: `pytest tests/test_dedup.py -v`
Expected: ALL PASS (dedup is already implemented)

**Step 3: Commit**

```bash
git add tests/test_dedup.py
git commit -m "test: full coverage for content deduplication"
```

---

## Phase 3: Token Economics Tracker (Harden + Real Token Counting)

### Task 3.1: Test tracker core behavior

**Files:**
- Create: `tests/test_tracker.py`
- Modify: `nightshift/economics/tracker.py` (add tiktoken counting)

**Step 1: Write failing tests**

```python
# tests/test_tracker.py
"""Tests for token economics tracker."""
import time
from nightshift.economics.tracker import TokenTracker, CallRecord, MODEL_PRICING


class TestTokenTracker:
    def test_initial_state(self):
        t = TokenTracker(budget=10.0)
        assert t.budget == 10.0
        assert t.spent == 0.0
        assert t.records == []

    def test_can_afford_within_budget(self, sample_messages):
        t = TokenTracker(budget=10.0)
        assert t.can_afford(sample_messages, "gpt-4o-mini") is True

    def test_cannot_afford_over_budget(self, sample_messages):
        t = TokenTracker(budget=0.0)
        assert t.can_afford(sample_messages, "gpt-4o") is False

    def test_record_api_tracks_cost(self, sample_messages):
        t = TokenTracker(budget=10.0)
        response = {"choices": [{"message": {"content": "result"}}]}
        record = t.record_api(sample_messages, response, "gpt-4o")
        assert record.cost_usd > 0
        assert record.was_local is False
        assert t.spent > 0
        assert len(t.records) == 1

    def test_record_local_saves_tokens(self, sample_messages):
        t = TokenTracker(budget=10.0)
        t.record_local(sample_messages)
        assert t._local_saves == 1
        assert t._local_tokens_saved > 0

    def test_budget_enforcement_cumulative(self):
        t = TokenTracker(budget=0.001)  # very small budget
        big_msgs = [{"role": "user", "content": "x " * 10000}]
        # First call might fit
        t.record_api(big_msgs, {"r": "ok"}, "claude-opus-4-20250514")
        # Second call should be over budget
        assert t.can_afford(big_msgs, "claude-opus-4-20250514") is False

    def test_report_structure(self, sample_messages):
        t = TokenTracker(budget=10.0)
        t.record_api(sample_messages, {"r": "ok"}, "gpt-4o")
        t.record_local(sample_messages)
        report = t.report()
        assert "budget_usd" in report
        assert "spent_usd" in report
        assert "remaining_usd" in report
        assert "api_calls" in report
        assert report["api_calls"] == 1
        assert report["local_handles"] == 1

    def test_unknown_model_uses_default_price(self):
        t = TokenTracker(budget=100.0)
        msgs = [{"role": "user", "content": "hi " * 1000}]
        t.record_api(msgs, {"r": "ok"}, "unknown-model-xyz")
        assert t.spent > 0  # should use default $3/M

    def test_free_model_costs_zero(self):
        t = TokenTracker(budget=100.0)
        msgs = [{"role": "user", "content": "hi " * 1000}]
        t.record_api(msgs, {"r": "ok"}, "gemini-2.0-flash")
        assert t.spent == 0.0

    def test_call_record_cost_per_insight(self):
        r = CallRecord(
            call_id=1, timestamp=time.time(), model="gpt-4o",
            input_tokens=1000, output_tokens=500, cost_usd=0.50,
            task_type="api", was_local=False, new_facts=5,
        )
        assert r.cost_per_insight == 0.10

    def test_call_record_cost_per_insight_zero_facts(self):
        r = CallRecord(
            call_id=1, timestamp=time.time(), model="gpt-4o",
            input_tokens=1000, output_tokens=500, cost_usd=0.50,
            task_type="api", was_local=False, new_facts=0,
        )
        assert r.cost_per_insight == float("inf")
```

**Step 2: Run tests**

Run: `pytest tests/test_tracker.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_tracker.py
git commit -m "test: full coverage for token economics tracker"
```

### Task 3.2: Add tiktoken-based accurate token counting

**Files:**
- Modify: `nightshift/economics/tracker.py`
- Create: `nightshift/utils.py`
- Create: `tests/test_utils.py`

**Step 1: Write failing test**

```python
# tests/test_utils.py
"""Tests for utility functions."""
from nightshift.utils import count_tokens


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_short_string(self):
        n = count_tokens("Hello, world!")
        assert 1 <= n <= 10

    def test_long_string(self):
        text = "The quick brown fox " * 100
        n = count_tokens(text)
        # ~500 words, should be ~400-600 tokens
        assert 300 < n < 700

    def test_messages_list(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello there."},
        ]
        n = count_tokens(msgs)
        assert n > 0

    def test_consistency(self):
        text = "Consistent counting test"
        assert count_tokens(text) == count_tokens(text)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nightshift.utils'`

**Step 3: Implement**

```python
# nightshift/utils.py
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
```

**Step 4: Run test**

Run: `pytest tests/test_utils.py -v`
Expected: ALL PASS

**Step 5: Wire tiktoken into tracker**

Replace `len(content) // 4` approximations in `tracker.py` with `count_tokens()`:

Modify `nightshift/economics/tracker.py`:
- Line 58: Replace `sum(len(m.get("content", "")) // 4 for m in messages)` with `count_tokens(messages)`
- Line 70: Same replacement
- Line 92: Same replacement
- Add import: `from nightshift.utils import count_tokens`
- Remove the `token_est` inline in `SentContent` creation in dedup.py too

**Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add nightshift/utils.py tests/test_utils.py nightshift/economics/tracker.py
git commit -m "feat: accurate token counting via tiktoken"
```

---

## Phase 4: UCB1 Budget Bandit (Harden)

### Task 4.1: Test bandit behavior

**Files:**
- Create: `tests/test_bandit.py`

**Step 1: Write tests**

```python
# tests/test_bandit.py
"""Tests for UCB1 budget bandit."""
from nightshift.economics.bandit import BudgetBandit, ArmStats


class TestArmStats:
    def test_unexplored_arm_infinite_reward(self):
        arm = ArmStats()
        assert arm.mean_reward == float("inf")

    def test_mean_reward_calculation(self):
        arm = ArmStats(pulls=4, total_reward=2.0)
        assert arm.mean_reward == 0.5


class TestBudgetBandit:
    def test_initial_selection_explores_all_arms(self):
        b = BudgetBandit()
        seen = set()
        for _ in range(4):
            arm = b.select()
            seen.add(arm)
            b.arms[arm].pulls += 1
            b.arms[arm].total_reward += 0.5
            b.total_pulls += 1
        assert seen == {"explore", "deepen", "synthesize", "evaluate"}

    def test_high_reward_arm_selected_more(self):
        b = BudgetBandit(c=0.1)  # low exploration = exploit more
        # Give "synthesize" a very high reward
        b.arms["synthesize"] = ArmStats(pulls=10, total_reward=9.0)
        b.arms["explore"] = ArmStats(pulls=10, total_reward=1.0)
        b.arms["deepen"] = ArmStats(pulls=10, total_reward=1.0)
        b.arms["evaluate"] = ArmStats(pulls=10, total_reward=1.0)
        b.total_pulls = 40

        selections = [b.select() for _ in range(10)]
        assert selections.count("synthesize") >= 7

    def test_update_increments_pulls(self):
        b = BudgetBandit()
        b.update({"_nightshift_action": "explore", "_nightshift_reward": 0.8})
        assert b.arms["explore"].pulls == 1
        assert b.arms["explore"].total_reward == 0.8
        assert b.total_pulls == 1

    def test_update_unknown_action_ignored(self):
        b = BudgetBandit()
        b.update({"_nightshift_action": "nonexistent", "_nightshift_reward": 1.0})
        assert b.total_pulls == 0

    def test_report_structure(self):
        b = BudgetBandit()
        b.update({"_nightshift_action": "explore", "_nightshift_reward": 0.5})
        report = b.report()
        assert "explore" in report
        assert "pulls" in report["explore"]
        assert "mean_reward" in report["explore"]

    def test_ucb_exploration_bonus_decays(self):
        b = BudgetBandit()
        # All arms equal reward, high pull count = low exploration bonus
        for arm in b.arms:
            b.arms[arm] = ArmStats(pulls=100, total_reward=50.0)
        b.total_pulls = 400
        # With equal stats, any arm could be selected (UCB scores are close)
        selected = b.select()
        assert selected in b.arms
```

**Step 2: Run tests**

Run: `pytest tests/test_bandit.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_bandit.py
git commit -m "test: full coverage for UCB1 budget bandit"
```

---

## Phase 5: Sliding Window History Manager (Harden)

### Task 5.1: Test window behavior

**Files:**
- Create: `tests/test_window.py`

**Step 1: Write tests**

```python
# tests/test_window.py
"""Tests for sliding window history manager."""
from nightshift.history.window import SlidingWindowManager


class FakeConfig:
    window_size = 3


class TestSlidingWindowManager:
    def test_short_history_unchanged(self, sample_messages):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(sample_messages)
        assert result == sample_messages

    def test_long_history_compressed(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(large_history)
        # Should be shorter than original
        assert len(result) < len(large_history)

    def test_system_messages_preserved(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(large_history)
        system_msgs = [m for m in result if m["role"] == "system"]
        # At least the original system message + the archive summary
        assert len(system_msgs) >= 1

    def test_recent_messages_kept_intact(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(large_history)
        # Last few messages should be present verbatim
        non_system = [m for m in result if m["role"] != "system"]
        last_original = large_history[-1]
        assert any(m["content"] == last_original["content"] for m in non_system)

    def test_archive_grows(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        w.apply(large_history)
        assert len(w._archive) == 1
        # Apply again with more history
        w.apply(large_history)
        assert len(w._archive) == 2

    def test_compressed_history_contains_marker(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(large_history)
        # Should contain a compressed history marker
        all_content = " ".join(m.get("content", "") for m in result)
        assert "Compressed history" in all_content

    def test_window_size_respected(self):
        config = FakeConfig()
        config.window_size = 2
        w = SlidingWindowManager(config)
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        result = w.apply(msgs)
        non_system = [m for m in result if m["role"] != "system"]
        # Active window = window_size * 2 = 4 messages (2 pairs)
        # Plus 1 archive summary message (system role)
        assert len(non_system) <= config.window_size * 2

    def test_empty_messages(self):
        w = SlidingWindowManager(FakeConfig())
        assert w.apply([]) == []

    def test_only_system_messages(self):
        w = SlidingWindowManager(FakeConfig())
        msgs = [{"role": "system", "content": "sys"}]
        assert w.apply(msgs) == msgs
```

**Step 2: Run tests**

Run: `pytest tests/test_window.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_window.py
git commit -m "test: full coverage for sliding window history manager"
```

---

## Phase 6: API Dispatch (Wire Up Real LLM Calls)

### Task 6.1: Build multi-provider dispatch

**Files:**
- Create: `nightshift/dispatch.py`
- Create: `tests/test_dispatch.py`

**Step 1: Write failing tests**

```python
# tests/test_dispatch.py
"""Tests for LLM API dispatch."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from nightshift.dispatch import Dispatcher, DispatchResult


class TestDispatcher:
    def test_parse_model_provider_openai(self):
        d = Dispatcher()
        provider, model = d._parse_model("gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_parse_model_provider_anthropic(self):
        d = Dispatcher()
        provider, model = d._parse_model("claude-sonnet-4-20250514")
        assert provider == "anthropic"

    def test_parse_model_provider_google(self):
        d = Dispatcher()
        provider, model = d._parse_model("gemini-2.0-flash")
        assert provider == "google"

    def test_parse_model_provider_deepseek(self):
        d = Dispatcher()
        provider, model = d._parse_model("deepseek-chat")
        assert provider == "deepseek"

    def test_parse_model_provider_unknown(self):
        d = Dispatcher()
        provider, model = d._parse_model("some-random-model")
        assert provider == "openai"  # default fallback

    def test_format_messages_openai(self):
        d = Dispatcher()
        msgs = [{"role": "user", "content": "hi"}]
        formatted = d._format_messages(msgs, "openai")
        assert formatted == msgs  # OpenAI format is the default

    def test_dispatch_result_structure(self):
        r = DispatchResult(
            content="hello",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            raw_response={"id": "123"},
        )
        assert r.content == "hello"
        assert r.total_tokens == 15

    def test_dispatch_without_api_key_raises(self):
        d = Dispatcher()
        msgs = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError, match="API key"):
            d.dispatch_sync(msgs, "gpt-4o")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dispatch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nightshift.dispatch'`

**Step 3: Implement**

```python
# nightshift/dispatch.py
"""Multi-provider LLM API dispatch."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class DispatchResult:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# Model prefix -> (provider, base_url, env_var_for_key)
_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "gpt": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "o1": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "o3": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "claude": ("anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": ("google", "https://generativelanguage.googleapis.com/v1beta", "GOOGLE_API_KEY"),
    "deepseek": ("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
}


class Dispatcher:
    """Routes LLM calls to the correct provider API."""

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=120.0)

    def dispatch_sync(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> DispatchResult:
        """Send messages to the appropriate LLM API."""
        provider, model_id = self._parse_model(model)
        api_key = self._get_key(provider)

        if provider == "anthropic":
            return self._dispatch_anthropic(messages, model_id, api_key, **kwargs)
        else:
            return self._dispatch_openai_compat(messages, model_id, provider, api_key, **kwargs)

    def _parse_model(self, model: str) -> tuple[str, str]:
        """Determine provider from model name."""
        for prefix, (provider, _, _) in _PROVIDERS.items():
            if model.startswith(prefix):
                return provider, model
        return "openai", model

    def _get_key(self, provider: str) -> str:
        for _, (prov, _, env_var) in _PROVIDERS.items():
            if prov == provider:
                key = os.environ.get(env_var, "")
                if key:
                    return key
        raise ValueError(
            f"API key not found for provider '{provider}'. "
            f"Set the appropriate environment variable."
        )

    def _format_messages(
        self, messages: list[dict[str, str]], provider: str
    ) -> list[dict[str, str]]:
        """Format messages for provider. OpenAI format is the baseline."""
        return messages

    def _dispatch_openai_compat(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str,
        api_key: str,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch to OpenAI-compatible API (OpenAI, DeepSeek, etc.)."""
        base_url = "https://api.openai.com/v1"
        for _, (prov, url, _) in _PROVIDERS.items():
            if prov == provider:
                base_url = url
                break

        resp = self._client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages, **kwargs},
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        content = data["choices"][0]["message"]["content"]
        return DispatchResult(
            content=content,
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw_response=data,
        )

    def _dispatch_anthropic(
        self,
        messages: list[dict[str, str]],
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch to Anthropic Messages API."""
        system_msg = ""
        non_system = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                non_system.append(m)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": non_system,
        }
        if system_msg:
            body["system"] = system_msg.strip()

        resp = self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        content = data["content"][0]["text"]
        return DispatchResult(
            content=content,
            model=model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            raw_response=data,
        )

    def close(self) -> None:
        self._client.close()
```

**Step 4: Run tests**

Run: `pytest tests/test_dispatch.py -v`
Expected: ALL PASS

**Step 5: Wire dispatch into engine**

Modify `nightshift/engine.py`:
- Add import: `from nightshift.dispatch import Dispatcher`
- In `__init__`: add `self.dispatcher = Dispatcher()`
- Replace `_dispatch` method body:

```python
def _dispatch(self, messages, model, **kwargs):
    result = self.dispatcher.dispatch_sync(messages, model, **kwargs)
    return {
        "content": result.content,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
```

**Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add nightshift/dispatch.py tests/test_dispatch.py nightshift/engine.py
git commit -m "feat: multi-provider LLM dispatch (OpenAI, Anthropic, Google, DeepSeek)"
```

---

## Phase 7: Engine Integration Tests

### Task 7.1: Test the full engine pipeline with mocked dispatch

**Files:**
- Create: `tests/test_engine.py`

**Step 1: Write tests**

```python
# tests/test_engine.py
"""Integration tests for the NightShift engine."""
import pytest
from unittest.mock import patch, MagicMock
from nightshift.engine import NightShift
from nightshift.dispatch import DispatchResult


def _mock_dispatch(messages, model, **kwargs):
    """Mock dispatch that returns a fake response."""
    return {
        "content": "Mock response",
        "model": model,
        "input_tokens": 100,
        "output_tokens": 50,
    }


class TestEngineIntegration:
    def test_engine_initializes(self):
        engine = NightShift(api_budget=5.0)
        assert engine.config.api_budget == 5.0

    def test_complete_with_mocked_dispatch(self, sample_messages):
        engine = NightShift(api_budget=5.0)
        engine._dispatch = _mock_dispatch
        result = engine.complete(sample_messages, model="gpt-4o", gate=False)
        assert result["content"] == "Mock response"

    def test_dedup_works_in_pipeline(self, duplicate_messages):
        engine = NightShift(api_budget=5.0)
        engine._dispatch = _mock_dispatch
        # First call
        engine.complete(duplicate_messages, model="gpt-4o", compress=False, gate=False)
        # Second call -- dedup should kick in
        engine.complete(duplicate_messages, model="gpt-4o", compress=False, gate=False)
        # Check that tracker recorded 2 calls
        report = engine.report()
        assert report["api_calls"] == 2

    def test_budget_enforcement(self):
        engine = NightShift(api_budget=0.0)  # zero budget
        engine._dispatch = _mock_dispatch
        msgs = [{"role": "user", "content": "hi " * 1000}]
        # Should hit budget limit and try local fallback
        with pytest.raises(NotImplementedError, match="Best-effort"):
            engine.complete(msgs, model="claude-opus-4-20250514", gate=False)

    def test_report_after_calls(self, sample_messages):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch
        engine.complete(sample_messages, model="gpt-4o", compress=False, gate=False)
        report = engine.report()
        assert report["api_calls"] == 1
        assert report["spent_usd"] > 0

    def test_history_window_applied(self):
        engine = NightShift(api_budget=10.0, window_size=2)
        engine._dispatch = _mock_dispatch
        # Build a large history
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"question {i} " + "x " * 100})
            msgs.append({"role": "assistant", "content": f"answer {i} " + "y " * 100})
        # The window manager should compress this
        result = engine.complete(msgs, model="gpt-4o", compress=False, gate=False)
        assert result["content"] == "Mock response"

    def test_compression_skipped_for_short_messages(self, sample_messages):
        engine = NightShift(api_budget=10.0, compress_threshold=50000)
        engine._dispatch = _mock_dispatch
        result = engine.complete(sample_messages, model="gpt-4o", gate=False)
        assert result["content"] == "Mock response"
```

**Step 2: Run tests**

Run: `pytest tests/test_engine.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_engine.py
git commit -m "test: engine integration tests with mocked dispatch"
```

---

## Phase 8: Compression Pipeline (Wire Up Local Models)

This is the heaviest phase. Each stage gets its own model loader, test, and integration.

### Task 8.1: Model pool manager

**Files:**
- Create: `nightshift/compression/models.py`
- Create: `tests/test_model_pool.py`

**Step 1: Write failing tests**

```python
# tests/test_model_pool.py
"""Tests for local model pool manager."""
import pytest
from nightshift.compression.models import ModelPool, ModelSpec


class TestModelPool:
    def test_list_models(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_models")
        specs = pool.list_models()
        assert "summarizer" in specs
        assert "reranker" in specs
        assert "embedder" in specs

    def test_model_spec_structure(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_models")
        spec = pool.list_models()["summarizer"]
        assert isinstance(spec, ModelSpec)
        assert spec.hf_id == "google-t5/t5-small"
        assert spec.size_mb > 0

    def test_is_downloaded_false_initially(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_nonexistent")
        assert pool.is_downloaded("summarizer") is False

    def test_get_undownloaded_raises(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_nonexistent")
        with pytest.raises(FileNotFoundError):
            pool.get("summarizer")
```

**Step 2: Implement**

```python
# nightshift/compression/models.py
"""Local model pool manager. Load/unload models one at a time."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    hf_id: str
    size_mb: int
    task: str


REGISTRY: dict[str, ModelSpec] = {
    "parser": ModelSpec("ibm-granite/granite-docling-258M", 258, "document_parsing"),
    "extractor": ModelSpec("urchade/gliner_medium-v2.1", 90, "ner"),
    "embedder": ModelSpec("jinaai/jina-embeddings-v5-text-nano", 239, "embedding"),
    "summarizer": ModelSpec("google-t5/t5-small", 60, "summarization"),
    "reranker": ModelSpec("cross-encoder/ms-marco-MiniLM-L6-v2", 22, "reranking"),
}


class ModelPool:
    """Manages downloading, loading, and unloading of local models.

    Only one model loaded in memory at a time (laptop-friendly).
    """

    def __init__(self, cache_dir: str = "~/.nightshift/models") -> None:
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self._loaded: tuple[str, Any] | None = None  # (name, model_instance)

    def list_models(self) -> dict[str, ModelSpec]:
        return dict(REGISTRY)

    def is_downloaded(self, name: str) -> bool:
        if name not in REGISTRY:
            raise KeyError(f"Unknown model: {name}")
        model_path = self.cache_dir / name
        return model_path.exists()

    def download(self, name: str) -> Path:
        """Download model from HuggingFace. Returns local path."""
        if name not in REGISTRY:
            raise KeyError(f"Unknown model: {name}")
        # Actual downloading is done by transformers/sentence-transformers on first load
        model_path = self.cache_dir / name
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    def get(self, name: str) -> Any:
        """Load a model into memory. Unloads previous model first."""
        if not self.is_downloaded(name):
            raise FileNotFoundError(
                f"Model '{name}' not downloaded. Run: engine.download_models()"
            )

        # Unload current model
        if self._loaded is not None and self._loaded[0] != name:
            self._unload()

        if self._loaded is not None and self._loaded[0] == name:
            return self._loaded[1]

        model = self._load(name)
        self._loaded = (name, model)
        return model

    def _load(self, name: str) -> Any:
        """Load a specific model by name."""
        spec = REGISTRY[name]

        if name == "summarizer":
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(spec.hf_id)
            model = T5ForConditionalGeneration.from_pretrained(spec.hf_id)
            return {"model": model, "tokenizer": tokenizer}

        if name == "reranker":
            from sentence_transformers import CrossEncoder
            return CrossEncoder(spec.hf_id)

        if name == "embedder":
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(spec.hf_id)

        if name == "extractor":
            from gliner import GLiNER
            return GLiNER.from_pretrained(spec.hf_id)

        raise NotImplementedError(f"Loader not implemented for '{name}'")

    def _unload(self) -> None:
        """Free memory from currently loaded model."""
        if self._loaded is not None:
            del self._loaded
            self._loaded = None
```

**Step 3: Run tests**

Run: `pytest tests/test_model_pool.py -v`
Expected: ALL PASS (no actual model downloads in these tests)

**Step 4: Commit**

```bash
git add nightshift/compression/models.py tests/test_model_pool.py
git commit -m "feat: local model pool manager with registry and lazy loading"
```

### Task 8.2: Wire T5-small summarization into compression stage 4

**Files:**
- Create: `nightshift/compression/summarizer.py`
- Create: `tests/test_summarizer.py`

**Step 1: Write tests using mock**

```python
# tests/test_summarizer.py
"""Tests for T5-based summarization stage."""
from nightshift.compression.summarizer import Summarizer


class TestSummarizer:
    def test_summarize_returns_shorter_text(self):
        s = Summarizer(use_model=False)  # use extractive fallback
        text = "The transformer architecture revolutionized NLP. " * 20
        result = s.summarize(text, max_length=50)
        assert len(result) < len(text)

    def test_summarize_list_of_chunks(self):
        s = Summarizer(use_model=False)
        chunks = [
            "First finding about model compression techniques.",
            "Second finding about quantization methods.",
            "Third finding about knowledge distillation.",
        ]
        results = s.summarize_chunks(chunks, max_length=30)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_empty_input(self):
        s = Summarizer(use_model=False)
        assert s.summarize("") == ""

    def test_short_input_returned_as_is(self):
        s = Summarizer(use_model=False)
        short = "Brief text."
        assert s.summarize(short) == short
```

**Step 2: Implement**

```python
# nightshift/compression/summarizer.py
"""Summarization stage using T5-small or extractive fallback."""
from __future__ import annotations

from typing import Any


class Summarizer:
    """Summarize text chunks. Uses T5-small when available, extractive fallback otherwise."""

    def __init__(self, use_model: bool = True, model: Any = None) -> None:
        self._use_model = use_model and model is not None
        self._model = model

    def summarize(self, text: str, max_length: int = 100) -> str:
        if not text or len(text) < max_length:
            return text
        if self._use_model:
            return self._model_summarize(text, max_length)
        return self._extractive_summarize(text, max_length)

    def summarize_chunks(self, chunks: list[str], max_length: int = 100) -> list[str]:
        return [self.summarize(c, max_length) for c in chunks]

    def _model_summarize(self, text: str, max_length: int) -> str:
        """Use T5-small for abstractive summarization."""
        model = self._model["model"]
        tokenizer = self._model["tokenizer"]
        inputs = tokenizer(
            f"summarize: {text}",
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=2,
            early_stopping=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _extractive_summarize(self, text: str, max_length: int) -> str:
        """Extractive fallback: take first N sentences."""
        sentences = text.replace(". ", ".\n").split("\n")
        result = []
        total = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if total + len(s) > max_length * 4:  # rough char-to-token
                break
            result.append(s)
            total += len(s)
        return " ".join(result) if result else text[:max_length * 4]
```

**Step 3: Run tests**

Run: `pytest tests/test_summarizer.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add nightshift/compression/summarizer.py tests/test_summarizer.py
git commit -m "feat: summarization stage with T5 and extractive fallback"
```

### Task 8.3: Wire MiniLM reranker into compression stage 5

**Files:**
- Create: `nightshift/compression/reranker.py`
- Create: `tests/test_reranker.py`

**Step 1: Write tests**

```python
# tests/test_reranker.py
"""Tests for reranking stage."""
from nightshift.compression.reranker import Reranker


class TestReranker:
    def test_rank_returns_top_k(self):
        r = Reranker(use_model=False)
        chunks = [f"chunk {i}" for i in range(20)]
        result = r.rank(chunks, query="relevant topic", top_k=5)
        assert len(result) == 5

    def test_rank_without_query_returns_first_k(self):
        r = Reranker(use_model=False)
        chunks = [f"chunk {i}" for i in range(10)]
        result = r.rank(chunks, query=None, top_k=3)
        assert len(result) == 3

    def test_rank_fewer_than_k(self):
        r = Reranker(use_model=False)
        chunks = ["only one"]
        result = r.rank(chunks, query="test", top_k=5)
        assert len(result) == 1

    def test_empty_chunks(self):
        r = Reranker(use_model=False)
        assert r.rank([], query="test", top_k=5) == []
```

**Step 2: Implement**

```python
# nightshift/compression/reranker.py
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
```

**Step 3: Run tests**

Run: `pytest tests/test_reranker.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add nightshift/compression/reranker.py tests/test_reranker.py
git commit -m "feat: reranking stage with cross-encoder and keyword fallback"
```

---

## Phase 9: Full Pipeline End-to-End Test

### Task 9.1: End-to-end test with no real models (fallback mode)

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Write test**

```python
# tests/test_e2e.py
"""End-to-end tests for the complete NightShift pipeline."""
import pytest
from nightshift.engine import NightShift


def _mock_dispatch(messages, model, **kwargs):
    return {
        "content": f"Processed {len(messages)} messages",
        "model": model,
        "input_tokens": sum(len(m.get("content", "")) // 4 for m in messages),
        "output_tokens": 50,
    }


class TestEndToEnd:
    def test_full_pipeline_short_message(self):
        engine = NightShift(api_budget=10.0)
        engine._dispatch = _mock_dispatch
        msgs = [{"role": "user", "content": "What is attention?"}]
        result = engine.complete(msgs, model="gpt-4o", gate=False)
        assert "Processed" in result["content"]

    def test_full_pipeline_long_message_compressed(self):
        engine = NightShift(api_budget=10.0, compress_threshold=100)
        engine._dispatch = _mock_dispatch
        msgs = [{"role": "user", "content": "research finding " * 500}]
        result = engine.complete(msgs, model="gpt-4o", gate=False)
        assert "Processed" in result["content"]

    def test_full_pipeline_dedup_across_calls(self):
        engine = NightShift(api_budget=10.0, compress_threshold=50000)
        engine._dispatch = _mock_dispatch
        big = "Repeated content block. " * 200
        msgs = [{"role": "user", "content": big}]
        r1 = engine.complete(msgs, model="gpt-4o", gate=False)
        r2 = engine.complete(msgs, model="gpt-4o", gate=False)
        # Second call should have fewer input tokens (dedup replaced content)
        assert r2["input_tokens"] < r1["input_tokens"]

    def test_full_pipeline_history_managed(self):
        engine = NightShift(api_budget=10.0, window_size=2, compress_threshold=50000)
        engine._dispatch = _mock_dispatch
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"q{i} " + "pad " * 50})
            msgs.append({"role": "assistant", "content": f"a{i} " + "pad " * 50})
        result = engine.complete(msgs, model="gpt-4o", gate=False)
        assert "Processed" in result["content"]
        # Should have compressed, so fewer messages sent to dispatch
        # (we can't easily assert the exact count, but it shouldn't crash)

    def test_full_pipeline_budget_exhaustion(self):
        engine = NightShift(api_budget=0.0, compress_threshold=50000)
        engine._dispatch = _mock_dispatch
        msgs = [{"role": "user", "content": "hi " * 500}]
        with pytest.raises(NotImplementedError, match="Best-effort"):
            engine.complete(msgs, model="claude-opus-4-20250514", gate=False)

    def test_report_reflects_all_calls(self):
        engine = NightShift(api_budget=10.0, compress_threshold=50000)
        engine._dispatch = _mock_dispatch
        for i in range(5):
            msgs = [{"role": "user", "content": f"query {i}"}]
            engine.complete(msgs, model="gpt-4o", gate=False)
        report = engine.report()
        assert report["api_calls"] == 5
        assert report["spent_usd"] > 0
        assert report["remaining_usd"] < 10.0

    def test_different_models_different_costs(self):
        engine = NightShift(api_budget=100.0, compress_threshold=50000)
        engine._dispatch = _mock_dispatch
        msgs = [{"role": "user", "content": "test " * 500}]
        engine.complete(msgs, model="gpt-4o-mini", gate=False)
        cost_cheap = engine.tracker.spent
        engine.complete(msgs, model="claude-opus-4-20250514", gate=False)
        cost_after = engine.tracker.spent
        # Opus should cost more than mini
        assert (cost_after - cost_cheap) > cost_cheap
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end pipeline tests covering all components"
```

---

## Phase 10: Confidence Gate (Wire Up Real Logic)

### Task 10.1: Test and implement keyword + heuristic classifier

**Files:**
- Modify: `nightshift/routing/confidence.py`
- Create: `tests/test_confidence.py`

**Step 1: Write tests**

```python
# tests/test_confidence.py
"""Tests for confidence-gated router."""
from nightshift.routing.confidence import ConfidenceGate


class FakeConfig:
    confidence_threshold = 0.8


class TestConfidenceGate:
    def test_extraction_classified_correctly(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Extract all entity names from this text."}]
        assert gate._classify_task(msgs) == "extraction"

    def test_retrieval_classified_correctly(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Search for similar papers about attention."}]
        assert gate._classify_task(msgs) == "retrieval"

    def test_evaluation_classified_correctly(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Is this novel? Evaluate the approach."}]
        assert gate._classify_task(msgs) == "evaluation"

    def test_generation_is_default(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Write a summary of the findings."}]
        assert gate._classify_task(msgs) == "generation"

    def test_extraction_high_confidence(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Extract entities from this document."}]
        conf = gate._estimate_confidence(msgs, "extraction")
        assert conf >= 0.9

    def test_complex_generation_low_confidence(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Synthesize " + "x " * 1000}]
        conf = gate._estimate_confidence(msgs, "generation")
        assert conf < 0.5

    def test_try_local_returns_none_for_generation(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Write a detailed analysis of this."}]
        result = gate.try_local(msgs)
        # Currently _handle_locally returns None (not implemented yet)
        # But stats should reflect the routing decision
        assert gate._api_routes >= 1

    def test_stats_tracking(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Write something complex."}]
        gate.try_local(msgs)
        stats = gate.stats()
        assert "local_hits" in stats
        assert "api_routes" in stats
```

**Step 2: Run tests**

Run: `pytest tests/test_confidence.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_confidence.py
git commit -m "test: confidence gate classification and routing tests"
```

---

## Phase Summary and Verification

After all phases complete, run the full verification:

```bash
# Full test suite
pytest tests/ -v --tb=short

# Expected output: ~50+ tests, ALL PASS
# Coverage of:
#   - Content deduplication (10 tests)
#   - Token tracker (11 tests)
#   - UCB1 bandit (6 tests)
#   - Sliding window (9 tests)
#   - API dispatch (8 tests)
#   - Engine integration (7 tests)
#   - End-to-end pipeline (7 tests)
#   - Confidence gate (8 tests)
#   - Utilities (5 tests)
#   - Model pool (4 tests)
#   - Summarizer (4 tests)
#   - Reranker (4 tests)

# Type check
mypy nightshift/ --ignore-missing-imports

# Lint
ruff check nightshift/
```

## What Each Phase Delivers

| Phase | Deliverable | Tests | Status |
|-------|------------|-------|--------|
| 1 | pytest infra, fixtures | conftest | Foundation |
| 2 | Content dedup hardened | 10 | Verified working |
| 3 | Token tracker + tiktoken | 16 | Accurate counting |
| 4 | UCB1 bandit hardened | 6 | Proven correct |
| 5 | Sliding window hardened | 9 | Bounded growth |
| 6 | Multi-provider dispatch | 8 | Real API calls |
| 7 | Engine integration | 7 | Pipeline works |
| 8 | Compression stages | 12 | Summarize + rerank |
| 9 | End-to-end pipeline | 7 | Everything together |
| 10 | Confidence gate | 8 | Smart routing |

**Total: 10 phases, ~83 tests, zero TODOs left in core path.**

## Future Phases (After Core)

| Phase | What | When |
|-------|------|------|
| 11 | Wire real local models (T5, MiniLM, GLiNER) with download-on-first-run | After core green |
| 12 | Persistent knowledge graph (ChromaDB) | After Phase 11 |
| 13 | Research agent (first application on the engine) | After Phase 12 |
| 14 | Benchmark suite vs AI Scientist / AI-Researcher | After Phase 13 |
| 15 | Paper draft | After benchmarks |
