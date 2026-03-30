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
    big_content = "This is a large block of text. " * 100
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": big_content},
    ]
