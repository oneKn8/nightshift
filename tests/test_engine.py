"""Integration tests for the NightShift engine."""
import pytest
from nightshift.engine import NightShift


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
        engine.complete(duplicate_messages, model="gpt-4o", compress=False, gate=False)
        engine.complete(duplicate_messages, model="gpt-4o", compress=False, gate=False)
        report = engine.report()
        assert report["api_calls"] == 2

    def test_budget_enforcement(self):
        engine = NightShift(api_budget=0.0)
        engine._dispatch = _mock_dispatch
        msgs = [{"role": "user", "content": "hi " * 1000}]
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
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"question {i} " + "x " * 100})
            msgs.append({"role": "assistant", "content": f"answer {i} " + "y " * 100})
        result = engine.complete(msgs, model="gpt-4o", compress=False, gate=False)
        assert result["content"] == "Mock response"

    def test_compression_skipped_for_short_messages(self, sample_messages):
        engine = NightShift(api_budget=10.0, compress_threshold=50000)
        engine._dispatch = _mock_dispatch
        result = engine.complete(sample_messages, model="gpt-4o", gate=False)
        assert result["content"] == "Mock response"
