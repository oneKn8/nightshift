"""End-to-end tests for the complete NightShift pipeline."""
import pytest
from nightshift.engine import NightShift


def _mock_dispatch(messages, model, **kwargs):
    input_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
    return {
        "content": f"Processed {len(messages)} messages",
        "model": model,
        "input_tokens": input_tokens,
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
        assert (cost_after - cost_cheap) > cost_cheap
