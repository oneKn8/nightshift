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
        t = TokenTracker(budget=0.001)
        big_msgs = [{"role": "user", "content": "x " * 10000}]
        t.record_api(big_msgs, {"r": "ok"}, "claude-opus-4-20250514")
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
        assert t.spent > 0

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
