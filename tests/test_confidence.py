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
        assert gate._api_routes >= 1

    def test_stats_tracking(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Write something complex."}]
        gate.try_local(msgs)
        stats = gate.stats()
        assert "local_hits" in stats
        assert "api_routes" in stats
