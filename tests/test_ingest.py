"""Tests for response fact extraction and knowledge graph ingestion."""

import pytest

from nightshift.history.window import SlidingWindowManager
from nightshift.history.knowledge import KnowledgeGraph


class FakeConfig:
    window_size = 5


@pytest.fixture
def kg(tmp_path):
    return KnowledgeGraph(path=str(tmp_path / "test_kg"))


@pytest.fixture
def window_with_kg(kg):
    window = SlidingWindowManager(FakeConfig())
    window.set_knowledge_graph(kg)
    return window, kg


class TestFactExtraction:
    def test_extracts_numbered_items(self):
        text = (
            "Key findings:\n"
            "1. Transformers outperform RNNs on long-range dependencies\n"
            "2. Self-attention scales quadratically with sequence length\n"
            "3. Pre-training on large corpora improves downstream performance\n"
        )
        facts = SlidingWindowManager._extract_facts(text)
        assert len(facts) == 3
        assert "Transformers outperform" in facts[0]

    def test_extracts_bullet_points(self):
        text = (
            "Summary:\n"
            "- Flash Attention reduces memory from O(n2) to O(n) for attention computation\n"
            "- LoRA adds trainable low-rank matrices to frozen pretrained weights\n"
            "- QLoRA combines 4-bit quantization with LoRA for further memory savings\n"
        )
        facts = SlidingWindowManager._extract_facts(text)
        assert len(facts) == 3
        assert "Flash Attention" in facts[0]

    def test_extracts_factual_sentences(self):
        text = (
            "The study found that larger models consistently outperform smaller ones "
            "on reasoning benchmarks. Results demonstrate that chain-of-thought prompting "
            "improves accuracy by 15-30%. The data suggests that scaling alone is insufficient "
            "without proper training methodology."
        )
        facts = SlidingWindowManager._extract_facts(text)
        assert len(facts) >= 2
        assert any("found that" in f.lower() for f in facts)

    def test_ignores_short_items(self):
        text = "1. Yes\n2. No\n3. Maybe\n"
        facts = SlidingWindowManager._extract_facts(text)
        assert len(facts) == 0  # All too short (<20 chars)

    def test_caps_at_20_facts(self):
        lines = [f"{i+1}. This is a sufficiently long fact number {i+1} about something important" for i in range(30)]
        text = "\n".join(lines)
        facts = SlidingWindowManager._extract_facts(text)
        assert len(facts) == 20

    def test_empty_text_returns_empty(self):
        assert SlidingWindowManager._extract_facts("") == []
        assert SlidingWindowManager._extract_facts("short") == []


class TestIngestResponse:
    def test_ingests_structured_response(self, window_with_kg):
        window, kg = window_with_kg
        response = {
            "content": (
                "Key findings:\n"
                "1. Neural scaling laws predict performance from compute budget\n"
                "2. Chinchilla-optimal training uses equal compute for data and parameters\n"
                "3. Mixture of Experts reduces inference cost while maintaining quality\n"
            ),
            "model": "gpt-5.4-mini",
        }
        window.ingest_response(response)
        assert kg.count() == 3

    def test_ingests_narrative_response(self, window_with_kg):
        window, kg = window_with_kg
        response = {
            "content": (
                "Recent research found that retrieval-augmented generation significantly "
                "improves factual accuracy. The study demonstrates that combining vector "
                "search with language models reduces hallucination rates by 40%."
            ),
            "model": "claude-sonnet-4-20250514",
        }
        window.ingest_response(response)
        assert kg.count() >= 1

    def test_skips_short_responses(self, window_with_kg):
        window, kg = window_with_kg
        window.ingest_response({"content": "OK", "model": "local"})
        assert kg.count() == 0

    def test_skips_without_kg(self):
        window = SlidingWindowManager(FakeConfig())
        # No KG set -- should not crash
        window.ingest_response({"content": "1. Some fact about something important", "model": "test"})

    def test_metadata_tracks_source_model(self, window_with_kg):
        window, kg = window_with_kg
        response = {
            "content": "- The experiment achieved state-of-the-art results on MMLU benchmark scores",
            "model": "gpt-5.4-mini",
        }
        window.ingest_response(response)
        results = kg.query("MMLU benchmark", top_k=1)
        assert len(results) == 1
        assert results[0]["metadata"]["source"] == "api:gpt-5.4-mini"

    def test_multiple_ingestions_accumulate(self, window_with_kg):
        window, kg = window_with_kg
        for i in range(3):
            response = {
                "content": f"- Finding {i}: This is an important discovery about topic {i} in research",
                "model": "test",
            }
            window.ingest_response(response)
        assert kg.count() == 3
