"""Tests for confidence gate local handling (extraction, retrieval, evaluation)."""

import shutil

import pytest

from nightshift.routing.confidence import ConfidenceGate
from nightshift.compression.summarizer import Summarizer
from nightshift.compression.reranker import Reranker
from nightshift.history.knowledge import KnowledgeGraph


class FakeConfig:
    confidence_threshold = 0.8


@pytest.fixture
def kg(tmp_path):
    kg = KnowledgeGraph(path=str(tmp_path / "test_kg"))
    kg.add(
        [
            "Transformers use self-attention mechanisms for sequence modeling",
            "BERT introduced bidirectional pre-training for NLP tasks",
            "GPT models are autoregressive language models trained on web text",
            "LoRA enables parameter-efficient fine-tuning of large models",
            "RLHF aligns language models with human preferences via reward models",
        ],
        metadata=[{"topic": "ML"}] * 5,
    )
    yield kg


@pytest.fixture
def gate_with_resources(kg):
    gate = ConfidenceGate(FakeConfig())
    summarizer = Summarizer(use_model=False)
    reranker = Reranker(use_model=False)
    gate.set_resources(summarizer=summarizer, reranker=reranker, kg=kg)
    return gate


class TestExtractionHandling:
    def test_extraction_returns_result(self, gate_with_resources):
        gate = gate_with_resources
        msgs = [{"role": "user", "content": "Extract the key concepts from: " + "x " * 200}]
        result = gate.try_local(msgs)
        assert result is not None
        assert result["model"] == "local"
        assert result["_task_type"] == "extraction"

    def test_extraction_without_summarizer_returns_none(self, kg):
        gate = ConfidenceGate(FakeConfig())
        gate.set_resources(kg=kg)
        # No summarizer set
        msgs = [{"role": "user", "content": "Extract entities from this long text. " + "x " * 200}]
        result = gate.try_local(msgs)
        assert result is None

    def test_extraction_increments_local_hits(self, gate_with_resources):
        gate = gate_with_resources
        msgs = [{"role": "user", "content": "Extract key findings from: " + "x " * 200}]
        gate.try_local(msgs)
        assert gate.stats()["local_hits"] >= 1


class TestRetrievalHandling:
    def test_retrieval_returns_facts(self, gate_with_resources):
        gate = gate_with_resources
        msgs = [{"role": "user", "content": "Search for what we know about transformers"}]
        result = gate.try_local(msgs)
        assert result is not None
        assert result["model"] == "local"
        assert result["_task_type"] == "retrieval"
        assert "attention" in result["content"].lower() or "transformer" in result["content"].lower()

    def test_retrieval_empty_kg_returns_none(self, tmp_path):
        gate = ConfidenceGate(FakeConfig())
        empty_kg = KnowledgeGraph(path=str(tmp_path / "empty_kg"))
        gate.set_resources(kg=empty_kg)
        msgs = [{"role": "user", "content": "Search for papers about quantum computing"}]
        result = gate.try_local(msgs)
        assert result is None

    def test_retrieval_reranks_results(self, gate_with_resources):
        gate = gate_with_resources
        msgs = [{"role": "user", "content": "Retrieve information about fine-tuning methods"}]
        result = gate.try_local(msgs)
        if result is not None:
            # Should have reranked and selected relevant facts
            assert "content" in result


class TestEvaluationHandling:
    def test_evaluation_finds_similar(self, gate_with_resources):
        gate = gate_with_resources
        # This is very similar to an existing fact
        msgs = [{"role": "user", "content": "Evaluate: Transformers use attention for sequence tasks"}]
        result = gate.try_local(msgs)
        if result is not None:
            assert result["_task_type"] == "evaluation"
            assert "content" in result

    def test_evaluation_confidence_with_kg(self, gate_with_resources):
        gate = gate_with_resources
        msgs = [{"role": "user", "content": "Evaluate this claim"}]
        conf = gate._estimate_confidence(msgs, "evaluation")
        assert conf >= 0.8  # Should be high because KG has data

    def test_evaluation_confidence_without_kg(self):
        gate = ConfidenceGate(FakeConfig())
        msgs = [{"role": "user", "content": "Evaluate this claim"}]
        conf = gate._estimate_confidence(msgs, "evaluation")
        assert conf == 0.5  # Default without KG


class TestGenerationRouting:
    def test_generation_goes_to_api(self, gate_with_resources):
        gate = gate_with_resources
        msgs = [{"role": "user", "content": "Write a detailed essay about machine learning."}]
        result = gate.try_local(msgs)
        # Generation should not be handled locally (threshold 0.8 > confidence 0.3-0.7)
        assert result is None
        assert gate.stats()["api_routes"] >= 1


class TestSetResources:
    def test_set_resources_partial(self):
        gate = ConfidenceGate(FakeConfig())
        summarizer = Summarizer(use_model=False)
        gate.set_resources(summarizer=summarizer)
        assert gate._summarizer is summarizer
        assert gate._reranker is None

    def test_set_resources_full(self, kg):
        gate = ConfidenceGate(FakeConfig())
        summarizer = Summarizer(use_model=False)
        reranker = Reranker(use_model=False)
        gate.set_resources(summarizer=summarizer, reranker=reranker, kg=kg)
        assert gate._summarizer is summarizer
        assert gate._reranker is reranker
        assert gate._kg is kg
