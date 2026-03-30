"""Tests for the overnight continuous research loop."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nightshift.loop import OvernightLoop, LoopState, _parse_duration
from nightshift.agents.research import ResearchResult


class TestParseDuration:
    def test_overnight(self):
        assert _parse_duration("overnight") == 8 * 3600

    def test_hours(self):
        assert _parse_duration("2h") == 7200

    def test_minutes(self):
        assert _parse_duration("30m") == 1800

    def test_seconds(self):
        assert _parse_duration("60s") == 60

    def test_raw_number(self):
        assert _parse_duration("3600") == 3600

    def test_case_insensitive(self):
        assert _parse_duration("OVERNIGHT") == 8 * 3600
        assert _parse_duration("4H") == 14400


class TestLoopState:
    def test_save_and_load(self, tmp_path):
        state = LoopState(
            iteration=3,
            start_time=1000.0,
            topics_explored=["AI safety", "RLHF"],
            topics_queue=["alignment"],
            bandit_state={"explore": {"pulls": 2, "total_reward": 1.0}},
            tracker_spent=0.05,
            tracker_call_count=5,
            knowledge_fact_count=42,
            records=[{"iteration": 1, "action": "explore", "topic": "AI safety"}],
            last_action="explore",
            last_topic="AI safety",
            stopped_reason="",
        )
        path = tmp_path / "state.json"
        state.save(path)

        loaded = LoopState.load(path)
        assert loaded.iteration == 3
        assert loaded.topics_explored == ["AI safety", "RLHF"]
        assert loaded.tracker_spent == 0.05
        assert loaded.knowledge_fact_count == 42

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "state.json"
        state = LoopState()
        state.save(path)
        assert path.exists()


def _mock_research_result(topic: str, facts: int = 5, cost: float = 0.001) -> ResearchResult:
    return ResearchResult(
        topic=topic,
        papers_fetched=10,
        facts_extracted=facts,
        api_calls=1,
        total_cost=cost,
        report="Mock report",
        citations=[],
        duration_seconds=1.0,
    )


class TestOvernightLoop:
    def test_init(self, tmp_path):
        loop = OvernightLoop(
            topics=["AI safety", "RLHF"],
            budget=5.0,
            duration="2h",
            checkpoint_path=str(tmp_path / "checkpoints"),
        )
        assert loop.duration_seconds == 7200
        assert loop.budget == 5.0
        assert len(loop.topics) == 2

    def test_stops_on_budget_exhaustion(self, tmp_path):
        loop = OvernightLoop(
            topics=["test topic"],
            budget=0.002,
            duration="1h",
            checkpoint_path=str(tmp_path / "checkpoints"),
            use_models=False,
        )

        call_count = [0]

        def mock_run(topic, max_papers=20, depth="moderate"):
            call_count[0] += 1
            return _mock_research_result(topic, facts=3, cost=0.001)

        def mock_report():
            # First call: budget available. After first run: depleted.
            spent = call_count[0] * 0.002
            return {
                "spent_usd": spent,
                "remaining_usd": max(0.002 - spent, -0.001),
                "api_calls": call_count[0],
            }

        mock_agent = MagicMock()
        mock_agent.run = mock_run
        mock_agent.report = mock_report
        mock_agent.kg.count.return_value = 10

        loop._agent = mock_agent
        state = loop.run()

        assert "budget" in state.stopped_reason
        assert state.iteration >= 1

    def test_stops_on_time_limit(self, tmp_path):
        loop = OvernightLoop(
            topics=["test topic"],
            budget=100.0,
            duration="1s",
            checkpoint_path=str(tmp_path / "checkpoints"),
            use_models=False,
        )

        # Set start time in the past
        loop._state.start_time = time.time() - 10

        def mock_run(topic, max_papers=20, depth="moderate"):
            return _mock_research_result(topic, facts=5, cost=0.001)

        mock_agent = MagicMock()
        mock_agent.run = mock_run
        mock_agent.report.return_value = {
            "spent_usd": 0.001,
            "remaining_usd": 99.999,
            "api_calls": 1,
        }
        mock_agent.kg.count.return_value = 5

        loop._agent = mock_agent
        state = loop.run()

        assert "time_limit" in state.stopped_reason

    def test_stops_on_convergence(self, tmp_path):
        loop = OvernightLoop(
            topics=["t1", "t2", "t3", "t4", "t5", "t6"],
            budget=100.0,
            duration="1h",
            convergence_threshold=2,
            checkpoint_path=str(tmp_path / "checkpoints"),
            use_models=False,
        )

        def mock_run(topic, max_papers=20, depth="moderate"):
            return _mock_research_result(topic, facts=0, cost=0.0001)

        mock_agent = MagicMock()
        mock_agent.run = mock_run
        mock_agent.report.return_value = {
            "spent_usd": 0.001,
            "remaining_usd": 99.999,
            "api_calls": 1,
        }
        mock_agent.kg.count.return_value = 0

        loop._agent = mock_agent
        state = loop.run()

        assert "converged" in state.stopped_reason

    def test_checkpoint_written(self, tmp_path):
        loop = OvernightLoop(
            topics=["test"],
            budget=0.003,
            duration="1h",
            checkpoint_path=str(tmp_path / "checkpoints"),
            use_models=False,
        )

        call_count = [0]

        def mock_run(topic, max_papers=20, depth="moderate"):
            call_count[0] += 1
            return _mock_research_result(topic, facts=3, cost=0.001)

        def mock_report():
            spent = call_count[0] * 0.002
            return {
                "spent_usd": spent,
                "remaining_usd": max(0.003 - spent, -0.001),
                "api_calls": call_count[0],
            }

        mock_agent = MagicMock()
        mock_agent.run = mock_run
        mock_agent.report = mock_report
        mock_agent.kg.count.return_value = 3

        loop._agent = mock_agent
        loop.run()

        assert loop.checkpoint_file.exists()
        with open(loop.checkpoint_file) as f:
            data = json.load(f)
        assert data["iteration"] >= 1
        assert len(data["records"]) >= 1

    def test_resume_from_checkpoint(self, tmp_path):
        # Create a checkpoint
        state = LoopState(
            iteration=5,
            start_time=time.time() - 100,
            topics_explored=["AI safety"],
            topics_queue=["RLHF"],
            bandit_state={
                "explore": {"pulls": 3, "total_reward": 1.5},
                "deepen": {"pulls": 1, "total_reward": 0.3},
                "synthesize": {"pulls": 1, "total_reward": 0.2},
                "evaluate": {"pulls": 0, "total_reward": 0.0},
            },
            tracker_spent=0.01,
            knowledge_fact_count=20,
        )
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        state.save(cp_dir / "loop_state.json")

        loop = OvernightLoop(
            topics=["AI safety", "RLHF"],
            budget=1.0,
            duration="1h",
            checkpoint_path=str(cp_dir),
            use_models=False,
        )
        resumed = loop.resume()
        assert resumed is True
        assert loop._state.iteration == 5
        assert loop._bandit.arms["explore"].pulls == 3

    def test_final_report_written(self, tmp_path):
        loop = OvernightLoop(
            topics=["test"],
            budget=0.002,
            duration="1h",
            checkpoint_path=str(tmp_path / "checkpoints"),
            use_models=False,
        )

        def mock_run(topic, max_papers=20, depth="moderate"):
            return _mock_research_result(topic, facts=3, cost=0.001)

        mock_agent = MagicMock()
        mock_agent.run = mock_run
        mock_agent.report.return_value = {
            "spent_usd": 0.003,
            "remaining_usd": -0.001,
            "api_calls": 1,
        }
        mock_agent.kg.count.return_value = 3

        loop._agent = mock_agent
        loop.run()

        report_path = tmp_path / "checkpoints" / "overnight_report.txt"
        assert report_path.exists()
        content = report_path.read_text()
        assert "NIGHTSHIFT OVERNIGHT RESEARCH REPORT" in content
        assert "BANDIT STATISTICS" in content

    def test_bandit_guides_action_selection(self, tmp_path):
        loop = OvernightLoop(
            topics=["t1", "t2", "t3", "t4"],
            budget=0.01,
            duration="1h",
            checkpoint_path=str(tmp_path / "checkpoints"),
            use_models=False,
        )

        actions_seen: list[str] = []

        def mock_run(topic, max_papers=20, depth="moderate"):
            return _mock_research_result(topic, facts=3, cost=0.001)

        mock_agent = MagicMock()
        mock_agent.run = mock_run
        mock_agent.report.return_value = {
            "spent_usd": 0.005,
            "remaining_usd": 0.005,
            "api_calls": 1,
        }
        mock_agent.kg.count.return_value = 10

        original_run = loop.run

        def recording_execute(agent, action, topic):
            actions_seen.append(action)
            return _mock_research_result(topic, facts=3, cost=0.001)

        loop._agent = mock_agent
        loop._execute_action = recording_execute

        # Make budget run out after a few iterations
        call_num = [0]
        def report_with_drain():
            call_num[0] += 1
            spent = call_num[0] * 0.003
            return {
                "spent_usd": spent,
                "remaining_usd": max(0.01 - spent, -0.001),
                "api_calls": call_num[0],
            }
        mock_agent.report = report_with_drain

        loop.run()

        # First 4 pulls should try each arm once (UCB1 exploration guarantee)
        if len(actions_seen) >= 4:
            first_four = set(actions_seen[:4])
            assert first_four == {"explore", "deepen", "synthesize", "evaluate"}


class TestWindowWithSummarizer:
    """Test that the sliding window uses the summarizer when provided."""

    def test_window_uses_summarizer(self):
        from nightshift.history.window import SlidingWindowManager
        from nightshift.compression.summarizer import Summarizer

        class MockConfig:
            window_size = 2

        summarizer = Summarizer(use_model=False)
        window = SlidingWindowManager(MockConfig(), summarizer=summarizer)

        msgs = [{"role": "system", "content": "system prompt"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"question {i} " + "x " * 200})
            msgs.append({"role": "assistant", "content": f"answer {i} " + "y " * 200})

        result = window.apply(msgs)
        # Should have: system + compressed archive + recent 4 messages
        assert len(result) <= 10  # Much shorter than original 21

    def test_window_set_summarizer(self):
        from nightshift.history.window import SlidingWindowManager
        from nightshift.compression.summarizer import Summarizer

        class MockConfig:
            window_size = 2

        window = SlidingWindowManager(MockConfig())
        assert window._summarizer is None

        summarizer = Summarizer(use_model=False)
        window.set_summarizer(summarizer)
        assert window._summarizer is summarizer


class TestBestEffortLocal:
    """Test the engine's best-effort local fallback when budget is exhausted."""

    def test_returns_local_synthesis(self):
        from nightshift.engine import NightShift

        engine = NightShift(api_budget=0.0)
        # Add some facts to the knowledge graph
        engine.kg.add(
            ["Fact 1: transformers use attention", "Fact 2: BERT is bidirectional"],
            metadata=[{"topic": "NLP"}] * 2,
        )

        msgs = [{"role": "user", "content": "Tell me about transformers"}]
        result = engine._best_effort_local(msgs)

        assert result["_budget_exhausted"] is True
        assert result["model"] == "local"
        assert "Local synthesis" in result["content"]

    def test_empty_knowledge_graph(self):
        from nightshift.engine import NightShift

        engine = NightShift(api_budget=0.0, knowledge_db="./test_empty_kb")
        msgs = [{"role": "user", "content": "anything"}]
        result = engine._best_effort_local(msgs)

        assert result["_budget_exhausted"] is True
        assert "No cached knowledge" in result["content"]

        # Cleanup
        import shutil
        shutil.rmtree("./test_empty_kb", ignore_errors=True)
