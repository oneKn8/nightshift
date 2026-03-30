"""Overnight continuous research loop with bandit-guided exploration."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nightshift.agents.research import ResearchAgent, ResearchResult
from nightshift.economics.bandit import BudgetBandit

log = logging.getLogger(__name__)


def _parse_duration(duration: str) -> float:
    """Parse duration string to seconds. 'overnight' = 8h."""
    duration = duration.strip().lower()
    if duration == "overnight":
        return 8 * 3600
    if duration.endswith("h"):
        return float(duration[:-1]) * 3600
    if duration.endswith("m"):
        return float(duration[:-1]) * 60
    if duration.endswith("s"):
        return float(duration[:-1])
    return float(duration)


@dataclass
class IterationRecord:
    iteration: int
    action: str
    topic: str
    papers_fetched: int
    facts_extracted: int
    cost_usd: float
    duration_seconds: float
    reward: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoopState:
    """Serializable checkpoint for the overnight loop."""
    iteration: int = 0
    start_time: float = 0.0
    topics_explored: list[str] = field(default_factory=list)
    topics_queue: list[str] = field(default_factory=list)
    bandit_state: dict[str, dict[str, float]] = field(default_factory=dict)
    tracker_spent: float = 0.0
    tracker_call_count: int = 0
    knowledge_fact_count: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    last_action: str = ""
    last_topic: str = ""
    stopped_reason: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> LoopState:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class OvernightLoop:
    """Multi-iteration research loop with bandit-guided topic exploration.

    Runs continuously until budget, time, or convergence criteria are met.
    Checkpoints state between iterations for resume capability.
    """

    def __init__(
        self,
        topics: list[str],
        budget: float = 10.0,
        duration: str = "overnight",
        model: str = "gpt-5.4-mini",
        knowledge_path: str = "./nightshift_kb",
        checkpoint_path: str = "./nightshift_checkpoints",
        max_papers_per_iter: int = 20,
        convergence_threshold: int = 2,
        use_models: bool = True,
    ) -> None:
        self.topics = list(topics)
        self.budget = budget
        self.duration_seconds = _parse_duration(duration)
        self.model = model
        self.knowledge_path = knowledge_path
        self.checkpoint_dir = Path(checkpoint_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_papers_per_iter = max_papers_per_iter
        self.convergence_threshold = convergence_threshold
        self.use_models = use_models

        self._agent: ResearchAgent | None = None
        self._bandit = BudgetBandit()
        self._state = LoopState()
        self._low_yield_streak = 0

    @property
    def checkpoint_file(self) -> Path:
        return self.checkpoint_dir / "loop_state.json"

    def _ensure_agent(self) -> ResearchAgent:
        if self._agent is None:
            self._agent = ResearchAgent(
                knowledge_path=self.knowledge_path,
                model=self.model,
                api_budget=self.budget,
                use_models=self.use_models,
            )
        return self._agent

    def resume(self) -> bool:
        """Resume from checkpoint if available. Returns True if resumed."""
        if self.checkpoint_file.exists():
            self._state = LoopState.load(self.checkpoint_file)
            # Restore bandit state
            for arm_name, stats in self._state.bandit_state.items():
                if arm_name in self._bandit.arms:
                    self._bandit.arms[arm_name].pulls = int(stats.get("pulls", 0))
                    self._bandit.arms[arm_name].total_reward = stats.get("total_reward", 0.0)
                    self._bandit.total_pulls += int(stats.get("pulls", 0))
            log.info(
                f"Resumed from iteration {self._state.iteration}, "
                f"${self._state.tracker_spent:.4f} spent, "
                f"{self._state.knowledge_fact_count} facts"
            )
            return True
        return False

    def run(self) -> LoopState:
        """Execute the overnight research loop."""
        agent = self._ensure_agent()
        self._state.start_time = self._state.start_time or time.time()
        self._state.topics_queue = self._state.topics_queue or list(self.topics)

        log.info(
            f"Starting overnight loop: {len(self.topics)} topics, "
            f"${self.budget} budget, {self.duration_seconds/3600:.1f}h duration"
        )

        while True:
            # Check stopping conditions
            stop_reason = self._check_stop_conditions(agent)
            if stop_reason:
                self._state.stopped_reason = stop_reason
                log.info(f"Loop stopped: {stop_reason}")
                break

            self._state.iteration += 1
            action = self._bandit.select()
            topic = self._pick_topic(action)

            log.info(
                f"Iteration {self._state.iteration}: "
                f"action={action}, topic={topic}"
            )

            iter_start = time.time()
            result = self._execute_action(agent, action, topic)
            iter_duration = time.time() - iter_start

            # Compute reward: facts per dollar (or per unit cost)
            reward = self._compute_reward(result, action)
            self._bandit.arms[action].pulls += 1
            self._bandit.arms[action].total_reward += reward
            self._bandit.total_pulls += 1

            # Track convergence
            if result.facts_extracted < self.convergence_threshold:
                self._low_yield_streak += 1
            else:
                self._low_yield_streak = 0

            record = IterationRecord(
                iteration=self._state.iteration,
                action=action,
                topic=topic,
                papers_fetched=result.papers_fetched,
                facts_extracted=result.facts_extracted,
                cost_usd=result.total_cost,
                duration_seconds=iter_duration,
                reward=reward,
            )
            self._state.records.append(record.__dict__)
            self._state.last_action = action
            self._state.last_topic = topic

            # Update state from agent
            engine_report = agent.report()
            self._state.tracker_spent = engine_report["spent_usd"]
            self._state.tracker_call_count = engine_report["api_calls"]
            self._state.knowledge_fact_count = agent.kg.count()

            # Snapshot bandit
            self._state.bandit_state = {
                name: {"pulls": s.pulls, "total_reward": s.total_reward}
                for name, s in self._bandit.arms.items()
            }

            # Checkpoint
            self._state.save(self.checkpoint_file)
            log.info(
                f"Iteration {self._state.iteration} done: "
                f"{result.facts_extracted} facts, ${result.total_cost:.4f}, "
                f"reward={reward:.3f}"
            )

        # Final checkpoint
        self._state.save(self.checkpoint_file)
        self._save_final_report()
        return self._state

    def _check_stop_conditions(self, agent: ResearchAgent) -> str | None:
        """Returns stop reason string, or None to continue."""
        elapsed = time.time() - self._state.start_time

        # Time limit
        if elapsed >= self.duration_seconds:
            return f"time_limit ({elapsed/3600:.1f}h elapsed)"

        # Budget
        report = agent.report()
        remaining = report["remaining_usd"]
        if remaining <= 0.001:
            return f"budget_exhausted (${report['spent_usd']:.4f} spent)"

        # Budget too low for even one cheap call
        if remaining < 0.0001:
            return f"budget_insufficient (${remaining:.6f} remaining)"

        # Convergence: N consecutive low-yield iterations
        if self._low_yield_streak >= 5:
            return f"converged ({self._low_yield_streak} low-yield iterations)"

        # No more topics to explore
        if not self._state.topics_queue and not self._state.topics_explored:
            return "no_topics"

        return None

    def _pick_topic(self, action: str) -> str:
        """Select topic based on action type."""
        if action == "explore":
            # Pick next unexplored topic
            if self._state.topics_queue:
                topic = self._state.topics_queue.pop(0)
                self._state.topics_explored.append(topic)
                return topic
            # All explored: generate sub-topic from existing knowledge
            return self._generate_subtopic()

        if action == "deepen":
            # Pick the topic with highest fact yield so far
            return self._best_topic() or self.topics[0]

        if action in ("synthesize", "evaluate"):
            # Synthesize/evaluate across all explored topics
            return " AND ".join(self._state.topics_explored[:3]) or self.topics[0]

        return self.topics[0]

    def _generate_subtopic(self) -> str:
        """Generate a sub-topic from existing explored topics."""
        if len(self._state.topics_explored) >= 2:
            # Cross-pollinate: combine two topics
            t1 = self._state.topics_explored[-1]
            t2 = self._state.topics_explored[-2]
            return f"{t1} intersection with {t2}"
        if self._state.topics_explored:
            return f"recent advances in {self._state.topics_explored[-1]}"
        return self.topics[0]

    def _best_topic(self) -> str | None:
        """Find the topic that yielded the most facts."""
        topic_facts: dict[str, int] = {}
        for rec in self._state.records:
            t = rec.get("topic", "")
            topic_facts[t] = topic_facts.get(t, 0) + rec.get("facts_extracted", 0)
        if not topic_facts:
            return None
        return max(topic_facts, key=topic_facts.get)  # type: ignore[arg-type]

    def _execute_action(
        self, agent: ResearchAgent, action: str, topic: str
    ) -> ResearchResult:
        """Execute the bandit-selected action."""
        if action == "explore":
            return agent.run(topic, max_papers=self.max_papers_per_iter, depth="moderate")
        if action == "deepen":
            return agent.run(topic, max_papers=self.max_papers_per_iter * 2, depth="comprehensive")
        if action == "synthesize":
            return agent.run(topic, max_papers=5, depth="comprehensive")
        if action == "evaluate":
            return agent.run(topic, max_papers=10, depth="shallow")
        return agent.run(topic, max_papers=self.max_papers_per_iter, depth="moderate")

    def _compute_reward(self, result: ResearchResult, action: str) -> float:
        """Compute reward for bandit update. Higher = better value."""
        if result.total_cost <= 0:
            # Free iteration (all local) -- moderate reward
            return 0.5 if result.facts_extracted > 0 else 0.1

        # Facts per dollar (normalized to [0, 1] range)
        facts_per_dollar = result.facts_extracted / max(result.total_cost, 0.0001)
        # Diminishing returns: log scale, cap at 1.0
        import math
        raw = math.log1p(facts_per_dollar) / 10.0
        return min(raw, 1.0)

    def _save_final_report(self) -> None:
        """Write a human-readable summary of the overnight run."""
        s = self._state
        total_facts = sum(r.get("facts_extracted", 0) for r in s.records)
        total_papers = sum(r.get("papers_fetched", 0) for r in s.records)
        total_cost = sum(r.get("cost_usd", 0) for r in s.records)
        elapsed = time.time() - s.start_time if s.start_time else 0

        report_path = self.checkpoint_dir / "overnight_report.txt"
        lines = [
            "NIGHTSHIFT OVERNIGHT RESEARCH REPORT",
            "=" * 40,
            f"Duration: {elapsed/3600:.1f} hours",
            f"Iterations: {s.iteration}",
            f"Topics explored: {len(s.topics_explored)}",
            f"Papers fetched: {total_papers}",
            f"Facts extracted: {total_facts}",
            f"Knowledge graph size: {s.knowledge_fact_count}",
            f"Total cost: ${total_cost:.4f}",
            f"Stop reason: {s.stopped_reason}",
            "",
            "BANDIT STATISTICS",
            "-" * 40,
        ]
        for arm, stats in s.bandit_state.items():
            pulls = int(stats.get("pulls", 0))
            reward = stats.get("total_reward", 0.0)
            mean = reward / max(pulls, 1)
            lines.append(f"  {arm}: {pulls} pulls, mean_reward={mean:.3f}")

        lines.extend([
            "",
            "ITERATION LOG",
            "-" * 40,
        ])
        for rec in s.records:
            lines.append(
                f"  #{rec['iteration']}: {rec['action']} on '{rec['topic']}' "
                f"-> {rec['facts_extracted']} facts, ${rec['cost_usd']:.4f}"
            )

        report_path.write_text("\n".join(lines))
        log.info(f"Report written to {report_path}")
