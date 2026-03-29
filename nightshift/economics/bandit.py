"""UCB1 bandit for budget-optimal exploration scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return float("inf")  # unexplored arms get priority
        return self.total_reward / self.pulls


class BudgetBandit:
    """UCB1 multi-armed bandit for allocating API budget across research actions.

    Arms:
    - explore: investigate new research direction
    - deepen: expand findings in promising direction
    - synthesize: connect and produce output
    - evaluate: verify/validate findings

    Early in research: favors explore (high uncertainty)
    Mid research: shifts to deepen (known-good directions)
    Late research: shifts to synthesize (diminishing exploration returns)
    """

    def __init__(self, c: float = 1.414) -> None:
        self.c = c  # exploration coefficient (sqrt(2) by default)
        self.arms: dict[str, ArmStats] = {
            "explore": ArmStats(),
            "deepen": ArmStats(),
            "synthesize": ArmStats(),
            "evaluate": ArmStats(),
        }
        self.total_pulls = 0

    def select(self) -> str:
        """Select the next action using UCB1."""
        if self.total_pulls < len(self.arms):
            # Pull each arm at least once
            for name, stats in self.arms.items():
                if stats.pulls == 0:
                    return name

        best_arm = ""
        best_score = -float("inf")
        for name, stats in self.arms.items():
            ucb = stats.mean_reward + self.c * math.sqrt(
                math.log(self.total_pulls) / max(stats.pulls, 1)
            )
            if ucb > best_score:
                best_score = ucb
                best_arm = name

        return best_arm

    def update(self, response: dict[str, Any]) -> None:
        """Update arm statistics based on API response quality."""
        # TODO: extract reward signal from response
        # reward = new_facts / tokens_spent
        # For now, just track pulls
        action = response.get("_nightshift_action", "explore")
        if action in self.arms:
            self.arms[action].pulls += 1
            self.arms[action].total_reward += response.get("_nightshift_reward", 0.5)
            self.total_pulls += 1

    def report(self) -> dict[str, dict[str, float]]:
        return {
            name: {"pulls": s.pulls, "mean_reward": s.mean_reward}
            for name, s in self.arms.items()
        }
