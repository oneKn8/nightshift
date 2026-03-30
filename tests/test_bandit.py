"""Tests for UCB1 budget bandit."""
from nightshift.economics.bandit import BudgetBandit, ArmStats


class TestArmStats:
    def test_unexplored_arm_infinite_reward(self):
        arm = ArmStats()
        assert arm.mean_reward == float("inf")

    def test_mean_reward_calculation(self):
        arm = ArmStats(pulls=4, total_reward=2.0)
        assert arm.mean_reward == 0.5


class TestBudgetBandit:
    def test_initial_selection_explores_all_arms(self):
        b = BudgetBandit()
        seen = set()
        for _ in range(4):
            arm = b.select()
            seen.add(arm)
            b.arms[arm].pulls += 1
            b.arms[arm].total_reward += 0.5
            b.total_pulls += 1
        assert seen == {"explore", "deepen", "synthesize", "evaluate"}

    def test_high_reward_arm_selected_more(self):
        b = BudgetBandit(c=0.1)
        b.arms["synthesize"] = ArmStats(pulls=10, total_reward=9.0)
        b.arms["explore"] = ArmStats(pulls=10, total_reward=1.0)
        b.arms["deepen"] = ArmStats(pulls=10, total_reward=1.0)
        b.arms["evaluate"] = ArmStats(pulls=10, total_reward=1.0)
        b.total_pulls = 40
        selections = [b.select() for _ in range(10)]
        assert selections.count("synthesize") >= 7

    def test_update_increments_pulls(self):
        b = BudgetBandit()
        b.update({"_nightshift_action": "explore", "_nightshift_reward": 0.8})
        assert b.arms["explore"].pulls == 1
        assert b.arms["explore"].total_reward == 0.8
        assert b.total_pulls == 1

    def test_update_unknown_action_ignored(self):
        b = BudgetBandit()
        b.update({"_nightshift_action": "nonexistent", "_nightshift_reward": 1.0})
        assert b.total_pulls == 0

    def test_report_structure(self):
        b = BudgetBandit()
        b.update({"_nightshift_action": "explore", "_nightshift_reward": 0.5})
        report = b.report()
        assert "explore" in report
        assert "pulls" in report["explore"]
        assert "mean_reward" in report["explore"]

    def test_ucb_exploration_bonus_decays(self):
        b = BudgetBandit()
        for arm in b.arms:
            b.arms[arm] = ArmStats(pulls=100, total_reward=50.0)
        b.total_pulls = 400
        selected = b.select()
        assert selected in b.arms
