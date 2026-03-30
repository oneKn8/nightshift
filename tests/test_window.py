"""Tests for sliding window history manager."""
from nightshift.history.window import SlidingWindowManager


class FakeConfig:
    window_size = 3


class TestSlidingWindowManager:
    def test_short_history_unchanged(self, sample_messages):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(sample_messages)
        assert result == sample_messages

    def test_long_history_compressed(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(large_history)
        assert len(result) < len(large_history)

    def test_system_messages_preserved(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(large_history)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) >= 1

    def test_recent_messages_kept_intact(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(large_history)
        non_system = [m for m in result if m["role"] != "system"]
        last_original = large_history[-1]
        assert any(m["content"] == last_original["content"] for m in non_system)

    def test_archive_grows(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        w.apply(large_history)
        assert len(w._archive) == 1
        w.apply(large_history)
        assert len(w._archive) == 2

    def test_compressed_history_contains_marker(self, large_history):
        w = SlidingWindowManager(FakeConfig())
        result = w.apply(large_history)
        all_content = " ".join(m.get("content", "") for m in result)
        assert "Compressed history" in all_content

    def test_window_size_respected(self):
        config = FakeConfig()
        config.window_size = 2
        w = SlidingWindowManager(config)
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        result = w.apply(msgs)
        non_system = [m for m in result if m["role"] != "system"]
        assert len(non_system) <= config.window_size * 2

    def test_empty_messages(self):
        w = SlidingWindowManager(FakeConfig())
        assert w.apply([]) == []

    def test_only_system_messages(self):
        w = SlidingWindowManager(FakeConfig())
        msgs = [{"role": "system", "content": "sys"}]
        assert w.apply(msgs) == msgs
