"""Tests for utility functions."""
from nightshift.utils import count_tokens


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_short_string(self):
        n = count_tokens("Hello, world!")
        assert 1 <= n <= 10

    def test_long_string(self):
        text = "The quick brown fox " * 100
        n = count_tokens(text)
        assert 300 < n < 700

    def test_messages_list(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello there."},
        ]
        n = count_tokens(msgs)
        assert n > 0

    def test_consistency(self):
        text = "Consistent counting test"
        assert count_tokens(text) == count_tokens(text)
