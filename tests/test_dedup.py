"""Tests for content deduplication."""
from nightshift.routing.dedup import ContentDedup, SentContent


class TestContentDedup:
    def test_first_message_passes_through(self, sample_messages):
        dedup = ContentDedup()
        result = dedup.process(sample_messages)
        assert result == sample_messages

    def test_short_content_never_deduped(self):
        dedup = ContentDedup()
        msgs = [{"role": "user", "content": "hello"}]
        result1 = dedup.process(msgs)
        result2 = dedup.process(msgs)
        # Short content should pass through unchanged even on second call
        assert result2[0]["content"] == "hello"

    def test_duplicate_long_content_replaced(self, duplicate_messages):
        dedup = ContentDedup()
        result1 = dedup.process(duplicate_messages)
        # First call: content passes through
        assert "This is a large block" in result1[1]["content"]

        result2 = dedup.process(duplicate_messages)
        # Second call: content replaced with reference
        assert "Previously provided" in result2[1]["content"]
        assert "unchanged" in result2[1]["content"]

    def test_different_content_not_deduped(self):
        dedup = ContentDedup()
        msg1 = [{"role": "user", "content": "A " * 200}]
        msg2 = [{"role": "user", "content": "B " * 200}]
        dedup.process(msg1)
        result = dedup.process(msg2)
        assert "Previously provided" not in result[0]["content"]

    def test_system_messages_preserved(self):
        dedup = ContentDedup()
        big = "system context " * 100
        msgs = [{"role": "system", "content": big}]
        dedup.process(msgs)
        result = dedup.process(msgs)
        # System message with same content should still get deduped
        assert "Previously provided" in result[0]["content"]

    def test_role_preserved_after_dedup(self, duplicate_messages):
        dedup = ContentDedup()
        dedup.process(duplicate_messages)
        result = dedup.process(duplicate_messages)
        assert result[1]["role"] == "user"

    def test_clear_resets_cache(self, duplicate_messages):
        dedup = ContentDedup()
        dedup.process(duplicate_messages)
        dedup.clear()
        result = dedup.process(duplicate_messages)
        # After clear, content should pass through again
        assert "This is a large block" in result[1]["content"]

    def test_token_count_in_reference(self, duplicate_messages):
        dedup = ContentDedup()
        dedup.process(duplicate_messages)
        result = dedup.process(duplicate_messages)
        # Reference should mention token count
        assert "tokens" in result[1]["content"]

    def test_hash_deterministic(self):
        h1 = ContentDedup._hash("test content")
        h2 = ContentDedup._hash("test content")
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_different_for_different_content(self):
        h1 = ContentDedup._hash("content A")
        h2 = ContentDedup._hash("content B")
        assert h1 != h2
