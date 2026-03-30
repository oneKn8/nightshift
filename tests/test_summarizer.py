"""Tests for T5-based summarization stage."""
from nightshift.compression.summarizer import Summarizer


class TestSummarizer:
    def test_summarize_returns_shorter_text(self):
        s = Summarizer(use_model=False)
        text = "The transformer architecture revolutionized NLP. " * 20
        result = s.summarize(text, max_length=50)
        assert len(result) < len(text)

    def test_summarize_list_of_chunks(self):
        s = Summarizer(use_model=False)
        chunks = [
            "First finding about model compression techniques. " * 10,
            "Second finding about quantization methods. " * 10,
            "Third finding about knowledge distillation. " * 10,
        ]
        results = s.summarize_chunks(chunks, max_length=30)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_empty_input(self):
        s = Summarizer(use_model=False)
        assert s.summarize("") == ""

    def test_short_input_returned_as_is(self):
        s = Summarizer(use_model=False)
        short = "Brief text."
        assert s.summarize(short) == short
