"""Tests for reranking stage."""
from nightshift.compression.reranker import Reranker


class TestReranker:
    def test_rank_returns_top_k(self):
        r = Reranker(use_model=False)
        chunks = [f"chunk {i}" for i in range(20)]
        result = r.rank(chunks, query="relevant topic", top_k=5)
        assert len(result) == 5

    def test_rank_without_query_returns_first_k(self):
        r = Reranker(use_model=False)
        chunks = [f"chunk {i}" for i in range(10)]
        result = r.rank(chunks, query=None, top_k=3)
        assert len(result) == 3

    def test_rank_fewer_than_k(self):
        r = Reranker(use_model=False)
        chunks = ["only one"]
        result = r.rank(chunks, query="test", top_k=5)
        assert len(result) == 1

    def test_empty_chunks(self):
        r = Reranker(use_model=False)
        assert r.rank([], query="test", top_k=5) == []

    def test_keyword_overlap_ranking(self):
        r = Reranker(use_model=False)
        chunks = [
            "machine learning is great",
            "the weather is nice today",
            "deep learning and machine learning models",
        ]
        result = r.rank(chunks, query="machine learning", top_k=2)
        # chunk with more keyword overlap should rank higher
        assert "machine learning" in result[0]
