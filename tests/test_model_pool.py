"""Tests for local model pool manager."""
import pytest
from nightshift.compression.models import ModelPool, ModelSpec


class TestModelPool:
    def test_list_models(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_models")
        specs = pool.list_models()
        assert "summarizer" in specs
        assert "reranker" in specs
        assert "embedder" in specs

    def test_model_spec_structure(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_models")
        spec = pool.list_models()["summarizer"]
        assert isinstance(spec, ModelSpec)
        assert spec.hf_id == "google-t5/t5-small"
        assert spec.size_mb > 0

    def test_is_downloaded_false_initially(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_nonexistent")
        assert pool.is_downloaded("summarizer") is False

    def test_get_undownloaded_raises(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_nonexistent")
        with pytest.raises(FileNotFoundError):
            pool.get("summarizer")

    def test_unknown_model_raises(self):
        pool = ModelPool(cache_dir="/tmp/nightshift_test_models")
        with pytest.raises(KeyError, match="Unknown model"):
            pool.is_downloaded("nonexistent_model")

    def test_download_creates_dir(self, tmp_path):
        pool = ModelPool(cache_dir=str(tmp_path / "models"))
        path = pool.download("summarizer")
        assert path.exists()
        assert pool.is_downloaded("summarizer") is True
