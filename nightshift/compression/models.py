"""Local model pool manager. Load/unload models one at a time."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    hf_id: str
    size_mb: int
    task: str


REGISTRY: dict[str, ModelSpec] = {
    "parser": ModelSpec("ibm-granite/granite-docling-258M", 258, "document_parsing"),
    "extractor": ModelSpec("urchade/gliner_medium-v2.1", 90, "ner"),
    "embedder": ModelSpec("jinaai/jina-embeddings-v5-text-nano", 239, "embedding"),
    "summarizer": ModelSpec("google-t5/t5-small", 60, "summarization"),
    "reranker": ModelSpec("cross-encoder/ms-marco-MiniLM-L6-v2", 22, "reranking"),
}


class ModelPool:
    """Manages downloading, loading, and unloading of local models.

    Only one model loaded in memory at a time (laptop-friendly).
    """

    def __init__(self, cache_dir: str = "~/.nightshift/models") -> None:
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self._loaded: tuple[str, Any] | None = None

    def list_models(self) -> dict[str, ModelSpec]:
        return dict(REGISTRY)

    def is_downloaded(self, name: str) -> bool:
        if name not in REGISTRY:
            raise KeyError(f"Unknown model: {name}")
        model_path = self.cache_dir / name
        return model_path.exists()

    def download(self, name: str) -> Path:
        """Download model from HuggingFace. Returns local path."""
        if name not in REGISTRY:
            raise KeyError(f"Unknown model: {name}")
        model_path = self.cache_dir / name
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    def get(self, name: str) -> Any:
        """Load a model into memory. Unloads previous model first."""
        if not self.is_downloaded(name):
            raise FileNotFoundError(
                f"Model '{name}' not downloaded. Run: engine.download_models()"
            )

        if self._loaded is not None and self._loaded[0] != name:
            self._unload()

        if self._loaded is not None and self._loaded[0] == name:
            return self._loaded[1]

        model = self._load(name)
        self._loaded = (name, model)
        return model

    def _load(self, name: str) -> Any:
        """Load a specific model by name."""
        spec = REGISTRY[name]

        if name == "summarizer":
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(spec.hf_id)
            model = T5ForConditionalGeneration.from_pretrained(spec.hf_id)
            return {"model": model, "tokenizer": tokenizer}

        if name == "reranker":
            from sentence_transformers import CrossEncoder
            return CrossEncoder(spec.hf_id)

        if name == "embedder":
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(spec.hf_id)

        if name == "extractor":
            from gliner import GLiNER
            return GLiNER.from_pretrained(spec.hf_id)

        raise NotImplementedError(f"Loader not implemented for '{name}'")

    def _unload(self) -> None:
        """Free memory from currently loaded model."""
        if self._loaded is not None:
            del self._loaded
            self._loaded = None
