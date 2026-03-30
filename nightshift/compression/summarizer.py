"""Summarization stage using T5-small or extractive fallback."""
from __future__ import annotations

from typing import Any


class Summarizer:
    """Summarize text chunks. Uses T5-small when available, extractive fallback otherwise."""

    def __init__(self, use_model: bool = True, model: Any = None) -> None:
        self._use_model = use_model and model is not None
        self._model = model

    def summarize(self, text: str, max_length: int = 100) -> str:
        if not text or len(text) < max_length:
            return text
        if self._use_model:
            return self._model_summarize(text, max_length)
        return self._extractive_summarize(text, max_length)

    def summarize_chunks(self, chunks: list[str], max_length: int = 100) -> list[str]:
        return [self.summarize(c, max_length) for c in chunks]

    def _model_summarize(self, text: str, max_length: int) -> str:
        """Use T5-small for abstractive summarization."""
        model = self._model["model"]
        tokenizer = self._model["tokenizer"]
        inputs = tokenizer(
            f"summarize: {text}",
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=2,
            early_stopping=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _extractive_summarize(self, text: str, max_length: int) -> str:
        """Extractive fallback: take first N sentences."""
        sentences = text.replace(". ", ".\n").split("\n")
        result = []
        total = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if total + len(s) > max_length * 4:
                break
            result.append(s)
            total += len(s)
        return " ".join(result) if result else text[:max_length * 4]
