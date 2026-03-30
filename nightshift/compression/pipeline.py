"""Progressive compression pipeline. 10M tokens --> 1.5K tokens locally."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from nightshift.compression.summarizer import Summarizer
from nightshift.compression.reranker import Reranker

log = logging.getLogger(__name__)


@dataclass
class CompressedContent:
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    content: str
    stages_applied: list[str]


class CompressionPipeline:
    """Runs raw input through local model stages to compress before API dispatch.

    Stages:
    1. Chunk (split into paragraphs/sections)
    2. Embed + Cluster (sentence-transformers, group similar chunks)
    3. Summarize (T5-small per cluster)
    4. Rank + Select (MiniLM cross-encoder, top-K by relevance)
    """

    def __init__(self, config: Any, use_models: bool = True) -> None:
        self.config = config
        self._use_models = use_models
        self._summarizer: Summarizer | None = None
        self._reranker: Reranker | None = None
        self._embedder: Any = None

    def _ensure_summarizer(self) -> Summarizer:
        if self._summarizer is None:
            if self._use_models:
                try:
                    from transformers import T5ForConditionalGeneration, T5Tokenizer
                    log.info("Loading T5-small summarizer...")
                    tok = T5Tokenizer.from_pretrained("google-t5/t5-small")
                    mdl = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
                    self._summarizer = Summarizer(
                        use_model=True, model={"model": mdl, "tokenizer": tok}
                    )
                    log.info("T5-small loaded")
                except Exception as e:
                    log.warning(f"Failed to load T5-small, using fallback: {e}")
                    self._summarizer = Summarizer(use_model=False)
            else:
                self._summarizer = Summarizer(use_model=False)
        return self._summarizer

    def _ensure_reranker(self) -> Reranker:
        if self._reranker is None:
            if self._use_models:
                try:
                    from sentence_transformers import CrossEncoder
                    log.info("Loading MiniLM reranker...")
                    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
                    self._reranker = Reranker(use_model=True, model=ce)
                    log.info("MiniLM loaded")
                except Exception as e:
                    log.warning(f"Failed to load MiniLM, using fallback: {e}")
                    self._reranker = Reranker(use_model=False)
            else:
                self._reranker = Reranker(use_model=False)
        return self._reranker

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            if self._use_models:
                try:
                    from sentence_transformers import SentenceTransformer
                    log.info("Loading embedding model...")
                    self._embedder = SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2"
                    )
                    log.info("Embedder loaded")
                except Exception as e:
                    log.warning(f"Failed to load embedder: {e}")
                    self._embedder = None
        return self._embedder

    def process(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Compress message content that exceeds threshold."""
        processed = []
        for msg in messages:
            token_est = len(msg.get("content", "")) // 4
            if token_est > self.config.compress_threshold:
                compressed = self._compress(msg["content"])
                processed.append({**msg, "content": compressed.content})
            else:
                processed.append(msg)
        return processed

    def _compress(
        self, content: str, query: str | None = None, target_chunks: int = 30
    ) -> CompressedContent:
        """Run content through compression stages."""
        original_tokens = len(content) // 4
        stages: list[str] = []

        # Stage 1: Chunk by paragraphs
        chunks = self._stage_chunk(content)
        stages.append("chunk")

        # Stage 2: Embed + cluster (deduplicate similar chunks)
        chunks = self._stage_embed_dedup(chunks)
        stages.append("embed_dedup")

        # Stage 3: Summarize each chunk
        summarizer = self._ensure_summarizer()
        summaries = summarizer.summarize_chunks(chunks, max_length=60)
        stages.append("summarize")

        # Stage 4: Rank by relevance and select top-K
        reranker = self._ensure_reranker()
        ranked = reranker.rank(summaries, query=query, top_k=target_chunks)
        stages.append("rank")

        result = "\n".join(ranked)
        compressed_tokens = len(result) // 4
        return CompressedContent(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=original_tokens / max(compressed_tokens, 1),
            content=result,
            stages_applied=stages,
        )

    def _stage_chunk(self, content: str) -> list[str]:
        """Split content into paragraph-level chunks."""
        paragraphs = content.split("\n\n")
        chunks = []
        for p in paragraphs:
            p = p.strip()
            if len(p) < 20:
                continue
            # Split very long paragraphs further
            if len(p) > 2000:
                sentences = p.replace(". ", ".\n").split("\n")
                current = ""
                for s in sentences:
                    if len(current) + len(s) > 1000:
                        if current:
                            chunks.append(current.strip())
                        current = s
                    else:
                        current += " " + s
                if current.strip():
                    chunks.append(current.strip())
            else:
                chunks.append(p)
        return chunks if chunks else [content[:2000]]

    def _stage_embed_dedup(self, chunks: list[str]) -> list[str]:
        """Remove near-duplicate chunks using embedding similarity."""
        if len(chunks) <= 5:
            return chunks

        embedder = self._ensure_embedder()
        if embedder is None:
            # Fallback: simple string dedup
            seen = set()
            deduped = []
            for c in chunks:
                key = c[:100].lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(c)
            return deduped

        import numpy as np

        embeddings = embedder.encode(chunks, show_progress_bar=False)
        keep = [True] * len(chunks)
        for i in range(len(chunks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(chunks)):
                if not keep[j]:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                ))
                if sim > 0.92:
                    keep[j] = False

        return [c for c, k in zip(chunks, keep) if k]
