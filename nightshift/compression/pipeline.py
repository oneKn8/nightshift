"""Progressive compression pipeline. 10M tokens --> 1.5K tokens locally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompressedContent:
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    content: str
    stages_applied: list[str]


class CompressionPipeline:
    """Runs raw input through local model stages to compress before API dispatch.

    Stage 1: Parse (Granite-Docling 258M) -- structure extraction
    Stage 2: Extract (GLiNER 90M) -- entities and relations
    Stage 3: Embed + Cluster (Jina v5 Nano 239M) -- semantic grouping
    Stage 4: Summarize (T5-small 60M) -- one-sentence per cluster
    Stage 5: Rank + Select (MiniLM 22M) -- top-K by relevance
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._models_loaded: dict[str, Any] = {}

    def process(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Compress message content that exceeds threshold."""
        processed = []
        for msg in messages:
            token_est = len(msg.get("content", "")) // 4  # rough estimate
            if token_est > self.config.compress_threshold:
                compressed = self._compress(msg["content"])
                processed.append({**msg, "content": compressed.content})
            else:
                processed.append(msg)
        return processed

    def _compress(self, content: str, query: str | None = None) -> CompressedContent:
        """Run content through compression stages."""
        original_tokens = len(content) // 4
        result = content
        stages: list[str] = []

        # Stage 1: Parse structure (if PDF/HTML detected)
        if self._looks_like_document(result):
            result = self._stage_parse(result)
            stages.append("parse")

        # Stage 2: Extract entities
        result = self._stage_extract(result)
        stages.append("extract")

        # Stage 3: Embed and cluster
        chunks = self._stage_embed_cluster(result)
        stages.append("embed_cluster")

        # Stage 4: Summarize clusters
        summaries = self._stage_summarize(chunks)
        stages.append("summarize")

        # Stage 5: Rank and select
        result = self._stage_rank(summaries, query)
        stages.append("rank")

        compressed_tokens = len(result) // 4
        return CompressedContent(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=original_tokens / max(compressed_tokens, 1),
            content=result,
            stages_applied=stages,
        )

    def _looks_like_document(self, content: str) -> bool:
        """Heuristic: does this look like a full document vs short text?"""
        return len(content) > 10000

    def _stage_parse(self, content: str) -> str:
        """Stage 1: Structural parsing via Granite-Docling."""
        # TODO: load granite-docling-258M, extract structure
        return content

    def _stage_extract(self, content: str) -> str:
        """Stage 2: Entity/relation extraction via GLiNER."""
        # TODO: load gliner-medium, extract entities
        return content

    def _stage_embed_cluster(self, content: str) -> list[str]:
        """Stage 3: Embed chunks and cluster by similarity."""
        # TODO: load jina-v5-nano, embed, cluster
        # For now, chunk by paragraphs
        return content.split("\n\n")

    def _stage_summarize(self, chunks: list[str]) -> list[str]:
        """Stage 4: Summarize each cluster to one sentence."""
        # TODO: load T5-small, summarize
        return chunks

    def _stage_rank(self, summaries: list[str], query: str | None) -> str:
        """Stage 5: Rank by relevance, select top-K."""
        # TODO: load MiniLM reranker, score, select top-K
        return "\n".join(summaries[:30])
