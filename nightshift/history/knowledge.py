"""Persistent knowledge graph using ChromaDB. Facts persist across sessions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class KnowledgeGraph:
    """Persistent vector store for research facts. Survives across sessions."""

    def __init__(self, path: str = "./nightshift_kb") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._collection = None
        self._client = None

    def _ensure_db(self) -> Any:
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._client.get_or_create_collection(
                name="knowledge",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, facts: list[str], metadata: list[dict[str, str]] | None = None) -> int:
        """Add facts to the knowledge graph. Returns number added."""
        if not facts:
            return 0
        col = self._ensure_db()
        ids = [f"fact_{col.count() + i}" for i in range(len(facts))]
        meta = metadata or [{}] * len(facts)
        col.add(documents=facts, ids=ids, metadatas=meta)
        return len(facts)

    def query(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Query knowledge graph for relevant facts."""
        col = self._ensure_db()
        if col.count() == 0:
            return []
        results = col.query(query_texts=[question], n_results=min(top_k, col.count()))
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"fact": doc, "metadata": meta, "distance": dist})
        return out

    def count(self) -> int:
        col = self._ensure_db()
        return col.count()

    def clear(self) -> None:
        if self._client is not None:
            self._client.delete_collection("knowledge")
            self._collection = None
