"""Autonomous research agent. Fetches papers, extracts facts, synthesizes reports."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from nightshift.agents.base import BaseAgent
from nightshift.history.knowledge import KnowledgeGraph
from nightshift.compression.summarizer import Summarizer
from nightshift.compression.reranker import Reranker

log = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    topic: str
    papers_fetched: int
    facts_extracted: int
    api_calls: int
    total_cost: float
    report: str
    citations: list[dict[str, str]]
    duration_seconds: float


class ResearchAgent(BaseAgent):
    """Autonomous research agent.

    Pipeline:
    1. Query Semantic Scholar for papers on a topic
    2. Extract key facts from abstracts locally (free)
    3. Embed facts into knowledge graph (free)
    4. Rank facts by relevance (free)
    5. Compress top findings (free)
    6. Send compressed findings to API for synthesis (paid, minimal tokens)
    7. Produce research report with citations
    """

    def __init__(
        self,
        knowledge_path: str = "./nightshift_kb",
        model: str = "gpt-5.4-mini",
        use_models: bool = True,
        **engine_kwargs: Any,
    ) -> None:
        super().__init__(**engine_kwargs)
        self.kg = KnowledgeGraph(path=knowledge_path)
        self.model = model
        self._use_models = use_models
        self._http = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "NightShift/0.1 (https://github.com/oneKn8/nightshift)"},
        )
        # Models loaded lazily from engine's compression pipeline on first use
        self._summarizer: Summarizer | None = None
        self._reranker: Reranker | None = None

    def _ensure_models(self) -> None:
        """Lazy-load T5-small + MiniLM through the engine's compression pipeline."""
        if self._summarizer is None:
            if self._use_models:
                self._summarizer = self.engine.compression._ensure_summarizer()
            else:
                self._summarizer = Summarizer(use_model=False)
        if self._reranker is None:
            if self._use_models:
                self._reranker = self.engine.compression._ensure_reranker()
            else:
                self._reranker = Reranker(use_model=False)

    def run(self, topic: str, max_papers: int = 30, depth: str = "moderate") -> ResearchResult:
        """Execute a full research run on a topic.

        Uses the engine's bandit to decide resource allocation:
        - explore: broad paper search (default for new topics)
        - deepen: more papers on known-good topic
        - synthesize: fewer papers, comprehensive report
        """
        start = time.time()
        self._ensure_models()

        # Consult bandit for action guidance
        action = self.engine.bandit.select()
        log.info(f"Starting research: {topic} (bandit suggests: {action})")

        # Adjust parameters based on bandit action
        if action == "deepen":
            max_papers = min(max_papers * 2, 100)
            depth = "comprehensive"
        elif action == "synthesize":
            max_papers = min(max_papers, 10)
            depth = "comprehensive"
        elif action == "evaluate":
            max_papers = min(max_papers, 10)
            depth = "shallow"

        # Phase 1: Check existing knowledge
        existing = self.kg.query(topic, top_k=10)
        existing_facts = [e["fact"] for e in existing if e["distance"] < 0.5]
        if existing_facts:
            log.info(f"Found {len(existing_facts)} existing facts in knowledge graph")

        # Phase 2: Fetch papers from Semantic Scholar
        papers = self._fetch_papers(topic, limit=max_papers)
        log.info(f"Fetched {len(papers)} papers")

        # Phase 3: Extract facts from abstracts (LOCAL, FREE)
        facts = []
        citations = []
        for paper in papers:
            abstract = paper.get("abstract", "")
            title = paper.get("title", "")
            year = paper.get("year", "")
            authors = ", ".join(
                a.get("name", "") for a in paper.get("authors", [])[:3]
            )
            if not abstract:
                continue

            # Extractive summarization of each abstract
            summary = self._summarizer.summarize(abstract, max_length=80)
            fact = f"[{year}] {title}: {summary}"
            facts.append(fact)
            citations.append({
                "title": title,
                "authors": authors,
                "year": str(year),
                "paperId": paper.get("paperId", ""),
            })

        log.info(f"Extracted {len(facts)} facts from paper abstracts")

        # Phase 4: Store in knowledge graph (LOCAL, FREE)
        if facts:
            self.kg.add(
                facts,
                metadata=[{"source": "semantic_scholar", "topic": topic}] * len(facts),
            )

        # Phase 5: Combine existing + new facts, rank by relevance
        all_facts = existing_facts + facts
        if not all_facts:
            return ResearchResult(
                topic=topic,
                papers_fetched=len(papers),
                facts_extracted=0,
                api_calls=0,
                total_cost=0.0,
                report="No papers found with abstracts for this topic.",
                citations=[],
                duration_seconds=time.time() - start,
            )

        ranked = self._reranker.rank(all_facts, query=topic, top_k=20)
        log.info(f"Ranked {len(ranked)} top facts")

        # Phase 6: Synthesize via API (PAID, but only ~1-2K tokens input)
        compressed = "\n".join(f"- {f}" for f in ranked)
        cost_before = self.engine.tracker.spent
        report = self._synthesize(topic, compressed, depth)
        cost_after = self.engine.tracker.spent
        api_cost = cost_after - cost_before

        # Update bandit with reward: new facts per dollar spent
        reward = len(facts) / max(api_cost, 0.0001) if facts else 0.1
        import math
        normalized_reward = min(math.log1p(reward) / 10.0, 1.0)
        self.engine.bandit.update({
            "_nightshift_action": action,
            "_nightshift_reward": normalized_reward,
        })

        duration = time.time() - start
        engine_report = self.report()

        result = ResearchResult(
            topic=topic,
            papers_fetched=len(papers),
            facts_extracted=len(facts),
            api_calls=engine_report["api_calls"],
            total_cost=engine_report["spent_usd"],
            report=report,
            citations=citations,
            duration_seconds=duration,
        )
        log.info(
            f"Research complete: {len(papers)} papers, {len(facts)} facts, "
            f"${engine_report['spent_usd']:.4f} spent, {duration:.1f}s, "
            f"action={action}, reward={normalized_reward:.3f}"
        )
        return result

    def _fetch_papers(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        """Fetch papers from Semantic Scholar API (free, no auth needed)."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,year,authors,citationCount,url",
        }
        for attempt in range(3):
            try:
                resp = self._http.get(url, params=params)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    log.info(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                papers = data.get("data", [])
                papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
                return papers
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    log.info(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                log.warning(f"Semantic Scholar API error: {e}")
                return []
            except Exception as e:
                log.warning(f"Semantic Scholar API error: {e}")
                return []
        log.warning("Semantic Scholar rate limit exceeded after 3 retries")
        return []

    def _synthesize(self, topic: str, facts: str, depth: str) -> str:
        """Send compressed facts to API for synthesis."""
        if depth == "shallow":
            instruction = "Write a 3-sentence summary of the key findings."
        elif depth == "comprehensive":
            instruction = (
                "Write a comprehensive research brief with sections: "
                "Executive Summary, Key Findings (numbered), Open Questions, "
                "and Recommendations. Be specific and cite the findings."
            )
        else:
            instruction = (
                "Write a research summary with: key findings (numbered), "
                "notable trends, and open questions. Be specific."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Synthesize the provided findings "
                    "into a clear, well-structured report. Only use information from "
                    "the provided facts. Do not hallucinate additional claims."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Findings from {len(facts.splitlines())} sources:\n\n"
                    f"{facts}\n\n"
                    f"{instruction}"
                ),
            },
        ]

        result = self.complete(messages, model=self.model, compress=False, gate=False)
        return result.get("content", "Synthesis failed.")

    def close(self) -> None:
        self._http.close()
