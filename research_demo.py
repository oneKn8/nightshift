"""NightShift Research Agent Demo.

Run autonomous research on any topic. Uses local models for extraction,
API only for synthesis. Tracks every dollar.
"""
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

from nightshift.agents.research import ResearchAgent

def main():
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "token optimization for LLM agents"

    print("=" * 70)
    print(f"NIGHTSHIFT RESEARCH AGENT")
    print(f"Topic: {topic}")
    print("=" * 70)

    agent = ResearchAgent(
        api_budget=1.0,
        model="gpt-5.4-mini",
        knowledge_path="./nightshift_kb",
    )

    result = agent.run(topic, max_papers=30, depth="moderate")

    print("\n" + "=" * 70)
    print("RESEARCH REPORT")
    print("=" * 70)
    print(result.report)

    print("\n" + "-" * 70)
    print("CITATIONS")
    print("-" * 70)
    for i, cite in enumerate(result.citations[:15], 1):
        print(f"  [{i}] {cite['title']} ({cite['year']}) - {cite['authors']}")

    print("\n" + "-" * 70)
    print("METRICS")
    print("-" * 70)
    print(f"  Papers fetched:    {result.papers_fetched}")
    print(f"  Facts extracted:   {result.facts_extracted}")
    print(f"  API calls:         {result.api_calls}")
    print(f"  Total cost:        ${result.total_cost:.4f}")
    print(f"  Duration:          {result.duration_seconds:.1f}s")
    print(f"  Knowledge graph:   {agent.kg.count()} facts stored")
    print("=" * 70)

    agent.close()


if __name__ == "__main__":
    main()
