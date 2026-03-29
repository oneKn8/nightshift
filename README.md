# NightShift

**The agent runtime that makes autonomous AI research 10x cheaper.**

AI Scientist costs $15/paper. AI-Researcher burns 2-5M tokens per task. AutoResearch hits context limits by iteration 60.

NightShift fixes the engine, not the agent. Run any AI research pipeline overnight on your laptop. Pay in time, not tokens.

## The Problem

Every autonomous research system has the same architecture:

```
[Read 100 papers] --> [Send everything to LLM] --> [Get output] --> [Repeat]
```

This is O(n) token cost where n = input size. We found five structural waste patterns across AI Scientist, AI-Researcher, and AutoResearch that account for **60-80% of all token spend**:

| # | Pattern | Example | Waste |
|---|---------|---------|-------|
| 1 | **History accumulation** | Full conversation replayed every turn, never compressed | 500K-1.5M tokens/session |
| 2 | **Redundant context** | Same code/templates/fewshot re-sent with every call | 2M tokens across 50 ideas |
| 3 | **Raw data in prompts** | 500K-2M JSON agent logs dumped into LLM input | 95% unnecessary |
| 4 | **Uniform model routing** | Frontier model used for simple JSON queries | 10-20x overspend |
| 5 | **Zero cost awareness** | No tracking, no budgets, no ROI measurement | Unoptimizable |

NightShift is a runtime layer that sits between your agent and the LLM API, eliminating these patterns automatically.

## How It Works

```
Without NightShift:
  Agent sends 10M tokens to API --> $15/task

With NightShift:
  Agent sends 10M tokens
    --> Progressive compression (6,666:1 local reduction)
    --> Confidence gate (skip API if local models handle it)
    --> Content deduplication (don't re-send what hasn't changed)
    --> Budget-optimal scheduling (spend $ where ROI is highest)
    --> API receives 1.5K tokens --> $1.50/task
```

## Core Components

### 1. Progressive Compression Pipeline

Local small models (sub-300M, run on any laptop) process raw input through stages:

```
10M tokens  -->  Parse (Granite-Docling 258M)
 2M tokens  -->  Extract (GLiNER 90M)
100K tokens -->  Embed + Cluster (Jina v5 Nano 239M)
 20K tokens -->  Summarize (T5-small 60M)
  4K tokens -->  Rank + Deduplicate (MiniLM 22M)
  1.5K tokens -> API
```

Each stage is free (local inference). The API only sees the final distillate.

### 2. Confidence-Gated Router

Not every decision needs the API. The router estimates local confidence before calling out:

- **High confidence locally** (entity extraction, classification, dedup) --> skip API
- **Low confidence** (synthesis, hypothesis generation, strategic decisions) --> call API
- **Ambiguous** (novelty evaluation, quality judgment) --> call API with compressed context

Reduces API calls by 30-50%.

### 3. Content Deduplication Layer

Tracks what content has already been sent to the API. Never re-sends:
- Static templates, fewshot examples, code that hasn't changed
- Conversation history that's already been summarized
- Documents that have already been processed

Uses content hashing + diff-based updates.

### 4. Sliding Window History Manager

Replaces unbounded conversation history with:
- **Active window**: Last N turns in full detail
- **Compressed archive**: Older turns summarized periodically
- **Persistent knowledge**: Facts extracted and stored in vector DB

Prevents the context accumulation that crashes AutoResearch at iteration 60.

### 5. Token Economics Engine

Every API call is tracked:

```json
{
  "call_id": "synth_042",
  "cost_usd": 0.45,
  "input_tokens": 1847,
  "new_insights": 4,
  "cost_per_insight": 0.112,
  "confidence_before": 0.6,
  "confidence_after": 0.85
}
```

UCB1 bandit algorithm allocates budget optimally across:
- **Explore**: New research directions (high variance)
- **Deepen**: Expand promising findings (medium variance)
- **Synthesize**: Connect and output (low variance)

### 6. Persistent Knowledge Graph

Research doesn't start from zero. Cross-session memory:
- Facts, entities, and relations persist across runs
- New research queries existing knowledge first
- The system gets faster and cheaper over time

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   YOUR AGENT                      │
│        (AI Scientist, custom, any agent)          │
└──────────────────────┬───────────────────────────┘
                       │ raw LLM calls
┌──────────────────────▼───────────────────────────┐
│                  NIGHTSHIFT RUNTIME               │
│                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ Progressive  │  │  Confidence  │  │ Content │ │
│  │ Compression  │  │    Gate      │  │  Dedup  │ │
│  └──────┬──────┘  └──────┬───────┘  └────┬────┘ │
│         └────────────────┼────────────────┘      │
│                          │                        │
│  ┌───────────────────────▼──────────────────────┐│
│  │          Token Economics Engine               ││
│  │  Budget tracker | Bandit scheduler | ROI log  ││
│  └───────────────────────┬──────────────────────┘│
│                          │                        │
│  ┌───────────────────────▼──────────────────────┐│
│  │     Sliding Window History Manager            ││
│  │  Active window | Compressed archive | KG      ││
│  └───────────────────────┬──────────────────────┘│
│                          │                        │
│  ┌───────────────────────▼──────────────────────┐│
│  │     Local Model Pool                          ││
│  │  Load/unload on demand | ONNX/GGUF inference  ││
│  │  Granite-Docling | GLiNER | Jina | T5 | MiniLM││
│  └──────────────────────────────────────────────┘│
└──────────────────────┬───────────────────────────┘
                       │ optimized calls only
┌──────────────────────▼───────────────────────────┐
│                   LLM API                         │
│         (Claude, GPT-4, Gemini, local)            │
└──────────────────────────────────────────────────┘
```

## Benchmarks

Based on our analysis of three major autonomous research systems:

| Metric | AI Scientist | AI-Researcher | With NightShift |
|--------|-------------|--------------|-----------------|
| Tokens per task | ~30M | 2.2-5.8M | **0.3-0.8M** |
| Cost per task | ~$15 | ~$15 | **$1.50-3** |
| Runs overnight on laptop | No | No | **Yes** |
| Gets faster over time | No | No | **Yes** |
| Budget-optimal | No | No | **Provably (UCB1)** |

## Quick Start

```bash
pip install nightshift-runtime

# Wrap any LLM client
from nightshift import NightShift

engine = NightShift(
    local_models="auto",        # downloads sub-300M models on first run
    api_budget="$5.00",         # hard cap
    knowledge_db="./research",  # persistent across sessions
)

# Drop-in replacement for your LLM calls
response = engine.complete(
    messages=messages,
    model="claude-sonnet-4-20250514",
    compress=True,              # enable progressive compression
    gate=True,                  # enable confidence gating
)

# Or wrap an existing agent
engine.wrap(your_agent_function, budget="$10", duration="overnight")
```

## Project Structure

```
nightshift/
├── nightshift/
│   ├── __init__.py
│   ├── engine.py              # Core runtime orchestrator
│   ├── compression/
│   │   ├── pipeline.py        # Progressive compression stages
│   │   ├── dedup.py           # Content deduplication
│   │   └── models.py          # Local model pool manager
│   ├── routing/
│   │   ├── confidence.py      # Confidence-gated router
│   │   ├── gate.py            # API vs local decision logic
│   │   └── model_select.py    # Task-appropriate model selection
│   ├── history/
│   │   ├── window.py          # Sliding window manager
│   │   ├── compressor.py      # History summarization
│   │   └── knowledge.py       # Persistent knowledge graph
│   ├── economics/
│   │   ├── tracker.py         # Token cost/value tracking
│   │   ├── bandit.py          # UCB1 budget allocation
│   │   └── reports.py         # ROI analysis and logging
│   └── agents/
│       ├── research.py        # Reference research agent
│       ├── pdf.py             # Reference PDF analysis agent
│       └── base.py            # Base agent class
├── tests/
├── benchmarks/
│   ├── vs_ai_scientist.py
│   ├── vs_ai_researcher.py
│   └── vs_autoresearch.py
├── docs/
│   ├── architecture.md
│   ├── waste-analysis.md      # The 5 patterns paper
│   └── token-economics.md     # Formal framework
├── pyproject.toml
├── LICENSE                    # MIT
└── README.md
```

## Research Paper

This project is accompanied by a research paper:

**"Token Economics: Cost-Optimal Autonomous AI Research via Progressive Compression and Confidence-Gated Routing"**

Key contributions:
1. Formal framework for token economics in multi-model agent systems
2. Progressive compression pipeline achieving 6,666:1 token reduction
3. Confidence-gated routing reducing API calls by 30-50%
4. UCB1-based budget-optimal exploration with convergence guarantees
5. Empirical evaluation: comparable quality at 10x lower cost

## Why "NightShift"

Because the best research happens while you sleep. Set your budget, point it at a problem, go to bed. Wake up to results.

## Contributing

NightShift is MIT licensed and open to contributions. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Built By

[Santo](https://github.com/oneKn8) -- I build the infrastructure most engineers `import`.

## License

MIT
