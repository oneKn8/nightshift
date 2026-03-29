# NightShift Architecture Specification

## Overview

NightShift is an agent runtime engine that optimizes token economics for autonomous AI systems. It sits between any AI agent and the LLM API, reducing token spend by 60-90% through five mechanisms: progressive compression, confidence-gated routing, content deduplication, sliding window history management, and budget-optimal scheduling.

## Design Principles

1. **Infrastructure, not application.** NightShift is a runtime layer. It doesn't know what your agent does. It optimizes how your agent talks to LLMs.
2. **Zero-copy optimization.** The agent's code doesn't change. NightShift intercepts and optimizes the call path transparently.
3. **Provably optimal.** Budget allocation uses UCB1 bandit with convergence guarantees, not heuristics.
4. **Laptop-first.** All local models run on CPU, sub-300M parameters, under 2GB total disk.
5. **Gets smarter over time.** Persistent knowledge graph amortizes cost across sessions.

## System Architecture

```
Agent Process
    │
    │  engine.complete(messages, model, ...)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    NightShift Engine                  │
│                                                      │
│  1. INTERCEPT                                        │
│     Parse incoming messages                          │
│     Hash content for dedup check                     │
│     Estimate input token count                       │
│                                                      │
│  2. COMPRESS                                         │
│     If content is raw text/docs:                     │
│       Stage 1: Parse structure (Granite-Docling)     │
│       Stage 2: Extract entities (GLiNER)             │
│       Stage 3: Embed + cluster (Jina v5 Nano)        │
│       Stage 4: Summarize clusters (T5-small)         │
│       Stage 5: Rank + select (MiniLM)                │
│     Replace raw content with compressed version      │
│                                                      │
│  3. DEDUPLICATE                                      │
│     Check content hash against sent_cache            │
│     If previously sent and unchanged: strip          │
│     If changed: send diff only                       │
│     Track: static content, fewshot examples,         │
│            code templates, system prompts             │
│                                                      │
│  4. MANAGE HISTORY                                   │
│     If msg_history length > window_size:             │
│       Summarize oldest N messages locally             │
│       Replace with compressed summary                │
│       Extract facts to knowledge graph               │
│     Inject relevant knowledge graph context          │
│                                                      │
│  5. GATE                                             │
│     Estimate confidence for local handling           │
│     If confidence > threshold:                       │
│       Route to local model (T5, Gemma 270M, etc.)    │
│       Return result without API call                 │
│     If confidence < threshold:                       │
│       Proceed to API                                 │
│                                                      │
│  6. SCHEDULE                                         │
│     Check remaining budget                           │
│     UCB1 bandit scores: explore vs deepen vs synth   │
│     If budget exhausted: return best-effort local    │
│     Log token economics for this call                │
│                                                      │
│  7. DISPATCH                                         │
│     Send optimized messages to API                   │
│     Record: cost, tokens, response quality           │
│     Update bandit statistics                         │
│     Update knowledge graph with new facts            │
│     Return response to agent                         │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Progressive Compression Pipeline

**Purpose:** Reduce raw input from millions of tokens to thousands before API dispatch.

**Stages:**

| Stage | Model | Input | Output | Compression |
|-------|-------|-------|--------|-------------|
| Parse | Granite-Docling 258M | Raw PDF/HTML/text | Structured markdown | ~2:1 |
| Extract | GLiNER 90M | Structured text | Entities + relations | ~20:1 |
| Embed | Jina v5 Nano 239M | Text chunks | Vectors in local DB | N/A (index) |
| Summarize | T5-small 60M | Relevant chunks | One-sentence summaries | ~50:1 |
| Rank | MiniLM 22M | Summaries | Top-K by relevance | ~5:1 |

**Aggregate compression:** 10M tokens --> 1.5K tokens (6,666:1)

**Model loading strategy:** Models are loaded on-demand, one at a time (memory constraint). After processing, unloaded. The pipeline processes in batch: all parsing first, then all extraction, etc. This minimizes model swap overhead.

**Interface:**
```python
class CompressionPipeline:
    def compress(
        self,
        content: str | list[str],
        query: str | None = None,     # optional relevance filter
        target_tokens: int = 1500,    # target output size
        stages: list[str] = ["all"],  # which stages to run
    ) -> CompressedContent:
        ...
```

### 2. Confidence-Gated Router

**Purpose:** Skip API calls when local models can handle the task with sufficient confidence.

**Decision logic:**

```
Task classification (by ModernBERT or heuristic):
  EXTRACTION tasks (NER, parsing, classification)
    --> Always local. Never needs API.

  RETRIEVAL tasks (search, similarity, reranking)
    --> Always local. Embedding + rerank is local-optimal.

  GENERATION tasks (writing, synthesis, hypothesis)
    --> Check complexity estimate
    --> Simple (template-fillable, short output): try local first
    --> Complex (multi-hop reasoning, long output): route to API

  EVALUATION tasks (novelty check, quality judgment)
    --> Compute embedding similarity to known items
    --> If clearly novel (distance > threshold): return locally
    --> If clearly duplicate (distance < threshold): return locally
    --> If ambiguous: route to API with compressed context
```

**Confidence estimation:**
- For extraction: GLiNER's entity confidence scores
- For retrieval: cosine similarity scores from embedding search
- For generation: perplexity of local model's output (high perplexity = low confidence)
- For evaluation: embedding distance distribution analysis

**Fallback:** If local result is used but downstream quality degrades, the router learns to lower its confidence threshold for that task type (online learning via exponential moving average).

### 3. Content Deduplication Layer

**Purpose:** Never re-send content the API has already seen.

**Mechanism:**

```python
class ContentDedup:
    def __init__(self):
        self.sent_cache: dict[str, ContentHash] = {}
        # Maps content_hash -> (first_sent_call_id, token_count)

    def process(self, messages: list[Message]) -> list[Message]:
        optimized = []
        for msg in messages:
            content_hash = hash(msg.content)
            if content_hash in self.sent_cache:
                # Already sent this exact content before
                # Replace with reference: "[Previously provided context, unchanged]"
                optimized.append(msg.with_content(
                    f"[Reference: content from call {self.sent_cache[content_hash].call_id}, "
                    f"{self.sent_cache[content_hash].tokens} tokens, unchanged]"
                ))
            else:
                self.sent_cache[content_hash] = ContentHash(...)
                optimized.append(msg)
        return optimized
```

**Handles the specific wastes found:**
- AI Scientist: experiment.py (4K tokens) re-sent 500 times --> sent once, referenced after
- AI Scientist: fewshot paper (17K tokens) re-sent 9 times --> sent once
- AI-Researcher: GitHub search results re-embedded in 5 agents --> sent once

### 4. Sliding Window History Manager

**Purpose:** Replace unbounded conversation history growth with bounded, compressed history.

**Three-tier memory:**

```
ACTIVE WINDOW (last N turns, full detail)
  Size: configurable, default 5 turns
  Content: exact messages as-is
  Purpose: recent context for coherent continuation

COMPRESSED ARCHIVE (older turns, summarized)
  Size: grows slowly (one summary paragraph per 10 turns)
  Content: locally-generated summaries of older conversations
  Purpose: long-term context without token explosion
  Compression: T5-small or local Gemma 270M generates summary

KNOWLEDGE GRAPH (extracted facts, persistent)
  Size: grows across sessions
  Content: entity-relation triples, key findings, decisions
  Purpose: cross-session memory, never re-discover known facts
  Storage: local vector DB (VectorVault or ChromaDB)
```

**Compression trigger:** When history exceeds `window_size` tokens, the oldest N turns are:
1. Summarized by local model (T5-small for short summaries, Gemma 270M for detailed)
2. Key facts extracted by GLiNER and stored in knowledge graph
3. Original messages replaced with summary in the history array

**This directly fixes:**
- AutoResearch: context crash at iteration 60 --> now runs indefinitely
- AI-Researcher: ML Agent accumulating 1.5M tokens --> capped at window_size
- AI Scientist: reflection loops doubling history each round --> windowed

### 5. Token Economics Engine

**Purpose:** Track cost/value of every API call. Allocate budget optimally.

**Per-call tracking:**
```python
@dataclass
class CallMetrics:
    call_id: str
    timestamp: float
    task_type: str          # explore | deepen | synthesize | evaluate
    input_tokens: int
    output_tokens: int
    cost_usd: float
    # Value metrics (computed post-hoc)
    new_facts: int          # facts added to knowledge graph
    confidence_delta: float # confidence change after this call
    novelty_score: float    # how novel was the output
    # Derived
    cost_per_insight: float
    roi: float              # value / cost
```

**Budget allocation via UCB1:**

The engine treats budget allocation as a multi-armed bandit problem:
- Arms: {explore_new_direction, deepen_existing, synthesize, evaluate}
- Reward: information_gained / tokens_spent
- Selection: UCB1 score = mean_reward + c * sqrt(ln(total_pulls) / arm_pulls)

Early in research: UCB1 favors exploration (high uncertainty).
Mid research: shifts to deepening (known-good directions).
Late research: shifts to synthesis (diminishing exploration returns).

**Budget enforcement:**
```python
if remaining_budget < estimated_call_cost:
    # Graceful degradation
    if can_handle_locally(task):
        return local_result
    else:
        return engine.synthesize_best_effort(knowledge_graph)
```

### 6. Local Model Pool

**Purpose:** Manage loading/unloading of local models on memory-constrained devices.

**Strategy:** Only one model loaded at a time. Pipeline processes in batch stages to minimize swaps.

**Supported formats:**
- ONNX Runtime (default, broadest compatibility)
- GGUF via llama.cpp (for generative models like Gemma 270M)
- PyTorch (fallback)

**Model manifest:**
```python
MODELS = {
    "parser":    {"id": "ibm-granite/granite-docling-258M",    "size_mb": 258, "format": "onnx"},
    "extractor": {"id": "urchade/gliner_medium-v2.1",          "size_mb": 90,  "format": "onnx"},
    "embedder":  {"id": "jinaai/jina-embeddings-v5-text-nano", "size_mb": 239, "format": "onnx"},
    "summarizer":{"id": "google-t5/t5-small",                  "size_mb": 60,  "format": "onnx"},
    "reranker":  {"id": "cross-encoder/ms-marco-MiniLM-L6-v2", "size_mb": 22,  "format": "onnx"},
    "classifier":{"id": "answerdotai/ModernBERT-base",         "size_mb": 149, "format": "onnx"},
    "generator": {"id": "google/gemma-3-270m-it",              "size_mb": 270, "format": "gguf"},
}
```

**Auto-download:** Models are downloaded on first use and cached in `~/.nightshift/models/`.

## Integration Patterns

### Pattern A: Drop-in LLM Client Replacement

```python
from nightshift import NightShift

engine = NightShift(api_budget="$5")

# Instead of: client.chat.completions.create(...)
response = engine.complete(
    messages=[{"role": "user", "content": massive_prompt}],
    model="claude-sonnet-4-20250514",
)
# NightShift compresses, deduplicates, gates, and dispatches
```

### Pattern B: Agent Wrapper

```python
from nightshift import NightShift

engine = NightShift(api_budget="$10", duration="8h")

@engine.wrap
def my_research_agent(topic: str):
    # Your agent code here
    # All LLM calls inside are intercepted and optimized
    ...

my_research_agent("efficient small model architectures 2026")
# Runs overnight within budget
```

### Pattern C: Middleware for Existing Systems

```python
from nightshift import NightShiftMiddleware

# Patch an existing OpenAI client
import openai
openai.ChatCompletion.create = NightShiftMiddleware(
    original=openai.ChatCompletion.create,
    budget="$5",
)
# All calls through this client are now optimized
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Compression ratio (raw text) | > 1000:1 |
| API call reduction | 30-50% via confidence gating |
| History growth | O(1) instead of O(n) via sliding window |
| Model swap latency | < 2s per swap |
| Total local model disk | < 2GB |
| RAM per model (peak) | < 1.5GB |
| Overnight session length | Unlimited (no context degradation) |
| Cross-session knowledge retention | 100% (persistent graph) |

## Testing Strategy

1. **Unit tests:** Each component (compression, routing, dedup, history, economics) tested independently
2. **Integration tests:** Full pipeline with mock LLM API
3. **Benchmark suite:** Run against AI Scientist, AI-Researcher, AutoResearch baselines
   - Metric: same output quality at lower token cost
   - Comparison: tokens spent, API calls made, wall-clock time, output quality (human eval)
4. **Overnight stress test:** 100+ iteration session without context degradation

## Implementation Phases

### Phase 1: Core Engine (Week 1-2)
- Engine class with intercept/dispatch
- Content deduplication
- Sliding window history manager
- Basic token tracking

### Phase 2: Compression Pipeline (Week 2-3)
- Local model pool with on-demand loading
- Progressive compression stages
- Batch processing pipeline

### Phase 3: Confidence Gating (Week 3-4)
- Task classification
- Confidence estimation per task type
- Routing logic with fallback

### Phase 4: Token Economics (Week 4-5)
- UCB1 bandit implementation
- Budget enforcement
- ROI tracking and reporting

### Phase 5: Benchmarks + Paper (Week 5-8)
- Benchmark suite against all three systems
- Research paper draft
- Blog post for launch
