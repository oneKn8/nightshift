"""NightShift demo: show the engine intercepting and optimizing a real LLM call."""
from dotenv import load_dotenv

load_dotenv()

from nightshift import NightShift

def main():
    engine = NightShift(api_budget=1.0)

    print("=" * 60)
    print("NIGHTSHIFT DEMO")
    print("=" * 60)

    # --- Call 1: Simple call ---
    print("\n[Call 1] Simple question to gpt-5.4-mini...")
    msgs = [
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": "What is the transformer architecture in one sentence?"},
    ]
    r1 = engine.complete(msgs, model="gpt-5.4-mini", gate=False)
    print(f"  Response: {r1['content']}")
    print(f"  Tokens: in={r1['input_tokens']}, out={r1['output_tokens']}")

    # --- Call 2: Duplicate content (dedup should kick in) ---
    big_context = (
        "The transformer architecture was introduced in the paper 'Attention Is All You Need' "
        "by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequences "
        "in parallel rather than sequentially like RNNs. The key innovation is multi-head "
        "attention which allows the model to attend to different representation subspaces. "
    ) * 20  # ~800 chars repeated = big block

    print("\n[Call 2] First call with large context block...")
    msgs2 = [
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": big_context + "\n\nSummarize the above in one sentence."},
    ]
    r2 = engine.complete(msgs2, model="gpt-5.4-mini", gate=False)
    print(f"  Response: {r2['content']}")
    print(f"  Tokens: in={r2['input_tokens']}, out={r2['output_tokens']}")

    print("\n[Call 3] SAME context again (dedup should replace with reference)...")
    msgs3 = [
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": big_context + "\n\nWhat year was this published?"},
    ]
    r3 = engine.complete(msgs3, model="gpt-5.4-mini", gate=False)
    print(f"  Response: {r3['content']}")
    print(f"  Tokens: in={r3['input_tokens']}, out={r3['output_tokens']}")

    # --- Report ---
    report = engine.report()
    print("\n" + "=" * 60)
    print("TOKEN ECONOMICS REPORT")
    print("=" * 60)
    print(f"  Budget:          ${report['budget_usd']:.4f}")
    print(f"  Spent:           ${report['spent_usd']:.4f}")
    print(f"  Remaining:       ${report['remaining_usd']:.4f}")
    print(f"  API calls:       {report['api_calls']}")
    print(f"  Local handles:   {report['local_handles']}")
    print(f"  Input tokens:    {report['total_input_tokens']}")
    print(f"  Output tokens:   {report['total_output_tokens']}")
    print(f"  Tokens saved:    {report['tokens_saved_by_local']}")
    print(f"  Avg cost/call:   ${report['avg_cost_per_call']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
