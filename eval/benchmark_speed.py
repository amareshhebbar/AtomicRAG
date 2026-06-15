import sys
import time
import json
import argparse
import statistics
from pathlib import Path
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.utils import build_prompt
except ImportError:
    print("[ERROR] torch/transformers not installed")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"

TEST_QUESTIONS = [
    "Where was the director of Inception born?",
    "What is the capital of the country where Einstein was born?",
    "Who founded the company that makes the iPhone?",
    "What language is spoken in the country that won the 2018 FIFA World Cup?",
    "What ocean borders the country where the Amazon river starts?",
    "What is the birth year of the spouse of the 2020 US president?",
    "In which city was the author of 1984 born?",
    "What is the currency of the country that invented the telephone?",
]

def benchmark_model(
    model_id: str,
    label: str,
    n_runs: int = 50,
    max_new_tokens: int = 200,
) -> dict:

    if not Path(model_id).exists() and not model_id.startswith("Qwen"):
        return {"label": label, "error": f"model not found: {model_id}"}

    print(f"\n  Benchmarking: {label}")
    print(f"  Model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    latencies_ms     = []
    output_token_counts = []

    print(f"  Warming up (3 runs)...")
    for q in TEST_QUESTIONS[:3]:
        prompt = build_prompt(q, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10, do_sample=False,
                          pad_token_id=tokenizer.pad_token_id)

    if device == "cuda":
        torch.cuda.synchronize()

    print(f"  Running {n_runs} inference calls...")
    questions_cycle = (TEST_QUESTIONS * ((n_runs // len(TEST_QUESTIONS)) + 1))[:n_runs]

    for i, question in enumerate(questions_cycle):
        prompt = build_prompt(question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000

        n_generated = output.shape[1] - inputs["input_ids"].shape[1]
        latencies_ms.append(elapsed_ms)
        output_token_counts.append(n_generated)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_runs}  last={elapsed_ms:.0f}ms")

    p50  = statistics.median(latencies_ms)
    p95  = statistics.quantiles(latencies_ms, n=20)[18] 
    p99  = statistics.quantiles(latencies_ms, n=100)[98] 
    mean = statistics.mean(latencies_ms)
    avg_tokens = statistics.mean(output_token_counts)
    tps = avg_tokens / (mean / 1000)

    cost_per_query = (mean / 1000) * (0.22 / 3600)
    cost_per_1k    = cost_per_query * 1000

    result = {
        "label":              label,
        "model_id":           model_id,
        "n_runs":             n_runs,
        "device":             device,
        "p50_ms":             round(p50, 1),
        "p95_ms":             round(p95, 1),
        "p99_ms":             round(p99, 1),
        "mean_ms":            round(mean, 1),
        "tokens_per_second":  round(tps, 1),
        "avg_output_tokens":  round(avg_tokens, 1),
        "cost_per_1k_usd":    round(cost_per_1k, 4),
    }

    print(f"  ✓ Done")
    return result


REFERENCE_BENCHMARKS = {
    "GPT-4o (API, prompted)": {
        "label":             "GPT-4o (API, prompted)",
        "p50_ms":            1800,
        "p95_ms":            3200,
        "mean_ms":           2000,
        "tokens_per_second": 60,
        "cost_per_1k_usd":   30.0,  
        "source":            "OpenAI pricing + typical TTFT benchmarks",
    },
    "GPT-3.5-turbo (API)": {
        "label":             "GPT-3.5-turbo (API, prompted)",
        "p50_ms":            600,
        "p95_ms":            1200,
        "mean_ms":           700,
        "tokens_per_second": 90,
        "cost_per_1k_usd":   0.5,
        "source":            "OpenAI pricing",
    },
}

def print_benchmark_table(results: list[dict]):
    print(f"\n  {'─'*70}")
    print(f"  {'Model':35}  {'p50':>6}  {'p95':>6}  {'TPS':>6}  {'$/1k':>8}")
    print(f"  {'─'*70}")

    for r in results:
        if "error" in r:
            print(f"  {r['label']:35}  ERROR: {r['error']}")
            continue
        label = r["label"][:35]
        p50   = f"{r.get('p50_ms', 0):.0f}ms"
        p95   = f"{r.get('p95_ms', 0):.0f}ms"
        tps   = f"{r.get('tokens_per_second', 0):.0f}"
        cost  = f"${r.get('cost_per_1k_usd', 0):.2f}"
        print(f"  {label:35}  {p50:>6}  {p95:>6}  {tps:>6}  {cost:>8}")

    print(f"  {'─'*70}")


def print_speedup(baseline: dict, finetuned: dict):
    if "error" in baseline or "error" in finetuned:
        return
    speedup = baseline.get("mean_ms", 1) / max(finetuned.get("mean_ms", 1), 1)
    cost_savings = 1 - (finetuned.get("cost_per_1k_usd", 0) /
                        max(baseline.get("cost_per_1k_usd", 1), 0.001))
    print(f"\n  vs GPT-4o:")
    print(f"    Speedup:      {speedup:.0f}x faster")
    print(f"    Cost savings: {cost_savings:.1%} cheaper per 1k queries")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default=None,
                        help="Fine-tuned model path. Default: outputs/final")
    parser.add_argument("--baseline",   type=str, default=None,
                        help="Baseline model (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--n_runs",     type=int, default=50)
    parser.add_argument("--no_wandb",   action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\n{'═'*56}")
    print(f"  AtomicRAG — Speed Benchmark")
    print(f"  n_runs: {args.n_runs}")
    print(f"{'═'*56}")

    all_results = []

    all_results.extend(REFERENCE_BENCHMARKS.values())

    if args.baseline:
        base_result = benchmark_model(args.baseline, "Base Qwen2.5-1.5B (prompted)", args.n_runs)
        all_results.append(base_result)

    model_path = args.model or str(ROOT / "outputs" / "final")
    if Path(model_path).exists() or model_path.startswith("Qwen"):
        ft_result = benchmark_model(model_path, "AtomicRAG fine-tuned", args.n_runs)
        all_results.append(ft_result)
    else:
        print(f"  [WARN] Fine-tuned model not found at {model_path}")
        print(f"  Run training first. Showing reference numbers only.")
        ft_result = None

    print_benchmark_table(all_results)

    if ft_result and "error" not in ft_result:
        print_speedup(REFERENCE_BENCHMARKS["GPT-4o (API, prompted)"], ft_result)
        
    out_path = RESULTS_DIR / "speed_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path.relative_to(ROOT)}")

    if not args.no_wandb and ft_result and "error" not in ft_result:
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "speed/p50_ms":   ft_result["p50_ms"],
                    "speed/p95_ms":   ft_result["p95_ms"],
                    "speed/tps":      ft_result["tokens_per_second"],
                    "speed/cost_1k":  ft_result["cost_per_1k_usd"],
                })
        except Exception:
            pass


if __name__ == "__main__":
    main()