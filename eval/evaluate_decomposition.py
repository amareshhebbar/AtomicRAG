import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_FILE = ROOT / "data" / "processed" / "test_sft.jsonl"
RESULTS_DIR = ROOT / "results"

def normalize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split to tokens."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred)
    gold_tokens = normalize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def best_match_f1(pred_hops: list[dict], gold_hops: list[dict]) -> tuple[float, float, float]:
    if not pred_hops or not gold_hops:
        return 0.0, 0.0, 0.0

    pred_texts = [h["sub_query"] for h in pred_hops]
    gold_texts = [h["sub_query"] for h in gold_hops]

    gold_scores = []
    for g in gold_texts:
        best = max(token_f1(p, g) for p in pred_texts)
        gold_scores.append(best)
        
    pred_scores = []
    for p in pred_texts:
        best = max(token_f1(p, g) for g in gold_texts)
        pred_scores.append(best)

    recall    = sum(gold_scores) / len(gold_scores)
    precision = sum(pred_scores) / len(pred_scores)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def dep_graph_correct(pred_hops: list[dict], gold_hops: list[dict]) -> bool:
    if len(pred_hops) != len(gold_hops):
        return False
    for p, g in zip(
        sorted(pred_hops, key=lambda h: h["hop"]),
        sorted(gold_hops, key=lambda h: h["hop"])
    ):
        pred_has_dep = len(p.get("depends_on", [])) > 0
        gold_has_dep = len(g.get("depends_on", [])) > 0
        if pred_has_dep != gold_has_dep:
            return False
    return True

def run_model_on_examples(
    model_id: str,
    examples: list[dict],
    max_new_tokens: int = 300,
) -> list[str | None]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.utils import build_prompt
    except ImportError:
        print("[ERROR] torch/transformers not installed")
        sys.exit(1)

    print(f"  Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  ✓ Model loaded")

    outputs = []
    for i, ex in enumerate(examples):
        messages = ex.get("messages", [])
        question  = next((m["content"] for m in messages if m["role"] == "user"), "")

        prompt = build_prompt(question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            outputs.append(generated)
        except Exception as e:
            outputs.append(None)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(examples)}")

    return outputs

def evaluate(model_id: str, examples: list[dict], label: str) -> dict:
    from src.utils import parse_decomp_output

    print(f"\n  Running inference — {label}")
    raw_outputs = run_model_on_examples(model_id, examples)

    metrics = defaultdict(list)
    failures = defaultdict(int)
    per_hop_results = defaultdict(list)   

    for ex, raw in zip(examples, raw_outputs):
        messages  = ex.get("messages", [])
        gold_raw  = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        try:
            gold_hops = json.loads(gold_raw)
        except Exception:
            continue

        hop_count = len(gold_hops)

        if raw is None:
            failures["inference_error"] += 1
            metrics["json_parse_rate"].append(0)
            metrics["hop_coverage_f1"].append(0)
            metrics["hop_count_acc"].append(0)
            metrics["dep_graph_acc"].append(0)
            continue

        pred_hops = parse_decomp_output(raw)

        if pred_hops is None:
            failures["json_parse_fail"] += 1
            metrics["json_parse_rate"].append(0)
            metrics["hop_coverage_f1"].append(0)
            metrics["hop_count_acc"].append(0)
            metrics["dep_graph_acc"].append(0)
            per_hop_results[hop_count].append(0)
            continue

        metrics["json_parse_rate"].append(1)

        hop_match = int(len(pred_hops) == len(gold_hops))
        metrics["hop_count_acc"].append(hop_match)

        prec, rec, f1 = best_match_f1(pred_hops, gold_hops)
        metrics["hop_coverage_f1"].append(f1)
        metrics["hop_precision"].append(prec)
        metrics["hop_recall"].append(rec)
        per_hop_results[hop_count].append(f1)
        
        dep_ok = int(dep_graph_correct(pred_hops, gold_hops))
        metrics["dep_graph_acc"].append(dep_ok)

    results = {k: round(sum(v) / len(v), 4) for k, v in metrics.items() if v}
    results["n_examples"]  = len(examples)
    results["n_failures"]  = dict(failures)
    results["per_hop_f1"]  = {
        str(k): round(sum(v) / len(v), 4)
        for k, v in per_hop_results.items() if v
    }
    results["label"] = label
    return results

def print_results(results: dict):
    label = results.get("label", "Model")
    print(f"\n  ┌─ {label} {'─' * (50 - len(label))}┐")
    print(f"  │  n_examples:      {results.get('n_examples', 0):>6}                        │")
    print(f"  │  json_parse_rate: {results.get('json_parse_rate', 0):>6.1%}  (% valid JSON output)   │")
    print(f"  │  hop_count_acc:   {results.get('hop_count_acc', 0):>6.1%}  (right number of hops)  │")
    print(f"  │  hop_precision:   {results.get('hop_precision', 0):>6.1%}                        │")
    print(f"  │  hop_recall:      {results.get('hop_recall', 0):>6.1%}                        │")
    print(f"  │  hop_coverage_f1: {results.get('hop_coverage_f1', 0):>6.1%}  ← HEADLINE METRIC      │")
    print(f"  │  dep_graph_acc:   {results.get('dep_graph_acc', 0):>6.1%}  (parallel vs sequential)│")
    print(f"  └{'─' * 54}┘")

    per_hop = results.get("per_hop_f1", {})
    if per_hop:
        print(f"\n  Per hop-count F1:")
        for k, v in sorted(per_hop.items()):
            bar = "█" * int(v * 20)
            print(f"    {k}-hop  {bar:<20}  {v:.1%}")

    if results.get("n_failures"):
        print(f"\n  Failures: {results['n_failures']}")


def print_comparison(baseline: dict, finetuned: dict):
    print(f"\n  {'─'*54}")
    print(f"  DELTA  (fine-tuned vs baseline)")
    print(f"  {'─'*54}")
    for metric in ["json_parse_rate", "hop_count_acc", "hop_coverage_f1", "dep_graph_acc"]:
        b = baseline.get(metric, 0)
        f = finetuned.get(metric, 0)
        delta = f - b
        sign  = "+" if delta >= 0 else ""
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        print(f"  {metric:20s}  {b:.1%} → {f:.1%}   {arrow} {sign}{delta:.1%}")


def load_test_examples(max_samples: int | None) -> list[dict]:
    if not TEST_FILE.exists():
        fallback = ROOT / "data" / "processed" / "train_sft.jsonl"
        if not fallback.exists():
            print(f"[ERROR] No test data found at {TEST_FILE}")
            print("  Run: python scripts/build_splits.py")
            sys.exit(1)
        path = fallback
        print(f"  [WARN] test_sft.jsonl empty — using train_sft.jsonl for demo")
    else:
        path = TEST_FILE

    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except Exception:
                    pass

    if max_samples:
        examples = examples[:max_samples]

    print(f"  Loaded {len(examples)} test examples from {path.name}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate decomposition quality.")
    parser.add_argument("--model",       type=str, default=None,
                        help="Fine-tuned model path or HF repo. Default: outputs/final")
    parser.add_argument("--baseline",    type=str, default=None,
                        help="Baseline model to compare against")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_wandb",    action="store_true")
    args = parser.parse_args()

    model_path = args.model or str(ROOT / "outputs" / "final")
    if not Path(model_path).exists() and not model_path.startswith("Qwen"):
        print(f"[ERROR] Model not found: {model_path}")
        print("  Run training first, or pass --model Qwen/Qwen2.5-1.5B-Instruct to eval base")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\n{'═'*56}")
    print(f"  AtomicRAG — Decomposition Evaluator")
    print(f"{'═'*56}")

    examples = load_test_examples(args.max_samples)

    ft_results = evaluate(model_path, examples, label="Fine-tuned")
    print_results(ft_results)

    if args.baseline:
        base_results = evaluate(args.baseline, examples, label="Baseline")
        print_results(base_results)
        print_comparison(base_results, ft_results)

    out_path = RESULTS_DIR / "decomposition_metrics.json"
    with open(out_path, "w") as f:
        payload = {"finetuned": ft_results}
        if args.baseline:
            payload["baseline"] = base_results
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {out_path.relative_to(ROOT)}")

    if not args.no_wandb:
        try:
            import wandb
            if wandb.run:
                wandb.log({f"eval/{k}": v for k, v in ft_results.items()
                           if isinstance(v, (int, float))})
        except Exception:
            pass

    print(f"\n  Headline: hop_coverage_f1 = {ft_results.get('hop_coverage_f1', 0):.1%}")


if __name__ == "__main__":
    main()