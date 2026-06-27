import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    from datasets import load_from_disk
except ImportError:
    print("[ERROR] Run: pip install datasets")
    sys.exit(1)

ROOT= Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"

SPLIT_FILES = {
    "train": PROC_DIR / "train_sft.jsonl",
    "val":PROC_DIR / "val_sft.jsonl",
    "test":  PROC_DIR / "test_sft.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def analyze_split(name: str, examples: list[dict], source_filter: str | None):
    if source_filter:
        examples = [e for e in examples if e.get("source") == source_filter]

    if not examples:
        print(f"\n  [{name.upper()}] No examples found.")
        return

    hop_counts= defaultdict(int)
    sources = defaultdict(int)
    complexities = defaultdict(int)
    domains = defaultdict(int)
    quality_tiers= defaultdict(int)
    sq_lengths= [] 
    total_hops= 0

    for ex in examples:
        hop_counts[ex.get("hop_count", "?")]     += 1
        sources[ex.get("source", "?")]            += 1
        complexities[ex.get("complexity", "?")]   += 1
        domains[ex.get("domain", "?")]            += 1
        quality_tiers[ex.get("quality_tier", "?")] += 1

        try:
            messages = ex.get("messages", [])
            asst= next(m["content"] for m in messages if m["role"] == "assistant")
            dep_graph = json.loads(asst)
            for hop in dep_graph:
                sq= hop.get("sub_query", "")
                sq_lengths.append(len(sq.split()))
                total_hops += 1
        except Exception:
            pass

    n = len(examples)
    print(f"\n  {'─' * 56}")
    print(f"  {name.upper()} SPLIT  —  {n:,} examples")
    print(f"  {'─' * 56}")

    print(f"\n  Hop counts:")
    for k, v in sorted(hop_counts.items()):
        bar = "█" * int(v / n * 40)
        print(f"    {k}-hop  {bar:40s}  {v:>6,}  ({v/n*100:5.1f}%)")

    print(f"\n  Sources:")
    for k, v in sorted(sources.items(), key=lambda x: -x[1]):
        bar = "█" * int(v / n * 40)
        print(f"    {k:20s}  {bar:40s}  {v:>6,}  ({v/n*100:5.1f}%)")

    print(f"\n  Complexity types:")
    for k, v in sorted(complexities.items(), key=lambda x: -x[1]):
        print(f"    {k:25s}  {v:>6,}  ({v/n*100:5.1f}%)")

    print(f"\n  Quality tiers:")
    for k, v in sorted(quality_tiers.items(), key=lambda x: -x[1]):
        print(f"    {k:15s}  {v:>6,}  ({v/n*100:5.1f}%)")

    if sq_lengths:
        avg = sum(sq_lengths) / len(sq_lengths)
        mn= min(sq_lengths)
        mx= max(sq_lengths)
        print(f"\n  Sub-query word length: min={mn}  avg={avg:.1f}  max={mx}")
        print(f"  Total sub-queries generated: {total_hops:,}")
        print(f"  Avg sub-queries per question: {total_hops/n:.2f}")


def check_hf_dataset():
    hf_path = PROC_DIR / "sft_dataset"
    if not hf_path.exists():
        print(f"\n  [HF Dataset] Not found at {hf_path.relative_to(ROOT)}")
        print(f"  Run: python scripts/build_splits.py")
        return

    print(f"\n  {'─' * 56}")
    print(f"  HuggingFace DatasetDict")
    print(f"  {'─' * 56}")
    try:
        dsd = load_from_disk(str(hf_path))
        print(f"  Splits: {list(dsd.keys())}")
        for split_name, ds in dsd.items():
            print(f"    {split_name:12s}  {len(ds):>7,} rows   columns: {ds.column_names}")
        print(f"\n  ✓ HF Dataset is valid and loadable.")
        print(f"\n  Quick load test:")
        print(f"    from datasets import load_from_disk")
        print(f"    import json")
        print(f"    dsd = load_from_disk('data/processed/sft_dataset')")
        print(f"    sample = dsd['train'][0]")
        sample = dsd["train"][0]
        msgs= json.loads(sample["messages_json"])
        print(f"    dsd['train'][0]['id']= {sample['id']}")
        print(f"    dsd['train'][0]['hop_count']= {sample['hop_count']}")
        print(f"    messages[1]['content'][:60] = {msgs[1]['content'][:60]}...")
    except Exception as e:
        print(f"  [ERROR] Could not load HF dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description="Print dataset statistics.")
    parser.add_argument("--split",  type=str, default=None,
                        help="Only analyze this split (train/val/test)")
    parser.add_argument("--source", type=str, default=None,
                        help="Filter by source (musique/hotpotqa/2wikimultihopqa/synthetic)")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  QueryDecomp — Dataset Statistics")
    print(f"{'═' * 60}")

    found_any = False
    splits_to_check = {args.split: SPLIT_FILES[args.split]} \
        if args.split and args.split in SPLIT_FILES \
        else SPLIT_FILES

    for name, path in splits_to_check.items():
        examples = load_jsonl(path)
        if examples:
            found_any = True
        analyze_split(name, examples, args.source)

    if not found_any:
        print("\n  No data found. Run the pipeline in order:")
        print("    python scripts/download_data.py")
        print("    python scripts/process_musique.py")
        print("    python scripts/process_hotpot.py")
        print("    python scripts/process_2wiki.py")
        print("    python scripts/build_splits.py")
        sys.exit(1)

    check_hf_dataset()

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()