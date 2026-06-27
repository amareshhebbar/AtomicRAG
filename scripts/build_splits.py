import json
import hashlib
import argparse
import sys
import random
from pathlib import Path
from typing import Optional
from collections import defaultdict

try:
    from datasets import Dataset, DatasetDict
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Run: pip install datasets tqdm")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR= ROOT / "data" / "processed"
SYN_DIR= ROOT / "data" / "synthetic"

SOURCE_FILES = {
    "musique": [PROC_DIR / "musique_train.jsonl",
                        PROC_DIR / "musique_val.jsonl"],
    "hotpotqa":[PROC_DIR / "hotpotqa_train.jsonl",
                        PROC_DIR / "hotpotqa_val.jsonl"],
    "2wikimultihopqa": [PROC_DIR / "2wiki_train.jsonl",
                        PROC_DIR / "2wiki_validation.jsonl",
                        PROC_DIR / "2wiki_test.jsonl"],
    "synthetic":  list(SYN_DIR.glob("*.jsonl")) if SYN_DIR.exists() else [],
}

RANDOM_SEED = 42



def parse_assistant_json(example: dict) -> Optional[list[dict]]:
    try:
        messages = example.get("messages", [])
        assistant_content = next(
            (m["content"] for m in messages if m["role"] == "assistant"), None
        )
        if not assistant_content:
            return None
        parsed = json.loads(assistant_content)
        if not isinstance(parsed, list):
            return None
        return parsed
    except (json.JSONDecodeError, KeyError):
        return None


def check_sub_query_lengths(dep_graph: list[dict]) -> bool:
    for hop in dep_graph:
        q= hop.get("sub_query", "")
        words = len(q.split())
        if words < 3 or words > 80:
            return False
    return True


def check_min_hops(dep_graph: list[dict]) -> bool:
    return len(dep_graph) >= 2


def check_token_overlap(question: str, dep_graph: list[dict]) -> bool:
    q_tokens= set(w.lower() for w in question.split() if len(w) > 3)
    all_text= " ".join(h.get("sub_query", "") for h in dep_graph).lower()
    sq_tokens= set(all_text.split())
    overlap = q_tokens & sq_tokens
    return len(overlap) >= 1


def check_valid_depends_on(dep_graph: list[dict]) -> bool:
    hop_nums = {h.get("hop") for h in dep_graph}
    for hop in dep_graph:
        for dep in hop.get("depends_on", []):
            if dep not in hop_nums:
                return False
    return True


def passes_quality_filter(example: dict) -> tuple[bool, str]:
    dep_graph = parse_assistant_json(example)

    if dep_graph is None:
        return False, "invalid_json"

    if not check_min_hops(dep_graph):
        return False, "too_few_hops"

    if not check_sub_query_lengths(dep_graph):
        return False, "bad_sq_length"

    messages = example.get("messages", [])
    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    if not check_token_overlap(question, dep_graph):
        return False, "no_token_overlap"

    if not check_valid_depends_on(dep_graph):
        return False, "invalid_depends_on"

    return True, ""



def question_hash(example: dict) -> str:
    messages = example.get("messages", [])
    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    normalized = " ".join(question.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()



def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return examples


def load_all_sources(sources: Optional[list[str]]) -> list[dict]:
    all_examples = []
    source_counts = {}

    keys = sources if sources else list(SOURCE_FILES.keys())

    for source in keys:
        files = SOURCE_FILES.get(source, [])
        if not files:
            if source != "synthetic":
                print(f"  [WARN] No files configured for source: {source}")
            continue

        count = 0
        for fpath in files:
            if not fpath.exists():
                print(f"  [SKIP] {fpath.relative_to(ROOT)} not found")
                continue
            examples = load_jsonl(fpath)
            all_examples.extend(examples)
            count += len(examples)
            print(f"  ✓ Loaded {len(examples):>7,}  ← {fpath.relative_to(ROOT)}")

        source_counts[source] = count

    print(f"\n  Total loaded (before filtering): {len(all_examples):,}")
    print(f"  Per source: {source_counts}")
    return all_examples



def filter_and_dedup(examples: list[dict]) -> tuple[list[dict], dict]:
    skip_reasons = defaultdict(int)
    seen_hashes= set()
    clean= []

    for ex in tqdm(examples, desc="  Filtering + deduplicating"):
        ok, reason = passes_quality_filter(ex)
        if not ok:
            skip_reasons[reason] += 1
            continue

        h = question_hash(ex)
        if h in seen_hashes:
            skip_reasons["duplicate"] += 1
            continue
        seen_hashes.add(h)

        clean.append(ex)

    stats = {
        "total_input":  len(examples),
        "total_output": len(clean),
        "skipped": dict(skip_reasons),
        "kept_pct":round(len(clean) / max(len(examples), 1) * 100, 1),
    }
    return clean, stats


def split_examples(
    examples: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    rng.shuffle(examples)

    n = len(examples)
    n_test = int(n * test_ratio)
    n_val= int(n * val_ratio)
    n_train= n - n_val - n_test

    train = examples[:n_train]
    val= examples[n_train : n_train + n_val]
    test= examples[n_train + n_val :]

    return train, val, test


def write_jsonl(examples: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  → {str(path.relative_to(ROOT)):55s}  {len(examples):>7,} examples  {size_mb:.2f} MB")


def build_hf_dataset(train: list[dict], val: list[dict], test: list[dict]) -> DatasetDict:
    def serialize(batch):
        return {
            "messages_json": [
                json.dumps(msgs, ensure_ascii=False)
                for msgs in batch["messages"]
            ]
        }

    def make_ds(examples: list[dict]) -> Dataset:
        rows = {
            "id": [],
            "domain":  [],
            "source":  [],
            "hop_count": [],
            "complexity":[],
            "messages":[],
            "ground_truth_answer":  [],
            "quality_tier": [],
        }
        for ex in examples:
            rows["id"].append(ex.get("id", ""))
            rows["domain"].append(ex.get("domain", "wikipedia"))
            rows["source"].append(ex.get("source", ""))
            rows["hop_count"].append(ex.get("hop_count", 2))
            rows["complexity"].append(ex.get("complexity", "bridge_2hop"))
            rows["messages"].append(ex.get("messages", []))
            rows["ground_truth_answer"].append(ex.get("ground_truth_answer", ""))
            rows["quality_tier"].append(ex.get("quality_tier", "heuristic"))

        ds = Dataset.from_dict(rows)
        ds = ds.map(
            lambda batch: {"messages_json": [
                json.dumps(m, ensure_ascii=False) for m in batch["messages"]
            ]},
            batched=True,
            desc="Serializing messages",
        )
        ds = ds.remove_columns(["messages"])
        return ds

    splits_dict = {}
    for split_name, data in [("train", train), ("validation", val), ("test", test)]:
        if data:  
            splits_dict[split_name] = make_ds(data)
        else:
            print(f"  ⚠  Skipping empty split: {split_name}")

    return DatasetDict(splits_dict)


def print_split_stats(train, val, test):
    print(f"\n{'─' * 60}")
    print(f"  FINAL SPLIT STATISTICS")
    print(f"{'─' * 60}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        hops= defaultdict(int)
        sources= defaultdict(int)
        domains= defaultdict(int)
        for ex in split:
            hops[ex.get("hop_count", "?")]    += 1
            sources[ex.get("source", "?")]    += 1
            domains[ex.get("domain", "?")]    += 1
        print(f"\n  {name.upper()} ({len(split):,} examples)")
        print(f"    Hop counts: {dict(sorted(hops.items()))}")
        print(f"    Sources: {dict(sources)}")
        print(f"    Domains: {dict(domains)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge, filter, deduplicate, and split all processed datasets."
    )
    parser.add_argument("--max_rows",   type=int,   default=None,
                        help="Limit total examples (for local testing)")
    parser.add_argument("--val_ratio",  type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=RANDOM_SEED)
    parser.add_argument("--sources",    type=str,   default=None,
                        help="Comma-separated sources to include. "
                             "Options: musique,hotpotqa,2wikimultihopqa,synthetic")
    parser.add_argument("--no_hf",      action="store_true",
                        help="Skip saving HuggingFace Dataset format.")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None

    print(f"\n{'═' * 60}")
    print(f"  QueryDecomp — Build Final SFT Splits")
    print(f"  Sources: {sources or 'all'}")
    print(f"  Val ratio:  {args.val_ratio}")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Seed:  {args.seed}")
    print(f"{'═' * 60}")

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading source files...")
    all_examples = load_all_sources(sources)

    if not all_examples:
        print("\n[ERROR] No examples loaded. Run process_*.py scripts first.")
        sys.exit(1)

    if args.max_rows and len(all_examples) > args.max_rows:
        random.Random(args.seed).shuffle(all_examples)
        all_examples = all_examples[: args.max_rows]
        print(f"  ⚡ Limited to {args.max_rows} rows (--max_rows flag)")

    print(f"\n  Applying quality filters and deduplication...")
    clean, stats = filter_and_dedup(all_examples)

    print(f"\n  Filter results:")
    print(f"    Input: {stats['total_input']:,}")
    print(f"    Output:{stats['total_output']:,}  ({stats['kept_pct']}% kept)")
    print(f"    Skipped:  {stats['skipped']}")

    print(f"\n  Splitting into train/val/test...")
    train, val, test = split_examples(clean, args.val_ratio, args.test_ratio, args.seed)

    print(f"\n  Writing JSONL files...")
    write_jsonl(train, PROC_DIR / "train_sft.jsonl")
    write_jsonl(val,   PROC_DIR / "val_sft.jsonl")
    write_jsonl(test,  PROC_DIR / "test_sft.jsonl")

    if not args.no_hf:
        print(f"\n  Building HuggingFace DatasetDict...")
        dsd = build_hf_dataset(train, val, test)
        save_path = PROC_DIR / "sft_dataset"
        dsd.save_to_disk(str(save_path))
        print(f"  → HF Dataset: {save_path.relative_to(ROOT)}/")
        for split_name in dsd.keys():
            print(f"    {split_name:12s} {len(dsd[split_name]):,}")

    print_split_stats(train, val, test)

    print(f"\n{'═' * 60}")
    print(f"  ✓ Done. Load in training with:")
    print(f"    from datasets import load_from_disk")
    print(f"    dsd = load_from_disk('data/processed/sft_dataset')")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()