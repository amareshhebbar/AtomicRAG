import json
import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    from datasets import Dataset, DatasetDict, load_dataset
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Run: pip install datasets pandas tqdm")
    sys.exit(1)

ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw" / "musique"
OUT_DIR    = ROOT / "data" / "processed"

SYSTEM_PROMPT = (
    "You are a query decomposition engine. "
    "Given a complex multi-hop question, decompose it into atomic sub-queries. "
    "Each sub-query must be answerable by retrieving a SINGLE document chunk. "
    "Return ONLY a valid JSON array. "
    "Each element must have exactly these keys:\n"
    "  hop        (int)       — 1-indexed hop number\n"
    "  sub_query  (string)    — the atomic question for this hop\n"
    "  depends_on (list[int]) — hop numbers this query depends on (empty for hop 1)\n"
    "No explanation. No markdown. Only the JSON array."
)


def decomposition_to_dependency_graph(decomp: list[dict]) -> list[dict]:
    graph = []
    for i, hop in enumerate(decomp):
        hop_num    = i + 1
        depends_on = [] if i == 0 else [i]   
        graph.append({
            "hop":        hop_num,
            "sub_query":  hop["question"].strip(),
            "depends_on": depends_on,
        })
    return graph


def build_messages(question: str, dep_graph: list[dict]) -> list[dict]:
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question.strip()},
        {"role": "assistant", "content": json.dumps(dep_graph, ensure_ascii=False)},
    ]


def infer_complexity(row: dict) -> str:
    n_hops = len(row.get("question_decomposition", []))
    if n_hops == 2:
        return "bridge_2hop"
    elif n_hops == 3:
        return "bridge_3hop"
    elif n_hops == 4:
        return "bridge_4hop"
    return f"bridge_{n_hops}hop"


def process_row(row: dict, idx: int, split: str) -> Optional[dict]:
    if not row.get("answerable", True):
        return None

    decomp_raw = row.get("question_decomposition", [])

    if isinstance(decomp_raw, dict):
        keys = list(decomp_raw.keys())
        n_items = len(decomp_raw[keys[0]]) if keys else 0
        decomp = [
            {k: decomp_raw[k][i] for k in keys}
            for i in range(n_items)
        ]
    elif hasattr(decomp_raw, "tolist"):
        decomp = decomp_raw.tolist()
    else:
        decomp = list(decomp_raw) if decomp_raw is not None else []

    if not decomp:
        return None

    for hop in decomp:
        if not isinstance(hop, dict):
            return None
        q = hop.get("question", "").strip()
        if not q:
            return None

    dep_graph = decomposition_to_dependency_graph(decomp)
    example = {
        "id":                   f"musique_{split}_{idx:06d}",
        "domain":               "wikipedia",
        "source":               "musique",
        "hop_count":            len(decomp),
        "complexity":           infer_complexity(row),
        "messages":             build_messages(row["question"], dep_graph),
        "ground_truth_answer":  row.get("answer", ""),
        "supporting_facts":     [
            {"id": h.get("id", ""), "answer": h.get("answer", "")}
            for h in decomp
        ],
    }
    return example

def process_split(split: str, max_rows: Optional[int]) -> list[dict]:
    parquet_path = RAW_DIR / f"{split}.parquet"
    if not parquet_path.exists():
        print(f"  [SKIP] {parquet_path} not found — run download_data.py first")
        return []

    print(f"\n  Loading {split} split from {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)

    if max_rows:
        df = df.head(max_rows)
        print(f"  ⚡ Limited to {max_rows} rows (--max_rows flag)")

    print(f"  Rows loaded: {len(df):,}")

    examples   = []
    skipped    = 0
    hop_counts = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {split}"):
        result = process_row(row.to_dict(), idx, split)
        if result is None:
            skipped += 1
            continue
        examples.append(result)
        hc = result["hop_count"]
        hop_counts[hc] = hop_counts.get(hc, 0) + 1

    print(f"  ✓ Converted:   {len(examples):,}")
    print(f"  ✗ Skipped:     {skipped:,}  (unanswerable or malformed)")
    print(f"  Hop distribution: {dict(sorted(hop_counts.items()))}")
    return examples


def write_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    size_kb = path.stat().st_size / 1024
    print(f"  → Written: {path.relative_to(ROOT)}  ({len(examples):,} examples, {size_kb:.1f} KB)")


def write_hf_dataset(train: list[dict], val: list[dict], out_dir: Path):
    def flatten_messages(examples):
        """Convert messages list to string for HF Dataset storage."""
        return {
            "messages_json": [json.dumps(m, ensure_ascii=False) for m in examples["messages"]]
        }

    train_ds = Dataset.from_list(train)
    val_ds   = Dataset.from_list(val)

    train_ds = train_ds.map(
        lambda ex: {"messages_json": json.dumps(ex["messages"])},
        desc="Serializing messages"
    )
    val_ds = val_ds.map(
        lambda ex: {"messages_json": json.dumps(ex["messages"])},
        desc="Serializing messages"
    )

    dsd = DatasetDict({"train": train_ds, "validation": val_ds})
    save_path = out_dir / "musique_hf"
    dsd.save_to_disk(str(save_path))
    print(f"  → HF Dataset saved: {save_path.relative_to(ROOT)}/")



def main():
    parser = argparse.ArgumentParser(description="Process MuSiQue dataset into querydecomp format.")
    parser.add_argument("--max_rows",  type=int, default=None,
                        help="Limit rows per split (for local testing).")
    parser.add_argument("--no_hf",    action="store_true",
                        help="Skip saving HuggingFace Dataset format.")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  MuSiQue Processor")
    print(f"  Input:  {RAW_DIR.relative_to(ROOT)}/")
    print(f"  Output: {OUT_DIR.relative_to(ROOT)}/")
    print(f"{'═' * 60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_examples = process_split("train",      args.max_rows)
    val_examples   = process_split("validation", args.max_rows)

    if not train_examples and not val_examples:
        print("\n[ERROR] No examples processed. Run download_data.py first.")
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print(f"  Writing output files...")

    write_jsonl(train_examples, OUT_DIR / "musique_train.jsonl")
    write_jsonl(val_examples,   OUT_DIR / "musique_val.jsonl")

    if not args.no_hf:
        print(f"  Saving HuggingFace Dataset format...")
        write_hf_dataset(train_examples, val_examples, OUT_DIR)

    if train_examples:
        print(f"\n{'─' * 60}")
        print("  SAMPLE OUTPUT (first training example):")
        print(f"{'─' * 60}")
        sample = train_examples[0]
        print(f"  id:         {sample['id']}")
        print(f"  hop_count:  {sample['hop_count']}")
        print(f"  complexity: {sample['complexity']}")
        print(f"  question:   {sample['messages'][1]['content'][:80]}...")
        print(f"  decomp:     {sample['messages'][2]['content']}")

    print(f"\n✓ MuSiQue processing complete.")


if __name__ == "__main__":
    main()