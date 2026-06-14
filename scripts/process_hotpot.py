import json
import argparse
import sys
import re
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Run: pip install datasets pandas tqdm")
    sys.exit(1)

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw" / "hotpotqa"
OUT_DIR  = ROOT / "data" / "processed"

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


def extract_supporting_titles(row: dict) -> list[str]:
    facts   = row.get("supporting_facts", {})
    titles  = facts.get("title", []) if isinstance(facts, dict) else []
    return list(dict.fromkeys(titles))   


def build_bridge_decomposition(question: str, titles: list[str]) -> list[dict]:
    if len(titles) < 2:
        titles = titles + ["[supporting document]"] if titles else ["[document A]", "[document B]"]

    t1, t2 = titles[0], titles[1]
    q_clean = question.rstrip("?").strip()

    return [
        {
            "hop":        1,
            "sub_query":  f"What information about {t1} is relevant to: {q_clean}?",
            "depends_on": [],
        },
        {
            "hop":        2,
            "sub_query":  f"Given what was found about {t1}, {question}",
            "depends_on": [1],
        },
    ]


def build_comparison_decomposition(question: str, titles: list[str]) -> list[dict]:
    if len(titles) < 2:
        titles = (titles + ["Entity B"])[:2]

    t1, t2 = titles[0], titles[1]

    return [
        {
            "hop":        1,
            "sub_query":  f"What is the relevant attribute of {t1} for the comparison?",
            "depends_on": [],  
        },
        {
            "hop":        2,
            "sub_query":  f"What is the relevant attribute of {t2} for the comparison?",
            "depends_on": [],   
        },
    ]


def process_row(row: dict, idx: int, split: str, filter_type: Optional[str]) -> Optional[dict]:
    """Process a single HotpotQA row."""
    q_type   = row.get("type", "bridge")
    question = row.get("question", "").strip()
    answer   = row.get("answer", "").strip()

    if not question:
        return None

    if filter_type and q_type != filter_type:
        return None

    titles = extract_supporting_titles(row)

    if q_type == "comparison":
        dep_graph  = build_comparison_decomposition(question, titles)
        complexity = "comparison_2hop"
    else:
        dep_graph  = build_bridge_decomposition(question, titles)
        complexity = "bridge_2hop"

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": json.dumps(dep_graph, ensure_ascii=False)},
    ]

    return {
        "id":                  f"hotpotqa_{split}_{idx:06d}",
        "domain":              "wikipedia",
        "source":              "hotpotqa",
        "hop_count":           2,
        "complexity":          complexity,
        "messages":            messages,
        "ground_truth_answer": answer,
        "supporting_facts":    titles,
        "quality_tier":        "heuristic",   
    }


def process_split(split: str, max_rows: Optional[int], filter_type: Optional[str]) -> list[dict]:
    parquet_path = RAW_DIR / f"{split}.parquet"
    if not parquet_path.exists():
        print(f"  [SKIP] {parquet_path} not found — run download_data.py first")
        return []

    print(f"\n  Loading {split} split...")
    df = pd.read_parquet(parquet_path)

    if max_rows:
        df = df.head(max_rows)

    print(f"  Rows: {len(df):,}")

    examples    = []
    skipped     = 0
    type_counts = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {split}"):
        result = process_row(row.to_dict(), idx, split, filter_type)
        if result is None:
            skipped += 1
            continue
        examples.append(result)
        c = result["complexity"]
        type_counts[c] = type_counts.get(c, 0) + 1

    print(f"  ✓ Converted:  {len(examples):,}")
    print(f"  ✗ Skipped:    {skipped:,}")
    print(f"  Types:        {type_counts}")
    return examples


def write_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    size_kb = path.stat().st_size / 1024
    print(f"  → {path.relative_to(ROOT)}  ({len(examples):,} examples, {size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Process HotpotQA into querydecomp format.")
    parser.add_argument("--max_rows",  type=int,   default=None)
    parser.add_argument("--type",      type=str,   default=None,
                        help="Filter by type: 'bridge' or 'comparison'")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  HotpotQA Processor")
    print(f"  Input:  {RAW_DIR.relative_to(ROOT)}/")
    print(f"{'═' * 60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_examples = process_split("train",      args.max_rows, args.type)
    val_examples   = process_split("validation", args.max_rows, args.type)

    if not train_examples and not val_examples:
        print("\n[ERROR] No examples processed. Run download_data.py first.")
        sys.exit(1)

    write_jsonl(train_examples, OUT_DIR / "hotpotqa_train.jsonl")
    write_jsonl(val_examples,   OUT_DIR / "hotpotqa_val.jsonl")

    if train_examples:
        print(f"\n  SAMPLE:")
        s = train_examples[0]
        print(f"  id:       {s['id']}")
        print(f"  question: {s['messages'][1]['content'][:80]}...")
        print(f"  decomp:   {s['messages'][2]['content']}")

    print(f"\n✓ HotpotQA processing complete.")


if __name__ == "__main__":
    main()