import json
import argparse
import sys
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
RAW_DIR  = ROOT / "data" / "raw" / "2wikimultihopqa"
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

TYPE_COMPLEXITY_MAP = {
    "bridge":       "bridge_2hop",
    "comparison":   "comparison_2hop",
    "compositional":"compositional_2hop",
    "inference":    "inference_2hop",
}


def get_evidence_titles(row: dict) -> list[str]:
    evidences = row.get("evidences", [])
    if isinstance(evidences, dict):
        titles = evidences.get("title", [])
    elif isinstance(evidences, list):
        titles = [e.get("title", "") if isinstance(e, dict) else "" for e in evidences]
    else:
        titles = []
    seen   = set()
    unique = []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def build_decomposition(question: str, q_type: str, titles: list[str]) -> list[dict]:
    if len(titles) < 2:
        titles = (titles + ["[supporting document]"] * 2)[:2]
    
    t1, t2 = titles[0], titles[1]

    if q_type in ("bridge", "compositional"):
        return [
            {
                "hop":        1,
                "sub_query":  f"What information does {t1} provide relevant to: {question}",
                "depends_on": [],
            },
            {
                "hop":        2,
                "sub_query":  f"Using the information from {t1}, {question}",
                "depends_on": [1],
            },
        ]
    else:
        return [
            {
                "hop":        1,
                "sub_query":  f"What relevant fact about {t1} relates to: {question}",
                "depends_on": [],
            },
            {
                "hop":        2,
                "sub_query":  f"What relevant fact about {t2} relates to: {question}",
                "depends_on": [],
            },
        ]


def process_row(row: dict, idx: int, split: str) -> Optional[dict]:
    question = row.get("question", "").strip()
    answer   = row.get("answer",   "").strip()
    q_type   = row.get("type",     "bridge").lower().strip()

    if not question:
        return None

    titles    = get_evidence_titles(row)
    dep_graph = build_decomposition(question, q_type, titles)
    messages  = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": json.dumps(dep_graph, ensure_ascii=False)},
    ]

    return {
        "id":                  f"2wiki_{split}_{idx:06d}",
        "domain":              "wikipedia",
        "source":              "2wikimultihopqa",
        "hop_count":           2,
        "complexity":          TYPE_COMPLEXITY_MAP.get(q_type, "bridge_2hop"),
        "messages":            messages,
        "ground_truth_answer": answer,
        "supporting_facts":    titles,
        "quality_tier":        "heuristic",
    }


def process_split(split: str, max_rows: Optional[int]) -> list[dict]:
    parquet_path = RAW_DIR / f"{split}.parquet"
    if not parquet_path.exists():
        print(f"  [SKIP] {parquet_path.name} not found — run download_data.py first")
        return []

    print(f"\n  Loading {split}...")
    df = pd.read_parquet(parquet_path)
    if max_rows:
        df = df.head(max_rows)
    print(f"  Rows: {len(df):,}")

    examples    = []
    skipped     = 0
    type_counts = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {split}"):
        result = process_row(row.to_dict(), idx, split)
        if result is None:
            skipped += 1
            continue
        examples.append(result)
        c = result["complexity"]
        type_counts[c] = type_counts.get(c, 0) + 1

    print(f"  ✓ Converted:  {len(examples):,}  ✗ Skipped: {skipped:,}")
    print(f"  Types:        {type_counts}")
    return examples


def write_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  → {path.relative_to(ROOT)}  ({len(examples):,} examples, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  2WikiMultiHopQA Processor")
    print(f"{'═' * 60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        examples = process_split(split, args.max_rows)
        if examples:
            write_jsonl(examples, OUT_DIR / f"2wiki_{split}.jsonl")

    print(f"\n✓ 2WikiMultiHopQA processing complete.")


if __name__ == "__main__":
    main()