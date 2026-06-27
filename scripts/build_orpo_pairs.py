import json
import random
import argparse
import sys
import re
from pathlib import Path
from typing import Optional
from copy import deepcopy

try:
    from datasets import Dataset, DatasetDict
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Run: pip install datasets tqdm")
    sys.exit(1)

ROOT= Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
RANDOM_SEED= 42

N_REJECTED_PER_EXAMPLE = 5


FAKE_PEOPLE = ["Dr. Alan Voss", "Marie Kessler", "Jonathan Crane", "Priya Mehta",
                  "Robert Finch", "Lena Hartmann", "David Osei", "Chen Wei"]
FAKE_PLACES = ["Veloria", "Dunstable", "Kraetheim", "Molvenia", "Santhera",
                  "New Colworth", "Port Albrecht", "Estravia"]
FAKE_COMPANIES = ["Vexar Systems", "Lumentech", "CoreAxis", "Proxima Labs",
                  "Stelara Group", "Nexfield Industries"]
FAKE_YEARS= ["1987", "1993", "2001", "2007", "2011", "2017", "2019"]

_ENTITY_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b')


def _find_entities(text: str) -> list[str]:
    candidates = _ENTITY_RE.findall(text)
    stopwords = {"What", "Who", "Where", "When", "Which", "How", "The", "In", "Of",
                 "Was", "Were", "Did", "Does", "Is", "Are", "Has", "Have"}
    return [e for e in candidates if e not in stopwords]


def _replace_entity(text: str, rng: random.Random) -> str:
    entities = _find_entities(text)
    if not entities:
        return text

    target = rng.choice(entities)
    pool= FAKE_PEOPLE + FAKE_PLACES + FAKE_COMPANIES
    replacement = rng.choice([p for p in pool if p != target])
    return text.replace(target, replacement, 1)


def rejection_merged_hops(dep_graph: list[dict], rng: random.Random) -> Optional[list[dict]]:
    if len(dep_graph) < 2:
        return None

    for i in range(len(dep_graph) - 1):
        hop_a = dep_graph[i]
        hop_b = dep_graph[i + 1]
        if hop_a["hop"] in hop_b["depends_on"]:
            merged_q = f"{hop_a['sub_query'].rstrip('?')} and {hop_b['sub_query'].lower()}"
            new_graph = [deepcopy(h) for h in dep_graph if h["hop"] not in (hop_a["hop"], hop_b["hop"])]
            merged_hop = {
                "hop":1,
                "sub_query":  merged_q,
                "depends_on": [],
            }
            remaining = sorted(new_graph, key=lambda h: h["hop"])
            result = [merged_hop]
            for j, h in enumerate(remaining, start=2):
                h["hop"] = j
                h["depends_on"] = []   
                result.append(h)
            return result

    return None


def rejection_hallucinated(dep_graph: list[dict], question: str, rng: random.Random) -> Optional[list[dict]]:
    new_graph = deepcopy(dep_graph)

    rng.shuffle(new_graph)
    for hop in new_graph:
        original = hop["sub_query"]
        replaced = _replace_entity(original, rng)
        if replaced != original:
            hop["sub_query"] = replaced
            return sorted(new_graph, key=lambda h: h["hop"])

    for hop in new_graph:
        years = re.findall(r'\b(19|20)\d{2}\b', hop["sub_query"])
        if years:
            fake_year = rng.choice([y for y in FAKE_YEARS if y not in years])
            hop["sub_query"] = re.sub(r'\b(19|20)\d{2}\b', fake_year, hop["sub_query"], count=1)
            return sorted(new_graph, key=lambda h: h["hop"])

    return None


def rejection_zero_decomp(question: str) -> list[dict]:
    return [{"hop": 1, "sub_query": question.strip(), "depends_on": []}]


def rejection_over_decomp(dep_graph: list[dict], rng: random.Random) -> list[dict]:
    new_graph = []
    hop_offset = 0

    for hop in dep_graph:
        sq = hop["sub_query"]
        split_hops = [
            {
                "hop":hop_offset + 1,
                "sub_query":  f"What is the subject being asked about in: {sq}",
                "depends_on": [],
            },
            {
                "hop":hop_offset + 2,
                "sub_query":  sq,
                "depends_on": [hop_offset + 1],
            },
            {
                "hop":hop_offset + 3,
                "sub_query":  f"Confirm the answer found for: {sq}",
                "depends_on": [hop_offset + 2],
            },
        ]
        new_graph.extend(split_hops)
        hop_offset += 3

    return new_graph


def rejection_missing_hop(dep_graph: list[dict], rng: random.Random) -> Optional[list[dict]]:
    if len(dep_graph) < 3:
        return None
    middle_idx = rng.randint(1, len(dep_graph) - 2)
    new_graph= [h for i, h in enumerate(dep_graph) if i != middle_idx]

    for i, hop in enumerate(new_graph):
        hop["hop"]= i + 1
        hop["depends_on"] = [] if i == 0 else [i]

    return new_graph



REJECTION_TYPES = ["merged_hops", "hallucinated", "zero_decomp", "over_decomp", "missing_hop"]


def build_rejected_variants(
    question:  str,
    dep_graph: list[dict],
    rng:  random.Random,
    n_max:int = N_REJECTED_PER_EXAMPLE,
) -> list[tuple[str, str]]:
    variants = []

    generators = {
        "merged_hops": lambda: rejection_merged_hops(dep_graph, rng),
        "hallucinated": lambda: rejection_hallucinated(dep_graph, question, rng),
        "zero_decomp":  lambda: rejection_zero_decomp(question),
        "over_decomp":  lambda: rejection_over_decomp(dep_graph, rng),
        "missing_hop":  lambda: rejection_missing_hop(dep_graph, rng),
    }

    for rtype in REJECTION_TYPES:
        if len(variants) >= n_max:
            break
        try:
            result = generators[rtype]()
        except Exception:
            continue

        if result is None:
            continue

        rejected_json = json.dumps(result, ensure_ascii=False)
        chosen_json= json.dumps(dep_graph, ensure_ascii=False)
        if rejected_json == chosen_json:
            continue

        variants.append((rejected_json, rtype))

    return variants


def process_sft_file(
    path:  Path,
    split: str,
    rng:random.Random,
    max_rows:Optional[int],
    n_rejected: int,
) -> list[dict]:
    if not path.exists():
        print(f"  [SKIP] {path.relative_to(ROOT)} not found")
        return []

    with open(path, encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    if max_rows:
        raw_lines = raw_lines[:max_rows]

    pairs = []
    pair_idx= 0
    type_counts= {t: 0 for t in REJECTION_TYPES}

    for line in tqdm(raw_lines, desc=f"  Building ORPO pairs [{split}]"):
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = ex.get("messages", [])
        question = next((m["content"] for m in messages if m["role"] == "user"), "")
        asst_raw = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        if not question or not asst_raw:
            continue

        try:
            dep_graph = json.loads(asst_raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(dep_graph, list) or len(dep_graph) < 2:
            continue

        chosen_json = json.dumps(dep_graph, ensure_ascii=False)
        variants = build_rejected_variants(question, dep_graph, rng, n_max=n_rejected)

        for rejected_json, rtype in variants:
            pairs.append({
                "id":f"orpo_{split}_{pair_idx:06d}",
                "question":  question,
                "chosen_json": chosen_json,
                "rejected_json":  rejected_json,
                "rejection_type": rtype,
                "source": ex.get("source", ""),
                "hop_count": ex.get("hop_count", 2),
            })
            type_counts[rtype] += 1
            pair_idx += 1

    print(f"  ✓ Generated {len(pairs):,} pairs from {len(raw_lines):,} examples")
    print(f"    Type breakdown: {type_counts}")
    return pairs


def write_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  → {path.relative_to(ROOT)}  ({len(examples):,} pairs, {size_mb:.2f} MB)")


def build_hf_dataset(train: list[dict], val: list[dict]) -> DatasetDict:
    def make_ds(examples: list[dict]) -> Dataset:
        if not examples:
            return Dataset.from_dict({
                "id": [], "question": [], "chosen_json": [],
                "rejected_json": [], "rejection_type": [], "source": [], "hop_count": [],
            })
        rows = {k: [] for k in examples[0].keys()}
        for ex in examples:
            for k, v in ex.items():
                rows[k].append(v)
        return Dataset.from_dict(rows)

    splits = {}
    if train:
        splits["train"] = make_ds(train)
    if val:
        splits["validation"] = make_ds(val)

    return DatasetDict(splits)


def main():
    parser = argparse.ArgumentParser(description="Build ORPO preference pairs.")
    parser.add_argument("--max_rows",   type=int, default=None,
                        help="Max SFT examples to process per split (local test)")
    parser.add_argument("--n_rejected", type=int, default=N_REJECTED_PER_EXAMPLE,
                        help="Max rejected variants per example (default: 5)")
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"\n{'═' * 60}")
    print(f"  QueryDecomp — Build ORPO Preference Pairs")
    print(f"  n_rejected per example: {args.n_rejected}")
    print(f"{'═' * 60}")

    train_pairs = process_sft_file(
        PROC_DIR / "train_sft.jsonl", "train", rng, args.max_rows, args.n_rejected
    )
    val_pairs = process_sft_file(
        PROC_DIR / "val_sft.jsonl", "val", rng, args.max_rows, args.n_rejected
    )

    if not train_pairs:
        print("\n  train_sft.jsonl empty — trying musique_train.jsonl...")
        train_pairs = process_sft_file(
            PROC_DIR / "musique_train.jsonl", "train", rng, args.max_rows, args.n_rejected
        )
        val_pairs = process_sft_file(
            PROC_DIR / "musique_val.jsonl", "val", rng, args.max_rows, args.n_rejected
        )

    if not train_pairs:
        print("\n[ERROR] No training data found. Run process_musique.py and build_splits.py first.")
        sys.exit(1)

    print(f"\n  Writing JSONL files...")
    write_jsonl(train_pairs, PROC_DIR / "orpo_train.jsonl")
    if val_pairs:
        write_jsonl(val_pairs, PROC_DIR / "orpo_val.jsonl")

    print(f"\n  Building HuggingFace DatasetDict...")
    dsd= build_hf_dataset(train_pairs, val_pairs)
    save_path = PROC_DIR / "orpo_dataset"
    dsd.save_to_disk(str(save_path))
    print(f"  → {save_path.relative_to(ROOT)}/")
    for split_name in dsd.keys():
        print(f"    {split_name}: {len(dsd[split_name]):,} pairs")

    if train_pairs:
        print(f"\n{'─' * 60}")
        print("  SAMPLE ORPO PAIR:")
        s = train_pairs[0]
        print(f"  Question: {s['question']}")
        print(f"  Chosen:{s['chosen_json']}")
        print(f"  Rejected: {s['rejected_json']}")
        print(f"  Reject type:{s['rejection_type']}")

    print(f"\n✓ ORPO pair generation complete.")
    print(f"  Total pairs: {len(train_pairs) + len(val_pairs):,}")
    print(f"  Expected ~5x your SFT training examples")


if __name__ == "__main__":
    main()