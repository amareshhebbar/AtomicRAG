import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    from datasets import load_dataset, DatasetDict
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Missing packages. Run:")
    print("  pip install datasets huggingface_hub tqdm")
    sys.exit(1)

ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "raw"
LOG_FILE  = ROOT / "data" / "download_log.json"


DATASETS = {
    "musique": {
        "hf_path":            "bdsaglam/musique",
        "config":             "default",
        "splits":             ["train", "validation"],
        "local_dir":          RAW_DIR / "musique",
        "trust_remote_code":  False,
        "description": (
            "MuSiQue — Gold standard. Explicit question_decomposition field with "
            "per-hop sub-questions and answer labels. 20K examples, 2-4 hops."
        ),
    },
    "hotpotqa": {
        "hf_path":            "hotpotqa/hotpot_qa",
        "config":             "distractor",
        "splits":             ["train", "validation"],
        "local_dir":          RAW_DIR / "hotpotqa",
        "trust_remote_code":  False,
        "description": (
            "HotpotQA (distractor setting) — 113K 2-hop questions with supporting "
            "facts labeled at sentence level. Largest volume dataset."
        ),
    },
    "2wikimultihopqa": {
        "hf_path":            "framolfese/2WikiMultihopQA",
        "config":             None,
        "splits":             ["train", "validation", "test"],
        "local_dir":          RAW_DIR / "2wikimultihopqa",
        "trust_remote_code":  False,
        "description": (
            "2WikiMultiHopQA — 167K examples across 4 reasoning types: bridge, "
            "comparison, compositional, inference. Most diverse hop types."
        ),
    },
}


def print_banner(text: str):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def save_dataset_info(name: str, dataset: DatasetDict, local_dir: Path):
    info = {
        "name":           name,
        "downloaded_at":  datetime.utcnow().isoformat() + "Z",
        "splits":         {},
    }
    for split, ds in dataset.items():
        info["splits"][split] = {
            "num_rows":    len(ds),
            "features":    {k: str(v) for k, v in ds.features.items()},
            "column_names": ds.column_names,
        }
    info_path = local_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ Metadata saved → {info_path.relative_to(ROOT)}")


def download_one(name: str, cfg: dict, dry_run: bool = False) -> bool:
    print_banner(f"Dataset: {name.upper()}")
    print(f"  HF path:     {cfg['hf_path']}")
    print(f"  Config:      {cfg['config'] or 'default'}")
    print(f"  Splits:      {', '.join(cfg['splits'])}")
    print(f"  Local dir:   {cfg['local_dir'].relative_to(ROOT)}")
    print(f"  Description: {cfg['description']}")

    if dry_run:
        print("  [DRY RUN] Skipping actual download.")
        return True

    local_dir: Path = cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    existing = list(local_dir.glob("**/*.arrow"))
    if existing:
        print(f"  ⚡ Already downloaded ({len(existing)} arrow file(s)). Skipping.")
        print(f"     Delete {local_dir.relative_to(ROOT)} to re-download.")
        return True

    try:
        load_kwargs = dict(
            path=cfg["hf_path"],
            cache_dir=str(local_dir),
            trust_remote_code=cfg["trust_remote_code"],
        )
        if cfg["config"]:
            load_kwargs["name"] = cfg["config"]

        print(f"\n  Downloading from HuggingFace...")
        dataset = load_dataset(**load_kwargs)

        available_splits = list(dataset.keys())
        requested        = cfg["splits"]
        missing          = [s for s in requested if s not in available_splits]
        if missing:
            print(f"  ⚠  Splits not found: {missing}. Available: {available_splits}")

        for split in available_splits:
            if split not in requested:
                continue
            out_path = local_dir / f"{split}.parquet"
            dataset[split].to_parquet(str(out_path))
            rows = len(dataset[split])
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {split:12s} → {out_path.name:30s}  {rows:>7,} rows  {size_mb:6.1f} MB")

        save_dataset_info(name, dataset, local_dir)
        return True

    except Exception as e:
        print(f"\n  [ERROR] Failed to download {name}: {e}")
        return False


def write_download_log(results: dict):
    log = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results":   results,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n✓ Download log → {LOG_FILE.relative_to(ROOT)}")


def print_summary(results: dict):
    print_banner("DOWNLOAD SUMMARY")
    ok  = [k for k, v in results.items() if v]
    err = [k for k, v in results.items() if not v]
    print(f"  Success: {', '.join(ok) if ok else 'none'}")
    print(f"  Failed:  {', '.join(err) if err else 'none'}")
    if err:
        print(f"\n  Retry failed datasets with:")
        print(f"    python scripts/download_data.py --datasets {','.join(err)}")



def main():
    parser = argparse.ArgumentParser(
        description="Download all querydecomp training datasets from HuggingFace."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset keys to download (default: all). "
             "Options: musique, hotpotqa, 2wikimultihopqa",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    if args.datasets:
        keys = [k.strip() for k in args.datasets.split(",")]
        invalid = [k for k in keys if k not in DATASETS]
        if invalid:
            print(f"[ERROR] Unknown dataset keys: {invalid}")
            print(f"  Valid options: {list(DATASETS.keys())}")
            sys.exit(1)
        to_download = {k: DATASETS[k] for k in keys}
    else:
        to_download = DATASETS

    print(f"\n{'═' * 60}")
    print(f"  QueryDecomp — Dataset Downloader")
    print(f"  Datasets to download: {list(to_download.keys())}")
    print(f"  Output root: {RAW_DIR.relative_to(ROOT)}/")
    if args.dry_run:
        print(f"  *** DRY RUN — no files will be written ***")
    print(f"{'═' * 60}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, cfg in to_download.items():
        results[name] = download_one(name, cfg, dry_run=args.dry_run)

    if not args.dry_run:
        write_download_log(results)

    print_summary(results)

    if any(not v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()