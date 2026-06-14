"""
train/merge_and_push.py
-----------------------
Merges a LoRA/DoRA adapter into the base model and saves a standalone model.
Optionally pushes to HuggingFace Hub.

Run after each training stage:

  After Stage 1:
    python train/merge_and_push.py --stage 1
    → outputs/stage1_merged/   (input for Stage 2)

  After Stage 2:
    python train/merge_and_push.py --stage 2
    → outputs/stage2_merged/   (input for Stage 3)

  After Stage 3:
    python train/merge_and_push.py --stage 3 --push
    → outputs/final/ + push to HuggingFace + GGUF
"""

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGE_CONFIGS = {
    "1": {
        "base_model":   None,                         # resolved from cfg.base_model_id
        "adapter_path": "outputs/stage1_qlora/final",
        "output_path":  "outputs/stage1_merged",
        "push":         False,
    },
    "2": {
        "base_model":   "outputs/stage1_merged",
        "adapter_path": "outputs/stage2_dora/final",
        "output_path":  "outputs/stage2_merged",
        "push":         False,
    },
    "3": {
        "base_model":   "outputs/stage2_merged",
        "adapter_path": "outputs/stage3_orpo/final",
        "output_path":  "outputs/final",
        "push":         True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge adapter + push to HuggingFace")
    parser.add_argument("--stage",     type=str, choices=["1","2","3"], default=None)
    parser.add_argument("--base",      type=str, default=None)
    parser.add_argument("--adapter",   type=str, default=None)
    parser.add_argument("--output",    type=str, default=None)
    parser.add_argument("--push",      action="store_true")
    parser.add_argument("--push_gguf", action="store_true")
    parser.add_argument("--repo_id",   type=str, default=None)
    return parser.parse_args()


def _check_internet() -> bool:
    """Quick check — try to reach HuggingFace. Returns True if online."""
    try:
        import socket
        socket.setdefaulttimeout(3)
        socket.getaddrinfo("huggingface.co", 443)
        return True
    except Exception:
        return False


def _load_tokenizer(base_model_id: str, adapter_path: str):
    """
    Load tokenizer with offline fallback.

    Priority:
      1. Load from adapter_path (saved there by training — always local)
      2. Load from base_model_id with local_files_only=True (HF cache)
      3. Load from base_model_id with network (RunPod / online)
    """
    from transformers import AutoTokenizer

    # 1. Tokenizer is saved in the adapter dir by every training script
    adapter_tok = Path(adapter_path) / "tokenizer.json"
    if adapter_tok.exists():
        print(f"  Loading tokenizer from adapter dir (offline-safe)...")
        tok = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"  ✓ Tokenizer loaded from {adapter_path}")
        return tok

    # 2. Try HF cache (no network)
    try:
        print(f"  Loading tokenizer from HF cache...")
        tok = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"  ✓ Tokenizer loaded from cache")
        return tok
    except Exception:
        pass

    # 3. Network (RunPod)
    print(f"  Downloading tokenizer from HuggingFace...")
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    print(f"  ✓ Tokenizer downloaded")
    return tok


def _load_base_model(base_model_id: str):
    """
    Load base model with offline fallback.
    On laptop: loads from HF cache (set when model was first downloaded).
    On RunPod: downloads if not cached.
    """
    import torch
    from transformers import AutoModelForCausalLM

    online = _check_internet()

    # Determine dtype — CPU gets float32, GPU gets bfloat16
    is_gpu = torch.cuda.is_available()
    dtype  = torch.bfloat16 if is_gpu else torch.float32
    device = "auto" if is_gpu else "cpu"

    load_kwargs = dict(
        pretrained_model_name_or_path=base_model_id,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # If base_model_id is a local path (Stage 2/3 merge), always local
    if Path(base_model_id).exists():
        print(f"  Loading base model from local path...")
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        print(f"  ✓ Base model loaded from disk")
        return model

    # HF model ID — try cache first, then network
    if not online:
        print(f"  No internet — loading from HF cache (local_files_only)...")
        load_kwargs["local_files_only"] = True

    print(f"  Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    print(f"  ✓ Base model loaded")
    return model


def do_merge(base_model_id: str, adapter_path: str, output_path: str,
             push_to_hub: bool, hub_repo_id: str):
    """Merge adapter into base model and save."""
    import torch
    from peft import PeftModel

    print(f"\n{'─'*56}")
    print(f"  Merging adapter")
    print(f"  Base:    {base_model_id}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {output_path}")
    print(f"{'─'*56}")

    tokenizer  = _load_tokenizer(base_model_id, adapter_path)
    base_model = _load_base_model(base_model_id)

    print(f"  Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print(f"  Merging and unloading adapter (~1 min)...")
    model = model.merge_and_unload()

    print(f"  Saving merged model → {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    total = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ Saved — {total:.2f}B params")

    if push_to_hub and hub_repo_id:
        if not _check_internet():
            print(f"  [WARN] No internet — skipping HF push")
            print(f"  Push manually later: python train/merge_and_push.py --stage 3 --push")
            return
        print(f"  Pushing to HuggingFace: {hub_repo_id}")
        model.push_to_hub(hub_repo_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_repo_id)
        print(f"  ✓ Live at: https://huggingface.co/{hub_repo_id}")


def export_gguf(model_path: str, output_dir: str, repo_id: str):
    """Export GGUF Q4_K_M for llama.cpp / Ollama."""
    import subprocess

    llama_cpp = ROOT / "llama.cpp"
    if not llama_cpp.exists():
        print(f"  [SKIP] llama.cpp not found — GGUF export skipped")
        print(f"  To enable: git clone https://github.com/ggerganov/llama.cpp.git")
        return

    gguf_name = f"{repo_id.split('/')[-1]}-Q4_K_M.gguf"
    gguf_path = Path(output_dir) / gguf_name

    print(f"\n  Exporting GGUF → {gguf_path}")
    r = subprocess.run(
        ["python", str(llama_cpp/"convert_hf_to_gguf.py"),
         model_path, "--outfile", str(gguf_path), "--outtype", "q4_k_m"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  [ERROR] GGUF export failed:\n{r.stderr}")
        return

    print(f"  ✓ GGUF saved ({gguf_path.stat().st_size/1e9:.2f} GB)")

    if repo_id and _check_internet():
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=str(gguf_path),
                path_in_repo=gguf_name,
                repo_id=repo_id,
            )
            print(f"  ✓ GGUF pushed to hub")
        except Exception as e:
            print(f"  [WARN] GGUF upload failed: {e}")


def main():
    args = parse_args()

    try:
        from src.config import get_config
    except ImportError as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    cfg = get_config("stage1")

    # Resolve paths
    if args.stage:
        sc           = STAGE_CONFIGS[args.stage]
        base_model   = args.base    or sc["base_model"] or cfg.base_model_id
        adapter_path = args.adapter or sc["adapter_path"]
        output_path  = args.output  or sc["output_path"]
        do_push      = args.push    or sc["push"]
    else:
        if not all([args.base, args.adapter, args.output]):
            print("[ERROR] Provide --stage OR all of --base, --adapter, --output")
            sys.exit(1)
        base_model   = args.base
        adapter_path = args.adapter
        output_path  = args.output
        do_push      = args.push

    # Make absolute
    if not Path(base_model).is_absolute() and not base_model.startswith("Qwen"):
        base_model = str(ROOT / base_model)
    if not Path(adapter_path).is_absolute():
        adapter_path = str(ROOT / adapter_path)
    if not Path(output_path).is_absolute():
        output_path = str(ROOT / output_path)

    repo_id = args.repo_id or cfg.hf_repo_id

    print(f"\n{'═'*56}")
    print(f"  Merge & Push — Stage {args.stage or 'custom'}")
    print(f"  Base:    {base_model}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {output_path}")
    print(f"  Push:    {do_push}  Repo: {repo_id if do_push else 'N/A'}")
    print(f"  Online:  {_check_internet()}")
    print(f"{'═'*56}\n")

    if not Path(adapter_path).exists():
        print(f"[ERROR] Adapter not found: {adapter_path}")
        print(f"  Did the training script finish successfully?")
        sys.exit(1)

    do_merge(
        base_model_id=base_model,
        adapter_path=adapter_path,
        output_path=output_path,
        push_to_hub=do_push,
        hub_repo_id=repo_id if do_push else "",
    )

    if args.push_gguf and args.stage == "3":
        export_gguf(output_path, output_path, repo_id)

    next_steps = {
        "1": "python train/stage2_dora.py",
        "2": "python train/stage3_orpo.py",
        "3": "python eval/evaluate_decomposition.py",
    }
    if args.stage and args.stage in next_steps:
        print(f"\n  Next: {next_steps[args.stage]}")

    print(f"\n✓ Done.\n")


if __name__ == "__main__":
    main()