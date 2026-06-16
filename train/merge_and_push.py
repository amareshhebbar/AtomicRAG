import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGE_CONFIGS = {
    "1": {
        "base_model":   None,
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
        "push":         False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",     type=str, choices=["1", "2", "3"], default=None)
    parser.add_argument("--base",      type=str, default=None)
    parser.add_argument("--adapter",   type=str, default=None)
    parser.add_argument("--output",    type=str, default=None)
    parser.add_argument("--push",      action="store_true")
    parser.add_argument("--push_gguf", action="store_true")
    parser.add_argument("--repo_id",   type=str, default=None)
    return parser.parse_args()


def _check_internet():
    try:
        import socket
        socket.setdefaulttimeout(3)
        socket.getaddrinfo("huggingface.co", 443)
        return True
    except Exception:
        return False


def _load_tokenizer(base_model_id, adapter_path):
    from transformers import AutoTokenizer
    if (Path(adapter_path) / "tokenizer.json").exists():
        print(f"  Loading tokenizer from adapter dir...")
        tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True, local_files_only=True)
        print(f"  Tokenizer loaded from {adapter_path}")
        return tok
    try:
        print(f"  Loading tokenizer from HF cache...")
        tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, local_files_only=True)
        print(f"  Tokenizer loaded from cache")
        return tok
    except Exception:
        pass
    print(f"  Downloading tokenizer...")
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    print(f"  Tokenizer downloaded")
    return tok


def _load_base_model(base_model_id):
    import torch
    from transformers import AutoModelForCausalLM
    is_gpu  = torch.cuda.is_available()
    dtype   = torch.bfloat16 if is_gpu else torch.float32
    device  = "auto" if is_gpu else "cpu"
    kwargs  = dict(pretrained_model_name_or_path=base_model_id, dtype=dtype,
                   device_map=device, trust_remote_code=True)
    if Path(base_model_id).exists():
        print(f"  Loading base model from local path...")
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        print(f"  Base model loaded from disk")
        return model
    if not _check_internet():
        print(f"  No internet — loading from HF cache...")
        kwargs["local_files_only"] = True
    print(f"  Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    print(f"  Base model loaded")
    return model


def do_merge(base_model_id, adapter_path, output_path, push_to_hub, hub_repo_id):
    print(f"\n{'─'*56}")
    print(f"  Merging adapter")
    print(f"  Base:    {base_model_id}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {output_path}")
    print(f"{'─'*56}")

    tokenizer  = _load_tokenizer(base_model_id, adapter_path)
    base_model = _load_base_model(base_model_id)

    from peft import PeftModel
    print(f"  Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"  Merging and unloading...")
    model = model.merge_and_unload()

    print(f"  Saving to {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    total = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Saved — {total:.2f}B params")

    if push_to_hub and hub_repo_id:
        if not _check_internet():
            print(f"  No internet — skipping HF push")
            print(f"  Push manually: python train/merge_and_push.py --stage 3 --push")
            return
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            print(f"  Pushing to HuggingFace: {hub_repo_id}")
            api.create_repo(repo_id=hub_repo_id, exist_ok=True, private=False)
            api.upload_folder(
                folder_path=output_path,
                repo_id=hub_repo_id,
                repo_type="model",
            )
            tokenizer.push_to_hub(hub_repo_id)
            print(f"  Live at: https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            print(f"  [ERROR] HF push failed: {e}")


def export_gguf(model_path, output_dir, repo_id):
    import subprocess
    llama_cpp = ROOT / "llama.cpp"
    if not llama_cpp.exists():
        print(f"  [SKIP] llama.cpp not found")
        return
    gguf_name = f"{repo_id.split('/')[-1]}-Q4_K_M.gguf"
    gguf_path = Path(output_dir) / gguf_name
    print(f"  Exporting GGUF → {gguf_path}")
    r = subprocess.run(
        ["python", str(llama_cpp / "convert_hf_to_gguf.py"),
         model_path, "--outfile", str(gguf_path), "--outtype", "q4_k_m"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  [ERROR] GGUF failed: {r.stderr}")
        return
    print(f"  GGUF saved ({gguf_path.stat().st_size/1e9:.2f} GB)")
    if repo_id and _check_internet():
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=str(gguf_path),
                                path_in_repo=gguf_name, repo_id=repo_id)
            print(f"  GGUF pushed to hub")
        except Exception as e:
            print(f"  [WARN] GGUF upload failed: {e}")


def main():
    args = parse_args()

    try:
        from src.config import get_config
    except ImportError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    cfg = get_config("stage1")

    if args.stage:
        sc           = STAGE_CONFIGS[args.stage]
        base_model   = args.base    or sc["base_model"] or cfg.base_model_id
        adapter_path = args.adapter or sc["adapter_path"]
        output_path  = args.output  or sc["output_path"]
        do_push      = args.push
    else:
        if not all([args.base, args.adapter, args.output]):
            print("[ERROR] Provide --stage OR all of --base, --adapter, --output")
            sys.exit(1)
        base_model   = args.base
        adapter_path = args.adapter
        output_path  = args.output
        do_push      = args.push

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

    next_steps = {"1": "train/stage2_dora.py", "2": "train/stage3_orpo.py",
                  "3": "eval/evaluate_decomposition.py"}
    if args.stage and args.stage in next_steps:
        print(f"\n  Next: python {next_steps[args.stage]}")

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()