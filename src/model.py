"""
src/model.py
------------
Model and tokenizer loading with BitsAndBytes quantization + LoRA/DoRA setup.

Handles:
  - 4-bit NF4 quantization (QLoRA Stage 1)
  - bf16 full precision (Stages 2 & 3)
  - LoRA adapter injection (Stage 1)
  - DoRA adapter injection (Stage 2)
  - Adapter merging for Stage 1→2 and Stage 2→3 transitions
  - Pushing merged model to HuggingFace Hub

Usage:
    from src.model import load_model_for_training, load_model_for_inference, merge_adapter

    # Training (Stage 1)
    model, tokenizer = load_model_for_training(cfg)

    # Inference only
    model, tokenizer = load_model_for_inference("outputs/stage3_orpo/final")

    # Merge adapter into base after a stage
    merge_adapter("Qwen/Qwen2.5-1.5B-Instruct", "outputs/stage1_qlora/final", "outputs/stage1_merged")
"""

import os
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Conditional imports — graceful degradation if packages missing
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] torch not installed — model loading will fail at runtime")

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARN] transformers/peft not installed — install on RunPod")

from src.config import BaseConfig


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str, padding_side: str = "right"):
    """
    Load tokenizer for Qwen2.5.

    padding_side="right" for SFT (decoder-only, left-pad causes issues with
    causal attention masks during training).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side=padding_side,
    )

    # Qwen2.5 uses <|endoftext|> as pad token by default
    # Make sure it's set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token ({tokenizer.eos_token!r})")

    print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size:,}  "
          f"pad={tokenizer.pad_token!r}  eos={tokenizer.eos_token!r}")

    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# BitsAndBytes config
# ─────────────────────────────────────────────────────────────────────────────

def get_bnb_config(cfg: BaseConfig) -> "BitsAndBytesConfig":
    """Build BitsAndBytes 4-bit quantization config from BaseConfig."""
    if not cfg.use_4bit:
        return None

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)   # e.g. torch.bfloat16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,             # "nf4"
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_double_quant,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LoRA / DoRA config
# ─────────────────────────────────────────────────────────────────────────────

def get_lora_config(cfg: BaseConfig) -> "LoraConfig":
    """
    Build PEFT LoraConfig from BaseConfig.
    Setting use_dora=True activates DoRA (Stage 2).

    DoRA decomposes weight updates into magnitude + direction:
      W = m * (W0 + BA) / ||W0 + BA||
    where m is a learned magnitude scalar per output feature,
    and BA is the standard LoRA low-rank update.
    """
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        use_dora=cfg.use_dora,                # False=LoRA, True=DoRA
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        inference_mode=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loader — for training
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_training(cfg: BaseConfig, model_id_override: Optional[str] = None):
    """
    Load model + tokenizer ready for training.

    For Stage 1: loads base model in 4-bit, applies LoRA
    For Stage 2: loads Stage 1 merged model in bf16, applies DoRA
    For Stage 3: loads Stage 2 merged model in bf16, applies LoRA for ORPO

    Args:
        cfg:               Config for the current stage
        model_id_override: Use this path instead of cfg.base_model_id
                           (used for Stage 2/3 to load merged intermediate models)

    Returns:
        (model, tokenizer)
    """
    model_id = model_id_override or cfg.base_model_id
    stage    = getattr(cfg, "stage", "unknown")

    print(f"\n{'─' * 56}")
    print(f"  Loading model for {stage}")
    print(f"  Model: {model_id}")
    print(f"  4-bit: {cfg.use_4bit}  DoRA: {cfg.use_dora}  r={cfg.lora_r}")
    print(f"{'─' * 56}")

    tokenizer = load_tokenizer(model_id, cfg.padding_side)

    bnb_config = get_bnb_config(cfg)   # None if use_4bit=False

    # ── Load base model ───────────────────────────────────────────────────────
    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        device_map=cfg.device_map,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    if not cfg.use_4bit:
        # CPU: use float32 (bf16 not supported on CPU)
        # GPU: use bf16 for memory efficiency
        on_cpu = getattr(cfg, "device_map", "auto") == "cpu"
        load_kwargs["dtype"] = torch.float32 if on_cpu else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    print(f"  ✓ Base model loaded — params: {count_params(model)}")

    # ── Prepare for k-bit training (only if 4-bit) ────────────────────────────
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
        )
        print(f"  ✓ Prepared for k-bit training")

    # ── Apply LoRA / DoRA ─────────────────────────────────────────────────────
    lora_cfg = get_lora_config(cfg)
    model    = get_peft_model(model, lora_cfg)

    adapter_type = "DoRA" if cfg.use_dora else "LoRA"
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total        = sum(p.numel() for p in model.parameters())
    pct          = 100 * trainable / total

    print(f"  ✓ {adapter_type} applied — "
          f"trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
    print(f"  Target modules: {cfg.target_modules}")

    if cfg.gradient_checkpointing and not cfg.use_4bit:
        model.gradient_checkpointing_enable()
        print(f"  ✓ Gradient checkpointing enabled")

    # Silence the annoying "Some layers were not initialized from the model checkpoint" warning
    model.config.use_cache = False   # incompatible with gradient checkpointing

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Inference loader — no LoRA, just the merged model
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_inference(model_path: str, device: str = "auto"):
    """
    Load a merged model (no LoRA adapter) for inference.
    Uses bf16 and auto device map.

    Args:
        model_path: local path or HF repo ID of merged model
        device:     "auto", "cuda", "cpu"
    """
    print(f"\n  Loading for inference: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  ✓ Ready — params: {count_params(model)}")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Adapter merger
# ─────────────────────────────────────────────────────────────────────────────

def merge_adapter(
    base_model_id:  str,
    adapter_path:   str,
    output_path:    str,
    push_to_hub:    bool = False,
    hub_repo_id:    str  = "",
):
    """
    Merge a LoRA/DoRA adapter into the base model weights and save as a
    standalone bf16 model.

    This is called between stages:
      - After Stage 1: merge into outputs/stage1_merged/
      - After Stage 2: merge into outputs/stage2_merged/
      - After Stage 3: merge into outputs/final/ (then optionally push to HF)

    Args:
        base_model_id: HF model ID or local path of the BASE model
        adapter_path:  path to the PEFT adapter (from training output)
        output_path:   where to save the merged bf16 model
        push_to_hub:   whether to push to HuggingFace Hub
        hub_repo_id:   HF repo ID (e.g. "AmareshHebbar/querydecomp-qwen2.5-1.5b")
    """
    print(f"\n{'─' * 56}")
    print(f"  Merging adapter")
    print(f"  Base:    {base_model_id}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {output_path}")
    print(f"{'─' * 56}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print(f"  Loading base model in bf16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"  Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print(f"  Merging and unloading adapter (this takes ~1 min)...")
    model = model.merge_and_unload()

    print(f"  Saving merged model → {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    merged_params = count_params(model)
    print(f"  ✓ Merged model saved — {merged_params}")

    if push_to_hub and hub_repo_id:
        print(f"  Pushing to HuggingFace Hub: {hub_repo_id}")
        model.push_to_hub(hub_repo_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_repo_id)
        print(f"  ✓ Pushed to hub: https://huggingface.co/{hub_repo_id}")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model) -> str:
    """Return a human-readable parameter count string."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_b   = total / 1e9
    return f"{total_b:.2f}B total, {trainable:,} trainable"


def get_gpu_memory_gb() -> float:
    """Return current GPU memory usage in GB."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def print_model_summary(model):
    """Print a quick summary of model architecture and VRAM usage."""
    print(f"\n  Model summary:")
    print(f"    Params:      {count_params(model)}")
    print(f"    VRAM used:   {get_gpu_memory_gb():.2f} GB")
    if hasattr(model, "config"):
        cfg = model.config
        print(f"    Hidden size: {getattr(cfg, 'hidden_size', '?')}")
        print(f"    Layers:      {getattr(cfg, 'num_hidden_layers', '?')}")
        print(f"    Heads:       {getattr(cfg, 'num_attention_heads', '?')}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (no GPU needed — just checks imports and config logic)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import get_config

    # Verify configs produce correct LoRA settings
    for stage in ["stage1", "stage2", "stage3"]:
        cfg = get_config(stage)
        print(f"  {stage}: use_4bit={cfg.use_4bit}  use_dora={cfg.use_dora}  "
              f"r={cfg.lora_r}  alpha={cfg.lora_alpha}")

    if PEFT_AVAILABLE:
        for stage in ["stage1", "stage2", "stage3"]:
            cfg      = get_config(stage)
            lora_cfg = get_lora_config(cfg)
            assert lora_cfg.r == cfg.lora_r
            assert lora_cfg.use_dora == cfg.use_dora
            print(f"  ✓ LoraConfig for {stage}: r={lora_cfg.r} dora={lora_cfg.use_dora}")

    print("\n✓ model.py self-test passed (no GPU needed)")