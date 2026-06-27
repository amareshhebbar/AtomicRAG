import os
from pathlib import Path
from typing import Optional

from src.config import get_config

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


def load_tokenizer(model_id: str, padding_side: str = "right"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side=padding_side,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token ({tokenizer.eos_token!r})")

    print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size:,}  "
          f"pad={tokenizer.pad_token!r}  eos={tokenizer.eos_token!r}")

    return tokenizer


def get_bnb_config(cfg: BaseConfig) -> "BitsAndBytesConfig":
    if not cfg.use_4bit:
        return None

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype) 

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,           
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_double_quant,
    )


def get_lora_config(cfg: BaseConfig) -> "LoraConfig":
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        use_dora=cfg.use_dora,              
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        inference_mode=False,
    )

def load_model_for_training(cfg: BaseConfig, model_id_override: Optional[str] = None):
    model_id = model_id_override or cfg.base_model_id
    stage = getattr(cfg, "stage", "unknown")

    print(f"\n{'─' * 56}")
    print(f"  Loading model for {stage}")
    print(f"  Model: {model_id}")
    print(f"  4-bit: {cfg.use_4bit}  DoRA: {cfg.use_dora}  r={cfg.lora_r}")
    print(f"{'─' * 56}")

    tokenizer = load_tokenizer(model_id, cfg.padding_side)

    bnb_config = get_bnb_config(cfg) 
    
    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        device_map=cfg.device_map,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    if not cfg.use_4bit:
        on_cpu = getattr(cfg, "device_map", "auto") == "cpu"
        load_kwargs["dtype"] = torch.float32 if on_cpu else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    print(f"  ✓ Base model loaded — params: {count_params(model)}")

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
        )
        print(f"  ✓ Prepared for k-bit training")
        
    lora_cfg = get_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)

    adapter_type = "DoRA" if cfg.use_dora else "LoRA"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total= sum(p.numel() for p in model.parameters())
    pct= 100 * trainable / total

    print(f"  ✓ {adapter_type} applied — "
          f"trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
    print(f"  Target modules: {cfg.target_modules}")

    if cfg.gradient_checkpointing and not cfg.use_4bit:
        model.gradient_checkpointing_enable()
        print(f"  ✓ Gradient checkpointing enabled")
        
    model.config.use_cache = False

    return model, tokenizer

def load_model_for_inference(model_path: str, device: str = "auto"):
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


def merge_adapter(
    base_model_id:  str,
    adapter_path:str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo_id: str= "",
):
    print(f"\n{'─' * 56}")
    print(f"  Merging adapter")
    print(f"  Base: {base_model_id}")
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


def count_params(model) -> str:
    total= sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_b= total / 1e9
    return f"{total_b:.2f}B total, {trainable:,} trainable"


def get_gpu_memory_gb() -> float:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def print_model_summary(model):
    print(f"\n  Model summary:")
    print(f"    Params: {count_params(model)}")
    print(f"    VRAM used:{get_gpu_memory_gb():.2f} GB")
    if hasattr(model, "config"):
        cfg = model.config
        print(f"    Hidden size: {getattr(cfg, 'hidden_size', '?')}")
        print(f"    Layers: {getattr(cfg, 'num_hidden_layers', '?')}")
        print(f"    Heads:  {getattr(cfg, 'num_attention_heads', '?')}")

if __name__ == "__main__":

    for stage in ["stage1", "stage2", "stage3"]:
        cfg = get_config(stage)
        print(f"  {stage}: use_4bit={cfg.use_4bit}  use_dora={cfg.use_dora}  "
              f"r={cfg.lora_r}  alpha={cfg.lora_alpha}")

    if PEFT_AVAILABLE:
        for stage in ["stage1", "stage2", "stage3"]:
            cfg = get_config(stage)
            lora_cfg = get_lora_config(cfg)
            assert lora_cfg.r == cfg.lora_r
            assert lora_cfg.use_dora == cfg.use_dora
            print(f"  ✓ LoraConfig for {stage}: r={lora_cfg.r} dora={lora_cfg.use_dora}")

    print("\n✓ model.py self-test passed (no GPU needed)")