"""
src/config.py
-------------
Single source of truth for ALL hyperparameters across all training stages.
Import this everywhere — never hardcode values in training scripts.

Usage:
    from src.config import Stage1Config, Stage2Config, Stage3Config, get_config
    cfg = get_config("stage1")           # returns Stage1Config
    cfg = get_config("stage1", lora_r=32)  # override any field
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Base — shared across all stages
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseConfig:
    # Model
    base_model_id:      str  = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision:     str  = "main"

    # Data paths (relative to ROOT)
    data_dir:           str  = "data/processed"
    hf_dataset_path:    str  = "data/processed/sft_dataset"
    orpo_dataset_path:  str  = "data/processed/orpo_dataset"
    output_dir:         str  = "outputs"          # overridden per stage

    # Tokenizer
    max_seq_length:     int  = 1024
    padding_side:       str  = "right"            # right for decoder-only SFT

    # Hardware
    use_4bit:           bool = True               # QLoRA 4-bit quantization
    bnb_4bit_compute_dtype: str = "bfloat16"      # compute in bf16
    bnb_4bit_quant_type:    str = "nf4"           # NormalFloat4
    bnb_double_quant:   bool = True               # quantize the quant constants
    device_map:         str  = "auto"

    # W&B
    wandb_project:      str  = "querydecomp"
    wandb_entity:       str  = ""                 # your W&B username
    report_to:          str  = "wandb"            # "none" to disable

    # HuggingFace Hub
    hf_repo_id:         str  = "AmareshHebbar/querydecomp-qwen2.5-1.5b"
    push_to_hub:        bool = False              # set True only in merge_and_push.py

    # Reproducibility
    seed:               int  = 42

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)
        print(f"Config saved → {path}")

    @property
    def root(self) -> Path:
        return ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — QLoRA SFT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage1Config(BaseConfig):
    stage:              str  = "stage1_qlora"
    output_dir:         str  = "outputs/stage1_qlora"
    run_name:           str  = "stage1-qlora-r16"

    # LoRA
    lora_r:             int  = 16
    lora_alpha:         int  = 32       # 2x rank is standard
    lora_dropout:       float = 0.05
    use_dora:           bool = False    # DoRA is Stage 2 only
    target_modules:     list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    num_train_epochs:   int   = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size:  int = 4
    gradient_accumulation_steps: int = 4    # effective batch = 16
    learning_rate:      float = 2e-4
    lr_scheduler_type:  str   = "cosine"
    warmup_ratio:       float = 0.05
    weight_decay:       float = 0.01
    max_grad_norm:      float = 1.0
    optim:              str   = "paged_adamw_32bit"   # memory-efficient

    # Efficiency
    gradient_checkpointing: bool = True
    bf16:               bool = True
    fp16:               bool = False    # bf16 is better on A5000
    dataloader_num_workers: int = 4

    # Logging & saving
    logging_steps:      int  = 10
    eval_steps:         int  = 100
    save_steps:         int  = 100
    save_total_limit:   int  = 3        # keep only 3 checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model:  str  = "eval_loss"
    greater_is_better:  bool = False

    # Data
    train_split:        str  = "train"
    eval_split:         str  = "validation"
    max_train_samples:  Optional[int] = None   # None = use all
    max_eval_samples:   Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — DoRA Refinement
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage2Config(BaseConfig):
    stage:              str  = "stage2_dora"
    output_dir:         str  = "outputs/stage2_dora"
    run_name:           str  = "stage2-dora"

    # Stage 2 loads from the merged Stage 1 model (not the original base)
    # Set this to the merged bf16 model path after Stage 1
    stage1_merged_path: str  = "outputs/stage1_merged"

    # DoRA — same structure as LoRA but use_dora=True
    lora_r:             int  = 8         # lower rank — refinement not learning
    lora_alpha:         int  = 16
    lora_dropout:       float = 0.05
    use_dora:           bool = True      # KEY DIFFERENCE from Stage 1
    target_modules:     list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training — lower LR, 1 epoch only (refinement)
    num_train_epochs:   int   = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate:      float = 5e-5    # 4x lower than Stage 1
    lr_scheduler_type:  str   = "cosine"
    warmup_ratio:       float = 0.05
    weight_decay:       float = 0.01
    max_grad_norm:      float = 1.0
    optim:              str   = "paged_adamw_32bit"

    # No 4-bit in Stage 2 — we merged Stage 1 into bf16
    use_4bit:           bool = False

    gradient_checkpointing: bool = True
    bf16:               bool = True
    fp16:               bool = False

    logging_steps:      int  = 10
    eval_steps:         int  = 50
    save_steps:         int  = 50
    save_total_limit:   int  = 2
    load_best_model_at_end: bool = True
    metric_for_best_model:  str  = "eval_loss"
    greater_is_better:  bool = False

    train_split:        str  = "train"
    eval_split:         str  = "validation"
    max_train_samples:  Optional[int] = None
    max_eval_samples:   Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — ORPO Alignment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage3Config(BaseConfig):
    stage:              str  = "stage3_orpo"
    output_dir:         str  = "outputs/stage3_orpo"
    run_name:           str  = "stage3-orpo"

    # Stage 3 loads from merged Stage 2 model
    stage2_merged_path: str  = "outputs/stage2_merged"

    # LoRA on top of Stage 2 merged model (for ORPO)
    lora_r:             int  = 8
    lora_alpha:         int  = 16
    lora_dropout:       float = 0.05
    use_dora:           bool = False
    target_modules:     list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ORPO-specific
    orpo_beta:          float = 0.1     # regularization weight
    max_seq_length:     int   = 2048    # longer — need room for chosen+rejected

    # Training — very low LR, 1 epoch
    num_train_epochs:   int   = 1
    per_device_train_batch_size: int = 2   # pairs = 2x sequence length
    gradient_accumulation_steps: int = 8   # effective batch = 16
    learning_rate:      float = 5e-6       # very low — alignment not SFT
    lr_scheduler_type:  str   = "cosine"
    warmup_ratio:       float = 0.05
    weight_decay:       float = 0.01
    max_grad_norm:      float = 1.0
    optim:              str   = "paged_adamw_32bit"

    use_4bit:           bool = False       # Stage 2 merged is bf16

    gradient_checkpointing: bool = True
    bf16:               bool = True
    fp16:               bool = False

    logging_steps:      int  = 5
    eval_steps:         int  = 50
    save_steps:         int  = 50
    save_total_limit:   int  = 2
    load_best_model_at_end: bool = True
    metric_for_best_model:  str  = "eval_loss"
    greater_is_better:  bool = False

    train_split:        str  = "train"
    eval_split:         str  = "validation"
    max_train_samples:  Optional[int] = None
    max_eval_samples:   Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Local test override — tiny everything for CPU/low-VRAM testing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalTestConfig(Stage1Config):
    stage:              str  = "local_test"
    output_dir:         str  = "outputs/stage1_qlora"   # same as Stage1 so merge works
    run_name:           str  = "local-test"

    # ── Force CPU — avoids OOM on low-VRAM laptops (GTX 1650 = 4GB) ─────────
    device_map:         str  = "cpu"        # load entirely on CPU
    use_4bit:           bool = False        # no quantization needed on CPU
    bf16:               bool = False        # bf16 not supported on CPU
    fp16:               bool = False        # fp16 also off for CPU
    optim:              str  = "adamw_torch" # paged_adamw_32bit needs bitsandbytes+GPU
    dataloader_num_workers: int = 0         # no multiprocessing on CPU

    max_seq_length:     int  = 256
    num_train_epochs:   int  = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_train_samples:  int  = 20
    max_eval_samples:   int  = 5
    logging_steps:      int  = 1
    eval_steps:         int  = 5
    save_steps:         int  = 5
    save_total_limit:   int  = 1
    load_best_model_at_end: bool = False    # needs matching eval, skip for tiny test
    report_to:          str  = "none"       # no W&B for local test
    gradient_checkpointing: bool = False    # not needed for tiny batches
    lora_r:             int  = 4
    lora_alpha:         int  = 8


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_MAP = {
    "stage1":     Stage1Config,
    "stage2":     Stage2Config,
    "stage3":     Stage3Config,
    "local_test": LocalTestConfig,
}


def get_config(stage: str, **overrides) -> BaseConfig:
    """
    Get a config for a given stage, with optional field overrides.

    Example:
        cfg = get_config("stage1", lora_r=32, learning_rate=1e-4)
    """
    if stage not in _CONFIG_MAP:
        raise ValueError(f"Unknown stage: {stage!r}. Choose from: {list(_CONFIG_MAP)}")

    cls = _CONFIG_MAP[stage]
    cfg = cls()

    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Config has no field: {key!r}")
        setattr(cfg, key, val)

    return cfg


def print_config(cfg: BaseConfig):
    print(f"\n{'═' * 56}")
    print(f"  Config: {cfg.stage}")
    print(f"{'═' * 56}")
    for k, v in asdict(cfg).items():
        print(f"  {k:35s} = {v}")
    print(f"{'═' * 56}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for stage in ["stage1", "stage2", "stage3", "local_test"]:
        cfg = get_config(stage)
        print(f"✓ {stage:12s}  lr={cfg.learning_rate}  epochs={cfg.num_train_epochs}  "
              f"r={cfg.lora_r}  dora={cfg.use_dora}  4bit={cfg.use_4bit}")

    # Test override
    cfg = get_config("stage1", lora_r=32, learning_rate=1e-4)
    assert cfg.lora_r == 32
    assert cfg.learning_rate == 1e-4
    print("\n✓ Override test passed")
    print("\n✓ All configs OK")