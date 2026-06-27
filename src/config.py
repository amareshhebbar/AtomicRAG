from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BaseConfig:
    base_model_id:  str= "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str= "main"
    data_dir:  str= "data/processed"
    hf_dataset_path:str= "data/processed/sft_dataset"
    orpo_dataset_path:str= "data/processed/orpo_dataset"
    output_dir:str= "outputs"
    max_seq_length: int= 1024
    padding_side:str= "right"
    use_4bit:  bool= False
    bnb_4bit_compute_dtype: str= "bfloat16"
    bnb_4bit_quant_type: str= "nf4"
    bnb_double_quant: bool= True
    device_map:str= "auto"
    wandb_project:  str= "querydecomp"
    wandb_entity:str= ""
    report_to: str= "wandb"
    hf_repo_id:str= "AmareshHebbar/querydecomp-qwen2.5-1.5b"
    push_to_hub: bool= False
    seed:int= 42
    stage1_merged_path:str= "outputs/stage1_merged"
    stage2_merged_path:str= "outputs/stage2_merged"
    orpo_beta: float = 0.1
    num_train_epochs: int= 1
    per_device_train_batch_size: int= 4
    per_device_eval_batch_size:  int= 4
    gradient_accumulation_steps: int= 4
    learning_rate:  float = 2e-4
    lr_scheduler_type:str= "cosine"
    warmup_ratio:float = 0.05
    weight_decay:float = 0.01
    max_grad_norm:  float = 1.0
    optim:str= "paged_adamw_32bit"
    gradient_checkpointing: bool= True
    bf16:bool= True
    fp16:bool= False
    dataloader_num_workers: int= 4
    logging_steps:  int= 10
    eval_steps:int= 100
    save_steps:int= 100
    save_total_limit: int= 3
    load_best_model_at_end: bool= True
    metric_for_best_model:  str= "eval_loss"
    greater_is_better:bool= False
    train_split: str= "train"
    eval_split:str= "validation"
    max_train_samples:Optional[int] = None
    max_eval_samples: Optional[int] = None
    lora_r: int= 16
    lora_alpha:int= 32
    lora_dropout:float = 0.05
    use_dora:  bool= False
    target_modules: list= field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)
        print(f"Config saved → {path}")

    @property
    def root(self) -> Path:
        return ROOT


@dataclass
class Stage1Config(BaseConfig):
    stage:str= "stage1_qlora"
    output_dir:str= "outputs/stage1_qlora"
    run_name:  str= "stage1-qlora-r16"
    use_4bit:  bool= False
    lora_r: int= 16
    lora_alpha:int= 32
    use_dora:  bool= False
    num_train_epochs: int= 3
    per_device_train_batch_size: int= 8
    per_device_eval_batch_size:  int= 8
    gradient_accumulation_steps: int= 2
    learning_rate:  float = 2e-4
    max_seq_length: int= 1024
    optim:str= "paged_adamw_32bit"
    gradient_checkpointing: bool= True
    bf16:bool= True
    dataloader_num_workers: int= 4
    logging_steps:  int= 10
    eval_steps:int= 200
    save_steps:int= 200
    save_total_limit: int= 3


@dataclass
class Stage2Config(BaseConfig):
    stage:str= "stage2_dora"
    output_dir:str= "outputs/stage2_dora"
    run_name:  str= "stage2-dora"
    stage1_merged_path:str= "outputs/stage1_merged"
    use_4bit:  bool= False
    lora_r: int= 8
    lora_alpha:int= 16
    use_dora:  bool= True
    num_train_epochs: int= 1
    per_device_train_batch_size: int= 8
    per_device_eval_batch_size:  int= 8
    gradient_accumulation_steps: int= 2
    learning_rate:  float = 5e-5
    max_seq_length: int= 1024
    optim:str= "paged_adamw_32bit"
    gradient_checkpointing: bool= True
    bf16:bool= True
    dataloader_num_workers: int= 4
    logging_steps:  int= 10
    eval_steps:int= 100
    save_steps:int= 100
    save_total_limit: int= 2


@dataclass
class Stage3Config(BaseConfig):
    stage:str= "stage3_orpo"
    output_dir:str= "outputs/stage3_orpo"
    run_name:  str= "stage3-orpo"
    stage2_merged_path:str= "outputs/stage2_merged"
    use_4bit:  bool= False
    lora_r: int= 8
    lora_alpha:int= 16
    use_dora:  bool= False
    orpo_beta: float = 0.1
    max_seq_length: int= 2048
    num_train_epochs: int= 1
    per_device_train_batch_size: int= 4
    per_device_eval_batch_size:  int= 4
    gradient_accumulation_steps: int= 4
    learning_rate:  float = 5e-6
    optim:str= "paged_adamw_32bit"
    gradient_checkpointing: bool= True
    bf16:bool= True
    dataloader_num_workers: int= 4
    logging_steps:  int= 5
    eval_steps:int= 100
    save_steps:int= 100
    save_total_limit: int= 2


@dataclass
class LocalTestConfig(BaseConfig):
    stage:str= "local_test"
    output_dir:str= "outputs/stage1_qlora"
    run_name:  str= "local-test"
    device_map:str= "cpu"
    use_4bit:  bool= False
    bf16:bool= False
    fp16:bool= False
    optim:str= "adamw_torch"
    dataloader_num_workers: int= 0
    gradient_checkpointing: bool= False
    report_to: str= "none"
    load_best_model_at_end: bool= False
    max_seq_length: int= 256
    num_train_epochs: int= 1
    per_device_train_batch_size: int= 1
    per_device_eval_batch_size:  int= 1
    gradient_accumulation_steps: int= 1
    max_train_samples:int= 20
    max_eval_samples: int= 5
    logging_steps:  int= 1
    eval_steps:int= 5
    save_steps:int= 5
    save_total_limit: int= 1
    lora_r: int= 4
    lora_alpha:int= 8
    use_dora:  bool= False


_CONFIG_MAP = {
    "stage1":Stage1Config,
    "stage2":Stage2Config,
    "stage3":Stage3Config,
    "local_test": LocalTestConfig,
}


def get_config(stage: str, **overrides) -> BaseConfig:
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


if __name__ == "__main__":
    for stage in ["stage1", "stage2", "stage3", "local_test"]:
        cfg = get_config(stage)
        print(f"✓ {stage:12s}  lr={cfg.learning_rate}  epochs={cfg.num_train_epochs}  "
              f"r={cfg.lora_r}  dora={cfg.use_dora}  4bit={cfg.use_4bit}  "
              f"batch={cfg.per_device_train_batch_size}  seq={cfg.max_seq_length}")
    cfg = get_config("stage1", lora_r=32, learning_rate=1e-4)
    assert cfg.lora_r == 32
    assert cfg.learning_rate == 1e-4
    print("\n✓ All configs OK")