import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: ORPO alignment")
    parser.add_argument("--local_test",  action="store_true",
                        help="CPU mode: no GPU, no W&B, tiny data")
    parser.add_argument("--orpo_beta",   type=float, default=None)
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--no_wandb",    action="store_true")
    parser.add_argument("--base_model",  type=str,   default=None,
                        help="Override base model path (default: outputs/stage2_merged)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.local_test:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import torch
        import wandb
        from trl import ORPOTrainer, ORPOConfig
        from src.config  import get_config, print_config
        from src.model   import load_model_for_training, print_model_summary
        from src.dataset import ORPODataset
    except ImportError as e:
        print(f"[ERROR] {e}")
        print("  Run: pip install torch transformers peft bitsandbytes trl accelerate wandb")
        sys.exit(1)

    on_cpu = args.local_test or not torch.cuda.is_available()
    overrides = {}
    if args.orpo_beta:   overrides["orpo_beta"]          = args.orpo_beta
    if args.max_samples: overrides["max_train_samples"]   = args.max_samples
    if args.no_wandb:    overrides["report_to"]           = "none"

    if args.local_test:
        overrides.update({
            "device_map":              "cpu",
            "use_4bit":                False,
            "bf16":                    False,
            "fp16":                    False,
            "optim":                   "adamw_torch",
            "dataloader_num_workers":  0,
            "load_best_model_at_end":  False,
            "report_to":               "none",
            "max_train_samples":       20,
            "max_eval_samples":        5,
            "max_seq_length":          512,
            "num_train_epochs":        1,
            "lora_r":                  4,
            "lora_alpha":              8,
            "logging_steps":           1,
            "eval_steps":              5,
            "save_steps":              5,
        })

    cfg = get_config("stage3", **overrides)
    print_config(cfg)
    print(f"\n  Mode: {'CPU (local_test)' if on_cpu else 'GPU (RunPod)'}")

    base_model_path = args.base_model
    if not base_model_path:
        merged = ROOT / "outputs" / "stage2_merged"
        if merged.exists():
            base_model_path = str(merged)
            print(f"  Using Stage 2 merged: {base_model_path}")
        else:
            base_model_path = cfg.base_model_id
            print(f"  Stage 2 merged not found → using base: {base_model_path}")

    output_dir        = ROOT / cfg.output_dir
    orpo_dataset_path = ROOT / cfg.orpo_dataset_path
    output_dir.mkdir(parents=True, exist_ok=True)

    if not orpo_dataset_path.exists():
        print(f"[ERROR] ORPO dataset not found: {orpo_dataset_path}")
        print("  Run: python scripts/build_orpo_pairs.py")
        sys.exit(1)

    if cfg.report_to == "wandb":
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=cfg.run_name,
            config=cfg.as_dict(),
            tags=["stage3", "orpo", "alignment", "qwen2.5-1.5b"],
        )
        print(f"  ✓ W&B: {cfg.wandb_project}/{cfg.run_name}")

    model, tokenizer = load_model_for_training(cfg, model_id_override=base_model_path)
    print_model_summary(model)

    train_ds = ORPODataset(str(orpo_dataset_path), tokenizer, cfg,
                           "train",      cfg.max_train_samples)
    has_val  = True
    try:
        eval_ds = ORPODataset(str(orpo_dataset_path), tokenizer, cfg,
                              "validation", cfg.max_eval_samples)
        if len(eval_ds) == 0:
            eval_ds  = None
            has_val  = False
    except Exception:
        eval_ds = None
        has_val = False

    use_bf16 = cfg.bf16 and not on_cpu
    use_fp16 = cfg.fp16 and not on_cpu

    orpo_args = ORPOConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = cfg.num_train_epochs,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        per_device_eval_batch_size  = cfg.per_device_eval_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        learning_rate               = cfg.learning_rate,
        lr_scheduler_type           = cfg.lr_scheduler_type,
        warmup_steps                = max(1, int(cfg.warmup_ratio * max(len(train_ds), 1))),
        weight_decay                = cfg.weight_decay,
        max_grad_norm               = cfg.max_grad_norm,
        optim                       = cfg.optim,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        gradient_checkpointing      = cfg.gradient_checkpointing,
        max_length                  = cfg.max_seq_length,
        max_prompt_length           = cfg.max_seq_length // 2,
        beta                        = cfg.orpo_beta,
        logging_steps               = cfg.logging_steps,
        eval_strategy               = "steps" if has_val else "no",
        eval_steps                  = cfg.eval_steps if has_val else None,
        save_strategy               = "steps",
        save_steps                  = cfg.save_steps,
        save_total_limit            = cfg.save_total_limit,
        load_best_model_at_end      = cfg.load_best_model_at_end and has_val,
        report_to                   = cfg.report_to,
        run_name                    = cfg.run_name,
        seed                        = cfg.seed,
        remove_unused_columns       = False,
        dataloader_num_workers      = cfg.dataloader_num_workers,
    )

    trainer = ORPOTrainer(
        model            = model,
        args             = orpo_args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        processing_class = tokenizer,
    )

    print(f"\n{'═'*56}")
    print(f"  Stage 3 ORPO — {'CPU local_test' if on_cpu else 'GPU RunPod'}")
    print(f"  Train pairs: {len(train_ds):,}")
    print(f"  beta={cfg.orpo_beta}  bf16={use_bf16}  optim={cfg.optim}")
    print(f"{'═'*56}\n")

    trainer.train()
    
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n  ✓ Stage 3 adapter saved → {final_dir.relative_to(ROOT)}")

    if cfg.report_to == "wandb":
        wandb.finish()

    print(f"  Next: python train/merge_and_push.py --stage 3 --push\n")


if __name__ == "__main__":
    main()