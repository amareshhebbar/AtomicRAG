import os
import sys
import json
import argparse
from pathlib import Path
try:
    import torch
    import wandb
    from transformers import TrainingArguments, Trainer
    from src.config  import get_config, print_config
    from src.model   import load_model_for_training, print_model_summary
    from src.dataset import SFTDataset, get_sft_collator, decode_batch_sample
    from src.utils   import parse_decomp_output, format_dep_graph, build_prompt
    from transformers import TrainerCallback
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    print("  Run: pip install torch transformers peft bitsandbytes trl accelerate wandb")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_test",    action="store_true")
    parser.add_argument("--lora_r",        type=int,   default=None)
    parser.add_argument("--lora_alpha",    type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs",    type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--max_samples",   type=int,   default=None)
    parser.add_argument("--resume_from",   type=str,   default=None)
    parser.add_argument("--no_wandb",      action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.local_test:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    on_cpu = args.local_test or not torch.cuda.is_available()

    stage = "local_test" if args.local_test else "stage1"

    overrides = {}
    if args.lora_r:        
        overrides["lora_r"] = args.lora_r
    if args.lora_alpha:    
        overrides["lora_alpha"] = args.lora_alpha
    if args.learning_rate: 
        overrides["learning_rate"]= args.learning_rate
    if args.num_epochs:    
        overrides["num_train_epochs"] = args.num_epochs
    if args.batch_size:    
        overrides["per_device_train_batch_size"] = args.batch_size
    if args.max_samples:   
        overrides["max_train_samples"]= args.max_samples
    if args.no_wandb:      
        overrides["report_to"] = "none"

    cfg = get_config(stage, **overrides)
    print_config(cfg)

    print(f"\n  Mode: {'CPU (local_test)' if on_cpu else 'GPU (RunPod)'}")
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")

    output_dir      = ROOT / cfg.output_dir
    hf_dataset_path = ROOT / cfg.hf_dataset_path
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(str(output_dir / "config.json"))
    
    if cfg.report_to == "wandb":
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=cfg.run_name,
            config=cfg.as_dict(),
            tags=["stage1", "qlora", "sft", "qwen2.5-1.5b"],
        )
        print(f"  ✓ W&B: {cfg.wandb_project}/{cfg.run_name}")

    model, tokenizer = load_model_for_training(cfg)
    print_model_summary(model)

    if not hf_dataset_path.exists():
        print(f"\n[ERROR] Dataset not found: {hf_dataset_path}")
        print("  Run: make data  or  make data MAX_ROWS=30")
        sys.exit(1)

    train_ds = SFTDataset(
        hf_dataset_path=str(hf_dataset_path),
        tokenizer=tokenizer,
        cfg=cfg,
        split="train",
        max_samples=cfg.max_train_samples,
    )
    eval_ds = SFTDataset(
        hf_dataset_path=str(hf_dataset_path),
        tokenizer=tokenizer,
        cfg=cfg,
        split="validation",
        max_samples=cfg.max_eval_samples,
    )

    collator = get_sft_collator(tokenizer, cfg.max_seq_length)

    if len(train_ds) > 0:
        batch = collator([train_ds[0]])
        n_active = (batch["labels"] != -100).sum().item()
        print(f"\n  Batch check: shape={batch['input_ids'].shape}  "
              f"active_label_tokens={n_active}")
        if on_cpu:
            print(decode_batch_sample(batch, tokenizer, 0))

    has_eval  = len(eval_ds) > 0
    load_best = cfg.load_best_model_at_end and has_eval

    use_bf16 = cfg.bf16 and not on_cpu
    use_fp16 = cfg.fp16 and not on_cpu

    training_args = TrainingArguments(
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
        dataloader_num_workers      = cfg.dataloader_num_workers,
        logging_steps               = cfg.logging_steps,
        eval_strategy               = "steps" if has_eval else "no",
        eval_steps                  = cfg.eval_steps if has_eval else None,
        save_strategy               = "steps",
        save_steps                  = cfg.save_steps,
        save_total_limit            = cfg.save_total_limit,
        load_best_model_at_end      = load_best,
        metric_for_best_model       = cfg.metric_for_best_model if load_best else None,
        greater_is_better           = cfg.greater_is_better,
        report_to                   = cfg.report_to,
        run_name                    = cfg.run_name,
        seed                        = cfg.seed,
        remove_unused_columns       = False,
        label_names                 = ["labels"],
    )
    
    

    class SampleOutputCallback(TrainerCallback):
        QUESTIONS = [
            "Where was the director of Inception born?",
            "What is the capital of the country where Einstein was born?",
        ]
        def on_evaluate(self, args, state, control, **kwargs):
            if cfg.report_to != "wandb": return
            if state.global_step % (cfg.eval_steps * 2) != 0: return
            m = kwargs.get("model")
            t = kwargs.get("processing_class") or tokenizer
            if m is None: return
            m.eval(); rows = []
            for q in self.QUESTIONS:
                inp = t(build_prompt(q, t), return_tensors="pt").to(m.device)
                with torch.no_grad():
                    out = m.generate(**inp, max_new_tokens=256, do_sample=False)
                gen = t.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                parsed = parse_decomp_output(gen)
                rows.append([state.global_step, q,
                             format_dep_graph(parsed) if parsed else f"FAILED:\n{gen}"])
            import wandb as _wb
            _wb.log({"sample_decompositions": _wb.Table(
                columns=["step","question","decomposition"], data=rows)})
            m.train()

    class VRAMLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not torch.cuda.is_available() or cfg.report_to != "wandb": return
            if logs:
                logs["vram_gb"] = torch.cuda.memory_allocated() / 1e9

    trainer = Trainer(
        model             = model,
        args              = training_args,
        train_dataset     = train_ds,
        eval_dataset      = eval_ds if has_eval else None,
        processing_class  = tokenizer,
        data_collator     = collator,
        callbacks         = [SampleOutputCallback(), VRAMLogCallback()],
    )

    print(f"\n{'═'*56}")
    print(f"  Stage 1 — {'CPU local test' if on_cpu else 'GPU RunPod'}")
    print(f"  Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")
    print(f"  bf16={use_bf16}  fp16={use_fp16}  optim={cfg.optim}")
    print(f"{'═'*56}\n")

    trainer.train(resume_from_checkpoint=args.resume_from or None)

    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n  ✓ Adapter saved → {final_dir.relative_to(ROOT)}")

    if cfg.report_to == "wandb":
        wandb.finish()

    print(f"\n  Next: python train/merge_and_push.py --stage 1\n")


if __name__ == "__main__":
    main()