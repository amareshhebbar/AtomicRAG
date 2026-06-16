import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_test",  action="store_true")
    parser.add_argument("--orpo_beta",   type=float, default=None)
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--no_wandb",    action="store_true")
    parser.add_argument("--base_model",  type=str,   default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.local_test:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import torch
        import torch.nn.functional as F
        import wandb
        from torch.utils.data import Dataset as TorchDataset
        from transformers import TrainingArguments, Trainer
        from src.config import get_config, print_config
        from src.model  import load_model_for_training, print_model_summary
    except ImportError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    on_cpu = args.local_test or not torch.cuda.is_available()

    overrides = {}
    if args.orpo_beta:   overrides["orpo_beta"]         = args.orpo_beta
    if args.max_samples: overrides["max_train_samples"]  = args.max_samples
    if args.no_wandb:    overrides["report_to"]          = "none"

    if args.local_test:
        overrides.update({
            "device_map":             "cpu",
            "use_4bit":               False,
            "bf16":                   False,
            "fp16":                   False,
            "optim":                  "adamw_torch",
            "dataloader_num_workers": 0,
            "load_best_model_at_end": False,
            "report_to":              "none",
            "max_train_samples":      20,
            "max_eval_samples":       5,
            "max_seq_length":         512,
            "num_train_epochs":       1,
            "lora_r":                 4,
            "lora_alpha":             8,
            "logging_steps":          1,
            "eval_steps":             5,
            "save_steps":             5,
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
            print(f"  Stage 2 merged not found, using base model")

    output_dir        = ROOT / cfg.output_dir
    orpo_dataset_path = ROOT / cfg.orpo_dataset_path
    output_dir.mkdir(parents=True, exist_ok=True)

    if not orpo_dataset_path.exists():
        print(f"[ERROR] ORPO dataset not found: {orpo_dataset_path}")
        sys.exit(1)

    if cfg.report_to == "wandb":
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=cfg.run_name,
            config=cfg.as_dict(),
            tags=["stage3", "orpo", "alignment"],
        )

    model, tokenizer = load_model_for_training(cfg, model_id_override=base_model_path)
    print_model_summary(model)

    from datasets import load_from_disk
    import json

    dsd = load_from_disk(str(orpo_dataset_path))

    class ORPOFlatDataset(TorchDataset):
        def __init__(self, hf_ds, tokenizer, max_len, max_samples):
            self.tokenizer = tokenizer
            self.max_len   = max_len
            self.examples  = []
            count = 0
            for row in hf_ds:
                if max_samples and count >= max_samples:
                    break
                prompt   = row.get("question", "")
                chosen   = row.get("chosen_json", "")
                rejected = row.get("rejected_json", "")
                if not prompt or not chosen or not rejected:
                    continue
                self.examples.append({
                    "prompt":   prompt,
                    "chosen":   chosen,
                    "rejected": rejected,
                })
                count += 1
            print(f"  ORPODataset: {len(self.examples)} valid pairs")

        def _tokenize(self, prompt, response):
            from src.utils import SYSTEM_PROMPT
            full = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )
            enc = self.tokenizer(
                full,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_tensors=None,
            )
            prompt_only = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompt_len = len(self.tokenizer(prompt_only, return_tensors=None)["input_ids"])
            ids    = enc["input_ids"]
            labels = [-100] * prompt_len + ids[prompt_len:]
            return {
                "input_ids":      torch.tensor(ids,                   dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels":         torch.tensor(labels,                dtype=torch.long),
            }

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            c  = self._tokenize(ex["prompt"], ex["chosen"])
            r  = self._tokenize(ex["prompt"], ex["rejected"])
            return {
                "chosen_input_ids":       c["input_ids"],
                "chosen_attention_mask":  c["attention_mask"],
                "chosen_labels":          c["labels"],
                "rejected_input_ids":     r["input_ids"],
                "rejected_attention_mask":r["attention_mask"],
                "rejected_labels":        r["labels"],
            }

    def orpo_collate(batch):
        def pad(seqs, pad_val):
            max_len = max(s.shape[0] for s in seqs)
            out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :s.shape[0]] = s
            return out

        return {
            "chosen_input_ids":       pad([b["chosen_input_ids"]   for b in batch], tokenizer.pad_token_id),
            "chosen_attention_mask":  pad([b["chosen_attention_mask"] for b in batch], 0),
            "chosen_labels":          pad([b["chosen_labels"]       for b in batch], -100),
            "rejected_input_ids":     pad([b["rejected_input_ids"]  for b in batch], tokenizer.pad_token_id),
            "rejected_attention_mask":pad([b["rejected_attention_mask"] for b in batch], 0),
            "rejected_labels":        pad([b["rejected_labels"]     for b in batch], -100),
        }

    train_split = dsd["train"]   if "train"      in dsd else None
    val_split   = dsd["validation"] if "validation" in dsd else None

    if train_split is None:
        print("[ERROR] No 'train' split in ORPO dataset")
        sys.exit(1)

    train_ds = ORPOFlatDataset(train_split, tokenizer, cfg.max_seq_length, cfg.max_train_samples)
    eval_ds  = ORPOFlatDataset(val_split,   tokenizer, cfg.max_seq_length, cfg.max_eval_samples) if val_split else None
    has_eval = eval_ds is not None and len(eval_ds) > 0

    beta = cfg.orpo_beta

    class ORPOTrainerCustom(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            chosen_ids   = inputs["chosen_input_ids"].to(model.device)
            chosen_mask  = inputs["chosen_attention_mask"].to(model.device)
            chosen_labels= inputs["chosen_labels"].to(model.device)
            rejected_ids = inputs["rejected_input_ids"].to(model.device)
            rejected_mask= inputs["rejected_attention_mask"].to(model.device)
            rejected_labels = inputs["rejected_labels"].to(model.device)

            chosen_out   = model(input_ids=chosen_ids,   attention_mask=chosen_mask,   labels=chosen_labels)
            rejected_out = model(input_ids=rejected_ids, attention_mask=rejected_mask, labels=rejected_labels)

            sft_loss = chosen_out.loss

            def seq_logprob(logits, labels, mask):
                log_probs = F.log_softmax(logits, dim=-1)
                active    = labels != -100
                token_lp  = torch.gather(log_probs, 2, labels.clamp(min=0).unsqueeze(2)).squeeze(2)
                token_lp  = token_lp * active.float()
                return token_lp.sum(dim=-1) / active.float().sum(dim=-1).clamp(min=1)

            c_lp = seq_logprob(chosen_out.logits,   chosen_labels,   chosen_mask)
            r_lp = seq_logprob(rejected_out.logits, rejected_labels, rejected_mask)

            log_odds  = (c_lp - r_lp) - (torch.log1p(-torch.exp(c_lp).clamp(max=1-1e-6)) -
                                           torch.log1p(-torch.exp(r_lp).clamp(max=1-1e-6)))
            orpo_loss = -F.logsigmoid(log_odds).mean()

            loss = sft_loss + beta * orpo_loss

            if return_outputs:
                return loss, chosen_out
            return loss

    use_bf16 = cfg.bf16 and not on_cpu
    use_fp16 = cfg.fp16 and not on_cpu
    load_best = cfg.load_best_model_at_end and has_eval

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
        label_names                 = ["chosen_labels"],
    )

    trainer = ORPOTrainerCustom(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds if has_eval else None,
        processing_class = tokenizer,
        data_collator    = orpo_collate,
    )

    print(f"\n{'═'*56}")
    print(f"  Stage 3 ORPO — {'CPU local_test' if on_cpu else 'GPU RunPod'}")
    print(f"  Train pairs: {len(train_ds)}")
    print(f"  beta={beta}  bf16={use_bf16}  optim={cfg.optim}")
    print(f"{'═'*56}\n")

    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n  Stage 3 adapter saved: {final_dir.relative_to(ROOT)}")

    if cfg.report_to == "wandb":
        wandb.finish()

    print(f"  Next: python train/merge_and_push.py --stage 3 --push\n")


if __name__ == "__main__":
    main()