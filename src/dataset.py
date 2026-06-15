import json
from pathlib import Path
from typing import Optional

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq
    from datasets import load_from_disk, Dataset as HFDataset
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("[WARN] torch/transformers not installed — install before training")
    class Dataset: 
        pass
    PreTrainedTokenizer = object
    HFDataset = object

from src.config import BaseConfig
from src.utils import build_training_text, SYSTEM_PROMPT


class SFTDataset(Dataset):
    def __init__(
        self,
        hf_dataset_path: str,
        tokenizer:        "PreTrainedTokenizer",
        cfg:              BaseConfig,
        split:            str = "train",
        max_samples:      Optional[int] = None,
    ):
        self.tokenizer   = tokenizer
        self.max_length  = cfg.max_seq_length
        self.split       = split

        dsd = load_from_disk(hf_dataset_path)

        if split not in dsd:
            available = list(dsd.keys())
            raise ValueError(
                f"Split '{split}' not found in dataset at {hf_dataset_path}. "
                f"Available: {available}"
            )

        hf_ds = dsd[split]

        if max_samples and max_samples < len(hf_ds):
            hf_ds = hf_ds.select(range(max_samples))

        print(f"  SFTDataset [{split}]: {len(hf_ds):,} examples  "
              f"max_len={self.max_length}")
        
        self.examples = self._tokenize_all(hf_ds)

        print(f"  ✓ Tokenized: {len(self.examples):,} valid examples "
              f"({len(hf_ds) - len(self.examples)} filtered by max_length)")

    def _tokenize_all(self, hf_ds: "HFDataset") -> list[dict]:
        results = []

        for row in hf_ds:
            try:
                messages = json.loads(row["messages_json"])
            except (json.JSONDecodeError, KeyError):
                continue

            tokenized = self._tokenize_one(messages)
            if tokenized is not None:
                results.append(tokenized)

        return results

    def _tokenize_one(self, messages: list[dict]) -> Optional[dict]:
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text     = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,   
        )

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        if len(tokenized["input_ids"]) >= self.max_length:
            return None  

        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_tokenized["input_ids"])

        input_ids = tokenized["input_ids"]
        labels    = [-100] * prompt_len + input_ids[prompt_len:]

        assert len(labels) == len(input_ids), (
            f"Label length mismatch: {len(labels)} vs {len(input_ids)}"
        )

        return {
            "input_ids":      torch.tensor(input_ids,                   dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(labels,                      dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class ORPODataset(Dataset):

    def __init__(
        self,
        hf_dataset_path: str,
        tokenizer:        "PreTrainedTokenizer",
        cfg:              BaseConfig,
        split:            str = "train",
        max_samples:      Optional[int] = None,
    ):
        self.tokenizer  = tokenizer
        self.max_length = cfg.max_seq_length

        dsd = load_from_disk(hf_dataset_path)

        if split not in dsd:
            available = list(dsd.keys())
            raise ValueError(
                f"Split '{split}' not in ORPO dataset at {hf_dataset_path}. "
                f"Available: {available}"
            )

        hf_ds = dsd[split]

        if max_samples and max_samples < len(hf_ds):
            hf_ds = hf_ds.select(range(max_samples))

        print(f"  ORPODataset [{split}]: {len(hf_ds):,} examples")
        self.examples = self._build_all(hf_ds)
        print(f"  ✓ Built: {len(self.examples):,} valid pairs")

    def _build_one(self, row: dict) -> Optional[dict]:
        try:
            question = row["question"]
            chosen   = row["chosen_json"]
            rejected = row["rejected_json"]
        except KeyError:
            return None

        if not question or not chosen or not rejected:
            return None
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt":            prompt,
            "chosen":            chosen.strip(),
            "rejected":          rejected.strip(),
            "rejection_type":    row.get("rejection_type", "unknown"),
        }

    def _build_all(self, hf_ds: "HFDataset") -> list[dict]:
        results = []
        for row in hf_ds:
            ex = self._build_one(row)
            if ex is not None:
                results.append(ex)
        return results

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def get_sft_collator(tokenizer: "PreTrainedTokenizer", max_length: int):
    """
    Data collator for SFT training.
    Pads sequences in a batch to the same length.
    label_pad_token_id=-100 means padded label positions don't contribute to loss.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,        
        label_pad_token_id=-100,
        return_tensors="pt",
    )


def decode_batch_sample(batch: dict, tokenizer, idx: int = 0) -> str:
    input_ids = batch["input_ids"][idx].tolist()
    labels    = batch["labels"][idx].tolist()

    prompt_tokens    = [t for t, l in zip(input_ids, labels) if l == -100]
    assistant_tokens = [t for t, l in zip(input_ids, labels) if l != -100]

    prompt_text    = tokenizer.decode(prompt_tokens,    skip_special_tokens=False)
    assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=False)

    return (
        f"=== Batch sample {idx} ===\n"
        f"[PROMPT ({len(prompt_tokens)} tokens, masked from loss)]:\n{prompt_text}\n\n"
        f"[ASSISTANT ({len(assistant_tokens)} tokens, contributes to loss)]:\n{assistant_text}"
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    hf_path = ROOT / "data" / "processed" / "sft_dataset"

    if not hf_path.exists():
        print(f"[ERROR] HF dataset not found at {hf_path}")
        print("  Run: python scripts/build_splits.py")
        sys.exit(1)

    if not DEPS_AVAILABLE:
        print("[ERROR] torch/transformers not installed")
        sys.exit(1)

    from transformers import AutoTokenizer
    from src.config import get_config

    cfg       = get_config("local_test")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nTesting SFTDataset...")
    ds = SFTDataset(
        hf_dataset_path=str(hf_path),
        tokenizer=tokenizer,
        cfg=cfg,
        split="train",
        max_samples=3,
    )

    print(f"\nDataset length: {len(ds)}")
    if len(ds) > 0:
        ex = ds[0]
        print(f"input_ids shape:      {ex['input_ids'].shape}")
        print(f"attention_mask shape: {ex['attention_mask'].shape}")
        print(f"labels shape:         {ex['labels'].shape}")

        labels    = ex["labels"].tolist()
        masked    = sum(1 for l in labels if l == -100)
        unmasked  = sum(1 for l in labels if l != -100)
        print(f"Masked tokens (prompt):     {masked}")
        print(f"Unmasked tokens (assistant): {unmasked}")
        assert unmasked > 0, "No assistant tokens found — label masking broken!"

        assistant_tokens = [
            ex["input_ids"][i].item()
            for i, l in enumerate(labels) if l != -100
        ]
        assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
        print(f"Assistant text (what model learns):\n  {assistant_text[:200]}")

    print("\n✓ SFTDataset test passed")