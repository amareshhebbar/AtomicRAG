"""
src/dataset.py
--------------
PyTorch Dataset classes for all training stages.

SFTDataset  — for Stage 1 (QLoRA) and Stage 2 (DoRA)
ORPODataset — for Stage 3 (ORPO alignment)

Both load from the HuggingFace DatasetDict saved by build_splits.py.
The 'messages_json' column (serialized JSON string) is deserialized back
to a list of dicts and passed through the model's chat template.

Usage:
    from src.dataset import SFTDataset, ORPODataset, get_data_collator

    train_ds = SFTDataset("data/processed/sft_dataset", tokenizer, cfg, split="train")
    eval_ds  = SFTDataset("data/processed/sft_dataset", tokenizer, cfg, split="validation")

    orpo_train = ORPODataset("data/processed/orpo_dataset", tokenizer, cfg, split="train")
"""

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
    # Stub so class definitions don't crash at import time when torch is missing
    class Dataset:  # type: ignore
        pass
    PreTrainedTokenizer = object
    HFDataset = object

from src.config import BaseConfig
from src.utils import build_training_text, SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# SFT Dataset — Stages 1 & 2
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (Stage 1 QLoRA, Stage 2 DoRA).

    Each example:
      Input:  [SYSTEM][USER question]
      Target: [ASSISTANT JSON dependency graph]

    Loss is computed ONLY on the assistant tokens (not system/user).
    We achieve this by setting input_ids labels to -100 for all non-assistant tokens.

    Args:
        hf_dataset_path: path to HF DatasetDict saved by build_splits.py
        tokenizer:       model tokenizer
        cfg:             training config
        split:           "train" or "validation"
        max_samples:     limit number of examples (for local testing)
    """

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

        # Load the HuggingFace dataset
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

        # Pre-tokenize everything upfront (faster than tokenizing per __getitem__)
        self.examples = self._tokenize_all(hf_ds)

        print(f"  ✓ Tokenized: {len(self.examples):,} valid examples "
              f"({len(hf_ds) - len(self.examples)} filtered by max_length)")

    def _tokenize_all(self, hf_ds: "HFDataset") -> list[dict]:
        """
        Tokenize all examples. Filter those exceeding max_length.
        Returns list of {'input_ids', 'attention_mask', 'labels'} dicts.
        """
        results = []

        for row in hf_ds:
            # Deserialize messages_json back to list[dict]
            try:
                messages = json.loads(row["messages_json"])
            except (json.JSONDecodeError, KeyError):
                continue

            tokenized = self._tokenize_one(messages)
            if tokenized is not None:
                results.append(tokenized)

        return results

    def _tokenize_one(self, messages: list[dict]) -> Optional[dict]:
        """
        Tokenize a single messages list.

        The full conversation (system + user + assistant) is tokenized as one sequence.
        Labels are -100 everywhere except the assistant turn, so loss is computed
        only on the JSON output the model needs to learn.

        Returns None if the sequence exceeds max_length.
        """
        # Full text with chat template (includes assistant turn)
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Text without assistant turn (to find where assistant tokens start)
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text     = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,   # adds "<|im_start|>assistant\n"
        )

        # Tokenize the full sequence
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        if len(tokenized["input_ids"]) >= self.max_length:
            return None   # too long — skip

        # Find how many tokens the prompt takes
        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_tokenized["input_ids"])

        # Build labels: -100 for prompt tokens, actual IDs for assistant tokens
        input_ids = tokenized["input_ids"]
        labels    = [-100] * prompt_len + input_ids[prompt_len:]

        # Safety: labels must be same length as input_ids
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


# ─────────────────────────────────────────────────────────────────────────────
# ORPO Dataset — Stage 3
# ─────────────────────────────────────────────────────────────────────────────

class ORPODataset(Dataset):
    """
    Dataset for ORPO alignment (Stage 3).

    Each example has:
      - prompt:   the user question (same system prompt applied)
      - chosen:   good decomposition (correct JSON dependency graph)
      - rejected: bad decomposition (one of 5 failure modes)

    The TRL ORPOTrainer expects a specific format:
      {
        "prompt":   tokenized prompt (system + user)
        "chosen":   tokenized full sequence ending with chosen response
        "rejected": tokenized full sequence ending with rejected response
      }

    We store the raw strings here and let the ORPOTrainer handle tokenization
    (it has its own collator that handles chosen/rejected padding correctly).

    Args:
        hf_dataset_path: path to HF DatasetDict of ORPO pairs
        tokenizer:       model tokenizer
        cfg:             Stage3Config
        split:           "train" or "validation"
        max_samples:     limit for local testing
    """

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
        """
        Build a single ORPO example.

        Returns a dict with string keys 'prompt', 'chosen', 'rejected'
        that TRL's ORPOTrainer can consume directly.
        """
        try:
            question = row["question"]
            chosen   = row["chosen_json"]
            rejected = row["rejected_json"]
        except KeyError:
            return None

        if not question or not chosen or not rejected:
            return None

        # Build the prompt string (system + user, no assistant)
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # chosen / rejected are the raw JSON strings (assistant turn content)
        # ORPOTrainer concatenates prompt + response internally
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


# ─────────────────────────────────────────────────────────────────────────────
# Data collator
# ─────────────────────────────────────────────────────────────────────────────

def get_sft_collator(tokenizer: "PreTrainedTokenizer", max_length: int):
    """
    Data collator for SFT training.
    Pads sequences in a batch to the same length.
    label_pad_token_id=-100 means padded label positions don't contribute to loss.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,          # for tensor core efficiency
        label_pad_token_id=-100,
        return_tensors="pt",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: inspect a batch
# ─────────────────────────────────────────────────────────────────────────────

def decode_batch_sample(batch: dict, tokenizer, idx: int = 0) -> str:
    """
    Decode one example from a batch for debugging.
    Shows which tokens have labels (what the model learns) vs masked (-100).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

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

        # Verify labels: should be -100 for prompt, actual IDs for assistant
        labels    = ex["labels"].tolist()
        masked    = sum(1 for l in labels if l == -100)
        unmasked  = sum(1 for l in labels if l != -100)
        print(f"Masked tokens (prompt):     {masked}")
        print(f"Unmasked tokens (assistant): {unmasked}")
        assert unmasked > 0, "No assistant tokens found — label masking broken!"

        # Decode to verify
        assistant_tokens = [
            ex["input_ids"][i].item()
            for i, l in enumerate(labels) if l != -100
        ]
        assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
        print(f"Assistant text (what model learns):\n  {assistant_text[:200]}")

    print("\n✓ SFTDataset test passed")