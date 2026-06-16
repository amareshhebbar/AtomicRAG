# AtomicRAG

Fine-tuned Qwen2.5-1.5B that breaks complex multi-hop questions into atomic sub-queries for RAG retrieval.

---

## The problem

When you ask a RAG system *"What changed in the refund policy after the 2023 restructuring that affects enterprise customers?"* — it embeds the entire question and retrieves garbage. The answer lives across 3 separate documents.

AtomicRAG breaks it into:

```json
[
  {"hop": 1, "sub_query": "When did the 2023 restructuring happen?",                "depends_on": []},
  {"hop": 2, "sub_query": "What refund policy was released after the restructuring?","depends_on": [1]},
  {"hop": 3, "sub_query": "Which tier of that policy applies to enterprise?",        "depends_on": [2]}
]
```

`depends_on: []` means retrieve in parallel. `depends_on: [1]` means wait for hop 1 first.

**Result: hit@3 goes from 31% to 67% on multi-hop questions.**

---

## Quick inference

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="AmareshHebbar/querydecomp-qwen2.5-1.5b")
out  = pipe("Who directed Inception and where were they born?", max_new_tokens=200, do_sample=False)
print(out[0]["generated_text"])
```

Via Ollama:
```bash
ollama run hf.co/AmareshHebbar/querydecomp-qwen2.5-1.5b
```

---

## Run it yourself

### Laptop — test with 30 rows (CPU, ~30 min, no cost)

```bash
git clone https://github.com/AmareshHebbar/AtomicRAG
cd AtomicRAG
cp .env.example .env
nano .env          # fill HF_TOKEN and WANDB_API_KEY
make MAX_ROWS=30
```

Runs the full pipeline on 30 rows per dataset. CPU only, no GPU, no push to HuggingFace. Use this to verify everything works before spending money on a pod.

### RunPod — full training (GPU, ~12h, pushes to HuggingFace)

```bash
git clone https://github.com/AmareshHebbar/AtomicRAG
cd AtomicRAG
cp .env.example .env
nano .env          # fill HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY
make
```

One command. Does everything.

---

## What make actually does

Both `make MAX_ROWS=30` and `make` run the same 8 stages. Only the data size and hardware change.

```
[1/8] Data        download datasets, process, build splits, build ORPO pairs
[2/8] Stage 1     QLoRA SFT — model learns the decomposition format
[3/8] Merge 1     merge adapter into base model
[4/8] Stage 2     DoRA refinement — improves decomposition precision
[5/8] Merge 2     merge adapter into base model
[6/8] Stage 3     ORPO alignment — model learns what bad decompositions look like
[7/8] Merge 3     final merge (RunPod also pushes to HuggingFace)
[8/8] Eval        F1, hit@k, speed benchmark, HTML report
```

| | `make MAX_ROWS=30` | `make` |
|---|---|---|
| Hardware | CPU | RTX A5000 GPU |
| Data size | 30 rows per dataset | All rows (~300K) |
| W&B logging | Off | On |
| HuggingFace push | Off | On |
| Time | ~30 min | ~12 hours |
| Cost | $0 | ~$2.64 |

---

## Other make commands

```bash
make setup             # create venv and install packages only
make data              # data pipeline only
make data MAX_ROWS=30  # data pipeline, 30 rows
make train             # all 3 training stages, skip data
make eval              # evaluation and report only
make reset             # wipe run state so everything re-runs next time
make clean             # delete venv and all outputs
make dry               # print every command without running anything
make help              # show all targets
```

If a RunPod pod crashes mid-run, just run `make` again. It reads `outputs/.run_state.json` and skips stages that already finished.

---

## Training — 3 stages explained

**Stage 1 — QLoRA SFT**

Trains the model to produce valid dependency graph JSON. Uses NF4 4-bit quantization so the model fits in GPU memory. 3 epochs, LoRA rank 16.

**Stage 2 — DoRA refinement**

DoRA decomposes each weight update into a magnitude component and a direction component and trains them separately. This gives better structured output than plain LoRA, especially on the `depends_on` field. 1 epoch on the Stage 1 merged model.

**Stage 3 — ORPO alignment**

Trains on (chosen, rejected) pairs without needing a separate reference model. The model learns to reject 5 specific failure modes:

| Rejection type | What it means |
|---|---|
| `merged_hops` | Two hops combined into one bloated sub-query |
| `hallucinated` | The entity in the sub-query is made up |
| `zero_decomp` | The original question returned unchanged |
| `over_decomp` | A 2-hop question split into 6 pointless sub-questions |
| `missing_hop` | The bridge hop dropped from a 3-hop chain |

---

## Data

| Dataset | Rows | What it provides |
|---|---|---|
| bdsaglam/musique | 44.7K | Gold sub-questions with explicit hop labels |
| hotpotqa/hotpot_qa | 113K | 2-hop bridge and comparison questions |
| framolfese/2WikiMultihopQA | 167K | Bridge, comparison, compositional, inference types |

After filtering and dedup: ~42K SFT training examples and ~71K ORPO pairs.

---

## Results

**Decomposition quality on MuSiQue validation:**

| Metric | Base model | Fine-tuned | Delta |
|---|---|---|---|
| JSON parse rate | 61% | 98% | +37% |
| Hop count accuracy | 43% | 84% | +41% |
| Hop coverage F1 | 0.38 | 0.79 | +0.41 |
| Dep graph accuracy | 21% | 71% | +50% |

**Retrieval improvement with BM25 on MuSiQue validation:**

| | Raw question | With decomposition | Delta |
|---|---|---|---|
| hit@1 | 18% | 41% | +23% |
| hit@3 | 31% | 67% | +36% |
| hit@5 | 39% | 78% | +39% |

**Speed vs GPT-4o:**

| | p50 | p95 | cost per 1k queries |
|---|---|---|---|
| GPT-4o (API, prompted) | 1800ms | 3200ms | $30.00 |
| AtomicRAG on A5000 | ~50ms | ~90ms | $0.02 |

---

## Environment variables

Copy `.env.example` to `.env` and fill these in:

```
HF_TOKEN          HuggingFace token with read and write access
WANDB_API_KEY     W&B API key
WANDB_ENTITY      your W&B username
WANDB_PROJECT     project name (default: querydecomp)
HF_REPO_ID        where to push the model (default: AmareshHebbar/querydecomp-qwen2.5-1.5b)
```

On RunPod you can set these in pod Settings > Environment Variables instead of the .env file.

---

## File structure

```
AtomicRAG/
├── Makefile                    run everything: make / make MAX_ROWS=30
├── requirements.txt            pip dependencies
├── .env.example                copy this to .env and fill in your tokens
│
├── scripts/
│   ├── download_data.py        download 3 datasets from HuggingFace
│   ├── process_musique.py      gold decompositions to JSONL
│   ├── process_hotpot.py       supporting facts to heuristic decompositions
│   ├── process_2wiki.py        4 reasoning types to unified format
│   ├── build_splits.py         merge, filter, dedup, save as HF DatasetDict
│   ├── build_orpo_pairs.py     5 rejection types per example
│   └── stats.py                dataset verification and statistics
│
├── src/
│   ├── config.py               all hyperparams in one dataclass per stage
│   ├── model.py                load model, apply LoRA/DoRA, merge adapters
│   ├── dataset.py              SFTDataset and ORPODataset with label masking
│   └── utils.py                prompt builder, JSON parser, dep graph validator
│
├── train/
│   ├── stage1_qlora.py         QLoRA SFT
│   ├── stage2_dora.py          DoRA refinement
│   ├── stage3_orpo.py          ORPO alignment (custom loss, no TRL dependency)
│   └── merge_and_push.py       merge adapter into bf16, push to HF Hub
│
├── eval/
│   ├── evaluate_decomposition.py   hop F1 and dep graph accuracy
│   ├── evaluate_retrieval.py       hit@k before and after decomposition
│   └── benchmark_speed.py          latency and cost vs GPT-4o
│
└── runpod/
    └── run_all.py              orchestrator used for benchmark report generation
```

---

*Amaresh Hebbar · [HuggingFace](https://huggingface.co/AmareshHebbar) · [W&B](https://wandb.ai/amareshhebbar)*