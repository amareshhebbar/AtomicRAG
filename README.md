# AtomicRAG

Fine-tuned Qwen2.5-1.5B that breaks complex questions into atomic sub-queries before RAG retrieval.

**The problem it solves:**

> *"What changed in the refund policy after the 2023 restructuring that affects enterprise customers?"*

A naive RAG system embeds this entire question and retrieves nothing useful. The answer lives across 3 separate documents. This model decomposes it into retrievable sub-queries with a dependency graph — so a retriever knows which hops to run in parallel and which need the previous answer first.

**Result:** hit@3 improves from **31% → 67%** on multi-hop questions. Inference: **~50ms**. Total training cost: **$2.64**.

---

## Output format

Not a flat list. A dependency graph:

```json
[
  {"hop": 1, "sub_query": "What year did the 2023 restructuring happen?",           "depends_on": []},
  {"hop": 2, "sub_query": "What was the refund policy version released after 2023?", "depends_on": [1]},
  {"hop": 3, "sub_query": "Which tier of that policy applies to enterprise customers?","depends_on": [2]}
]
```

`depends_on: []` = parallel retrieval safe. `depends_on: [1]` = wait for hop 1 answer first.

This is what no other decomposition model outputs at sub-7B scale.

---

## Use it

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="AmareshHebbar/querydecomp-qwen2.5-1.5b")
out  = pipe("Who directed Inception and where were they born?", max_new_tokens=200)
print(out[0]["generated_text"])
```

Via Ollama (GGUF):
```bash
ollama run hf.co/AmareshHebbar/querydecomp-qwen2.5-1.5b
```

---

## Training — 3-stage curriculum

| Stage | Method | Why | Time | Cost |
|---|---|---|---|---|
| 1 | QLoRA SFT (r=16, NF4 4-bit) | Learn the structure of decomposition | ~5h | $1.10 |
| 2 | DoRA refinement (r=8, bf16) | Calibrate magnitude vs direction separately | ~2.5h | $0.55 |
| 3 | ORPO alignment | Learn what bad decompositions look like | ~3.5h | $0.77 |
| — | Merge + push | Final bf16 model + GGUF to HF Hub | ~30m | $0.22 |

**Total: ~$2.64 on RTX A5000 spot.**

Stage 3 ORPO trains the model to reject 5 specific failure modes:

| Rejection type | Example |
|---|---|
| `merged_hops` | "Who directed Inception and where were they born?" as ONE sub-query |
| `hallucinated` | Replacing "Christopher Nolan" with "Alan Voss" in the sub-query |
| `zero_decomp` | Returning the original question unchanged |
| `over_decomp` | Splitting a 2-hop into 6 trivial sub-questions |
| `missing_hop` | Dropping the bridge query from a 3-hop chain |

---

## Data

| Dataset | Size | What it provides |
|---|---|---|
| MuSiQue | 44.7K | Gold standard — explicit per-hop sub-questions with answers |
| HotpotQA | 113K | 2-hop bridge + comparison questions |
| 2WikiMultiHopQA | 167K | 4 reasoning types: bridge, comparison, compositional, inference |
| Synthetic (Ollama) | ~12K | Domain expansion: legal, financial, medical |

After filtering and dedup: **~42K SFT examples** + **~71K ORPO pairs**.

---

## Run the full pipeline yourself

On a fresh RunPod A5000 pod, with `HF_TOKEN` and `WANDB_API_KEY` set:

```bash
git clone https://github.com/AmareshHebbar/AtomicRAG
cd AtomicRAG
python runpod/run_all.py
```

That's it. One command. It handles:
- GPU check + auth
- pip installs
- Dataset download + processing + dedup
- All 3 training stages with merge between each
- Push to HuggingFace + GGUF export
- Evaluation

If the pod crashes mid-run, rerun the same command — it reads state from `outputs/.run_state.json` and resumes from where it left off.

```bash
# Jump to a specific phase
python runpod/run_all.py --from_phase 6    # start from Stage 2

# Data only
python runpod/run_all.py --only_data

# See what would run without doing anything
python runpod/run_all.py --dry_run
```

---

## File structure

```
AtomicRAG/
│
├── scripts/
│   ├── download_data.py        downloads MuSiQue + HotpotQA + 2Wiki from HF
│   ├── process_musique.py      gold decompositions → dependency graph JSONL
│   ├── process_hotpot.py       heuristic decompositions from supporting facts
│   ├── process_2wiki.py        4 reasoning types → unified format
│   ├── build_splits.py         merge + filter + dedup + HF DatasetDict
│   ├── build_orpo_pairs.py     5 rejection types per example → ORPO pairs
│   └── stats.py                dataset statistics and verification
│
├── src/
│   ├── config.py               all hyperparams in one dataclass per stage
│   ├── model.py                model loading (4-bit/bf16), LoRA/DoRA, merge
│   ├── dataset.py              SFTDataset + ORPODataset (label masking built in)
│   └── utils.py                prompt builder, JSON parser, dep graph validator
│
├── train/
│   ├── stage1_qlora.py         QLoRA SFT with W&B logging + sample callbacks
│   ├── stage2_dora.py          DoRA refinement on merged Stage 1 model
│   ├── stage3_orpo.py          ORPO alignment using TRL ORPOTrainer
│   └── merge_and_push.py       merge adapter → bf16 → HF Hub + GGUF
│
├── eval/
│   ├── evaluate_decomposition.py   sub-query F1, hop coverage, dep graph accuracy
│   ├── evaluate_retrieval.py       hit@k before vs after decomposition
│   └── benchmark_speed.py          p50/p95 latency vs GPT-4o reference
│
├── configs/
│   ├── local_test.yaml         CPU-safe tiny run (20 examples, no GPU)
│   └── runpod_a5000.yaml       full A5000 config with VRAM breakdown
│
└── runpod/
    ├── run_all.py              master orchestrator — runs entire pipeline
    ├── setup.sh                one-shot pod setup
    └── train_stage*.sh         individual stage launchers
```

---

## Eval results

| Metric | Base model | Fine-tuned | Delta |
|---|---|---|---|
| JSON parse rate | 61% | 98% | +37% |
| Hop count accuracy | 43% | 84% | +41% |
| Hop coverage F1 | 0.38 | 0.79 | +0.41 |
| Dep graph accuracy | 21% | 71% | +50% |

Retrieval (BM25, MuSiQue validation):

| | Raw question | Decomposed | Delta |
|---|---|---|---|
| hit@1 | 18% | 41% | +23% |
| hit@3 | 31% | 67% | +36% |
| hit@5 | 39% | 78% | +39% |

Speed vs GPT-4o:

| | p50 | p95 | $/1k queries |
|---|---|---|---|
| GPT-4o (API, prompted) | 1800ms | 3200ms | $30.00 |
| AtomicRAG (A5000) | ~50ms | ~90ms | $0.02 |

---

## Why fine-tuning, not prompting

GPT-4o with a good system prompt gets ~61% hop coverage F1 on MuSiQue. This model gets 79%. The gap exists because:

1. Decomposition quality is domain-specific. Legal hops differ from financial hops differ from medical hops. The model learns these patterns at the weight level.
2. The `depends_on` structure (parallel vs sequential) requires understanding causality between sub-queries — something prompting can't reliably produce at inference time.
3. At 50ms vs 2000ms, this is viable inside a real RAG pipeline. GPT-4o is not.

---

## Connects to

- **recall** — my RAG retrieval engine. AtomicRAG sits in front of it as the query layer.
- **ShiftLeft** — autonomous bug-fixing agent. AtomicRAG is the triage layer that decomposes a bug report before the fix agent starts.
- **hard-coat** — hallucination interceptor. The `rejected` training examples in Stage 3 directly use hard-coat's semantic similarity detection to flag hallucinated entities.

---

*Amaresh Hebbar · [GitHub](https://github.com/AmareshHebbar) · [HuggingFace](https://huggingface.co/AmareshHebbar) · [W&B](https://wandb.ai/amareshhebbar)*