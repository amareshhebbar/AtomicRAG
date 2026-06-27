#   make                  → full pipeline, all data        (RunPod / production)
#   make MAX_ROWS=30      → full pipeline, 30 rows, CPU    (laptop testing)
#
# Both run the EXACT same stages:
#   data → stage1 → merge1 → stage2 → merge2 → stage3 → merge+push → eval → report

# Other targets:
#   make setup            → venv + install only (no training)
#   make data             → data pipeline only
#   make data MAX_ROWS=30 → data pipeline, 30 rows
#   make train            → all 3 training stages (data already built)
#   make eval             → evaluation + report only
#   make dry              → print what would run, execute nothing
#   make reset            → wipe run state so everything re-runs
#   make clean            → delete venv + all outputs
#   make env              → create .env from .env.example
#   make help             → show this message

VENV         := myenv
PYTHON       := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
RUN_ALL      := runpod/run_all.py
REQUIREMENTS := requirements.txt
ENV_FILE     := .env
ENV_EXAMPLE  := .env.example
STATE_FILE   := outputs/.run_state.json
PYTHON3      := python3

export PYTHONPATH := $(shell pwd)

MAX_ROWS ?=

ifdef MAX_ROWS
    DATA_FLAGS   := --max_rows $(MAX_ROWS)
    TRAIN_FLAGS  := --local_test
    SKIP_PUSH    := true
    _MODE        := LAPTOP TEST ($(MAX_ROWS) rows · CPU · no W&B · no HF push)
else
    DATA_FLAGS   :=
    TRAIN_FLAGS  :=
    SKIP_PUSH    :=
    _MODE        := PRODUCTION (full data · GPU · W&B · push to HuggingFace)
endif


.DEFAULT_GOAL := all
.PHONY: all setup venv install install-torch run check \
        data train eval dry reset clean env help \
        _data_step _stage1 _merge1 _stage2 _merge2 _stage3 _finalize _eval _report

check:
	@echo ""
	@echo "  ── Preflight checks ──────────────────────────────────"
	@echo -n "  CUDA:          "
	@$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'NOT FOUND'; print('OK  (' + torch.cuda.get_device_name(0) + ')')" 2>/dev/null || \
		(echo "FAIL — no GPU detected. Check pod template or torch install." && exit 1)
	@echo -n "  HF_TOKEN:      "
	@test -n "$$HF_TOKEN" || (echo "FAIL — run: export \$$(cat .env | xargs)" && exit 1)
	@$(PYTHON) -c "\
import os, urllib.request, urllib.error; \
req = urllib.request.Request('https://huggingface.co/api/whoami', headers={'Authorization': 'Bearer ' + os.environ['HF_TOKEN']}); \
r = urllib.request.urlopen(req, timeout=10); print('OK  (' + __import__('json').loads(r.read())['name'] + ')')" 2>/dev/null || \
		echo "WARN — HF API unreachable (token set, continuing anyway)"
	@echo -n "  WANDB_API_KEY: "
	@test -n "$$WANDB_API_KEY" || (echo "not set — using WANDB_MODE=disabled" )
	@$(PYTHON) -c "\
import os, urllib.request, json; \
req = urllib.request.Request('https://api.wandb.ai/graphql', \
  data=b'{\"query\":\"{viewer{entity}}\"}', \
  headers={'Authorization': 'Bearer ' + os.environ['WANDB_API_KEY'], 'Content-Type': 'application/json'}); \
r = urllib.request.urlopen(req, timeout=10); d = json.loads(r.read()); \
entity = d['data']['viewer']['entity']; print('OK  (' + entity + ')')" 2>/dev/null || \
		echo "WARN — W&B unreachable (continuing with WANDB_MODE=disabled)"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  ✓ Checks done — starting pipeline"
	@echo ""


all: env setup check _data_step _stage1 _merge1 _stage2 _merge2 _stage3 _finalize _eval _report
	@echo ""
	@echo "  ✓ Full pipeline complete — $(_MODE)"
	@echo ""


env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo ""; \
		echo "  .env not found — copying from .env.example"; \
		echo "  Fill in HF_TOKEN and WANDB_API_KEY, then run make again"; \
		echo ""; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
		echo "  Created .env — edit:  nano .env"; \
		echo ""; \
		exit 1; \
	else \
		echo "  ✓ .env exists"; \
	fi

venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	@echo ""
	@echo "  Creating virtual environment: $(VENV)/"
	$(PYTHON3) -m venv $(VENV)
	@echo "  ✓ venv created"

install: venv
	@echo ""
	@echo "  Installing from $(REQUIREMENTS)..."
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r $(REQUIREMENTS)
	@echo "  ✓ Base packages installed"

install-torch: venv
	@echo ""
	@echo "  Detecting GPU and installing torch..."
	@$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'cuda not available'; print('  ✓ torch ' + torch.__version__ + ' already installed')" 2>/dev/null || \
	( \
		CUDA_VER=$$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $$NF}' | cut -d. -f1); \
		if [ "$$CUDA_VER" = "" ]; then \
			echo "  No GPU → CPU torch"; \
			$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet; \
		elif [ "$$CUDA_VER" -ge 12 ]; then \
			echo "  CUDA $$CUDA_VER → cu128"; \
			$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet; \
		else \
			echo "  CUDA $$CUDA_VER → cu118"; \
			$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet; \
		fi; \
		echo "  ✓ torch installed"; \
	)


setup: env venv install install-torch
	@echo ""
	@echo "  ✓ Setup complete"
	@echo "  Laptop:  make MAX_ROWS=30"
	@echo "  RunPod:  make"
	@echo ""


_data_step: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [1/8] DATA PIPELINE — $(_MODE)"
	@echo "  ════════════════════════════════════════════════"
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/process_musique.py  $(DATA_FLAGS)
	$(PYTHON) scripts/process_hotpot.py   $(DATA_FLAGS)
	$(PYTHON) scripts/process_2wiki.py    $(DATA_FLAGS)
	$(PYTHON) scripts/build_splits.py     $(DATA_FLAGS)
	$(PYTHON) scripts/build_orpo_pairs.py $(DATA_FLAGS)
	$(PYTHON) scripts/stats.py
	@echo "  ✓ Data done"

_stage1: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [2/8] STAGE 1 — QLoRA SFT"
ifdef MAX_ROWS
	@echo "         CPU mode · local_test · no W&B"
else
	@echo "         GPU · 4-bit NF4 · r=16 · W&B: stage1-qlora-r16"
endif
	@echo "  ════════════════════════════════════════════════"
	$(PYTHON) train/stage1_qlora.py $(TRAIN_FLAGS)
	@echo "  ✓ Stage 1 done"

_merge1: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [3/8] MERGE STAGE 1 → bf16 model"
	@echo "  ════════════════════════════════════════════════"
	$(PYTHON) train/merge_and_push.py --stage 1
	@echo "  ✓ Stage 1 merged → outputs/stage1_merged/"

_stage2: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [4/8] STAGE 2 — DoRA Refinement"
ifdef MAX_ROWS
	@echo "         CPU mode · local_test"
else
	@echo "         GPU · r=8 · bf16 · W&B: stage2-dora"
endif
	@echo "  ════════════════════════════════════════════════"
	$(PYTHON) train/stage2_dora.py $(TRAIN_FLAGS)
	@echo "  ✓ Stage 2 done"

_merge2: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [5/8] MERGE STAGE 2 → bf16 model"
	@echo "  ════════════════════════════════════════════════"
	$(PYTHON) train/merge_and_push.py --stage 2
	@echo "  ✓ Stage 2 merged → outputs/stage2_merged/"

_stage3: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [6/8] STAGE 3 — ORPO Alignment"
ifdef MAX_ROWS
	@echo "         CPU mode · local_test"
else
	@echo "         GPU · beta=0.1 · bf16 · W&B: stage3-orpo"
endif
	@echo "  ════════════════════════════════════════════════"
	$(PYTHON) train/stage3_orpo.py $(TRAIN_FLAGS)
	@echo "  ✓ Stage 3 done"

_finalize: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [7/8] MERGE STAGE 3 + PUSH"
	@echo "  ════════════════════════════════════════════════"
ifdef MAX_ROWS
	@echo "  Laptop: merge only (no HF push)"
	$(PYTHON) train/merge_and_push.py --stage 3
else
	@echo "  RunPod: merge + push to HuggingFace + GGUF"
	$(PYTHON) train/merge_and_push.py --stage 3 --push --push_gguf
endif
	@echo "  ✓ Final model → outputs/final/"

_eval: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [8a/8] EVALUATION"
	@echo "  ════════════════════════════════════════════════"
	-$(PYTHON) eval/evaluate_decomposition.py || echo "  ⚠ evaluate_decomposition skipped"
	-$(PYTHON) eval/evaluate_retrieval.py     || echo "  ⚠ evaluate_retrieval skipped"
	-$(PYTHON) eval/benchmark_speed.py        || echo "  ⚠ benchmark_speed skipped"
	@echo "  ✓ Evaluation done (results in results/)"

_report: $(VENV)/bin/activate
	@echo ""
	@echo "  ════════════════════════════════════════════════"
	@echo "  [8b/8] BENCHMARK REPORT"
	@echo "  ════════════════════════════════════════════════"
	$(PYTHON) runpod/run_all.py --from_phase 11
	@echo "  ✓ Report → results/benchmark_report.html"


run: $(VENV)/bin/activate
	@echo ""
	@echo "  QueryDecomp — $(_MODE)"
	@echo ""
ifdef MAX_ROWS
	$(MAKE) _data_step _stage1 _merge1 _stage2 _merge2 _stage3 _finalize _eval _report MAX_ROWS=$(MAX_ROWS)
else
	$(PYTHON) $(RUN_ALL)
endif

data: $(VENV)/bin/activate
	@echo ""
	@echo "  Data pipeline — $(_MODE)"
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/process_musique.py  $(DATA_FLAGS)
	$(PYTHON) scripts/process_hotpot.py   $(DATA_FLAGS)
	$(PYTHON) scripts/process_2wiki.py    $(DATA_FLAGS)
	$(PYTHON) scripts/build_splits.py     $(DATA_FLAGS)
	$(PYTHON) scripts/build_orpo_pairs.py $(DATA_FLAGS)
	$(PYTHON) scripts/stats.py

train: $(VENV)/bin/activate
	@echo ""
	@echo "  Training — $(_MODE)"
ifdef MAX_ROWS
	$(MAKE) _stage1 _merge1 _stage2 _merge2 _stage3 _finalize MAX_ROWS=$(MAX_ROWS)
else
	$(PYTHON) $(RUN_ALL) --skip_data
endif

eval: $(VENV)/bin/activate
	@echo ""
	@echo "  Evaluation + report"
	$(PYTHON) $(RUN_ALL) --from_phase 10

dry: $(VENV)/bin/activate
	@echo ""
	@echo "  DRY RUN — $(_MODE)"
	$(PYTHON) $(RUN_ALL) --dry_run

reset:
	@echo ""
	@if [ -f $(STATE_FILE) ]; then \
		rm $(STATE_FILE); echo "  ✓ State cleared"; \
	else \
		echo "  Nothing to reset"; \
	fi
	@echo ""

clean:
	@echo ""
	@echo "  Removing venv and outputs..."
	rm -rf $(VENV) outputs/ results/ data/processed/ data/raw/
	@echo "  ✓ Cleaned (.env kept)"
	@echo ""

help:
	@echo ""
	@echo "  querydecomp — Makefile"
	@echo ""
	@echo "  ── Run ───────────────────────────────────────────────"
	@echo "  make                   full pipeline  (RunPod/GPU)"
	@echo "  make MAX_ROWS=30       full pipeline  (laptop/CPU/30 rows)"
	@echo ""
	@echo "  ── Both run the same 8 stages: ───────────────────────"
	@echo "  [1] Data download + process + split + ORPO pairs"
	@echo "  [2] Stage 1 QLoRA SFT"
	@echo "  [3] Merge Stage 1 → bf16"
	@echo "  [4] Stage 2 DoRA refinement"
	@echo "  [5] Merge Stage 2 → bf16"
	@echo "  [6] Stage 3 ORPO alignment"
	@echo "  [7] Merge Stage 3 + push to HuggingFace  (RunPod only)"
	@echo "  [8] Eval + benchmark report"
	@echo ""
	@echo "  ── Individual steps ──────────────────────────────────"
	@echo "  make setup             venv + install only"
	@echo "  make data              data pipeline only"
	@echo "  make data MAX_ROWS=30  data, 30 rows"
	@echo "  make train             all 3 training stages"
	@echo "  make train MAX_ROWS=30 training, CPU mode"
	@echo "  make eval              evaluation + report only"
	@echo ""
	@echo "  ── Utils ─────────────────────────────────────────────"
	@echo "  make dry               show commands, run nothing"
	@echo "  make reset             clear run state"
	@echo "  make clean             delete venv + outputs"
	@echo "  make help              this message"
	@echo ""
	@echo "  ── Laptop → RunPod workflow ──────────────────────────"
	@echo "  make MAX_ROWS=30       test locally (no HF push)"
	@echo "  git push               push your code"
	@echo "  [RunPod] make          full training + push to HF"
	@echo ""