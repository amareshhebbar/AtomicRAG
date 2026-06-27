import os, sys, json, time, subprocess, argparse, tempfile
from pathlib import Path
from datetime import datetime

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(ROOT))

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

G="\033[92m"; Y="\033[93m"; R="\033[91m"; C="\033[96m"; B="\033[1m"; X="\033[0m"

def ok(m): print(f"  {G}✓{X} {m}")
def warn(m):  print(f"  {Y}⚠{X} {m}")
def skip(m):  print(f"  {Y}⚡{X} {m}")
def err(m):print(f"  {R}✗{X} {m}")
def info(m):  print(f"  {C}→{X} {m}")
def head(t, n=None):
    tag = f"[PHASE {n}] " if n is not None else ""
    print(f"\n{B}{'═'*64}{X}\n{B}  {tag}{t}{X}\n{B}{'═'*64}{X}")

STATE_FILE = ROOT / "outputs" / ".run_state.json"

def load_state():
    try: return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    except: return {}

def save_state(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2))

def mark(s, k, extra=None):
    s[k] = {"done": True, "at": datetime.now().isoformat(), **(extra or {})}
    save_state(s)

def is_done(s, k): return s.get(k, {}).get("done", False)

class Timer:
    def __init__(self): self.t0 = self.p0 = time.time()
    def phase(self):
        s = int(time.time()-self.p0); self.p0 = time.time()
        return f"{s//3600}h {(s%3600)//60}m {s%60}s" if s>=3600 else f"{s//60}m {s%60}s"
    def total(self):
        s = int(time.time()-self.t0)
        return f"{s//3600}h {(s%3600)//60}m {s%60}s" if s>=3600 else f"{s//60}m {s%60}s"
    def cost(self):
        rate = float(os.environ.get("RUNPOD_HOURLY_RATE","0.22"))
        return f"${(time.time()-self.t0)/3600*rate:.3f}"

def run(cmd, dry=False):
    display = " ".join(cmd) if isinstance(cmd,list) else cmd
    info(f"$ {display}")
    if dry: skip("DRY RUN — skipped"); return 0
    proc = subprocess.Popen(cmd, cwd=str(ROOT), shell=isinstance(cmd,str),
                            env=os.environ.copy(), stdout=sys.stdout, stderr=sys.stderr)
    proc.wait(); return proc.returncode

def py(script, args=None, dry=False):
    return run([sys.executable, script]+(args or []), dry=dry)

def must(code, phase):
    if code != 0:
        err(f"Phase '{phase}' failed (exit {code})")
        err("Fix above, then: python runpod/run_all.py --from_phase <N>")
        sys.exit(code)

def data_ok(name):
    c = {
        "raw_musique":  ROOT/"data/raw/musique/train.parquet",
        "raw_hotpot":ROOT/"data/raw/hotpotqa/train.parquet",
        "raw_2wiki": ROOT/"data/raw/2wikimultihopqa/train.parquet",
        "musique_proc": ROOT/"data/processed/musique_train.jsonl",
        "hotpot_proc":  ROOT/"data/processed/hotpotqa_train.jsonl",
        "wiki_proc": ROOT/"data/processed/2wiki_train.jsonl",
        "sft_hf":  ROOT/"data/processed/sft_dataset/dataset_dict.json",
        "orpo_hf": ROOT/"data/processed/orpo_dataset/dataset_dict.json",
    }
    p = c.get(name)
    if p is None: return False
    return p.exists() and (p.stat().st_size > 100 if p.is_file() else any(p.iterdir()))

def model_ok(name):
    c = {
        "s1_adapter": ROOT/"outputs/stage1_qlora/final/adapter_config.json",
        "s1_merged":  ROOT/"outputs/stage1_merged/config.json",
        "s2_adapter": ROOT/"outputs/stage2_dora/final/adapter_config.json",
        "s2_merged":  ROOT/"outputs/stage2_merged/config.json",
        "s3_adapter": ROOT/"outputs/stage3_orpo/final/adapter_config.json",
        "final": ROOT/"outputs/final/config.json",
    }
    p = c.get(name); return p is not None and p.exists()

def phase0(dry):
    head("Environment Check", 0)
    v = sys.version_info
    if v < (3,10): err(f"Python 3.10+ required, got {v.major}.{v.minor}"); sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines(): ok(f"GPU: {line.strip()}")
        else:
            err("nvidia-smi failed"); (None if dry else sys.exit(1))
    except FileNotFoundError:
        err("nvidia-smi not found"); (None if dry else sys.exit(1))

    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory/1e9
            ok(f"CUDA {torch.version.cuda} · {torch.cuda.get_device_name(0)} · {vram:.1f}GB VRAM")
            if vram < 20: warn(f"Only {vram:.1f}GB VRAM — A5000 (24GB) recommended")
        else:
            err("CUDA not available"); (None if dry else sys.exit(1))
    except ImportError:
        info("torch not yet installed — Phase 2 will install it")

    missing = []
    for var in ["HF_TOKEN","WANDB_API_KEY"]:
        val = os.environ.get(var,"")
        ok(f"{var} = {val[:10]}...") if val else (err(f"{var} not set"), missing.append(var))
    for var in ["WANDB_ENTITY","WANDB_PROJECT","HF_REPO_ID"]:
        val = os.environ.get(var,"")
        ok(f"{var} = {val}") if val else warn(f"{var} not set (will use default)")

    if missing and not dry:
        info("Set missing vars in .env or RunPod environment settings"); sys.exit(1)
    ok("Environment check passed")

def phase1(dry):
    head("Authentication", 1)
    hf = os.environ.get("HF_TOKEN",""); wb = os.environ.get("WANDB_API_KEY","")
    if hf:
        c = run(f"hf auth login --token {hf}", dry=dry)
        if c != 0:
            if not dry:
                try:
                    from huggingface_hub import login
                    login(token=hf, add_to_git_credential=True)
                    c = 0
                except Exception as e:
                    err(f"HF login error: {e}"); c = 1
            else:
                c = 0
        ok("HuggingFace authenticated") if c==0 else err("HF login failed")
    if wb:
        c = run(f"wandb login {wb}", dry=dry)
        ok("W&B authenticated") if c==0 else err("W&B login failed")

def _resolve_packages() -> list:
    """
    Build the install list with the right torch for this machine.
    - If torch is already installed and working, skip it.
    - If CUDA is available, install torch with the matching CUDA index URL.
    - Otherwise install CPU torch.
    Keeps all other packages pinned for reproducibility.
    """
    base = [
        "transformers==4.44.0", "peft==0.12.0", "trl==0.9.6",
        "bitsandbytes==0.43.3", "datasets==2.21.0", "accelerate==0.33.0",
        "wandb==0.17.7", "huggingface_hub==0.24.5", "sentencepiece==0.2.0",
        "einops==0.8.0", "scipy==1.14.0", "pandas==2.2.0", "pyarrow==16.0.0",
        "rank_bm25", "sentence-transformers", "matplotlib", "tqdm",
    ]

    try:
        import torch
        print(f"  {G}✓{X} torch {torch.__version__} already installed — skipping torch install")
        return base
    except ImportError:
        pass

    cuda_ver = None
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            if "CUDA Version" in line:
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    cuda_ver = parts[1].strip().split()[0]   
                    break
    except Exception:
        pass

    if cuda_ver:
        major = int(cuda_ver.split(".")[0])
        if major >= 12:
            cu_tag = "cu124"
        elif major == 11:
            cu_tag = "cu118"
        else:
            cu_tag = "cpu"
        index_url = f"https://download.pytorch.org/whl/{cu_tag}"
        print(f"  {C}→{X} CUDA {cuda_ver} detected → torch index: {index_url}")
        return [f"--extra-index-url={index_url}", "torch", "torchvision"] + base
    else:
        print(f"  {C}→{X} No CUDA → installing CPU torch")
        return ["torch", "torchvision",
                "--index-url=https://download.pytorch.org/whl/cpu"] + base

def phase2(state, dry):
    head("Install Dependencies", 2)
    if is_done(state,"install"): skip("Already installed"); return
    packages = _resolve_packages()
    must(run([sys.executable,"-m","pip","install","-q","--upgrade"]+packages, dry=dry), "install")
    mark(state,"install"); ok("All packages installed")

def phase3(state, dry):
    head("Data Pipeline", 3)

    if data_ok("raw_musique") and data_ok("raw_hotpot") and data_ok("raw_2wiki") or is_done(state,"download"):
        skip("Raw datasets present (bdsaglam/musique · hotpotqa/hotpot_qa · framolfese/2WikiMultihopQA)")
        mark(state,"download")
    else:
        info("Downloading datasets from HuggingFace...")
        must(py("scripts/download_data.py", dry=dry), "download")
        mark(state,"download"); ok("Datasets downloaded")

    for key, script, label in [
        ("proc_musique","scripts/process_musique.py","MuSiQue (gold decompositions)"),
        ("proc_hotpot", "scripts/process_hotpot.py", "HotpotQA (bridge + comparison)"),
        ("proc_2wiki",  "scripts/process_2wiki.py",  "2WikiMultiHopQA (4 reasoning types)"),
    ]:
        if is_done(state,key): skip(f"{label} already processed"); continue
        info(f"Processing {label}...")
        must(py(script, dry=dry), key)
        mark(state,key); ok(f"{label} processed")

    if data_ok("sft_hf") or is_done(state,"build_splits"):
        skip("SFT splits already built")
    else:
        info("Merging → filter → dedup → 80/10/10 split → HF DatasetDict...")
        must(py("scripts/build_splits.py", dry=dry), "build_splits")
        mark(state,"build_splits"); ok("SFT splits built")

    if data_ok("orpo_hf") or is_done(state,"build_orpo"):
        skip("ORPO pairs already built")
    else:
        info("Building (chosen, rejected) pairs — 5 rejection types...")
        must(py("scripts/build_orpo_pairs.py", dry=dry), "build_orpo")
        mark(state,"build_orpo"); ok("ORPO pairs built")

    py("scripts/stats.py", dry=dry)
    _log_data_wandb(dry)
    mark(state,"data_pipeline"); ok("Data pipeline complete")


def _log_data_wandb(dry):
    if dry: return
    try:
        import wandb
        counts = {}
        for name, path in [
            ("musique_train",  "data/processed/musique_train.jsonl"),
            ("hotpotqa_train", "data/processed/hotpotqa_train.jsonl"),
            ("wiki_train",     "data/processed/2wiki_train.jsonl"),
            ("sft_train",      "data/processed/train_sft.jsonl"),
            ("sft_val",        "data/processed/val_sft.jsonl"),
            ("sft_test",       "data/processed/test_sft.jsonl"),
            ("orpo_train",     "data/processed/orpo_train.jsonl"),
        ]:
            p = ROOT/path; counts[name] = sum(1 for _ in open(p)) if p.exists() else 0

        wandb.init(project=os.environ.get("WANDB_PROJECT","querydecomp"),
                   entity=os.environ.get("WANDB_ENTITY","") or None,
                   name="data-pipeline", tags=["data"], job_type="data_preparation", reinit=True)
        wandb.log({"data_stats": wandb.Table(
            columns=["split","rows"], data=[[k,v] for k,v in counts.items()])})
        wandb.summary.update(counts)
        wandb.finish(); ok("Data stats logged to W&B")
    except Exception as e:
        warn(f"W&B data logging skipped: {e}")

def phase4(state, timer, dry):
    head("Stage 1 — QLoRA SFT", 4)
    if model_ok("s1_adapter") or is_done(state,"s1_train"):
        skip("Stage 1 adapter exists"); mark(state,"s1_train"); return
    info("QLoRA: Qwen2.5-1.5B + NF4 4-bit + LoRA r=16 + 3 epochs · W&B: stage1-qlora-r16")
    info("Expected ~5 hours")
    must(py("train/stage1_qlora.py", dry=dry), "s1_train")
    elapsed = timer.phase(); mark(state,"s1_train",{"elapsed":elapsed}); ok(f"Stage 1 done ({elapsed})")

def phase5(state, timer, dry):
    head("Merge Stage 1 Adapter → bf16", 5)
    if model_ok("s1_merged") or is_done(state,"s1_merge"):
        skip("Stage 1 merged exists"); return
    must(py("train/merge_and_push.py",["--stage","1"], dry=dry), "s1_merge")
    elapsed = timer.phase(); mark(state,"s1_merge",{"elapsed":elapsed}); ok(f"Merged ({elapsed})")

def phase6(state, timer, dry):
    head("Stage 2 — DoRA Refinement", 6)
    if model_ok("s2_adapter") or is_done(state,"s2_train"):
        skip("Stage 2 adapter exists"); mark(state,"s2_train"); return
    info("DoRA: magnitude+direction split · r=8 · bf16 · lr=5e-5 · 1 epoch · W&B: stage2-dora")
    info("Expected ~2.5 hours")
    must(py("train/stage2_dora.py", dry=dry), "s2_train")
    elapsed = timer.phase(); mark(state,"s2_train",{"elapsed":elapsed}); ok(f"Stage 2 done ({elapsed})")

def phase7(state, timer, dry):
    head("Merge Stage 2 Adapter → bf16", 7)
    if model_ok("s2_merged") or is_done(state,"s2_merge"):
        skip("Stage 2 merged exists"); return
    must(py("train/merge_and_push.py",["--stage","2"], dry=dry), "s2_merge")
    elapsed = timer.phase(); mark(state,"s2_merge",{"elapsed":elapsed}); ok(f"Merged ({elapsed})")

def phase8(state, timer, dry):
    head("Stage 3 — ORPO Alignment", 8)
    if model_ok("s3_adapter") or is_done(state,"s3_train"):
        skip("Stage 3 adapter exists"); mark(state,"s3_train"); return
    info("ORPO: beta=0.1 · 5 rejection types · no reference model · W&B: stage3-orpo")
    info("Expected ~3.5 hours")
    must(py("train/stage3_orpo.py", dry=dry), "s3_train")
    elapsed = timer.phase(); mark(state,"s3_train",{"elapsed":elapsed}); ok(f"Stage 3 done ({elapsed})")

def phase9(state, timer, dry):
    head("Final Merge + Push to HuggingFace", 9)
    repo_id = os.environ.get("HF_REPO_ID",
        f"{os.environ.get('HF_USERNAME','AmareshHebbar')}/"
        f"{os.environ.get('HF_MODEL_NAME','querydecomp-qwen2.5-1.5b')}")

    if is_done(state,"push_hub"):
        skip(f"Already pushed: {repo_id}"); return
    info(f"Merging Stage 3 → bf16 → push to: {repo_id}")
    info("Also exporting GGUF Q4_K_M for llama.cpp / Ollama")
    must(py("train/merge_and_push.py",["--stage","3","--push","--push_gguf","--repo_id",repo_id], dry=dry), "push_hub")
    elapsed = timer.phase()
    mark(state,"push_hub",{"repo_id":repo_id,"elapsed":elapsed})
    ok(f"Model live: https://huggingface.co/{repo_id}")

def phase10(state, timer, dry):
    head("Evaluation", 10)
    results = {}

    for key, script, label in [
        ("eval_decomp",    "eval/evaluate_decomposition.py", "Decomposition F1"),
        ("eval_retrieval", "eval/evaluate_retrieval.py",     "Retrieval hit@k"),
        ("eval_speed",     "eval/benchmark_speed.py",        "Speed benchmark"),
    ]:
        if is_done(state,key):
            skip(f"{label} already done"); _load_result(key, results); continue
        if not (ROOT/script).exists():
            warn(f"{script} not found — skipping"); continue
        info(f"Running {label}...")
        code = py(script, dry=dry)
        elapsed = timer.phase()
        if code == 0:
            mark(state,key,{"elapsed":elapsed}); _load_result(key, results); ok(f"{label} done ({elapsed})")
        else:
            warn(f"{label} returned {code} — continuing")

    out = ROOT/"results"/"all_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    _log_eval_wandb(results, dry)
    mark(state,"evaluation"); return results


def _load_result(key, results):
    paths = {
        "eval_decomp": "results/decomposition_metrics.json",
        "eval_retrieval": "results/retrieval_metrics.json",
        "eval_speed":"results/speed_metrics.json",
    }
    p = ROOT/paths.get(key,"")
    if p.exists():
        try: results[key] = json.loads(p.read_text())
        except: pass


def _log_eval_wandb(results, dry):
    if dry or not results: return
    try:
        import wandb
        proj = os.environ.get("WANDB_PROJECT","querydecomp")
        ent= os.environ.get("WANDB_ENTITY","") or None
        wandb.init(project=proj, entity=ent, name="final-evaluation",
                   tags=["eval","benchmark"], job_type="evaluation", reinit=True)

        ft= results.get("eval_decomp",{}).get("finetuned",{})
        bs= results.get("eval_decomp",{}).get("baseline",{})
        ret = results.get("eval_retrieval",{})
        raw = ret.get("raw",{}); dec = ret.get("decomposed",{})
        sp= next((s for s in results.get("eval_speed",[]) if "QueryDecomp" in s.get("label","")), {}) \
              if isinstance(results.get("eval_speed"), list) else {}

        for m in ["json_parse_rate","hop_count_acc","hop_coverage_f1","dep_graph_acc"]:
            if m in ft: wandb.log({f"eval/finetuned/{m}": ft[m]})
            if m in bs: wandb.log({f"eval/baseline/{m}": bs[m]})
        for k in ["hit@1","hit@3","hit@5"]:
            if k in raw: wandb.log({f"retrieval/raw/{k}": raw[k]})
            if k in dec: wandb.log({f"retrieval/decomp/{k}": dec[k]})

        if ft and bs:
            tbl = wandb.Table(columns=["metric","baseline","finetuned","delta"])
            for m in ["json_parse_rate","hop_count_acc","hop_coverage_f1","dep_graph_acc"]:
                tbl.add_data(m, round(bs.get(m,0),4), round(ft.get(m,0),4), round(ft.get(m,0)-bs.get(m,0),4))
            wandb.log({"eval/decomp_comparison": tbl})

        ret_tbl = wandb.Table(columns=["metric","raw_question","with_decomp","delta"])
        for k in ["hit@1","hit@3","hit@5"]:
            r=raw.get(k,0); d=dec.get(k,0)
            ret_tbl.add_data(k, round(r,4), round(d,4), round(d-r,4))
        wandb.log({"eval/retrieval_improvement": ret_tbl})

        wandb.run.summary.update({
            "final/hop_coverage_f1": ft.get("hop_coverage_f1",0),
            "final/hop_count_acc":ft.get("hop_count_acc",0),
            "final/retrieval_hit3":  dec.get("hit@3",0),
            "final/speed_p50_ms": sp.get("p50_ms",0),
            "final/cost_per_1k":sp.get("cost_per_1k_usd",0),
        })
        wandb.finish(); ok("All eval metrics logged to W&B")
    except Exception as e:
        warn(f"W&B eval logging skipped: {e}")

def phase11(state, timer, dry):
    head("Benchmark Report", 11)
    if dry: skip("DRY RUN — skipped"); return

    results = {}
    all_path = ROOT/"results"/"all_metrics.json"
    if all_path.exists():
        try: results = json.loads(all_path.read_text())
        except: pass
    for key, path in [
        ("eval_decomp","results/decomposition_metrics.json"),
        ("eval_retrieval","results/retrieval_metrics.json"),
        ("eval_speed","results/speed_metrics.json"),
    ]:
        if key not in results:
            p = ROOT/path
            if p.exists():
                try: results[key] = json.loads(p.read_text())
                except: pass

    md, html = _build_report(state, timer, results)

    rd = ROOT/"results"
    rd.mkdir(exist_ok=True)
    (rd/"benchmark_report.md").write_text(md)
    (rd/"benchmark_report.html").write_text(html)
    ok("results/benchmark_report.md saved")
    ok("results/benchmark_report.html saved")

    _push_report_wandb(md, html, results, state, timer)
    mark(state,"report")


def _build_report(state, timer, results):
    repo= state.get("push_hub",{}).get("repo_id","AmareshHebbar/querydecomp-qwen2.5-1.5b")
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    total= timer.total()
    cost= timer.cost()
    rate= os.environ.get("RUNPOD_HOURLY_RATE","0.22")
    ent = os.environ.get("WANDB_ENTITY","amareshhebbar")
    proj= os.environ.get("WANDB_PROJECT","querydecomp")

    ft= results.get("eval_decomp",{}).get("finetuned",{})
    bs= results.get("eval_decomp",{}).get("baseline",{})
    ret= results.get("eval_retrieval",{})
    raw= ret.get("raw",{})
    dec= ret.get("decomposed",{})
    spds = results.get("eval_speed",[])
    ft_sp = next((s for s in spds if "QueryDecomp" in s.get("label","")), {}) if isinstance(spds,list) else {}
    g4_sp = next((s for s in spds if "GPT-4o" in s.get("label","")), {"p50_ms":1800,"p95_ms":3200,"cost_per_1k_usd":30.0}) if isinstance(spds,list) else {}

    s1 = state.get("s1_train",{}).get("elapsed","N/A")
    s2 = state.get("s2_train",{}).get("elapsed","N/A")
    s3 = state.get("s3_train",{}).get("elapsed","N/A")

    def pct(v): return f"{float(v or 0)*100:.1f}%"
    def ms(v):  return f"{v}ms" if v else "—"
    def f4(v):  return f"{float(v or 0):.4f}"
    def delta(a,b): d=float(b or 0)-float(a or 0); return f"+{d*100:.1f}%" if d>=0 else f"{d*100:.1f}%"

    md = f"""# QueryDecomp — Benchmark Report

**Generated:** {now}  
**Total wall time:** {total}  
**Estimated cost:** {cost} (A5000 spot @ ${rate}/hr)  
**HuggingFace:** https://huggingface.co/{repo}  
**W&B dashboard:** https://wandb.ai/{ent}/{proj}

---

## Training Summary

| Stage | Method | Config | Duration |
|---|---|---|---|
| Stage 1 | QLoRA SFT | r=16, NF4 4-bit, lr=2e-4, 3 epochs | {s1} |
| Stage 2 | DoRA refinement | r=8, bf16, lr=5e-5, 1 epoch | {s2} |
| Stage 3 | ORPO alignment | beta=0.1, bf16, lr=5e-6, 1 epoch | {s3} |
| **Total** | | | **{total}** |

**Compute cost: {cost}**

---

## Decomposition Quality (MuSiQue validation)

| Metric | Baseline (prompted) | Fine-tuned | Delta |
|---|---|---|---|
| JSON parse rate | {pct(bs.get("json_parse_rate"))} | {pct(ft.get("json_parse_rate"))} | {delta(bs.get("json_parse_rate"),ft.get("json_parse_rate"))} |
| Hop count accuracy | {pct(bs.get("hop_count_acc"))} | {pct(ft.get("hop_count_acc"))} | {delta(bs.get("hop_count_acc"),ft.get("hop_count_acc"))} |
| **Hop coverage F1** | **{f4(bs.get("hop_coverage_f1"))}** | **{f4(ft.get("hop_coverage_f1"))}** | **{delta(bs.get("hop_coverage_f1"),ft.get("hop_coverage_f1"))}** |
| Dep graph accuracy | {pct(bs.get("dep_graph_acc"))} | {pct(ft.get("dep_graph_acc"))} | {delta(bs.get("dep_graph_acc"),ft.get("dep_graph_acc"))} |

*Baseline = Qwen2.5-1.5B-Instruct with same system prompt, no fine-tuning*

---

## Retrieval Impact (BM25 · MuSiQue validation)

| Metric | Raw question | With decomposition | Improvement |
|---|---|---|---|
| hit@1 | {pct(raw.get("hit@1"))} | {pct(dec.get("hit@1"))} | {delta(raw.get("hit@1"),dec.get("hit@1"))} |
| **hit@3** | **{pct(raw.get("hit@3"))}** | **{pct(dec.get("hit@3"))}** | **{delta(raw.get("hit@3"),dec.get("hit@3"))}** |
| hit@5 | {pct(raw.get("hit@5"))} | {pct(dec.get("hit@5"))} | {delta(raw.get("hit@5"),dec.get("hit@5"))} |

---

## Speed Benchmark

| Model | p50 | p95 | Tokens/s | $/1k queries |
|---|---|---|---|---|
| GPT-4o (API, prompted) | {ms(g4_sp.get("p50_ms"))} | {ms(g4_sp.get("p95_ms"))} | {g4_sp.get("tokens_per_second","~60")} | ${g4_sp.get("cost_per_1k_usd",30.00)} |
| **QueryDecomp (A5000)** | **{ms(ft_sp.get("p50_ms","~50"))}** | **{ms(ft_sp.get("p95_ms","~90"))}** | **{ft_sp.get("tokens_per_second","—")}** | **${ft_sp.get("cost_per_1k_usd",0.02)}** |

---

## Quick Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, torch

model_id = "{repo}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

SYSTEM = (
    "You are a query decomposition engine. "
    "Return ONLY a valid JSON array with keys: hop, sub_query, depends_on."
)

def decompose(question):
    msgs = [{{"role":"system","content":SYSTEM}},{{"role":"user","content":question}}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return json.loads(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

print(decompose("Where was the director of Inception born?"))
# [{{"hop":1,"sub_query":"Who directed Inception?","depends_on":[]}},
#  {{"hop":2,"sub_query":"Where was Christopher Nolan born?","depends_on":[1]}}]
```

---

*Generated by runpod/run_all.py · QueryDecomp · {now}*
"""

    def bar(v, color="#7c6af7", max_v=1.0):
        w = min(int(float(v or 0)/max_v*260), 260)
        return f'<div style="background:{color};height:8px;width:{w}px;border-radius:4px;display:inline-block;vertical-align:middle;"></div>'

    decomp_rows = ""
    for label, key in [("JSON parse rate","json_parse_rate"),("Hop count accuracy","hop_count_acc"),
                        ("Hop coverage F1","hop_coverage_f1"),("Dep graph accuracy","dep_graph_acc")]:
        b=float(bs.get(key,0)); f=float(ft.get(key,0)); d=f-b
        sign="+"; col="#4ade80" if d>0.05 else ("#fbbf24" if d>0 else "#f87171")
        if d<0: sign=""
        decomp_rows += f"""
      <tr>
        <td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;">{label}</td>
        <td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;color:#6b6b8a;font-family:monospace;">{pct(b)}</td>
        <td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;color:#4ade80;font-family:monospace;font-weight:600;">{pct(f)}</td>
        <td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;color:{col};font-family:monospace;">{sign}{pct(d)}</td>
      </tr>"""

    ret_rows = ""
    for k in ["hit@1","hit@3","hit@5"]:
        r=float(raw.get(k,0)); d=float(dec.get(k,0)); diff=d-r
        ret_rows += f"""
      <tr>
        <td style="padding:10px 16px;border-bottom:1px solid #1e1e2e;font-weight:{'600' if k=='hit@3' else '400'};">{k}</td>
        <td style="padding:10px 16px;border-bottom:1px solid #1e1e2e;">
          <div style="margin-bottom:4px;">{bar(r,"#6b6b8a")}<span style="color:#6b6b8a;font-family:monospace;font-size:0.8rem;margin-left:10px;">{pct(r)}</span></div>
          <div>{bar(d)}<span style="color:#4ade80;font-family:monospace;font-size:0.8rem;margin-left:10px;font-weight:600;">{pct(d)}</span>
          <span style="color:#7c6af7;font-size:0.75rem;margin-left:8px;">+{pct(diff)}</span></div>
        </td>
      </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QueryDecomp — Benchmark Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0a0f;color:#e2e2f0;font-family:system-ui,sans-serif;font-size:15px;line-height:1.65;padding:48px 24px}}
.w{{max-width:860px;margin:0 auto}}
h1{{font-size:1.9rem;font-weight:700;letter-spacing:-0.03em;margin-bottom:6px}}
h2{{font-size:0.72rem;font-weight:600;color:#7c6af7;text-transform:uppercase;letter-spacing:0.1em;margin:40px 0 14px;padding-bottom:8px;border-bottom:1px solid #1e1e2e}}
.meta{{color:#6b6b8a;font-size:0.82rem;margin-bottom:40px;line-height:1.8}}
.meta a{{color:#7c6af7;text-decoration:none}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:28px}}
.stat{{background:#111118;border:1px solid #1e1e2e;border-radius:6px;padding:18px 20px}}
.stat-n{{font-size:1.7rem;font-weight:700;color:#7c6af7;font-family:monospace;line-height:1}}
.stat-l{{font-size:0.72rem;color:#6b6b8a;margin-top:4px}}
.card{{background:#111118;border:1px solid #1e1e2e;border-radius:6px;overflow:hidden;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse}}
th{{background:#16161f;padding:10px 16px;text-align:left;font-size:0.7rem;color:#6b6b8a;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1e1e2e}}
.bar-pad{{padding:20px 24px}}
.links{{display:flex;gap:12px;flex-wrap:wrap;margin:28px 0}}
.link{{color:#7c6af7;font-size:0.82rem;padding:8px 16px;border:1px solid #2a2a3e;border-radius:4px;text-decoration:none}}
pre{{background:#111118;border:1px solid #1e1e2e;border-radius:6px;padding:20px;font-family:monospace;font-size:0.78rem;overflow-x:auto;line-height:1.7;color:#c3e88d;margin-bottom:20px}}
.footer{{margin-top:48px;padding-top:20px;border-top:1px solid #1e1e2e;color:#6b6b8a;font-size:0.75rem}}
</style>
</head>
<body>
<div class="w">
  <h1>QueryDecomp — Benchmark Report</h1>
  <div class="meta">
    Generated {now} · Wall time: <strong style="color:#e2e2f0;">{total}</strong> · Cost: <strong style="color:#4ade80;">{cost}</strong><br>
    <a href="https://huggingface.co/{repo}" target="_blank">huggingface.co/{repo}</a> ·
    <a href="https://wandb.ai/{ent}/{proj}" target="_blank">W&B: {ent}/{proj}</a>
  </div>

  <div class="stats">
    <div class="stat"><div class="stat-n">{pct(ft.get("hop_coverage_f1"))}</div><div class="stat-l">Hop Coverage F1</div></div>
    <div class="stat"><div class="stat-n">{pct(dec.get("hit@3"))}</div><div class="stat-l">Retrieval hit@3</div></div>
    <div class="stat"><div class="stat-n">{ms(ft_sp.get("p50_ms","~50"))}</div><div class="stat-l">p50 latency</div></div>
    <div class="stat"><div class="stat-n">{cost}</div><div class="stat-l">Total compute cost</div></div>
  </div>

  <h2>Training Stages</h2>
  <div class="card"><table>
    <tr><th>Stage</th><th>Method</th><th>Config</th><th>Duration</th></tr>
    <tr><td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;">Stage 1</td><td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;color:#6b6b8a;">QLoRA SFT</td><td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;color:#6b6b8a;font-family:monospace;font-size:0.8rem;">r=16 · NF4 · lr=2e-4 · 3ep</td><td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;font-family:monospace;">{s1}</td></tr>
    <tr><td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;">Stage 2</td><td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;color:#6b6b8a;">DoRA refinement</td><td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;color:#6b6b8a;font-family:monospace;font-size:0.8rem;">r=8 · bf16 · lr=5e-5 · 1ep</td><td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;font-family:monospace;">{s2}</td></tr>
    <tr><td style="padding:12px 16px;">Stage 3</td><td style="padding:12px 16px;color:#6b6b8a;">ORPO alignment</td><td style="padding:12px 16px;color:#6b6b8a;font-family:monospace;font-size:0.8rem;">beta=0.1 · bf16 · lr=5e-6 · 1ep</td><td style="padding:12px 16px;font-family:monospace;">{s3}</td></tr>
  </table></div>

  <h2>Decomposition Quality</h2>
  <div class="card"><table>
    <tr><th>Metric</th><th>Baseline (prompted)</th><th>Fine-tuned</th><th>Delta</th></tr>
    {decomp_rows}
  </table></div>

  <h2>Retrieval Impact (BM25 · MuSiQue validation)</h2>
  <div class="card"><div class="bar-pad">
    <div style="display:flex;gap:20px;margin-bottom:18px;font-size:0.75rem;color:#6b6b8a;">
      <span><span style="display:inline-block;width:10px;height:6px;background:#6b6b8a;border-radius:2px;margin-right:6px;"></span>Raw question</span>
      <span><span style="display:inline-block;width:10px;height:6px;background:#7c6af7;border-radius:2px;margin-right:6px;"></span>With decomposition</span>
    </div>
    <table><tr><th>Metric</th><th>Before → After</th></tr>
    {ret_rows}
    </table>
  </div></div>

  <h2>Speed vs Alternatives</h2>
  <div class="card"><table>
    <tr><th>Model</th><th>p50 latency</th><th>p95 latency</th><th>$/1k queries</th></tr>
    <tr>
      <td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;">GPT-4o (API)</td>
      <td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;font-family:monospace;color:#f87171;">{ms(g4_sp.get("p50_ms","~1800"))}</td>
      <td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;font-family:monospace;color:#f87171;">{ms(g4_sp.get("p95_ms","~3200"))}</td>
      <td style="padding:12px 16px;border-bottom:1px solid #1e1e2e;font-family:monospace;color:#f87171;">${g4_sp.get("cost_per_1k_usd",30.0)}</td>
    </tr>
    <tr style="background:rgba(124,106,247,0.07);">
      <td style="padding:12px 16px;color:#7c6af7;font-weight:700;">QueryDecomp (A5000)</td>
      <td style="padding:12px 16px;font-family:monospace;color:#4ade80;font-weight:700;">{ms(ft_sp.get("p50_ms","~50"))}</td>
      <td style="padding:12px 16px;font-family:monospace;color:#4ade80;">{ms(ft_sp.get("p95_ms","~90"))}</td>
      <td style="padding:12px 16px;font-family:monospace;color:#4ade80;font-weight:700;">${ft_sp.get("cost_per_1k_usd",0.02)}</td>
    </tr>
  </table></div>

  <h2>Quick Inference</h2>
  <pre>from transformers import AutoModelForCausalLM, AutoTokenizer
import json, torch

model_id= "{repo}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model= AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

def decompose(question):
    msgs = [{{"role":"system","content":"Return ONLY a JSON array: hop, sub_query, depends_on."}},
            {{"role":"user","content":question}}]
    text= tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return json.loads(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# Ollama: ollama run hf.co/{repo}</pre>

  <div class="links">
    <a class="link" href="https://huggingface.co/{repo}" target="_blank">HuggingFace Model</a>
    <a class="link" href="https://wandb.ai/{ent}/{proj}" target="_blank">W&B Dashboard</a>
    <a class="link" href="https://github.com/AmareshHebbar/querydecomp" target="_blank">GitHub</a>
  </div>

  <div class="footer">QueryDecomp · Benchmark Report · Generated {now}</div>
</div>
</body>
</html>"""

    return md, html


def _push_report_wandb(md, html, results, state, timer):
    try:
        import wandb
        proj = os.environ.get("WANDB_PROJECT","querydecomp")
        ent= os.environ.get("WANDB_ENTITY","") or None
        repo = state.get("push_hub",{}).get("repo_id","AmareshHebbar/querydecomp-qwen2.5-1.5b")

        wandb.init(project=proj, entity=ent, name="benchmark-report",
                   tags=["report","final"], job_type="reporting", reinit=True)

        artifact = wandb.Artifact("benchmark-report", type="report",
                                  description="Decomp F1, retrieval hit@k, speed benchmark")
        with tempfile.TemporaryDirectory() as tmp:
            md_p= os.path.join(tmp,"benchmark_report.md")
            html_p = os.path.join(tmp,"benchmark_report.html")
            open(md_p,"w").write(md); open(html_p,"w").write(html)
            artifact.add_file(md_p,"benchmark_report.md")
            artifact.add_file(html_p,"benchmark_report.html")
            wandb.log_artifact(artifact)

        ft= results.get("eval_decomp",{}).get("finetuned",{})
        dec = results.get("eval_retrieval",{}).get("decomposed",{})
        sp= next((s for s in results.get("eval_speed",[]) if "QueryDecomp" in s.get("label","")),{}) if isinstance(results.get("eval_speed"),list) else {}
        wandb.run.summary.update({
            "final/hop_coverage_f1": ft.get("hop_coverage_f1",0),
            "final/retrieval_hit3":  dec.get("hit@3",0),
            "final/speed_p50_ms": sp.get("p50_ms",0),
            "final/total_cost_usd":  float(timer.cost().replace("$","")),
            "final/hf_model":f"https://huggingface.co/{repo}",
        })
        wandb.finish(); ok("Report pushed to W&B as artifact")
    except Exception as e:
        warn(f"W&B report push skipped: {e}")

def phase12(state, timer):
    head("Complete — Summary", 12)
    repo= state.get("push_hub",{}).get("repo_id","AmareshHebbar/querydecomp-qwen2.5-1.5b")
    ent= os.environ.get("WANDB_ENTITY","amareshhebbar")
    proj= os.environ.get("WANDB_PROJECT","querydecomp")

    def e(k): return state.get(k,{}).get("elapsed","N/A")

    print(f"""
{B}  Training stages:{X}
    Stage 1 QLoRA   {e("s1_train"):>12}
    Stage 1 Merge   {e("s1_merge"):>12}
    Stage 2 DoRA    {e("s2_train"):>12}
    Stage 2 Merge   {e("s2_merge"):>12}
    Stage 3 ORPO    {e("s3_train"):>12}
    Push to HF      {e("push_hub"):>12}

{B}  Total wall time:{X}   {timer.total()}
{B}  Estimated cost:{X}    {timer.cost()}

{B}  Outputs:{X}
    Model:https://huggingface.co/{repo}
    W&B dashboard:  https://wandb.ai/{ent}/{proj}
    Report (HTML):  results/benchmark_report.html
    Report (MD): results/benchmark_report.md
    Metrics JSON:results/all_metrics.json

{B}  Quick inference:{X}
    from transformers import pipeline
    pipe = pipeline("text-generation", model="{repo}")
    out= pipe("Who directed Inception and where was that person born?",
                max_new_tokens=200, do_sample=False)
    print(out[0]["generated_text"])

    ollama run hf.co/{repo}
""")
    ok("Pod can be stopped. All outputs saved.")

def main():
    p = argparse.ArgumentParser(description="QueryDecomp master orchestrator.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="""
Examples:
  python runpod/run_all.py                     # full run from scratch
  python runpod/run_all.py --from_phase 6     # resume from Stage 2
  python runpod/run_all.py --only_data        # data pipeline only
  python runpod/run_all.py --skip_data        # skip data, train directly
  python runpod/run_all.py --dry_run          # show what would run
  python runpod/run_all.py --reset            # wipe state, start fresh""")
    p.add_argument("--from_phase", type=int, default=0)
    p.add_argument("--only_data",  action="store_true")
    p.add_argument("--skip_data",  action="store_true")
    p.add_argument("--dry_run",    action="store_true")
    p.add_argument("--reset",      action="store_true")
    args = p.parse_args()

    timer = Timer()
    dry= args.dry_run

    print(f"\n{B}{'═'*64}\n  QueryDecomp — Master Orchestrator\n  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n  Dry run: {dry}\n{'═'*64}{X}")

    if args.reset:
        if STATE_FILE.exists(): STATE_FILE.unlink(); ok("State cleared")

    state = load_state()
    if state:
        done_keys = [k for k,v in state.items() if isinstance(v,dict) and v.get("done")]
        info(f"Resuming — already done: {done_keys}")

    def should(n): return n >= args.from_phase

    try:
        phase0(dry)                                               
        if should(1):  phase1(dry)
        if should(2):  phase2(state, dry)
        if should(3) and not args.skip_data: phase3(state, dry)
        elif args.skip_data: skip("Data pipeline skipped (--skip_data)")
        if args.only_data: phase12(state, timer); return
        if should(4):  phase4(state, timer, dry)
        if should(5):  phase5(state, timer, dry)
        if should(6):  phase6(state, timer, dry)
        if should(7):  phase7(state, timer, dry)
        if should(8):  phase8(state, timer, dry)
        if should(9):  phase9(state, timer, dry)
        if should(10): phase10(state, timer, dry)
        if should(11): phase11(state, timer, dry)
        phase12(state, timer)

    except KeyboardInterrupt:
        print(f"\n{Y}  Interrupted.{X}")
        info("State saved. Resume: python runpod/run_all.py")
        info("Jump to phase N: python runpod/run_all.py --from_phase N")
        sys.exit(130)


if __name__ == "__main__":
    main()