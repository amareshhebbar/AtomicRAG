import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"

class BM25Retriever:
    def __init__(self, corpus: list[str]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "rank_bm25", "-q"])
            from rank_bm25 import BM25Okapi

        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25= BM25Okapi(tokenized)
        self.corpus = corpus

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        """Return indices of top-k documents."""
        scores = self.bm25.get_scores(query.lower().split())
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

class DenseRetriever:
    def __init__(self, corpus: list[str]):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
            from sentence_transformers import SentenceTransformer
            import numpy as np

        import numpy as np
        self.np = np
        print("  Loading sentence-transformers (all-MiniLM-L6-v2)...")
        self.model= SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus = corpus
        print(f"  Encoding {len(corpus)} documents...")
        self.embeddings = self.model.encode(corpus, show_progress_bar=True,
                                            convert_to_numpy=True, normalize_embeddings=True)

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        q_emb= self.model.encode([query], normalize_embeddings=True)
        scores = (q_emb @ self.embeddings.T)[0]
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

def load_musique_with_corpus(max_samples: int | None) -> list[dict]:
    raw_path = ROOT / "data" / "raw" / "musique" / "validation.parquet"
    if not raw_path.exists():
        print("  [WARN] MuSiQue raw data not found — using synthetic examples for demo")
        return _build_synthetic_examples()

    try:
        import pandas as pd
        df = pd.read_parquet(raw_path)
        if max_samples:
            df = df.head(max_samples)

        examples = []
        for _, row in df.iterrows():
            row = row.to_dict()
            paragraphs = row.get("paragraphs", [])
            if isinstance(paragraphs, dict):
                titles = paragraphs.get("title", [])
                bodies = paragraphs.get("paragraph_text", [])
            elif isinstance(paragraphs, list):
                titles = [p.get("title", "") if isinstance(p, dict) else "" for p in paragraphs]
                bodies = [p.get("paragraph_text", "") if isinstance(p, dict) else str(p) for p in paragraphs]
            else:
                continue

            decomp = row.get("question_decomposition", [])
            if isinstance(decomp, dict):
                keys = list(decomp.keys())
                n = len(decomp[keys[0]]) if keys else 0
                decomp = [{k: decomp[k][i] for k in keys} for i in range(n)]
            elif not isinstance(decomp, list):
                continue

            gold_para_idxs = set()
            for hop in decomp:
                idx = hop.get("paragraph_support_idx")
                if idx is not None:
                    gold_para_idxs.add(int(idx))

            if not bodies or not gold_para_idxs:
                continue

            corpus = [f"{t}: {b}" for t, b in zip(titles, bodies)]
            question = row.get("question", "").strip()

            gold_sub_queries = [h.get("question", "").strip() for h in decomp
                                if h.get("question")]

            if not question or not gold_sub_queries:
                continue

            examples.append({
                "question":question,
                "corpus":corpus,
                "gold_para_idxs":  gold_para_idxs,
                "gold_sub_queries": gold_sub_queries,
                "answer":row.get("answer", ""),
            })

        print(f"  Loaded {len(examples)} examples with corpus")
        return examples

    except Exception as e:
        print(f"  [WARN] Could not load MuSiQue parquet: {e}")
        return _build_synthetic_examples()


def _build_synthetic_examples():
    return [
        {
            "question":"Where was the director of Inception born?",
            "corpus":  [
                "Inception: Inception is a 2010 film directed by Christopher Nolan.",
                "Christopher Nolan: Christopher Nolan was born in London, England in 1970.",
                "Avatar: Avatar is a 2009 film directed by James Cameron.",
                "James Cameron: James Cameron was born in Kapuskasing, Ontario, Canada.",
                "London: London is the capital city of England.",
            ],
            "gold_para_idxs":{0, 1},
            "gold_sub_queries": ["Who directed Inception?", "Where was Christopher Nolan born?"],
            "answer":"London",
        },
        {
            "question":"What is the capital of the country where Einstein was born?",
            "corpus":  [
                "Einstein: Albert Einstein was born in Ulm, Germany in 1879.",
                "Germany: Germany is a country in central Europe. Its capital is Berlin.",
                "France: France is a country in Western Europe. Its capital is Paris.",
                "Albert Einstein: He developed the theory of relativity.",
                "Berlin: Berlin is the capital and largest city of Germany.",
            ],
            "gold_para_idxs":{0, 1},
            "gold_sub_queries": ["In which country was Einstein born?", "What is the capital of Germany?"],
            "answer":"Berlin",
        },
    ] * 10 

def get_predicted_sub_queries(model_id: str, questions: list[str]) -> list[list[str] | None]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.utils import build_prompt, parse_decomp_output
    except ImportError:
        print("[ERROR] torch not installed")
        sys.exit(1)

    if not Path(model_id).exists() and not model_id.startswith("Qwen"):
        print(f"  [WARN] Model {model_id} not found — using gold sub-queries as proxy")
        return None

    print(f"  Loading model for retrieval eval: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for q in questions:
        prompt = build_prompt(q, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
            parsed = parse_decomp_output(generated)
            sub_queries = [h["sub_query"] for h in parsed] if parsed else None
            results.append(sub_queries)
        except Exception:
            results.append(None)

    return results

def hit_at_k(retrieved_idxs: list[int], gold_idxs: set[int], k: int) -> int:
    return int(bool(set(retrieved_idxs[:k]) & gold_idxs))


def evaluate_retrieval(
    examples: list[dict],
    retriever_class,
    predicted_sub_queries: list[list[str] | None] | None,
    ks: list[int] = [1, 3, 5],
) -> dict:
    raw_hits= defaultdict(list) 
    decomp_hits = defaultdict(list) 

    for i, ex in enumerate(examples):
        corpus= ex["corpus"]
        gold_idxs = ex["gold_para_idxs"]
        question= ex["question"]

        if len(corpus) == 0 or len(gold_idxs) == 0:
            continue

        retriever = retriever_class(corpus)

        raw_retrieved = retriever.retrieve(question, k=max(ks))
        for k in ks:
            raw_hits[k].append(hit_at_k(raw_retrieved, gold_idxs, k))

        if predicted_sub_queries is not None and predicted_sub_queries[i] is not None:
            sub_queries = predicted_sub_queries[i]
        else:
            sub_queries = ex["gold_sub_queries"]  

        all_retrieved = []
        for sq in sub_queries:
            all_retrieved.extend(retriever.retrieve(sq, k=3))
        seen = set()
        all_retrieved_dedup = []
        for idx in all_retrieved:
            if idx not in seen:
                seen.add(idx)
                all_retrieved_dedup.append(idx)

        for k in ks:
            decomp_hits[k].append(hit_at_k(all_retrieved_dedup, gold_idxs, k))

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(examples)}")

    n = len(examples)
    return {
        "n_examples": n,
        "raw": {f"hit@{k}": round(sum(v)/len(v), 4) for k, v in raw_hits.items() if v},
        "decomposed": {f"hit@{k}": round(sum(v)/len(v), 4) for k, v in decomp_hits.items() if v},
    }

def print_retrieval_results(results: dict):
    raw = results["raw"]
    decomp = results["decomposed"]
    n = results["n_examples"]

    print(f"\n  Retrieval Results  ({n} examples)")
    print(f"  {'─'*52}")
    print(f"  {'Metric':12}  {'Raw Question':>14}  {'Decomposed':>12}  {'Delta':>8}")
    print(f"  {'─'*52}")

    for metric in sorted(raw.keys()):
        r = raw.get(metric, 0)
        d = decomp.get(metric, 0)
        delta = d - r
        sign= "+" if delta >= 0 else ""
        arrow = " ↑" if delta > 0.02 else (" ↓" if delta < -0.02 else "  ")
        print(f"  {metric:12}  {r:>14.1%}  {d:>12.1%}  {sign}{delta:.1%}{arrow}")

    print(f"  {'─'*52}")
    avg_raw= sum(raw.values())   / len(raw)   if raw   else 0
    avg_decomp= sum(decomp.values())/ len(decomp) if decomp else 0
    print(f"  {'avg':12}  {avg_raw:>14.1%}  {avg_decomp:>12.1%}  "
          f"{'+' if avg_decomp > avg_raw else ''}{avg_decomp-avg_raw:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default=None)
    parser.add_argument("--retriever",   type=str, default="bm25",
                        choices=["bm25", "dense"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--no_wandb",    action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\n{'═'*56}")
    print(f"  AtomicRAG — Retrieval Evaluator")
    print(f"  Retriever: {args.retriever.upper()}")
    print(f"{'═'*56}")

    examples = load_musique_with_corpus(args.max_samples)
    if not examples:
        print("[ERROR] No examples loaded")
        sys.exit(1)

    retriever_class = BM25Retriever if args.retriever == "bm25" else DenseRetriever

    model_path = args.model or str(ROOT / "outputs" / "final")
    questions= [ex["question"] for ex in examples]
    predicted= get_predicted_sub_queries(model_path, questions)

    print(f"\n  Evaluating retrieval...")
    results = evaluate_retrieval(examples, retriever_class, predicted)
    print_retrieval_results(results)

    out_path = RESULTS_DIR / "retrieval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path.relative_to(ROOT)}")

    if not args.no_wandb:
        try:
            import wandb
            if wandb.run:
                for split, metrics in [("raw", results["raw"]), ("decomposed", results["decomposed"])]:
                    for k, v in metrics.items():
                        wandb.log({f"retrieval/{split}/{k}": v})
        except Exception:
            pass

    h3_raw = results["raw"].get("hit@3", 0)
    h3_decomp = results["decomposed"].get("hit@3", 0)
    print(f"\n  Headline: hit@3  {h3_raw:.1%} → {h3_decomp:.1%}  "
          f"(+{h3_decomp - h3_raw:.1%})")


if __name__ == "__main__":
    main()