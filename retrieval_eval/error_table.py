#!/usr/bin/env python3
"""Where do the remaining errors live? Buckets the per-question results of one run.

  error_table.py results/title_content__heldout.json
Buckets: rank 1 | rank 2-3 sibling (wrong page shares >=2 title words with gold) |
rank 2-3 other | rank 4-10 | >10 or absent, with examples.
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # Webiks-Hebrew-RAGbot-Demo checkout
DATA = os.environ.get("RAG_DATA_DIR", os.path.join(REPO, "data"))
MODEL = os.environ.get("RAG_MODEL_DIR", os.path.join(REPO, "app", "artifacts", "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"))
RESULTS = os.environ.get("RAG_RESULTS_DIR", os.path.join(REPO, "results"))
TRAINER = os.environ.get("RAG_TRAINER_DIR", os.path.join(os.path.dirname(REPO), "Webiks-Hebrew-RAGbot-Trainer"))
sys.path.insert(0, HERE)
from eval_retrieval import load_eval_set

def toks(s): return set(w for w in re.findall(r"[֐-׿]+", s or "") if len(w) > 2)

def main(path):
    R = json.load(open(path)); per = R["per_query"]
    d = json.load(open(os.path.join(DATA, "paragraph_corpus.json"), encoding="utf-8"))
    title = {int(d["doc_id"][k]): d["title"][k] for k in d["doc_id"]}
    gold = load_eval_set(R["metrics"]["eval_set"])
    b = {"rank 1": [], "rank 2-3, sibling page first": [], "rank 2-3, other": [], "rank 4-10": [], "rank >10 or absent": []}
    for m in per:
        r = m["first_gold_rank"]; q = m["question"]
        if r == 1: b["rank 1"].append(m); continue
        if r and r <= 3:
            top = title.get(m["top10_doc_ids"][0], ""); shared = max(len(toks(top) & toks(title[g])) for g in gold[q])
            b["rank 2-3, sibling page first" if shared >= 2 else "rank 2-3, other"].append(m)
        elif r and r <= 10: b["rank 4-10"].append(m)
        else: b["rank >10 or absent"].append(m)
    n = len(per)
    print(f"{R['metrics']['variant']} on {R['metrics']['eval_set']} (n={n})")
    for k, v in b.items():
        print(f"  {k:<32} {len(v):>4}  ({100*len(v)/n:4.1f}%)")
    print("\nexamples:")
    for k in ["rank 2-3, sibling page first", "rank 2-3, other", "rank >10 or absent"]:
        for m in b[k][:3]:
            g = next(iter(gold[m["question"]]))
            print(f"  [{k}] {m['question'][:55]!r}\n      gold: {title[g][:50]} | top1: {title.get(m['top10_doc_ids'][0], '?')[:50]}")

if __name__ == "__main__":
    main(sys.argv[1])
