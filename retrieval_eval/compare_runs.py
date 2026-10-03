#!/usr/bin/env python3
"""Compare two eval_retrieval.py result files: metric deltas and per-question flips.

  compare_runs.py results/content__heldout.json results/title_content__heldout.json [--show 10]
"""
import argparse, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # Webiks-Hebrew-RAGbot-Demo checkout
DATA = os.environ.get("RAG_DATA_DIR", os.path.join(REPO, "data"))
MODEL = os.environ.get("RAG_MODEL_DIR", os.path.join(REPO, "app", "artifacts", "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"))
RESULTS = os.environ.get("RAG_RESULTS_DIR", os.path.join(REPO, "results"))
TRAINER = os.environ.get("RAG_TRAINER_DIR", os.path.join(os.path.dirname(REPO), "Webiks-Hebrew-RAGbot-Trainer"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    A, B = json.load(open(args.a)), json.load(open(args.b))
    d = json.load(open(os.path.join(DATA, "paragraph_corpus.json"), encoding="utf-8"))
    title = {int(d["doc_id"][k]): d["title"][k] for k in d["doc_id"]}
    print(f"{'metric':<10}{'A: '+A['metrics']['variant']:>18}{'B: '+B['metrics']['variant']:>18}{'delta':>10}")
    for m in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10"]:
        a, b = A["metrics"][m], B["metrics"][m]
        print(f"{m:<10}{a:>18.3f}{b:>18.3f}{b-a:>+10.3f}")
    print(f"{'not found':<10}{A['metrics']['not_found_in_top200_paragraphs']:>18}{B['metrics']['not_found_in_top200_paragraphs']:>18}")
    pa = {m["question"]: m for m in A["per_query"]}; pb = {m["question"]: m for m in B["per_query"]}
    rank = lambda m: m["first_gold_rank"] or 10**6
    better = [q for q in pa if rank(pb[q]) < rank(pa[q])]
    worse = [q for q in pa if rank(pb[q]) > rank(pa[q])]
    same = len(pa) - len(better) - len(worse)
    to1 = sum(1 for q in better if rank(pb[q]) == 1); from1 = sum(1 for q in worse if rank(pa[q]) == 1)
    print(f"\nper question: improved {len(better)} (of which now rank 1: {to1}) | worse {len(worse)} (of which lost rank 1: {from1}) | unchanged {same}")
    by_bucket = lambda qs, src: {"2-3": sum(1 for q in qs if 2 <= rank(src[q]) <= 3), "4-10": sum(1 for q in qs if 4 <= rank(src[q]) <= 10), ">10": sum(1 for q in qs if rank(src[q]) > 10)}
    print("improved questions, by their rank in A:", by_bucket(better, pa))
    print("worsened questions, by their rank in B:", by_bucket(worse, pb))
    def show(qs, label):
        print(f"\n--- {label} (up to {args.show})")
        for q in sorted(qs, key=lambda q: abs(rank(pa[q]) - rank(pb[q])), reverse=True)[:args.show]:
            ga, gb = pa[q], pb[q]
            print(f"* {q[:62]!r}  rank {ga['first_gold_rank']} -> {gb['first_gold_rank']}\n    A top1: {title.get(ga['top10_doc_ids'][0], '?')[:60]}\n    B top1: {title.get(gb['top10_doc_ids'][0], '?')[:60]}")
    show(better, "biggest improvements"); show(worse, "biggest regressions")

if __name__ == "__main__":
    main()
