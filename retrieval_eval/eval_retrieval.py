#!/usr/bin/env python3
"""Document-level retrieval evaluation on cached corpus embeddings.

Reproduces the metric definitions of Webiks' CustomInformationRetrievalEvaluator
(Trainer/utils.py): paragraphs are scored by cosine, each document gets the max score of
its paragraphs, documents are ranked, and hit@k / MRR@k / nDCG@k are computed against the
gold doc_ids of each unique question. Difference: corpus vectors come from data/emb/<variant>.npy
instead of being re-encoded on every call. Equivalence to their evaluator is checked by
validate_eval.py on a small corpus.

Usage:
  eval_retrieval.py --variant content --eval-set heldout          # their seed-42 10% split
  eval_retrieval.py --variant title_content --eval-set heldout --out results/title_content.json
Writes a JSON with metrics and a per-question CSV (rank of first gold doc, top-10 titles).
"""
import argparse, csv, json, math, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # Webiks-Hebrew-RAGbot-Demo checkout
DATA = os.environ.get("RAG_DATA_DIR", os.path.join(REPO, "data"))
MODEL = os.environ.get("RAG_MODEL_DIR", os.path.join(REPO, "app", "artifacts", "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"))
RESULTS = os.environ.get("RAG_RESULTS_DIR", os.path.join(REPO, "results"))
TRAINER = os.environ.get("RAG_TRAINER_DIR", os.path.join(os.path.dirname(REPO), "Webiks-Hebrew-RAGbot-Trainer"))

QA = os.path.join(DATA, "Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv")

def load_eval_set(name, qa_path=None):
    qa_path = qa_path or QA
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(qa_path, encoding="utf-8")
    if name == "heldout":
        uq = df["question"].unique()
        _, val_q = train_test_split(uq, test_size=0.1, random_state=42)   # exactly Trainer/utils.split_train_eval
        df = df[df["question"].isin(val_q)]
    elif name == "all":
        pass
    elif os.path.exists(name):
        df = pd.read_csv(name, encoding="utf-8")
    else:
        raise SystemExit(f"unknown eval set {name}")
    gold = {}
    for q, g in df.groupby("question", sort=False):
        gold[q] = set(int(x) for x in g["doc_id"])
    return gold  # question -> set(doc_id)

def doc_rank_metrics(ranked_doc_ids, gold, ks=(1, 3, 5, 10), mrr_k=10, ndcg_k=10):
    """ranked_doc_ids: list of doc_ids best-first. Returns dict of per-query metrics."""
    out = {}
    for k in ks:
        out[f"hit@{k}"] = 1.0 if any(d in gold for d in ranked_doc_ids[:k]) else 0.0
    rr = 0.0
    for i, d in enumerate(ranked_doc_ids[:mrr_k]):
        if d in gold:
            rr = 1.0 / (i + 1); break
    out[f"mrr@{mrr_k}"] = rr
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_doc_ids[:ndcg_k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), ndcg_k)))
    out[f"ndcg@{ndcg_k}"] = dcg / idcg if idcg else 0.0
    first = next((i + 1 for i, d in enumerate(ranked_doc_ids) if d in gold), None)
    out["first_gold_rank"] = first
    return out

def evaluate(corpus_emb, corpus_doc_ids, query_emb, questions, gold_map, top_paragraphs=200):
    """corpus_emb: N x D (not necessarily normalised); corpus_doc_ids: N ints; query_emb: Q x D."""
    cn = corpus_emb / np.linalg.norm(corpus_emb, axis=1, keepdims=True)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    per_q = []
    for qi, q in enumerate(questions):
        scores = cn @ qn[qi]
        top = np.argpartition(-scores, min(top_paragraphs, len(scores) - 1))[:top_paragraphs]
        top = top[np.argsort(-scores[top])]
        seen, ranked = set(), []
        for pi in top:                      # max paragraph score per doc == first appearance in sorted order
            d = int(corpus_doc_ids[pi])
            if d not in seen:
                seen.add(d); ranked.append(d)
        m = doc_rank_metrics(ranked, gold_map[q])
        m["question"] = q; m["top10_doc_ids"] = ranked[:10]
        per_q.append(m)
    agg = {}
    for key in per_q[0]:
        if key in ("question", "top10_doc_ids", "first_gold_rank"): continue
        agg[key] = float(np.mean([m[key] for m in per_q]))
    agg["n_queries"] = len(per_q)
    agg["not_found_in_top200_paragraphs"] = sum(1 for m in per_q if m["first_gold_rank"] is None)
    return agg, per_q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="content")
    ap.add_argument("--emb-dir", default=os.path.join(DATA, "emb"))
    ap.add_argument("--corpus", default=os.path.join(DATA, "paragraph_corpus.json"))
    ap.add_argument("--eval-set", default="heldout")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(RESULTS, f"{a.variant}__{os.path.basename(a.eval_set)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    emb = np.load(os.path.join(a.emb_dir, f"{a.variant}.npy"))
    keys = json.load(open(os.path.join(a.emb_dir, f"{a.variant}.keys.json")))
    d = json.load(open(a.corpus, encoding="utf-8"))
    doc_ids = np.array([int(d["doc_id"][k]) for k in keys])
    titles = {int(d["doc_id"][k]): d["title"][k] for k in keys}
    gold_map = load_eval_set(a.eval_set)
    questions = list(gold_map)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device); model.eval()
    t0 = time.time()
    q_emb = model.encode(questions, batch_size=32, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    agg, per_q = evaluate(emb, doc_ids, q_emb, questions, gold_map)
    agg.update({"variant": a.variant, "eval_set": a.eval_set, "corpus_paragraphs": int(len(keys)), "seconds": round(time.time() - t0, 1)})
    json.dump({"metrics": agg, "per_query": per_q}, open(out, "w"), ensure_ascii=False, indent=1)
    with open(out.replace(".json", ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["question", "first_gold_rank", "gold_titles", "top5_titles"])
        for m in per_q:
            w.writerow([m["question"], m["first_gold_rank"], " | ".join(titles.get(g, str(g)) for g in gold_map[m["question"]]),
                        " | ".join(titles.get(x, str(x)) for x in m["top10_doc_ids"][:5])])
    print(json.dumps(agg, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
