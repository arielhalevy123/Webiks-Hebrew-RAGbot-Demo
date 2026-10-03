#!/usr/bin/env python3
"""Experiment 3: separate title vector fused with the paragraph vector.

score(q, paragraph) = w * cos(q, title_vec[doc]) + (1 - w) * cos(q, para_vec)
Weight chosen on even-indexed questions, reported on odd-indexed ones, and vice versa, so
the reported number is never tuned on the questions it is measured on.
"""
import json, numpy as np, time
from sentence_transformers import SentenceTransformer
from eval_retrieval import load_eval_set, evaluate, MODEL

d = json.load(open("data/paragraph_corpus.json", encoding="utf-8"))
gold = load_eval_set("heldout"); qs = list(gold)
model = SentenceTransformer(MODEL, device="mps"); model.eval()
q = model.encode(qs, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
titles = sorted({d["title"][k] for k in d["title"]})
t0 = time.time(); tv = model.encode(titles, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
print(f"encoded {len(titles)} unique titles in {time.time()-t0:.0f}s", flush=True)
tix = {t: i for i, t in enumerate(titles)}
tv = tv / np.linalg.norm(tv, axis=1, keepdims=True); qn = q / np.linalg.norm(q, axis=1, keepdims=True)

def run(variant, w, idx):
    emb = np.load(f"data/emb/{variant}.npy"); keys = json.load(open(f"data/emb/{variant}.keys.json"))
    doc = np.array([int(d["doc_id"][k]) for k in keys]); ti = np.array([tix[d["title"][k]] for k in keys])
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    # fused "embedding" trick: cos is linear in q, so fused score = q · (w*title + (1-w)*para)
    fused = w * tv[ti] + (1 - w) * en
    sub = [qs[i] for i in idx]
    r, _ = evaluate(fused, doc, qn[idx], sub, gold)
    return r

even = list(range(0, len(qs), 2)); odd = list(range(1, len(qs), 2))
for variant in ["content", "title_content"]:
    print(f"\n=== base vectors: {variant} ===")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        a = run(variant, w, even); b = run(variant, w, odd)
        print(f"w={w:.1f}  even: hit@1 {a['hit@1']:.3f} hit@3 {a['hit@3']:.3f} mrr {a['mrr@10']:.3f} | odd: hit@1 {b['hit@1']:.3f} hit@3 {b['hit@3']:.3f} mrr {b['mrr@10']:.3f}")
    # honest estimate: pick w on one half by mrr, report on the other half, both directions
    ws = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_on_even = max(ws, key=lambda w: run(variant, w, even)["mrr@10"]); rep_odd = run(variant, best_on_even, odd)
    best_on_odd = max(ws, key=lambda w: run(variant, w, odd)["mrr@10"]); rep_even = run(variant, best_on_odd, even)
    base_odd = run(variant, 0.0, odd); base_even = run(variant, 0.0, even)
    print(f"cross-fitted: w chosen on even = {best_on_even} -> odd hit@1 {rep_odd['hit@1']:.3f} (w=0: {base_odd['hit@1']:.3f}), hit@3 {rep_odd['hit@3']:.3f} ({base_odd['hit@3']:.3f})")
    print(f"              w chosen on odd  = {best_on_odd} -> even hit@1 {rep_even['hit@1']:.3f} (w=0: {base_even['hit@1']:.3f}), hit@3 {rep_even['hit@3']:.3f} ({base_even['hit@3']:.3f})")
