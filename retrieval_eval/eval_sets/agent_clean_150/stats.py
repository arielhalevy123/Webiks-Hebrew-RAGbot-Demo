#!/usr/bin/env python3
"""Paired statistics for baseline (content) vs shipped (fused_w0.3) on the agent set:
bootstrap 95% CI of the hit@1 / hit@3 / MRR@10 deltas (10,000 resamples, seed 0), exact two-sided
sign test on per-question rank changes, and a sensitivity run excluding the corpus pages whose
stored content belongs to a different page (title/content mismatch, found after evaluation)."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
MISMATCH_DOCS = {10075, 11477}   # see agent_questions_method.md

def load(v, s):
    return {m["question"]: m for m in json.load(open(os.path.join(R, f"{v}__{s}.csv.json")))["per_query"]}

def signtest(b, w):
    n = b + w; k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

def report(s, exclude=frozenset()):
    import csv
    gold = {r["question"]: int(r["doc_id"]) for r in csv.DictReader(open(os.path.join(HERE, f"{s}.csv"), encoding="utf-8"))}
    A, B = load("content", s), load("fused_w0.3", s)
    qs = [q for q in A if gold[q] not in exclude]
    rng = np.random.default_rng(0); out = {"set": s, "n": len(qs), "excluded_docs": sorted(exclude)}
    for m in ["hit@1", "hit@3", "mrr@10"]:
        a = np.array([A[q][m] for q in qs]); b = np.array([B[q][m] for q in qs]); d = b - a
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        out[m] = {"content": round(a.mean(), 3), "fused_w0.3": round(b.mean(), 3), "delta": round(d.mean(), 3),
                  "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    rk = lambda m: m["first_gold_rank"] or 10 ** 6
    better = sum(rk(B[q]) < rk(A[q]) for q in qs); worse = sum(rk(B[q]) > rk(A[q]) for q in qs)
    out.update({"improved": better, "worse": worse, "unchanged": len(qs) - better - worse, "sign_test_p": round(signtest(better, worse), 3)})
    return out

if __name__ == "__main__":
    res = [report("agent_questions"), report("agent_questions_strict"),
           report("agent_questions", MISMATCH_DOCS), report("agent_questions_strict", MISMATCH_DOCS)]
    json.dump(res, open(os.path.join(R, "stats.json"), "w"), ensure_ascii=False, indent=1)
    for r in res: print(json.dumps(r, ensure_ascii=False))
