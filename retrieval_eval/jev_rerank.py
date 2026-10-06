#!/usr/bin/env python3
"""Experiment 8 (07.10): TypeSafe Jev as the optional second-stage rerank, vs the gpt-4o-mini title pick.

Candidates, base ranking and evaluator are exactly those of title_llm.py (base retriever title_content,
top-30 pages per question, ranked by best paragraph), so the baselines match that script to the digit.

Pattern (TypeSafe re-ranking cookbook, docs.typesafe.ai/cookbooks/rerank_typesafe): one Noul question per
(question, candidate) pair, state = {question, candidate}; the shortlist is sorted by the returned noul
(probability of "yes"), highest first. Ties and unscored questions keep retrieval order (stable sort).
Every pair is scored independently, so there is no position bias and no shared context limit; the only
limit is 32k tokens of state per call, which every paragraph in the corpus fits (longest ~23k tokens).

Variants (all on the same 30 candidates):
  J1  title only (direct analogue of the gpt-4o-mini pick)
  J2  title + the page's best-matching paragraph, truncated to ~300 Jev tokens (~365 Hebrew characters)
  J3  title + the page's best-matching paragraph, full text (all 30 candidates)
"Best-matching paragraph" = the paragraph that gave the page its rank in the base retriever.

Sets: vendor held-out 296 and agent_clean_150 only (IdoAgai's public set is deliberately not sent).
Every Jev response is cached in $RAG_DATA_DIR/jev_rerank_calls.jsonl keyed by a hash of the exact request
(no corpus text is stored), so re-runs are free and deterministic. gpt-4o-mini picks are read from
title_llm.py's cache; no OpenAI call is made.

Run from the Demo checkout:  RAG_DATA_DIR=../data python retrieval_eval/jev_rerank.py [--limit N] [--variants J1,J2,J3]
"""
import argparse, hashlib, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from title_llm import SETS, BASE, N_CAND, CACHE as GPT_CACHE, page_ranking, summarise, REPO  # noqa: E402  (also loads app/.env)
from eval_retrieval import load_eval_set, doc_rank_metrics, MODEL, DATA  # noqa: E402

USE_SETS = ["vendor_heldout_296", "agent_clean_150"]
URL = "https://api.typesafe.ai/v1/systemone"
JEV = "jev-latest"
PRICE_PER_MTOK = 0.042          # input only; output tokens are free (docs.typesafe.ai/models)
BUDGET_USD = 5.0
J2_CHARS = 365                  # ~300 Jev tokens: Hebrew measured at ~0.82 tokens per character
CALLS = os.path.join(DATA, "jev_rerank_calls.jsonl")
QLAT = os.path.join(DATA, "jev_rerank_qlat.jsonl")
OUT = os.path.join(REPO, "results", "jev_rerank")
MAX_RPS, MAX_TPS = 60, 70_000   # below the documented 80 req/s and 100k tok/s

INSTR = {
    "J1": ("The state holds a question a member of the public asked (in Hebrew) and one candidate page from "
           "Kol-Zchut, the Israeli rights-information website, given by its title. "
           "Is this the page that answers the question?"),
    "J2": ("The state holds a question a member of the public asked (in Hebrew) and one candidate page from "
           "Kol-Zchut, the Israeli rights-information website: its title and the start of the page paragraph "
           "that best matches the question. Is this the page that answers the question?"),
}
INSTR["J3"] = INSTR["J2"].replace("the start of the page paragraph", "the page paragraph")
CRITERIA = {
    "true": ("The page is about the specific right, benefit, procedure or situation the question asks about, "
             "and fits the population and sub-case the question describes (who the person is, their status, "
             "the specific circumstance)."),
    "false": ("The page is only on a related or broader topic, covers a different population or sub-case, "
              "or would not answer the question."),
}


def clean(p):
    return re.sub(r"\s*\n\s*", "\n", p).strip()


def truncate(p, n):
    if len(p) <= n:
        return p
    cut = p[:n]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n * 0.8 else cut) + " ..."


def request_body(variant, q, title, para):
    state = {"question": q, "page_title": title}
    if variant == "J2":
        state["page_paragraph_start"] = truncate(para, J2_CHARS)
    elif variant == "J3":
        state["page_paragraph"] = para
    return {"model": JEV, "state": state,
            "questions": {"answers_question": {"type": "noul", "instructions": INSTR[variant], "criteria": CRITERIA}}}


def body_key(variant, body):
    return variant + ":" + hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def est_tokens(body):
    return int(0.85 * len(json.dumps(body["state"], ensure_ascii=False))) + 330


class Limiter:
    """Sliding one-second window over requests and estimated tokens."""
    def __init__(self, rps, tps):
        self.rps, self.tps, self.lock, self.win = rps, tps, threading.Lock(), []

    def wait(self, tokens):
        while True:
            with self.lock:
                now = time.time()
                self.win = [(t, k) for t, k in self.win if now - t < 1.0]
                if len(self.win) < self.rps and sum(k for _, k in self.win) + tokens <= self.tps:
                    self.win.append((now, tokens)); return
            time.sleep(0.02)


def p90(x):
    return float(np.percentile(x, 90)) if len(x) else None


def boot_ci(a, b, n=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N questions per set (smoke test)")
    ap.add_argument("--variants", default="J1,J2,J3")
    ap.add_argument("--dry-run", action="store_true", help="estimate tokens and cost, call nothing")
    args = ap.parse_args()
    variants = args.variants.split(",")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and not args.dry_run:
        raise SystemExit("TYPESAFE_API_KEY not set (expected in app/.env)")
    from sentence_transformers import SentenceTransformer
    os.makedirs(OUT, exist_ok=True)

    d = json.load(open(os.path.join(DATA, "paragraph_corpus.json"), encoding="utf-8"))
    title_of = {}
    for k in d["doc_id"]:
        title_of.setdefault(int(d["doc_id"][k]), d["title"][k])
    e = np.load(os.path.join(DATA, "emb", f"{BASE}.npy")); keys = json.load(open(os.path.join(DATA, "emb", f"{BASE}.keys.json")))
    base_emb = e / np.linalg.norm(e, axis=1, keepdims=True)
    base_doc = np.array([int(d["doc_id"][k]) for k in keys])
    paras_of = {}
    for i, dd in enumerate(base_doc):
        paras_of.setdefault(int(dd), []).append(i)
    model = SentenceTransformer(MODEL, device="mps"); model.eval()

    gpt = {}
    for line in open(GPT_CACHE, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "pick" and r["base"] == BASE:
            gpt[r["q"]] = r

    cache, qlat = {}, {}
    if os.path.exists(CALLS):
        for line in open(CALLS, encoding="utf-8"):
            r = json.loads(line); cache[r["key"]] = r
    if os.path.exists(QLAT):
        for line in open(QLAT, encoding="utf-8"):
            r = json.loads(line); qlat[r["qkey"]] = r
    lock = threading.Lock()
    fh, fq = open(CALLS, "a", encoding="utf-8"), open(QLAT, "a", encoding="utf-8")
    limiter = Limiter(MAX_RPS, MAX_TPS)
    spent = {"in": 0, "out": 0, "calls": 0, "failed_calls": 0}

    def call(variant, q, dd, body):
        k = body_key(variant, body)
        if k in cache:
            return cache[k]
        for attempt in range(5):
            limiter.wait(est_tokens(body))
            try:
                t0 = time.time()
                r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=30)
                sec = round(time.time() - t0, 3)
                if r.status_code in (429, 529) or r.status_code >= 500:
                    time.sleep(min(2 ** attempt, 16)); continue
                if r.status_code != 200:
                    break                            # 4xx other than 429: do not retry
                js = r.json()
                rec = {"key": k, "variant": variant, "q": q, "doc_id": dd, "noul": js["answers"]["answers_question"]["noul"],
                       "in": js["usage"]["input_tokens"], "out": js["usage"]["output_tokens"], "sec": sec, "model": js["model"]}
                with lock:
                    cache[k] = rec; spent["in"] += rec["in"]; spent["out"] += rec["out"]; spent["calls"] += 1
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                return rec
            except (requests.RequestException, ValueError, KeyError):
                time.sleep(min(2 ** attempt, 16))
        with lock:
            spent["failed_calls"] += 1
        return None

    pool = ThreadPoolExecutor(max_workers=N_CAND)
    report = {"_config": {"jev_model_alias": JEV, "pattern": "one Noul per (question, candidate) pair, sort by noul desc, stable",
                          "instructions": INSTR, "criteria": CRITERIA, "j2_chars": J2_CHARS, "n_candidates": N_CAND,
                          "base": BASE, "limit": args.limit}}
    projected_total = 0.0
    for set_name in USE_SETS:
        gold = load_eval_set(SETS[set_name]); qs = list(gold)
        if args.limit:
            qs = qs[:args.limit]
        Q = model.encode(qs, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        base_rank, cands, best_para = [], {}, {}
        for i, q in enumerate(qs):
            s = base_emb @ Q[i]
            br = page_ranking(s, base_doc); base_rank.append(br); cands[q] = br[:N_CAND]
            for dd in cands[q]:
                pi = paras_of[dd]
                best_para[(q, dd)] = clean(d["content"][keys[pi[int(np.argmax(s[pi]))]]])
        bodies = {v: {q: [(dd, request_body(v, q, title_of[dd], best_para[(q, dd)])) for dd in cands[q]] for q in qs} for v in variants}

        # cost guard: estimate the uncached part of this set before calling anything
        est = sum(est_tokens(b) for v in variants for q in qs for _, b in bodies[v][q] if body_key(v, b) not in cache)
        est_usd = est / 1e6 * PRICE_PER_MTOK
        already = (spent["in"]) / 1e6 * PRICE_PER_MTOK
        print(f"[{set_name}] n={len(qs)} uncached est. {est:,} tokens = ${est_usd:.3f} (spent so far this run ${already:.3f})", flush=True)
        projected_total += est_usd
        if already + est_usd > BUDGET_USD:
            raise SystemExit(f"projected spend ${already + est_usd:.2f} exceeds the ${BUDGET_USD} cap; stopping")
        if args.dry_run:
            continue

        jev_rank = {v: [] for v in variants}
        fallbacks = {v: 0 for v in variants}; ties_top = {v: 0 for v in variants}; lat = {v: [] for v in variants}
        toks = {v: 0 for v in variants}
        for v in variants:
            t_set = time.time()
            for i, q in enumerate(qs):
                items = bodies[v][q]
                qkey = v + ":" + hashlib.sha256("|".join(body_key(v, b) for _, b in items).encode()).hexdigest()[:24]
                need = any(body_key(v, b) not in cache for _, b in items)
                t0 = time.time()
                recs = list(pool.map(lambda it: call(v, q, it[0], it[1]), items))
                wall = round(time.time() - t0, 3)
                if need and all(recs):
                    with lock:
                        qlat[qkey] = {"qkey": qkey, "wall_sec": wall}; fq.write(json.dumps(qlat[qkey]) + "\n"); fq.flush()
                if qkey in qlat:
                    lat[v].append(qlat[qkey]["wall_sec"])
                if not all(recs):
                    fallbacks[v] += 1; jev_rank[v].append(base_rank[i]); continue
                toks[v] += sum(r["in"] for r in recs)
                order = sorted(range(len(items)), key=lambda j: -recs[j]["noul"])     # stable: ties keep retrieval order
                top = recs[order[0]]["noul"]
                ties_top[v] += sum(1 for r in recs if r["noul"] == top) > 1
                rk = [items[j][0] for j in order]
                jev_rank[v].append(rk + base_rank[i][N_CAND:])
                if (i + 1) % 50 == 0:
                    print(f"  {set_name} {v}: {i+1}/{len(qs)} ({time.time()-t_set:.0f}s)", flush=True)
            print(f"[{set_name}] {v} done in {time.time()-t_set:.0f}s, fallbacks {fallbacks[v]}", flush=True)

        # rankings and metrics
        gpt_rank, gpt_lat, gpt_missing, gpt_cand_mismatch = [], [], 0, 0
        for i, q in enumerate(qs):
            rec = gpt.get(q); picked = []
            if rec is None:
                gpt_missing += 1
            else:
                gpt_cand_mismatch += rec["cands"] != cands[q]
                gpt_lat.append(rec.get("sec"))
                try:
                    for n in json.loads(rec["out"]).get("ranking", []):
                        n = int(n)
                        if 1 <= n <= len(cands[q]) and cands[q][n - 1] not in picked:
                            picked.append(cands[q][n - 1])
                except Exception:
                    pass
            gpt_rank.append(picked + [dd for dd in base_rank[i] if dd not in picked])
        runs = {"base (title_content, step 1)": base_rank, "gpt-4o-mini title pick": gpt_rank}
        names = {"J1": "J1 Jev: title only", "J2": "J2 Jev: title + paragraph ~300 tok", "J3": "J3 Jev: title + full paragraph"}
        for v in variants:
            runs[names[v]] = jev_rank[v]
        per = {name: [] for name in runs}
        for name, rks in runs.items():
            for i, q in enumerate(qs):
                m = doc_rank_metrics(rks[i], gold[q]); m["question"] = q; m["top10_doc_ids"] = rks[i][:10]
                per[name].append(m)
        rank = lambda m: m["first_gold_rank"] or 10 ** 6
        res = {"n": len(qs), "gold_in_30_candidates": float(np.mean([bool(set(cands[q]) & gold[q]) for q in qs])),
               "gpt_cache_missing": gpt_missing, "gpt_cache_candidate_mismatch": int(gpt_cand_mismatch),
               "gpt_latency_sec": {"median": float(np.median(gpt_lat)) if gpt_lat else None, "p90": p90(gpt_lat),
                                   "note": "single call, measured 05.10 by title_llm.py"},
               "variants": {}}
        b, g = per["base (title_content, step 1)"], per["gpt-4o-mini title pick"]
        for name, pq in per.items():
            agg = summarise(pq)
            agg["vs_step1_better"] = sum(rank(m) < rank(x) for m, x in zip(pq, b))
            agg["vs_step1_worse"] = sum(rank(m) > rank(x) for m, x in zip(pq, b))
            agg["vs_gpt_better"] = sum(rank(m) < rank(x) for m, x in zip(pq, g))
            agg["vs_gpt_worse"] = sum(rank(m) > rank(x) for m, x in zip(pq, g))
            for met in ("hit@3", "mrr@10"):
                agg[f"{met}_delta_vs_gpt_mean_ci95"] = boot_ci([m[met] for m in pq], [x[met] for x in g])
                agg[f"{met}_delta_vs_step1_mean_ci95"] = boot_ci([m[met] for m in pq], [x[met] for x in b])
            v = next((k for k, nm in names.items() if nm == name), None)
            if v:
                agg.update({"fallback_questions": fallbacks[v], "top1_tied_questions": ties_top[v],
                            "latency_sec_per_question": {"median": float(np.median(lat[v])) if lat[v] else None, "p90": p90(lat[v]),
                                                          "n_measured": len(lat[v])},
                            "input_tokens": toks[v], "usd": round(toks[v] / 1e6 * PRICE_PER_MTOK, 4)})
            res["variants"][name] = agg
            if not args.limit:
                json.dump({"metrics": agg, "per_query": pq}, open(os.path.join(OUT, f"{set_name}__{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}.json"), "w"), ensure_ascii=False)
        report[set_name] = res

    fh.close(); fq.close(); pool.shutdown()
    jev_models = sorted({r["model"] for r in cache.values()})
    report["_jev"] = {"models_seen": jev_models, "this_run_new_calls": spent["calls"], "this_run_failed_calls": spent["failed_calls"],
                      "this_run_input_tokens": spent["in"], "this_run_usd": round(spent["in"] / 1e6 * PRICE_PER_MTOK, 4),
                      "projected_usd_before_run": round(projected_total, 4)}
    if args.dry_run:
        print(json.dumps(report["_jev"], indent=1)); return
    json.dump(report, open(os.path.join(OUT, "summary.json" if not args.limit else "smoke_summary.json"), "w"), ensure_ascii=False, indent=1)

    for set_name in USE_SETS:
        res = report[set_name]
        print(f"\n=== {set_name} (n={res['n']}, gold in 30 candidates {res['gold_in_30_candidates']:.1%}; gpt cache missing {res['gpt_cache_missing']}, cand mismatch {res['gpt_cache_candidate_mismatch']})")
        print(f"{'variant':<36}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>7}  {'vs base b/w':>11}  {'vs gpt b/w':>10}  hit@3 d vs gpt [CI]   MRR d vs gpt [CI]")
        for name, a in res["variants"].items():
            print(f"{name:<36}{a['hit@1']:>7.3f}{a['hit@3']:>7.3f}{a['hit@5']:>7.3f}{a['mrr@10']:>7.3f}  {a['vs_step1_better']:>4}/{a['vs_step1_worse']:<6}  "
                  f"{a['vs_gpt_better']:>4}/{a['vs_gpt_worse']:<5}  {a['hit@3_delta_vs_gpt_mean_ci95']}  {a['mrr@10_delta_vs_gpt_mean_ci95']}")
            if "latency_sec_per_question" in a:
                print(f"{'':<36}latency med {a['latency_sec_per_question']['median']} p90 {a['latency_sec_per_question']['p90']}, "
                      f"tokens {a['input_tokens']:,} (${a['usd']}), fallbacks {a['fallback_questions']}, top-1 ties {a['top1_tied_questions']}")
    print("\nJev:", report["_jev"])


if __name__ == "__main__":
    main()
