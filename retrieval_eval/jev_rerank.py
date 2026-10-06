#!/usr/bin/env python3
"""Experiment 9 (07.10): TypeSafe Jev as the optional second-stage rerank, vs the gpt-4o-mini title pick.

Candidates, base ranking and evaluator are exactly those of title_llm.py (base retriever title_content,
top-30 pages per question, ranked by best paragraph), so the baselines match that script to the digit.

Pattern: one Jev Choice question per user question. State = {question}; the options are the 30 candidate
page titles in retrieval order (a duplicate title gets its doc id appended). Jev returns a probability per
option and the shortlist is sorted by it, highest first; ties and zero-probability pages keep retrieval
order (stable sort), and a failed call keeps the plain retrieval order.

Variants (both on the same 30 candidates, one call per question):
  C1  options are the titles only (direct analogue of the gpt-4o-mini pick; this is what app/src ships)
  C2  each option also carries the start of the page's best-matching paragraph, ~150 Jev tokens
"Best-matching paragraph" = the paragraph that gave the page its rank in the base retriever.

Sets: vendor held-out 296 and agent_clean_150 only (IdoAgai's public set is deliberately not sent).
Every Jev response is cached in $RAG_DATA_DIR/jev_rerank_calls.jsonl keyed by a hash of the exact request
(a record holds the question and one probability for each candidate title, no page text), so re-runs are free
and deterministic; a committed copy is in results/jev_rerank/*.jsonl.gz. gpt-4o-mini picks are read from
title_llm.py's cache; no OpenAI call is made.

Run from the Demo checkout:  RAG_DATA_DIR=../data python retrieval_eval/jev_rerank.py [--limit N] [--variants C1,C2]
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
C2_CHARS = 180                  # ~150 Jev tokens per option snippet
CALLS = os.path.join(DATA, "jev_rerank_calls.jsonl")
QLAT = os.path.join(DATA, "jev_rerank_qlat.jsonl")
OUT = os.path.join(REPO, "results", "jev_rerank")
MAX_RPS, MAX_TPS = 60, 70_000   # below the documented 80 req/s and 100k tok/s

INSTR = {}
INSTR["C1"] = ("The state holds a question a member of the public asked (in Hebrew). Each option is a candidate page "
               "from Kol-Zchut, the Israeli rights-information website, named by its page title. Which page answers "
               "the question? Choose the page about the specific right, benefit, procedure or situation the question "
               "asks about that fits the population and sub-case the question describes (who the person is, their "
               "status, the specific circumstance), rather than a page on a related or broader topic.")
INSTR["C2"] = INSTR["C1"].replace("named by its page title.", "named by its page title and described by the start "
                                  "of the page paragraph that best matches the question.")


def clean(p):
    return re.sub(r"\s*\n\s*", "\n", p).strip()


def truncate(p, n):
    if len(p) <= n:
        return p
    cut = p[:n]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n * 0.8 else cut) + " ..."


def choice_body(variant, q, cand_docs, title_of, para_of):
    """One Choice over all candidates; option keys are titles (doc id appended to duplicates), in retrieval order."""
    seen, opts, key2doc = {}, {}, {}
    for dd in cand_docs:
        t = title_of[dd].strip()
        k = t if t not in seen else f"{t} (doc {dd})"
        seen[t] = True; key2doc[k] = dd
        opts[k] = truncate(para_of(dd), C2_CHARS) if variant == "C2" else None
    body = {"model": JEV, "state": {"question": q},
            "questions": {"answering_page": {"type": "choice", "instructions": INSTR[variant], "criteria": opts}}}
    return body, key2doc


def body_key(variant, body):
    return variant + ":" + hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def est_tokens(body):
    return int(0.85 * len(json.dumps(body, ensure_ascii=False))) + 330


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
    ap.add_argument("--variants", default="C1,C2")
    ap.add_argument("--dry-run", action="store_true", help="estimate tokens and cost, call nothing")
    args = ap.parse_args()
    variants = args.variants.split(",")
    if not set(variants) <= set(INSTR):
        raise SystemExit(f"unknown variant(s); choose from {sorted(INSTR)}")
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
                ans = js["answers"]["answering_page"]
                rec = {"key": k, "variant": variant, "q": q, "doc_id": dd, "probs": ans.get("probabilities"),
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

    pool = ThreadPoolExecutor(max_workers=4)
    report = {"_config": {"jev_model_alias": JEV,
                          "pattern": "one Choice per question over the candidate titles, sort by probability desc, stable",
                          "instructions": {v: INSTR[v] for v in variants}, "c2_chars": C2_CHARS, "n_candidates": N_CAND,
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
        bodies, key2doc = {}, {}
        for v in variants:
            bodies[v] = {}
            for q in qs:
                b, k2d = choice_body(v, q, cands[q], title_of, lambda dd: best_para[(q, dd)])
                bodies[v][q] = [(None, b)]; key2doc[(v, q)] = k2d

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
        toks = {v: 0 for v in variants}; missing_opts = {v: 0 for v in variants}; nonzero = {v: [] for v in variants}
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
                probs, k2d = recs[0]["probs"] or {}, key2doc[(v, q)]
                sc = {k2d[k]: float(p) for k, p in probs.items() if k in k2d}
                if len(sc) < len(k2d):
                    missing_opts[v] += len(k2d) - len(sc)
                scores = [sc.get(dd, 0.0) for dd in cands[q]]; docs = cands[q]
                order = sorted(range(len(docs)), key=lambda j: -scores[j])     # stable: ties keep retrieval order
                top = scores[order[0]]
                ties_top[v] += sum(1 for x in scores if x == top) > 1
                nonzero[v].append(sum(1 for x in scores if x > 0))
                rk = [docs[j] for j in order]
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
        names = {"C1": "C1 Jev Choice: 30 titles", "C2": "C2 Jev Choice: 30 titles + ~150 tok snippet"}
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
                            "options_missing_in_reply": missing_opts[v], "median_candidates_with_nonzero_score": float(np.median(nonzero[v])) if nonzero[v] else None,
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
