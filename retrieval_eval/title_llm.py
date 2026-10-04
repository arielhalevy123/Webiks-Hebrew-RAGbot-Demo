#!/usr/bin/env python3
"""Experiment 7 (Ariel's idea, 05.10): let an LLM use the candidate page titles at query time.

Base retriever: title_content (title prepended to the embedded text). For each question the base
retriever's top-30 pages are taken (ranked by best paragraph), and gpt-4o-mini sees the question and
those 30 titles. Two variants:
  A  "pick":    the LLM orders the titles it thinks answer the question (up to 10); final ranking =
                the LLM's order, then the remaining candidates in retrieval order.
  B  "rewrite": the LLM rewrites the question in one sentence using terms from the titles that fit it;
                the rewrite is searched with the same index; reported alone and fused with the original
                search by reciprocal rank (original + rewrite).
Evaluated on the vendor held-out split (296), the agent-written clean set (150, retrieval_eval/eval_sets/)
and IdoAgai's public generated set (300, fetched by retrieval_eval/fetch_idoagai_set.py). LLM outputs are cached, so re-runs are free and deterministic.

Run from the Demo checkout:  RAG_DATA_DIR=../data python retrieval_eval/title_llm.py
"""
import json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
load_dotenv(os.path.join(REPO, "app", ".env"))
sys.path.insert(0, HERE)
from eval_retrieval import load_eval_set, doc_rank_metrics, MODEL, DATA  # noqa: E402

SETS = {
    "vendor_heldout_296": "heldout",
    "agent_clean_150": os.path.join(HERE, "eval_sets", "agent_clean_150", "agent_questions.csv"),
    "ido_generated_300": os.path.join(DATA, "eval_sets", "idoagai_generated_300.csv"),  # fetch_idoagai_set.py
}
BASE = os.environ.get("TITLE_LLM_BASE", "title_content")
N_CAND = 30
CACHE = os.path.join(DATA, "llm_title_queries.jsonl")
OUT = os.path.join(REPO, "results", "title_llm", BASE)
LLM = "gpt-4o-mini"

sys.path.insert(0, os.path.join(REPO, "app", "src"))
from title_rerank import PICK_PROMPT as PICK  # noqa: E402  (the prompt the server uses; identical to the one measured)
REWRITE = ("לפניך שאלה של אזרח ורשימת כותרות של עמודים מאתר כל-זכות שעשויים להיות רלוונטיים. "
           "נסח מחדש את השאלה במשפט אחד בעברית, תוך שימוש במונחים מהכותרות שמתאימות לשאלה, "
           "בלי להוסיף עובדות שאינן בשאלה ובלי להעתיק כותרת שלמה. החזר רק את הניסוח החדש.")


def page_ranking(scores, doc, top_par=400):
    idx = np.argpartition(-scores, top_par)[:top_par]
    idx = idx[np.argsort(-scores[idx])]
    seen, out = set(), []
    for i in idx:
        dd = int(doc[i])
        if dd not in seen:
            seen.add(dd); out.append(dd)
    return out


def rrf(rankings, k=60):
    s = {}
    for r in rankings:
        for rank, dd in enumerate(r):
            s[dd] = s.get(dd, 0.0) + 1.0 / (k + rank + 1)
    return [dd for dd, _ in sorted(s.items(), key=lambda x: -x[1])]


def summarise(per):
    agg = {k: float(np.mean([m[k] for m in per])) for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10"]}
    agg["n_queries"] = len(per)
    agg["not_found"] = sum(1 for m in per if m["first_gold_rank"] is None)
    return agg


def main():
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    client = OpenAI(api_key=os.environ["OAI_API_KEY"])
    os.makedirs(OUT, exist_ok=True)

    d = json.load(open(os.path.join(DATA, "paragraph_corpus.json"), encoding="utf-8"))
    title_of = {}
    for k in d["doc_id"]:
        title_of.setdefault(int(d["doc_id"][k]), d["title"][k])

    def load(v):
        e = np.load(os.path.join(DATA, "emb", f"{v}.npy")); keys = json.load(open(os.path.join(DATA, "emb", f"{v}.keys.json")))
        return e / np.linalg.norm(e, axis=1, keepdims=True), np.array([int(d["doc_id"][k]) for k in keys])
    base_emb, base_doc = load(BASE)
    cont_emb, cont_doc = load("content")
    model = SentenceTransformer(MODEL, device="mps"); model.eval()

    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            r = json.loads(line); cache[(r["kind"], r["base"], r["q"])] = r
    lock = threading.Lock(); fh = open(CACHE, "a", encoding="utf-8"); usage = [0, 0]; lat = []

    def call(kind, q, cands):
        key = (kind, BASE, q)
        if key in cache:
            return
        listing = "\n".join(f"{i+1}. {title_of[dd]}" for i, dd in enumerate(cands))
        user = f"שאלה: {q}\n\nכותרות:\n{listing}"
        for attempt in range(5):
            try:
                t0 = time.time()
                kw = dict(model=LLM, temperature=0, max_tokens=120,
                          messages=[{"role": "system", "content": PICK if kind == "pick" else REWRITE},
                                    {"role": "user", "content": user}])
                if kind == "pick":
                    kw["response_format"] = {"type": "json_object"}
                r = client.chat.completions.create(**kw)
                txt = (r.choices[0].message.content or "").strip()
                rec = {"kind": kind, "base": BASE, "q": q, "cands": cands, "out": txt, "sec": round(time.time() - t0, 3)}
                with lock:
                    cache[key] = rec; usage[0] += r.usage.prompt_tokens; usage[1] += r.usage.completion_tokens; lat.append(rec["sec"])
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                return
            except Exception:
                time.sleep(2 ** attempt)

    report = {}
    for set_name, spec in SETS.items():
        if spec != "heldout" and not os.path.exists(spec):
            print(f"[{set_name}] skipped: {spec} not found (run retrieval_eval/fetch_idoagai_set.py for the IdoAgai set)")
            continue
        gold = load_eval_set(spec); qs = list(gold)
        Q = model.encode(qs, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        base_rank = [page_ranking(base_emb @ Q[i], base_doc) for i in range(len(qs))]
        cont_rank = [page_ranking(cont_emb @ Q[i], cont_doc) for i in range(len(qs))]
        cands = {q: base_rank[i][:N_CAND] for i, q in enumerate(qs)}

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=12) as ex:
            for q in qs:
                ex.submit(call, "pick", q, cands[q]); ex.submit(call, "rewrite", q, cands[q])
        print(f"[{set_name}] LLM calls done in {time.time()-t0:.0f}s", flush=True)

        rewrites = [cache[("rewrite", BASE, q)]["out"] if ("rewrite", BASE, q) in cache else q for q in qs]
        R = model.encode(rewrites, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        R = R / np.linalg.norm(R, axis=1, keepdims=True)

        variants = {"content (original system)": [], f"{BASE} (base retriever)": [], "A: LLM picks from 30 titles": [],
                    "B: rewrite with title words, alone": [], "B: original + rewrite (RRF)": []}
        in_pool = 0; bad_json = 0
        for i, q in enumerate(qs):
            g = gold[q]
            in_pool += bool(set(cands[q]) & g)
            rec = cache.get(("pick", BASE, q))
            picked = []
            if rec:
                try:
                    nums = json.loads(rec["out"]).get("ranking", [])
                    for n in nums:
                        n = int(n)
                        if 1 <= n <= len(cands[q]) and cands[q][n - 1] not in picked:
                            picked.append(cands[q][n - 1])
                except Exception:
                    bad_json += 1
            a_rank = picked + [dd for dd in base_rank[i] if dd not in picked]
            rw_rank = page_ranking(base_emb @ R[i], base_doc)
            rankings = {"content (original system)": cont_rank[i], f"{BASE} (base retriever)": base_rank[i],
                        "A: LLM picks from 30 titles": a_rank, "B: rewrite with title words, alone": rw_rank,
                        "B: original + rewrite (RRF)": rrf([base_rank[i], rw_rank])}
            for name, rk in rankings.items():
                m = doc_rank_metrics(rk, g); m["question"] = q; m["top10_doc_ids"] = rk[:10]
                if name.startswith("B"):
                    m["rewrite"] = rewrites[i]
                variants[name].append(m)

        rank = lambda m: m["first_gold_rank"] or 10 ** 6
        res = {"n": len(qs), "gold_in_30_candidates": in_pool / len(qs), "bad_json": bad_json, "variants": {}}
        ref_c = {m["question"]: m for m in variants["content (original system)"]}
        ref_b = {m["question"]: m for m in variants[f"{BASE} (base retriever)"]}
        rng = np.random.default_rng(0)
        for name, per in variants.items():
            agg = summarise(per)
            agg["vs_content_better"] = sum(1 for m in per if rank(m) < rank(ref_c[m["question"]]))
            agg["vs_content_worse"] = sum(1 for m in per if rank(m) > rank(ref_c[m["question"]]))
            agg["vs_step1_better"] = sum(1 for m in per if rank(m) < rank(ref_b[m["question"]]))
            agg["vs_step1_worse"] = sum(1 for m in per if rank(m) > rank(ref_b[m["question"]]))
            dlt = np.array([m["hit@1"] - ref_c[m["question"]]["hit@1"] for m in per])
            bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
            agg["hit@1_delta_vs_content_ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
            res["variants"][name] = agg
            json.dump({"metrics": agg, "per_query": per}, open(os.path.join(OUT, f"{set_name}__{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}.json"), "w"), ensure_ascii=False)
        report[set_name] = res

    fh.close()
    cost = usage[0] / 1e6 * 0.15 + usage[1] / 1e6 * 0.60
    report["_llm"] = {"model": LLM, "new_calls_prompt_tokens": usage[0], "new_calls_completion_tokens": usage[1],
                      "new_calls_cost_usd": round(cost, 4), "median_call_seconds": float(np.median(lat)) if lat else None}
    json.dump(report, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)

    for set_name, res in report.items():
        if set_name.startswith("_"):
            continue
        print(f"\n=== {set_name}  (n={res['n']}, gold page among the 30 candidates: {res['gold_in_30_candidates']:.1%}, unparsable picks: {res['bad_json']})")
        print(f"{'variant':<38}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}{'MRR':>7}  {'vs content b/w':>14}  {'vs base b/w':>12}  hit@1 Δ CI vs content")
        for name, a in res["variants"].items():
            print(f"{name:<38}{a['hit@1']:>7.3f}{a['hit@3']:>7.3f}{a['hit@5']:>7.3f}{a['hit@10']:>8.3f}{a['mrr@10']:>7.3f}  "
                  f"{a['vs_content_better']:>6}/{a['vs_content_worse']:<7}  {a['vs_step1_better']:>5}/{a['vs_step1_worse']:<6}  {a['hit@1_delta_vs_content_ci95']}")
    print("\nLLM:", report["_llm"])


if __name__ == "__main__":
    main()
