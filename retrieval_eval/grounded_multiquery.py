#!/usr/bin/env python3
"""Experiment 8 (Ariel's idea, 05.10): grounded multi-query.

For each question the base retriever (title_content) returns its top pages. gpt-4o-mini sees the
question and those pages and writes 3 rewrites of the question that use the pages' terms. Each
rewrite is embedded and searched with the same index, and the page rankings of the original
question and the 3 rewrites are fused by reciprocal rank (RRF).
  C30: grounded on the top 30 page titles.
  C20: grounded on the top 20 pages, title + the first 300 characters of the page's best paragraph.
Compared with the base retriever and with experiment 7's "LLM picks among 30 titles" (A).
Outputs cached; run from the Demo checkout: RAG_DATA_DIR=../data python retrieval_eval/grounded_multiquery.py
"""
import json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
load_dotenv(os.path.join(REPO, "app", ".env"))
sys.path.insert(0, HERE)
from eval_retrieval import load_eval_set, doc_rank_metrics, MODEL, DATA  # noqa: E402
from title_llm import page_ranking, rrf, summarise, SETS  # noqa: E402

BASE = "title_content"
CACHE = os.path.join(DATA, "llm_grounded_rewrites.jsonl")
OUT = os.path.join(REPO, "results", "grounded_multiquery")
LLM = "gpt-4o-mini"
PROMPT = ("לפניך שאלה של אזרח ורשימת עמודים מאתר כל-זכות שעשויים להיות רלוונטיים. "
          "נסח את השאלה מחדש ב-3 דרכים שונות בעברית. כל ניסוח ישתמש במונחים מהעמודים שנראים מתאימים לשאלה, "
          "ישמור על הכוונה המקורית, ולא יוסיף עובדות שאינן בשאלה. "
          "החזר JSON בלבד: {\"rewrites\": [\"...\", \"...\", \"...\"]}")


def main():
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    client = OpenAI(api_key=os.environ["OAI_API_KEY"]); os.makedirs(OUT, exist_ok=True)
    d = json.load(open(os.path.join(DATA, "paragraph_corpus.json"), encoding="utf-8"))
    title_of = {}
    for k in d["doc_id"]:
        title_of.setdefault(int(d["doc_id"][k]), d["title"][k])
    e = np.load(os.path.join(DATA, "emb", f"{BASE}.npy")); keys = json.load(open(os.path.join(DATA, "emb", f"{BASE}.keys.json")))
    emb = e / np.linalg.norm(e, axis=1, keepdims=True); doc = np.array([int(d["doc_id"][k]) for k in keys])
    model = SentenceTransformer(MODEL, device="mps"); model.eval()

    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            r = json.loads(line); cache[(r["kind"], r["q"])] = r
    lock = threading.Lock(); fh = open(CACHE, "a", encoding="utf-8"); usage = [0, 0]; lat = []

    def call(kind, q, listing):
        if (kind, q) in cache:
            return
        for attempt in range(5):
            try:
                t0 = time.time()
                r = client.chat.completions.create(model=LLM, temperature=0, max_tokens=300, response_format={"type": "json_object"},
                                                   messages=[{"role": "system", "content": PROMPT},
                                                             {"role": "user", "content": f"שאלה: {q}\n\nעמודים:\n{listing}"}])
                txt = (r.choices[0].message.content or "").strip()
                try:
                    rw = [str(x).strip() for x in json.loads(txt).get("rewrites", []) if str(x).strip()][:3]
                except Exception:
                    rw = []
                rec = {"kind": kind, "q": q, "rewrites": rw, "sec": round(time.time() - t0, 3)}
                with lock:
                    cache[(kind, q)] = rec; usage[0] += r.usage.prompt_tokens; usage[1] += r.usage.completion_tokens; lat.append(rec["sec"])
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                return
            except Exception:
                time.sleep(2 ** attempt)

    # best paragraph text per page under the base index, for C20 snippets
    report = {}
    for set_name, spec in SETS.items():
        if spec != "heldout" and not os.path.exists(spec):
            print(f"[{set_name}] skipped (missing {spec})"); continue
        gold = load_eval_set(spec); qs = list(gold)
        Q = model.encode(qs, batch_size=32, convert_to_numpy=True, show_progress_bar=False); Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        base_rank, snippets = [], []
        for i in range(len(qs)):
            s = emb @ Q[i]; idx = np.argpartition(-s, 400)[:400]; idx = idx[np.argsort(-s[idx])]
            seen, order, snip = set(), [], {}
            for j in idx:
                dd = int(doc[j])
                if dd not in seen:
                    seen.add(dd); order.append(dd); snip[dd] = re.sub(r"\s+", " ", d["content"][keys[j]] or "").strip()[:300]
            base_rank.append(order); snippets.append(snip)
        lists = {}
        for i, q in enumerate(qs):
            lists[("C30", q)] = "\n".join(f"- {title_of[dd]}" for dd in base_rank[i][:30])
            lists[("C20", q)] = "\n".join(f"- {title_of[dd]}: {snippets[i][dd]}" for dd in base_rank[i][:20])
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=12) as ex:
            for (kind, q), listing in lists.items():
                ex.submit(call, kind, q, listing)
        print(f"[{set_name}] LLM calls done in {time.time()-t0:.0f}s", flush=True)

        def rewrites_rank(kind):
            texts, spans = [], []
            for q in qs:
                rw = cache.get((kind, q), {}).get("rewrites", [])
                spans.append((len(texts), len(rw))); texts += rw
            V = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False) if texts else np.zeros((0, emb.shape[1]))
            if len(V): V = V / np.linalg.norm(V, axis=1, keepdims=True)
            return [[page_ranking(emb @ V[s0 + j], doc) for j in range(n)] for s0, n in spans]

        rr = {k: rewrites_rank(k) for k in ("C30", "C20")}
        pick = json.load(open(os.path.join(REPO, "results", "title_llm", "title_content", f"{set_name}__A_LLM_picks_from_30_titles.json")))["per_query"] \
            if os.path.exists(os.path.join(REPO, "results", "title_llm", "title_content", f"{set_name}__A_LLM_picks_from_30_titles.json")) else None
        variants = {"title in text (base)": [lambda i: base_rank[i]],
                    "C30: original + 3 rewrites from 30 titles (RRF)": [lambda i: rrf([base_rank[i]] + rr["C30"][i])],
                    "C30: 3 rewrites only (RRF)": [lambda i: rrf(rr["C30"][i]) if rr["C30"][i] else base_rank[i]],
                    "C20: original + 3 rewrites from 20 titles+snippets (RRF)": [lambda i: rrf([base_rank[i]] + rr["C20"][i])],
                    "C20: 3 rewrites only (RRF)": [lambda i: rrf(rr["C20"][i]) if rr["C20"][i] else base_rank[i]]}
        res = {}; ref = None; rng = np.random.default_rng(0)
        rank = lambda m: m["first_gold_rank"] or 10 ** 6
        for name, (fn,) in variants.items():
            per = []
            for i, q in enumerate(qs):
                rk = fn(i); m = doc_rank_metrics(rk, gold[q]); m["question"] = q; m["top10_doc_ids"] = rk[:10]; per.append(m)
            agg = summarise(per)
            if ref is None: ref = {m["question"]: m for m in per}
            agg["vs_base_better"] = sum(1 for m in per if rank(m) < rank(ref[m["question"]]))
            agg["vs_base_worse"] = sum(1 for m in per if rank(m) > rank(ref[m["question"]]))
            dlt = np.array([m["hit@1"] - ref[m["question"]]["hit@1"] for m in per]); bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
            agg["hit@1_delta_vs_base_ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
            res[name] = agg
            if not set_name.startswith("ido"):
                json.dump({"metrics": agg, "per_query": per}, open(os.path.join(OUT, f"{set_name}__{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}.json"), "w"), ensure_ascii=False)
        if pick:
            pm = {k: float(np.mean([m[k] for m in pick])) for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"]}
            res["A: LLM picks from 30 titles (experiment 7)"] = pm
        report[set_name] = res
    fh.close()
    report["_llm"] = {"model": LLM, "new_prompt_tokens": usage[0], "new_completion_tokens": usage[1],
                      "new_cost_usd": round(usage[0] / 1e6 * 0.15 + usage[1] / 1e6 * 0.60, 4), "median_call_seconds": float(np.median(lat)) if lat else None}
    json.dump(report, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)
    for s, res in report.items():
        if s.startswith("_"): continue
        print(f"\n=== {s}")
        print(f"{'variant':<58}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}{'MRR':>7}  b/w vs base   hit@1 Δ CI")
        for name, a in res.items():
            print(f"{name:<58}{a['hit@1']:>7.3f}{a['hit@3']:>7.3f}{a['hit@5']:>7.3f}{a['hit@10']:>8.3f}{a['mrr@10']:>7.3f}  "
                  f"{str(a.get('vs_base_better',''))+'/'+str(a.get('vs_base_worse','')):>9}   {a.get('hit@1_delta_vs_base_ci95','')}")
    print("\nLLM:", report["_llm"])


if __name__ == "__main__":
    main()
