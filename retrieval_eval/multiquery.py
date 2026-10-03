#!/usr/bin/env python3
"""Experiment 6 (Ariel's idea): generate 4 paraphrases per question, search with each plus the
original, fuse the page rankings (reciprocal-rank voting), evaluate on the shipped index."""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(HERE), "app", ".env"))
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from eval_retrieval import load_eval_set, evaluate, doc_rank_metrics, MODEL

PROMPT = ("נסח מחדש את שאלת האזרח הבאה ב-4 דרכים שונות בעברית: פעם רשמית, פעם מדוברת, פעם קצרה עם מילות מפתח בלבד, "
          "ופעם שמזכירה את שם הזכות/הנושא המשוער. שמור על הכוונה, אל תוסיף עובדות. החזר 4 שורות בלבד, שורה לכל ניסוח, בלי מספור.")
CACHE = os.path.join(os.path.dirname(HERE), "data/query_paraphrases.jsonl")

def main():
    client = OpenAI(api_key=os.environ["OAI_API_KEY"])
    gold = load_eval_set("heldout"); qs = list(gold)
    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            r = json.loads(l); cache[r["q"]] = r["paras"]
    lock = threading.Lock(); f = open(CACHE, "a", encoding="utf-8"); usage = [0, 0]
    def one(q):
        if q in cache: return
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.4, max_tokens=240,
                        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": q}])
                paras = [l.strip("-•* \t") for l in (r.choices[0].message.content or "").split("\n") if l.strip()][:4]
                with lock:
                    cache[q] = paras; usage[0] += r.usage.prompt_tokens; usage[1] += r.usage.completion_tokens
                    f.write(json.dumps({"q": q, "paras": paras}, ensure_ascii=False) + "\n"); f.flush()
                return
            except Exception:
                time.sleep(2 ** attempt)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        for q in qs: ex.submit(one, q)
    f.close()
    print(f"LLM calls done in {time.time()-t0:.0f}s, spent ${usage[0]/1e6*0.15 + usage[1]/1e6*0.60:.3f}", flush=True)
    print("example:", qs[0][:60], "->", cache[qs[0]])

    d = json.load(open(os.path.join(os.path.dirname(HERE), "data/paragraph_corpus.json"), encoding="utf-8"))
    emb = np.load(os.path.join(os.path.dirname(HERE), "data/emb/fused_w0.3.npy")); keys = json.load(open(os.path.join(os.path.dirname(HERE), "data/emb/fused_w0.3.keys.json")))
    doc = np.array([int(d["doc_id"][k]) for k in keys])
    cn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    model = SentenceTransformer(MODEL, device="mps"); model.eval()

    def page_ranking(qvec, top=200):
        s = cn @ (qvec / np.linalg.norm(qvec)); idx = np.argpartition(-s, top)[:top]; idx = idx[np.argsort(-s[idx])]
        seen, out = set(), []
        for i in idx:
            dd = int(doc[i])
            if dd not in seen: seen.add(dd); out.append(dd)
        return out

    all_texts = []; spans = []
    for q in qs:
        texts = [q] + cache[q][:4]; spans.append((len(all_texts), len(texts))); all_texts += texts
    E = model.encode(all_texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

    def fuse(rankings, k=60, weights=None):
        score = {}
        for j, r in enumerate(rankings):
            w = 1.0 if weights is None else weights[j]
            for rank, dd in enumerate(r): score[dd] = score.get(dd, 0.0) + w / (k + rank + 1)
        return [dd for dd, _ in sorted(score.items(), key=lambda x: -x[1])]

    configs = {"original only": lambda rs: rs[0], "4 paraphrases, vote (RRF)": lambda rs: fuse(rs[1:]),
               "original + 4 paraphrases, vote (RRF)": lambda rs: fuse(rs), "original ×2 + 4 paraphrases": lambda rs: fuse(rs, weights=[2, 1, 1, 1, 1])}
    results = {}
    for name, fn in configs.items():
        per = []
        for qi, q in enumerate(qs):
            s0, n = spans[qi]; rs = [page_ranking(E[s0 + j]) for j in range(n)]
            ranked = fn(rs); m = doc_rank_metrics(ranked, gold[q]); m["question"] = q; m["top10_doc_ids"] = ranked[:10]; per.append(m)
        agg = {k: float(np.mean([m[k] for m in per])) for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10"]}
        agg["n_queries"] = len(per); agg["not_found_in_top200_paragraphs"] = sum(1 for m in per if m["first_gold_rank"] is None)
        agg["variant"] = f"fused_w0.3+multiquery_{name}"; agg["eval_set"] = "heldout"; results[name] = (agg, per)
        json.dump({"metrics": agg, "per_query": per}, open(os.path.join(os.path.dirname(HERE), "results", f"fused_w0.3__multiquery_{name.split(',')[0].replace(' ', '_')}__heldout.json"), "w"), ensure_ascii=False)
    base = {m["question"]: m for m in results["original only"][1]}
    rank = lambda m: m["first_gold_rank"] or 10**6
    print(f"\n{'config':<40}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}{'mrr':>7}{'better/worse':>14}")
    for name, (agg, per) in results.items():
        b = sum(1 for m in per if rank(m) < rank(base[m['question']])); w = sum(1 for m in per if rank(m) > rank(base[m['question']]))
        print(f"{name:<40}{agg['hit@1']:>7.3f}{agg['hit@3']:>7.3f}{agg['hit@5']:>7.3f}{agg['hit@10']:>8.3f}{agg['mrr@10']:>7.3f}{b:>7}/{w:<6}")

if __name__ == "__main__":
    main()
