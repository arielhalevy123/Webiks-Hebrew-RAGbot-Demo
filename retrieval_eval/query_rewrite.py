#!/usr/bin/env python3
"""Experiment 5: query-side LLM help, measured on the shipped index (fused_w0.3 vectors).

Three variants, each 296 calls to gpt-4o-mini, cached to data/query_rewrites.jsonl:
  rewrite  : the question restated as a clear, complete Hebrew search query naming the likely topic
  concat   : original question + rewrite, embedded together
  hyde     : a short Kol-Zchut-style paragraph that would answer the question (HyDE); embed that
Baseline for comparison: the original question (= results/fused_w0.3__heldout.json).
"""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(HERE), "app", ".env"))
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from eval_retrieval import load_eval_set, evaluate, MODEL

PROMPTS = {
    "rewrite": "נסח מחדש את שאלת האזרח הבאה כשאילתת חיפוש אחת, ברורה ומלאה, בעברית, שמזכירה את שם הזכות/הקצבה/הנושא שככל הנראה מדובר בו. שמור על הכוונה המקורית, אל תוסיף פרטים שלא נאמרו. החזר שורה אחת בלבד.",
    "hyde": "כתוב פסקה קצרה (2-3 משפטים) בסגנון אתר כל-זכות שהייתה עונה על שאלת האזרח הבאה: מה הזכות, מי זכאי, ואיך מממשים. אל תפנה לאזרח, כתוב כמו דף מידע. החזר את הפסקה בלבד.",
}
CACHE = os.path.join(os.path.dirname(HERE), "data/query_rewrites.jsonl")

def main():
    client = OpenAI(api_key=os.environ["OAI_API_KEY"])
    gold = load_eval_set("heldout"); qs = list(gold)
    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            r = json.loads(l); cache[(r["kind"], r["q"])] = r["out"]
    lock = threading.Lock(); f = open(CACHE, "a", encoding="utf-8"); usage = [0, 0]
    def one(kind, q):
        if (kind, q) in cache: return
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.2, max_tokens=160,
                        messages=[{"role": "system", "content": PROMPTS[kind]}, {"role": "user", "content": q}])
                out = (r.choices[0].message.content or "").strip()
                with lock:
                    cache[(kind, q)] = out; usage[0] += r.usage.prompt_tokens; usage[1] += r.usage.completion_tokens
                    f.write(json.dumps({"kind": kind, "q": q, "out": out}, ensure_ascii=False) + "\n"); f.flush()
                return
            except Exception:
                time.sleep(2 ** attempt)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        for kind in PROMPTS:
            for q in qs: ex.submit(one, kind, q)
    f.close()
    print(f"LLM calls done in {time.time()-t0:.0f}s, spent ${usage[0]/1e6*0.15 + usage[1]/1e6*0.60:.3f}", flush=True)
    for q in qs[:2]:
        print("Q:", q[:80]); print("  rewrite:", cache[("rewrite", q)][:100]); print("  hyde:", cache[("hyde", q)][:100])

    d = json.load(open(os.path.join(os.path.dirname(HERE), "data/paragraph_corpus.json"), encoding="utf-8"))
    emb = np.load(os.path.join(os.path.dirname(HERE), "data/emb/fused_w0.3.npy")); keys = json.load(open(os.path.join(os.path.dirname(HERE), "data/emb/fused_w0.3.keys.json")))
    doc = np.array([int(d["doc_id"][k]) for k in keys])
    model = SentenceTransformer(MODEL, device="mps"); model.eval()
    variants = {
        "original": qs,
        "rewrite": [cache[("rewrite", q)] for q in qs],
        "concat": [f"{q}\n{cache[('rewrite', q)]}" for q in qs],
        "hyde": [cache[("hyde", q)] for q in qs],
        "orig+hyde": [f"{q}\n{cache[('hyde', q)]}" for q in qs],
    }
    print(f"\n{'query variant':<14}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@10':>8}{'mrr@10':>8}{'absent':>8}")
    for name, texts in variants.items():
        qe = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        r, per = evaluate(emb, doc, qe, qs, gold)
        json.dump({"metrics": {**r, "variant": f"fused_w0.3+query_{name}", "eval_set": "heldout"}, "per_query": per},
                  open(os.path.join(os.path.dirname(HERE), "results", f"fused_w0.3__query_{name}__heldout.json"), "w"), ensure_ascii=False)
        print(f"{name:<14}{r['hit@1']:>8.3f}{r['hit@3']:>8.3f}{r['hit@5']:>8.3f}{r['hit@10']:>8.3f}{r['mrr@10']:>8.3f}{r['not_found_in_top200_paragraphs']:>8}")

if __name__ == "__main__":
    main()
