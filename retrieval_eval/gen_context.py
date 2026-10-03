#!/usr/bin/env python3
"""Experiment 4: LLM-written context per paragraph, generated once at index time.

For every paragraph, ask a small model for 1-2 Hebrew sentences saying what the paragraph
covers within its page and which questions it answers. The text is later prepended before
embedding (contextual retrieval). Output is appended to data/llm_context.jsonl as it goes
(resumable); token usage is accumulated so the cost is measured, not guessed.

  gen_context.py --limit 50          # dry run: measure tokens, print projected cost, stop
  gen_context.py                     # full corpus, stops if projected spend exceeds --max-usd
"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(HERE), "app", ".env"))
PRICE_IN, PRICE_OUT = 0.15, 0.60   # USD per 1M tokens, gpt-4o-mini (check against the current price list)

SYSTEM = ("אתה עוזר לבנות אינדקס חיפוש לאתר כל-זכות. לכל פסקה תכתוב הקשר קצר שיעזור למצוא אותה. "
          "כתוב בעברית, משפט אחד או שניים בלבד, בלי כותרות ובלי הקדמה: מה הפסקה הזו מתארת בתוך הדף, "
          "ועל איזה סוג שאלות של אזרח היא עונה. אל תחזור על הטקסט של הפסקה ואל תוסיף מידע שלא בה.")

def build_user(title, headings, para):
    h = " | ".join(headings[:12]) if headings else "-"
    return f"דף: {title}\nסעיפי הדף: {h}\n\nהפסקה:\n{para[:2500]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-usd", type=float, default=6.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data", "llm_context.jsonl"))
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OAI_API_KEY"])

    d = json.load(open(os.path.join(os.path.dirname(HERE), "data", "paragraph_corpus.json"), encoding="utf-8"))
    keys = list(d["content"])
    by_doc = {}
    for k in keys: by_doc.setdefault(d["doc_id"][k], []).append(k)
    def headings_of(doc_id):
        hs = []
        for k in by_doc[doc_id]:
            first = next((l.strip() for l in (d["content"][k] or "").split("\n") if l.strip()), "")
            if 0 < len(first) < 40: hs.append(first)
        return hs
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            try: done.add(json.loads(line)["key"])
            except Exception: pass
    todo = [k for k in keys if k not in done]
    if a.limit: todo = todo[:a.limit]
    print(f"paragraphs total {len(keys)}, already done {len(done)}, this run {len(todo)}, model {a.model}", flush=True)

    lock = threading.Lock(); usage = {"in": 0, "out": 0, "n": 0, "err": 0}; t0 = time.time()
    out_f = open(a.out, "a", encoding="utf-8")

    def one(k):
        title = d["title"][k] or ""; para = d["content"][k] or ""
        for attempt in range(5):
            try:
                r = client.chat.completions.create(model=a.model, temperature=0.2, max_tokens=120,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": build_user(title, headings_of(d["doc_id"][k]), para)}])
                txt = (r.choices[0].message.content or "").strip()
                with lock:
                    usage["in"] += r.usage.prompt_tokens; usage["out"] += r.usage.completion_tokens; usage["n"] += 1
                    out_f.write(json.dumps({"key": k, "doc_id": d["doc_id"][k], "context": txt,
                                            "in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}, ensure_ascii=False) + "\n"); out_f.flush()
                return True
            except Exception as e:
                time.sleep(2 ** attempt)
        with lock: usage["err"] += 1
        return False

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, k) for k in todo]
        for i, f in enumerate(as_completed(futs), 1):
            if i % 200 == 0 or i == len(todo):
                with lock:
                    spent = usage["in"] / 1e6 * PRICE_IN + usage["out"] / 1e6 * PRICE_OUT
                    per = spent / max(usage["n"], 1); proj = per * (len(keys) - len(done))
                    rate = usage["n"] / max(time.time() - t0, 1)
                print(f"{i}/{len(todo)} done | spent ${spent:.3f} | avg in {usage['in']/max(usage['n'],1):.0f} out {usage['out']/max(usage['n'],1):.0f} tok | "
                      f"projected full corpus ${proj:.2f} | {rate:.1f}/s eta {((len(keys)-len(done)-usage['n'])/max(rate,0.01))/60:.0f} min | errors {usage['err']}", flush=True)
                if proj > a.max_usd and not a.limit:
                    print(f"STOP: projected ${proj:.2f} exceeds cap ${a.max_usd}", flush=True); os._exit(2)
    out_f.close()
    spent = usage["in"] / 1e6 * PRICE_IN + usage["out"] / 1e6 * PRICE_OUT
    print(f"finished: {usage['n']} ok, {usage['err']} failed, spent ${spent:.3f}, projected full ${spent / max(usage['n'],1) * len(keys):.2f}", flush=True)

if __name__ == "__main__":
    main()
