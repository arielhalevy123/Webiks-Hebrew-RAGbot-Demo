#!/usr/bin/env python3
"""Download IdoAgai's public generated evaluation set (300 questions) and convert it to the
evaluator's CSV format (question, doc_id), plus the 'strict' subset of questions that share no
3+-letter word with their page title. The questions are his work and stay in his repository;
this script only fetches them for evaluation.

Source: https://github.com/IdoAgai/Webiks-Hebrew-RAGbot-Demo (branch cross-encoder-reranking),
evaluation/datasets/clean_eval_set.json
Output: <RAG_DATA_DIR>/eval_sets/idoagai_generated_300.csv and ..._strict.csv
"""
import csv, json, os, re, urllib.request

URL = ("https://raw.githubusercontent.com/IdoAgai/Webiks-Hebrew-RAGbot-Demo/"
       "cross-encoder-reranking/evaluation/datasets/clean_eval_set.json")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RAG_DATA_DIR", os.path.join(os.path.dirname(HERE), "data"))
FINALS = str.maketrans("ךםןףץ", "כמנפצ")


def words(text):
    out = set()
    for w in re.sub(r"[^א-ת ]", " ", text).split():
        w = w.translate(FINALS)
        if len(w) >= 3:
            out.add(w)
            for k in (1, 2):
                if len(w) - k >= 3 and all(c in "והבלמשכ" for c in w[:k]):
                    out.add(w[k:])
    return out


def main():
    rows = json.loads(urllib.request.urlopen(URL, timeout=60).read().decode("utf-8"))
    out_dir = os.path.join(DATA, "eval_sets"); os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, "idoagai_generated_300.csv"); strict = full.replace(".csv", "_strict.csv")
    with open(full, "w", newline="", encoding="utf-8") as f, open(strict, "w", newline="", encoding="utf-8") as g:
        wf, wg = csv.writer(f), csv.writer(g); wf.writerow(["question", "doc_id"]); wg.writerow(["question", "doc_id"])
        n_strict = 0
        for r in rows:
            wf.writerow([r["question"], int(r["doc_id"])])
            if not (words(r["question"]) & words(r["title"])):
                wg.writerow([r["question"], int(r["doc_id"])]); n_strict += 1
    print(f"{len(rows)} questions -> {full}; {n_strict} strict -> {strict}")


if __name__ == "__main__":
    main()
