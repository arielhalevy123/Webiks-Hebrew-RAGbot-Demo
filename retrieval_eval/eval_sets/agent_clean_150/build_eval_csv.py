#!/usr/bin/env python3
"""Join the agent-written questions (agent_questions_raw.tsv: index<TAB>question, written from page
content only) with the page key (agent_sample_key.json) AFTER writing, and build:
  agent_questions.csv          question, doc_id, title   (all 150)
  agent_questions_strict.csv   the subset whose question shares no content word (3+ Hebrew
                               letters) with the gold page title
Overlap test (deliberately loose, so the strict subset is conservative): tokens are reduced to
Hebrew letters only (so צה"ל -> צהל), final letters normalised, length >= 3; a question token and a
title token overlap if equal, or equal after stripping up to two leading prefix letters (ו ה ב ל מ ש כ)
from either one. No stopword list (a shared function word also counts as overlap).
"""
import csv, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = str.maketrans("ךםןףץ", "כמנפצ")
PREF = "והבלמשכ"

def toks(s):
    out = set()
    for w in s.split():
        w = re.sub(r"[^א-ת]", "", w).translate(FINAL)
        if len(w) >= 3: out.add(w)
    return out

def variants(w):
    v = {w}
    if len(w) > 3 and w[0] in PREF:
        v.add(w[1:])
        if len(w) > 4 and w[1] in PREF: v.add(w[2:])
    return {x for x in v if len(x) >= 3}

def shared(q, t):
    tv = set().union(*[variants(w) for w in toks(t)]) if toks(t) else set()
    return sorted(w for w in toks(q) if variants(w) & tv)

def main():
    key = {p["i"]: p for p in json.load(open(os.path.join(HERE, "agent_sample_key.json"), encoding="utf-8"))["pages"]}
    rows = []
    for line in open(os.path.join(HERE, "agent_questions_raw.tsv"), encoding="utf-8"):
        i, q = line.rstrip("\n").split("\t")
        p = key[int(i)]
        rows.append({"question": q.strip(), "doc_id": p["doc_id"], "title": p["title"], "shared": " ".join(shared(q, p["title"]))})
    assert len({r["question"] for r in rows}) == len(rows), "duplicate questions"
    with open(os.path.join(HERE, "agent_questions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "doc_id", "title", "shared"]); w.writeheader(); w.writerows(rows)
    strict = [r for r in rows if not r["shared"]]
    with open(os.path.join(HERE, "agent_questions_strict.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "doc_id", "title", "shared"]); w.writeheader(); w.writerows(strict)
    print(f"all: {len(rows)}  strict (no shared title word): {len(strict)}")

if __name__ == "__main__":
    main()
