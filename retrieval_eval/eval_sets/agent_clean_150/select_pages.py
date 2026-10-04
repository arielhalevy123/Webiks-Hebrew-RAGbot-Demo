#!/usr/bin/env python3
"""Select a clean page sample for the agent-written retrieval eval set.

Clean = the page's doc_id never appears in the vendor QA file (the embedder was trained on it).
Steps (counts are printed and recorded in agent_questions_method.md):
  1. all pages in paragraph_corpus.json
  2. drop pages with any row in the QA csv
  3. drop the 60 pages already given to Ariel for the human set (pages_key.json), so the
     agent set and the human set stay independent and so the agent writer, who had seen a
     few of those titles, never writes for them
  4. drop pages whose non-boilerplate content (unique paragraphs, boilerplate contact /
     source-list blocks removed) is under MIN_CHARS characters
  5. random sample of N pages with SEED; pages whose title contains "חרבות ברזל" (the
     wartime twin pages) are capped at WAR_CAP; the rest is a plain random draw.

Outputs:
  agent_sample_key.json     index -> doc_id, title   (the key; not read while writing)
  <scratch>/agent_sample_content.txt   index + content only, no title (what the writer reads)
"""
import json, os, random, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "src", "data")
SEED, N, MIN_CHARS, WAR_CAP = 20261005, 150, 300, 15
BOILER = ("מוקדים ממשלתיים", "גורמים מסייעים", "מקורות משפטיים ורשמיים", "גורם ממשלתי")
WAR = "חרבות ברזל"

def clean_for_display(t):
    t = re.sub(r"\((/he/|https?://)[^)]*\)", "", t)        # drop link targets (they carry page names)
    return re.sub(r"[ \t]+", " ", t).strip()

def main(content_out):
    d = json.load(open(os.path.join(DATA, "paragraph_corpus.json"), encoding="utf-8"))
    qa = pd.read_csv(os.path.join(DATA, "Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv"))
    qa_docs = set(int(x) for x in qa["doc_id"])
    human = set(int(p["doc_id"]) for p in json.load(open(os.path.join(HERE, "pages_key.json"), encoding="utf-8")))
    pages, title = {}, {}
    for k in sorted(d["doc_id"], key=int):                   # paragraph order within page = row-id order
        doc = int(d["doc_id"][k]); title[doc] = d["title"][k]
        pages.setdefault(doc, []).append(d["content"][k])
    counts = {"1_all_pages": len(pages)}
    clean = [p for p in pages if p not in qa_docs]; counts["2_not_in_QA"] = len(clean)
    clean = [p for p in clean if p not in human]; counts["3_not_in_human_60"] = len(clean)
    body = {}
    for p in clean:
        seen, keep = set(), []
        for c in pages[p]:
            s = c.strip()
            if s.startswith(BOILER) or s in seen: continue
            seen.add(s); keep.append(s)
        body[p] = keep
    elig = sorted(p for p in clean if sum(len(c) for c in body[p]) >= MIN_CHARS)
    counts["4_min_chars"] = len(elig)
    counts["4b_eligible_wartime"] = sum(1 for p in elig if WAR in title[p])
    rng = random.Random(SEED); order = elig[:]; rng.shuffle(order)
    sample, war = [], 0
    for p in order:
        if WAR in title[p]:
            if war >= WAR_CAP: continue
            war += 1
        sample.append(p)
        if len(sample) == N: break
    counts["5_sampled"] = len(sample); counts["5b_sampled_wartime"] = war
    key = [{"i": i + 1, "doc_id": p, "title": title[p]} for i, p in enumerate(sample)]
    json.dump({"seed": SEED, "counts": counts, "pages": key}, open(os.path.join(HERE, "agent_sample_key.json"), "w"),
              ensure_ascii=False, indent=1)
    with open(content_out, "w", encoding="utf-8") as f:     # NO titles in this file
        for i, p in enumerate(sample):
            txt = clean_for_display("\n".join(body[p]))
            f.write(f"=== {i + 1}\n{txt[:900]}\n\n")
    print(json.dumps(counts, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "agent_sample_content.txt"))
