#!/usr/bin/env python3
"""Prove eval_retrieval.py gives the same numbers as Webiks' own evaluator.

Builds a small corpus (all paragraphs of the gold docs of 40 held-out questions + random
distractor docs), runs (a) their CustomInformationRetrievalEvaluator and (b) our evaluate()
on identical inputs, and prints both. Runs on CPU so it does not fight the MPS encode job.
"""
import os, sys, json, random
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # Webiks-Hebrew-RAGbot-Demo checkout
DATA = os.environ.get("RAG_DATA_DIR", os.path.join(REPO, "data"))
MODEL = os.environ.get("RAG_MODEL_DIR", os.path.join(REPO, "app", "artifacts", "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"))
RESULTS = os.environ.get("RAG_RESULTS_DIR", os.path.join(REPO, "results"))
TRAINER = os.environ.get("RAG_TRAINER_DIR", os.path.join(os.path.dirname(REPO), "Webiks-Hebrew-RAGbot-Trainer"))
sys.path.insert(0, TRAINER)
from utils import CustomInformationRetrievalEvaluator          # theirs
from eval_retrieval import load_eval_set, evaluate, MODEL       # ours
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

random.seed(7)
gold_map = load_eval_set("heldout")
qs = random.sample(list(gold_map), 40)
corpus = json.load(open(os.path.join(DATA, "paragraph_corpus.json"), encoding="utf-8"))
cols = {k: v for k, v in corpus.items() if isinstance(v, dict)}
keys = list(cols["content"])
by_doc = {}
for k in keys: by_doc.setdefault(int(cols["doc_id"][k]), []).append(k)
want = set().union(*(gold_map[q] for q in qs))
distract = random.sample([d for d in by_doc if d not in want], 120)
sel = [k for d in sorted(want | set(distract)) for k in by_doc[d]]
corpus_df = pd.DataFrame({"doc_id": [int(cols["doc_id"][k]) for k in sel], "content": [cols["content"][k] or "" for k in sel],
                          "title": [cols["title"][k] for k in sel], "link": [cols["link"][k] for k in sel]})
print(f"mini corpus: {len(sel)} paragraphs, {len(want)} gold docs + {len(distract)} distractors, {len(qs)} questions", flush=True)

model = SentenceTransformer(MODEL, device="cpu"); model.eval()
# (a) theirs
queries = {f"q_{i}": q for i, q in enumerate(qs)}
relevant = {f"q_{i}": [f"d_{d}" for d in gold_map[q]] for i, q in enumerate(qs)}
ev = CustomInformationRetrievalEvaluator(corpus=None, queries=queries, relevant_docs=relevant, corpus_df=corpus_df,
        ranking_csv_path=os.path.join(RESULTS, "validate_theirs.csv"),
        mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10], accuracy_at_k=[1, 3, 5, 10], precision_recall_at_k=[10],
        corpus_chunk_size=1000, show_progress_bar=False, batch_size=16, write_csv=False,
        main_score_function="cosine", score_functions={"cosine": cos_sim}, k=10, hit_at_k=[1, 3, 5, 10])
os.makedirs(RESULTS, exist_ok=True)
theirs = ev(model, output_path=RESULTS)
# (b) ours, on the same vectors
cemb = model.encode(corpus_df["content"].tolist(), batch_size=16, convert_to_numpy=True, show_progress_bar=False)
qemb = model.encode(qs, batch_size=16, convert_to_numpy=True, show_progress_bar=False)
ours, _ = evaluate(cemb, corpus_df["doc_id"].to_numpy(), qemb, qs, gold_map, top_paragraphs=len(sel))
def pick(d, *subs): return {k: round(v, 4) for k, v in d.items() if isinstance(v, float) and any(s in k for s in subs)}
print("THEIRS:", pick(theirs, "accuracy@", "mrr@10", "ndcg@10"))
print("OURS  :", {k: round(v, 4) for k, v in ours.items() if k.startswith(("hit@", "mrr@", "ndcg@"))})
