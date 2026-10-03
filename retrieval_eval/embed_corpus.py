#!/usr/bin/env python3
"""Encode the paragraph corpus once per representation variant and cache the vectors.

Why: their evaluator (Trainer/utils.py) re-encodes all 24,487 paragraphs on every call
(~40 min on MPS). Caching lets us evaluate many variants and also bulk-load Elasticsearch
without re-encoding. Vectors are from the shipped fine-tuned model, unchanged.

Variants (what text goes into the embedder; stored text is never changed):
  content        : the paragraph body only  (= the shipped system, baseline)
  title_content  : "<page title>\n<body>"    (cheap contextual embedding)
  ctx_title_content : "<LLM-written context>\n<page title>\n<body>"  (needs data/llm_context.jsonl, see gen_context.py)

Output: data/emb/<variant>.npy (float32, N x 1024) + data/emb/<variant>.keys.json (paragraph keys in row order)
"""
import argparse, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # Webiks-Hebrew-RAGbot-Demo checkout
DATA = os.environ.get("RAG_DATA_DIR", os.path.join(REPO, "data"))
MODEL = os.environ.get("RAG_MODEL_DIR", os.path.join(REPO, "app", "artifacts", "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"))
RESULTS = os.environ.get("RAG_RESULTS_DIR", os.path.join(REPO, "results"))
TRAINER = os.environ.get("RAG_TRAINER_DIR", os.path.join(os.path.dirname(REPO), "Webiks-Hebrew-RAGbot-Trainer"))

def load_context(path):
    ctx = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line); ctx[r["key"]] = r["context"]
        except Exception:
            pass
    return ctx

def texts_for(variant, cols, keys, ctx_path=None):
    if variant == "ctx_title_content":
        ctx = load_context(ctx_path)
        missing = [k for k in keys if k not in ctx]
        if missing:
            raise SystemExit(f"{len(missing)} paragraphs have no generated context in {ctx_path}; run gen_context.py to completion first")
        return [f"{ctx[k]}\n{cols['title'][k] or ''}\n{cols['content'][k] or ''}" for k in keys]
    if variant == "content":
        return [cols["content"][k] or "" for k in keys]
    if variant == "title_content":
        return [f"{cols['title'][k] or ''}\n{cols['content'][k] or ''}" for k in keys]
    raise SystemExit(f"unknown variant {variant}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(DATA, "paragraph_corpus.json"))
    ap.add_argument("--variant", default="content")
    ap.add_argument("--out-dir", default=os.path.join(DATA, "emb"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--context", default=None, help="llm_context.jsonl for the ctx_title_content variant (default: <data dir>/llm_context.jsonl)")
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)
    d = json.load(open(a.corpus, encoding="utf-8"))
    cols = {k: v for k, v in d.items() if isinstance(v, dict)}
    keys = list(cols["content"].keys())
    ctx_path = a.context or os.path.join(os.path.dirname(a.corpus), "llm_context.jsonl")
    texts = texts_for(a.variant, cols, keys, ctx_path)
    print(f"variant={a.variant} paragraphs={len(keys)} device={device}", flush=True)

    model = SentenceTransformer(MODEL, device=device); model.eval()
    t0 = time.time()
    emb = model.encode(texts, batch_size=a.batch, convert_to_numpy=True, show_progress_bar=False,
                       normalize_embeddings=False).astype(np.float32)
    print(f"encoded in {time.time()-t0:.0f}s, shape={emb.shape}", flush=True)
    np.save(os.path.join(a.out_dir, f"{a.variant}.npy"), emb)
    json.dump(keys, open(os.path.join(a.out_dir, f"{a.variant}.keys.json"), "w"))
    print("saved", os.path.join(a.out_dir, f"{a.variant}.npy"), flush=True)

if __name__ == "__main__":
    main()
