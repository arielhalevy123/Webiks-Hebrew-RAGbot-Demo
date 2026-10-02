#!/usr/bin/env python3
"""Build a single-vector fused cache from a base variant and the cached title vectors.

  build_fused.py --base title_content --w 0.3          -> data/emb/fused_w0.3.npy
  build_fused.py --base ctx_title_content --w 0.3      -> data/emb/ctx_fused_w0.3.npy
stored = normalise(w * unit(title_vec[doc]) + (1 - w) * unit(base_vec)). Title vectors are
encoded once into data/emb/_titles.npy (+ _titles.json) if missing.
"""
import argparse, json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="title_content"); ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--emb-dir", default=os.path.join(os.path.dirname(HERE), "data", "emb")); ap.add_argument("--corpus", default=os.path.join(os.path.dirname(HERE), "data", "paragraph_corpus.json"))
    ap.add_argument("--model", default=os.path.join(os.path.dirname(HERE), "app", "artifacts", "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"))
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    d = json.load(open(a.corpus, encoding="utf-8"))
    keys = json.load(open(os.path.join(a.emb_dir, f"{a.base}.keys.json"))); emb = np.load(os.path.join(a.emb_dir, f"{a.base}.npy"))
    tpath = os.path.join(a.emb_dir, "_titles.npy")
    if not os.path.exists(tpath):
        from sentence_transformers import SentenceTransformer
        import torch
        titles = sorted({d["title"][k] for k in d["title"]})
        m = SentenceTransformer(a.model, device="mps" if torch.backends.mps.is_available() else "cpu")
        tv = m.encode(titles, batch_size=64, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(tpath, tv); json.dump(titles, open(tpath.replace(".npy", ".json"), "w"), ensure_ascii=False)
    titles = json.load(open(tpath.replace(".npy", ".json"))); tv = np.load(tpath); tix = {t: i for i, t in enumerate(titles)}
    tv = tv / np.linalg.norm(tv, axis=1, keepdims=True); en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    ti = np.array([tix[d["title"][k]] for k in keys])
    fused = a.w * tv[ti] + (1 - a.w) * en; fused = (fused / np.linalg.norm(fused, axis=1, keepdims=True)).astype(np.float32)
    name = a.out_name or (f"fused_w{a.w:.1f}" if a.base == "title_content" else f"{a.base}_fused_w{a.w:.1f}")
    np.save(os.path.join(a.emb_dir, f"{name}.npy"), fused); json.dump(keys, open(os.path.join(a.emb_dir, f"{name}.keys.json"), "w"))
    print("saved", name, fused.shape)

if __name__ == "__main__":
    main()
