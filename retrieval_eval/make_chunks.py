#!/usr/bin/env python3
"""Re-cut the packed Webiks paragraphs into smaller token windows.

Output has the same schema as the original corpus (dict columns doc_id/title/content/link
keyed by chunk id), so embed_corpus.py and eval_retrieval.py work on it unchanged and
documents are still ranked by their best chunk. Each chunk keeps its parent paragraph key
in `parent` for tracing.

  make_chunks.py --max-tokens 256 --overlap 32 --out data/paragraph_corpus_chunk256.json
Splits on sentence/line boundaries where possible, falls back to token windows.
"""
import argparse, json, os, re
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # Webiks-Hebrew-RAGbot-Demo checkout
DATA = os.environ.get("RAG_DATA_DIR", os.path.join(REPO, "data"))
MODEL = os.environ.get("RAG_MODEL_DIR", os.path.join(REPO, "app", "artifacts", "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"))
RESULTS = os.environ.get("RAG_RESULTS_DIR", os.path.join(REPO, "results"))
TRAINER = os.environ.get("RAG_TRAINER_DIR", os.path.join(os.path.dirname(REPO), "Webiks-Hebrew-RAGbot-Trainer"))
SENT = re.compile(r"(?<=[\.\!\?:;])\s+|\n+")

def split_units(text):
    units = [u.strip() for u in SENT.split(text) if u and u.strip()]
    return units or [text.strip()]

def chunk_text(text, tok, max_tokens, overlap):
    units = split_units(text)
    lens = [len(tok.encode(u, add_special_tokens=False)) for u in units]
    chunks, cur, cur_len = [], [], 0
    i = 0
    while i < len(units):
        u, L = units[i], lens[i]
        if L > max_tokens:                     # a single huge unit: hard-split by tokens
            ids = tok.encode(u, add_special_tokens=False)
            for s in range(0, len(ids), max_tokens - overlap):
                chunks.append(tok.decode(ids[s:s + max_tokens]))
            i += 1; cur, cur_len = [], 0; continue
        if cur_len + L <= max_tokens:
            cur.append(u); cur_len += L; i += 1
        else:
            chunks.append(" ".join(cur))
            # overlap: keep trailing units up to `overlap` tokens
            keep, keep_len = [], 0
            for j in range(len(cur) - 1, -1, -1):
                lj = lens[i - len(cur) + j]
                if keep_len + lj > overlap: break
                keep.insert(0, cur[j]); keep_len += lj
            cur, cur_len = keep, keep_len
    if cur: chunks.append(" ".join(cur))
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(DATA, "paragraph_corpus.json"))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    d = json.load(open(a.corpus, encoding="utf-8"))
    cols = {k: v for k, v in d.items() if isinstance(v, dict)}
    out = {c: {} for c in cols}; out["parent"] = {}
    n_par, n_chunks, untouched = 0, 0, 0
    for k in cols["content"]:
        text = cols["content"][k] or ""
        n_par += 1
        if len(tok.encode(text, add_special_tokens=False)) <= a.max_tokens:
            pieces = [text]; untouched += 1
        else:
            pieces = chunk_text(text, tok, a.max_tokens, a.overlap)
        for j, p in enumerate(pieces):
            cid = f"{k}_{j}"
            for c in cols: out[c][cid] = cols[c][k]
            out["content"][cid] = p; out["parent"][cid] = k; n_chunks += 1
    for k, v in d.items():
        if not isinstance(v, dict): out[k] = v
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"paragraphs={n_par} -> chunks={n_chunks} (left whole: {untouched}); max_tokens={a.max_tokens} overlap={a.overlap}; wrote {a.out}")

if __name__ == "__main__":
    main()
