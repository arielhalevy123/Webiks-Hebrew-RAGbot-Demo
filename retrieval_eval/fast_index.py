#!/usr/bin/env python3
"""Batched indexer for the Webiks paragraph corpus.

Produces the same Elasticsearch documents as the Demo's /initialize_elastic_from_json
(same index naming, same field names, same model), but encodes in batches on the best
available torch device (MPS on Apple Silicon) and bulk-writes to ES. Ingestion speed
only; the stored vectors are those of the shipped model, so retrieval behaviour is
unchanged. Written 02.10.2026 because the Demo path encodes one paragraph per call
(~65/min on CPU here) and the experiments ahead need many re-embeddings.

Usage:
  fast_index.py --corpus ../data/paragraph_corpus_subset.json [--index embedded_fusion]
                [--model-dir Webiks-Hebrew-RAGbot-Demo/app/artifacts/<model>]
                [--config app/src/doc-config.json] [--embed-field content | title_content]
                [--batch 32] [--no-delete]

By default the stored vectors follow the document config exactly as Engine.embed_document does:
`embed_context_fields` (e.g. ["title"]) are prepended to the embedded text, and a positive
`embed_title_weight` fuses a separately embedded title vector in. --embed-field overrides that.
"""
import argparse, json, math, os, sys, time
import numpy as np
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # Webiks-Hebrew-RAGbot-Demo checkout
DATA = os.environ.get("RAG_DATA_DIR", os.path.join(REPO, "data"))
MODEL = os.environ.get("RAG_MODEL_DIR", os.path.join(REPO, "app", "artifacts", "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"))
RESULTS = os.environ.get("RAG_RESULTS_DIR", os.path.join(REPO, "results"))
TRAINER = os.environ.get("RAG_TRAINER_DIR", os.path.join(os.path.dirname(REPO), "Webiks-Hebrew-RAGbot-Trainer"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index", default=os.getenv("ES_EMBEDDING_INDEX", "embedded_fusion"))
    ap.add_argument("--index-length", type=int, default=int(os.getenv("ES_EMBEDDING_INDEX_LENGTH", "1000")))
    ap.add_argument("--model-dir", default=MODEL)
    ap.add_argument("--model-name", default="Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0",
                    help="suffix used in the vector field name; must match doc-config.json model_name")
    ap.add_argument("--config", default=os.path.join(REPO, "app", "src", "doc-config.json"),
                    help="document config to follow (embed_context_fields, embed_title_weight)")
    ap.add_argument("--embed-field", default=None,
                    help="override: content | title_content (default: derived from --config)")
    ap.add_argument("--es", default="http://localhost:9200")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--no-delete", action="store_true", help="keep existing <index>* indices")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--emb-dir", default=None, help="load <emb-dir>/<variant>.npy + .keys.json (from embed_corpus.py) instead of encoding; --variant names the cache")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8")) if os.path.exists(a.config) else {}
    context_fields = list(cfg.get("embed_context_fields", []))
    title_weight = float(cfg.get("embed_title_weight", 0.0))
    if a.embed_field is None:
        a.embed_field = "title_content" if "title" in context_fields else "content"
    else:
        title_weight = 0.0 if a.embed_field == "content" else title_weight
    cache_name = a.embed_field if title_weight <= 0 else f"fused_w{title_weight:g}"
    print(f"representation: embedded text = {a.embed_field}; title vector weight = {title_weight:g} "
          f"(config {a.config if cfg else 'not found'})", flush=True)

    import torch
    from elasticsearch import Elasticsearch, helpers
    from sentence_transformers import SentenceTransformer

    cache = None
    if a.emb_dir:
        cache_keys = json.load(open(os.path.join(a.emb_dir, f"{cache_name}.keys.json")))
        cache = {k: i for i, k in enumerate(cache_keys)}
        cache_vecs = np.load(os.path.join(a.emb_dir, f"{cache_name}.npy"))
        print(f"using cached vectors {a.emb_dir}/{cache_name}.npy ({cache_vecs.shape})", flush=True)
        model = None
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={device} model={a.model_dir}", flush=True)
        model = SentenceTransformer(a.model_dir, device=device)
        model.eval()

    d = json.load(open(a.corpus, encoding="utf-8"))
    cols = {k: v for k, v in d.items() if isinstance(v, dict)}
    keys = list(cols["content"].keys())
    if a.limit: keys = keys[:a.limit]
    print(f"paragraphs={len(keys)} documents={len(set(cols['doc_id'][k] for k in keys))}", flush=True)

    es = Elasticsearch(a.es)
    if not a.no_delete:
        existing = es.indices.get(index=f"{a.index}*", ignore_unavailable=True)
        for ix in existing:
            es.indices.delete(index=ix); print(f"deleted {ix}", flush=True)

    def text_for(k):
        c = cols["content"][k] or ""
        if a.embed_field == "title_content":
            return f"{cols['title'][k] or ''}\n{c}"
        return c

    vec_field = f"content_{a.model_name}_vectors"
    t0 = time.time(); done = 0; title_cache = {}
    for i in range(0, len(keys), a.batch):
        chunk = keys[i:i + a.batch]
        if cache is not None:
            vecs = cache_vecs[[cache[k] for k in chunk]]
        else:
            vecs = model.encode([text_for(k) for k in chunk], batch_size=a.batch, convert_to_numpy=True, show_progress_bar=False)
            if title_weight > 0:   # same formula as Engine.embed_document
                vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
                titles = [cols["title"][k] or "" for k in chunk]
                missing = [t for t in set(titles) if t and t not in title_cache]
                if missing:
                    tv = model.encode(missing, batch_size=a.batch, convert_to_numpy=True, show_progress_bar=False)
                    for t, v in zip(missing, tv):
                        title_cache[t] = v / max(np.linalg.norm(v), 1e-12)
                fused = np.array([title_weight * title_cache[t] + (1 - title_weight) * v if t else v
                                  for t, v in zip(titles, vecs)])
                vecs = fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)
        actions = []
        for k, v in zip(chunk, vecs):
            doc_id = int(cols["doc_id"][k])
            doc = {col: cols[col][k] for col in cols}
            doc["doc_id"] = doc_id
            doc[vec_field] = v.tolist()
            doc["last_update"] = datetime.now().isoformat()
            actions.append({"_index": f"{a.index}_{round(doc_id / a.index_length)}", "_source": doc})
        helpers.bulk(es, actions, refresh=False)
        done += len(chunk)
        if (i // a.batch) % 10 == 0 or done == len(keys):
            rate = done / (time.time() - t0)
            print(f"{done}/{len(keys)}  {rate:.0f} para/s  eta {((len(keys)-done)/max(rate,1e-6)):.0f}s", flush=True)
    es.indices.refresh(index=f"{a.index}*")
    total = sum(int(x["docs.count"]) for x in es.cat.indices(index=f"{a.index}*", format="json"))
    print(f"done in {time.time()-t0:.0f}s, ES now holds {total} paragraphs under {a.index}*", flush=True)

if __name__ == "__main__":
    main()
