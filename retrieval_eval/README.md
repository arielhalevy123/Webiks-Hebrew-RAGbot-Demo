# retrieval_eval — measuring the retrieval stage

Scripts used for the retrieval-improvement submission. They work on cached corpus vectors
so that one variant scores in seconds instead of re-encoding 24,487 paragraphs per run.

Paths default to this checkout (`data/`, `results/`, `app/artifacts/<model>`) and can be
overridden with `RAG_DATA_DIR`, `RAG_RESULTS_DIR`, `RAG_MODEL_DIR`, `RAG_TRAINER_DIR`.

| script | what it does |
|---|---|
| `embed_corpus.py --variant content\|title_content` | encode the corpus once per representation, cache to `data/emb/<variant>.npy` (MPS/CUDA/CPU) |
| `eval_retrieval.py --variant X --eval-set heldout` | document-level hit@k / MRR@10 / nDCG@10 on the Trainer's seed-42 10% split; writes JSON + per-question CSV |
| `validate_eval.py` | runs Webiks' `CustomInformationRetrievalEvaluator` and ours on one small corpus and prints both (they match to 4 decimals) |
| `compare_runs.py A.json B.json` | metric deltas and per-question improvements / regressions between two runs |
| `error_table.py run.json` | buckets the remaining errors (sibling page first, rank 4-10, deep misses) with examples |
| `make_chunks.py --max-tokens 256 --overlap 32 --out ...` | re-cuts packed paragraphs into smaller windows (experiment 2) |
| `fast_index.py --corpus ... [--emb-dir data/emb --embed-field title_content]` | bulk-loads Elasticsearch with the same documents `/initialize_elastic_from_json` would create, from cached vectors in ~20 s or by batched encoding in minutes; vectors verified identical to the Demo path |

Inputs expected in `data/`: `paragraph_corpus.json` (Kol-Zchut paragraph corpus) and
`Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv` (QA set). Both are linked from
the Trainer README; neither is committed here.

Typical run:

```bash
python retrieval_eval/embed_corpus.py --variant content
python retrieval_eval/embed_corpus.py --variant title_content
python retrieval_eval/eval_retrieval.py --variant content
python retrieval_eval/eval_retrieval.py --variant title_content
python retrieval_eval/compare_runs.py results/content__heldout.json results/title_content__heldout.json
```
