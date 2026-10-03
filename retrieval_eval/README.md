# retrieval_eval — measuring the retrieval stage

Scripts used for the retrieval-improvement submission. They work on cached corpus vectors
so that one variant scores in seconds instead of re-encoding 24,487 paragraphs per run.

Paths default to this checkout (`data/`, `results/`, `app/artifacts/<model>`) and can be
overridden with `RAG_DATA_DIR`, `RAG_RESULTS_DIR`, `RAG_MODEL_DIR`, `RAG_TRAINER_DIR`.

| script | what it does |
|---|---|
| `embed_corpus.py --variant content\|title_content\|ctx_title_content` | encode the corpus once per representation, cache to `data/emb/<variant>.npy` (MPS/CUDA/CPU) |
| `eval_retrieval.py --variant X --eval-set heldout` | document-level hit@k / MRR@10 / nDCG@10 on the Trainer's seed-42 10% split; writes JSON + per-question CSV |
| `validate_eval.py` | runs Webiks' `CustomInformationRetrievalEvaluator` and ours on one small corpus and prints both (they match to 4 decimals) |
| `compare_runs.py A.json B.json` | metric deltas and per-question improvements / regressions between two runs |
| `error_table.py run.json` | buckets the remaining errors (sibling page first, rank 4-10, deep misses) with examples |
| `make_chunks.py --max-tokens 256 --overlap 32 --out ...` | re-cuts packed paragraphs into smaller windows (experiment 2) |
| `title_fusion.py` / `build_fused.py` | embed the 7,007 distinct page titles once and build the fused cache `fused_w<w>.npy` = normalise(w·unit(title) + (1−w)·unit(text)); the cross-fitted weight sweep lives here (experiment 3, shipped at w=0.3) |
| `gen_context.py` | write a 1–2 sentence context per paragraph with gpt-4o-mini for the `ctx_title_content` variant (experiment 4, rejected); output `data/llm_context.jsonl`, a gzipped copy is in `results/` |
| `query_rewrite.py` | query-side LLM variants: rewrite, hypothetical answer (HyDE), concatenations (experiment 5, not shipped) |
| `multiquery.py` | 4 paraphrases per question, searched separately, pages fused by reciprocal rank (experiment 6, not shipped) |
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

Shipped variant and the full QA file:

```bash
python retrieval_eval/build_fused.py --base title_content --w 0.3     # writes data/emb/fused_w0.3.npy (title_fusion.py is the weight sweep)
python retrieval_eval/eval_retrieval.py --variant fused_w0.3 --eval-set heldout   # 296 questions (seed-42 split)
python retrieval_eval/eval_retrieval.py --variant fused_w0.3 --eval-set all       # all 2,951 questions
python retrieval_eval/compare_runs.py results/content__heldout.json results/fused_w0.3__heldout.json
python retrieval_eval/error_table.py results/fused_w0.3__heldout.json
```

Tests: `PYTHONPATH=app/src DOCUMENT_DEFINITION_CONFIG=app/src/doc-config.json pytest tests -q`
(needs `pytest-mock`; 41 pass, the 6 `test_main.py` errors are identical on upstream).
