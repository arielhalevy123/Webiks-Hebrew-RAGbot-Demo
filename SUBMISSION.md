# Hebrew RAG Pipeline Enhancement — Retrieval Submission

Ariel Halevy, October 2026. Code: [engine fork, PR #1 merged](https://github.com/arielhalevy123/Webiks-Hebrew-RAGbot/pull/1) and this Demo fork, branch `title-context-embedding`. Full experiment record, error analysis and sources: `APPENDIX.md`.

## 1. The improvement, and why this direction

**What changed.** Each paragraph's stored vector now carries the title of the page it belongs to, in two opt-in steps set in the document config (`app/src/doc-config.json`):
1. `embed_context_fields: ["title"]` — the text embedded at indexing time becomes `"<page title>\n<paragraph>"`.
2. `embed_title_weight: 0.3` — the title is also embedded on its own and fused into the stored vector: `normalise(0.3·unit(title) + 0.7·unit(text))`.

Cosine similarity is linear in the query, so searching that single fused vector equals `0.3·cos(q, title) + 0.7·cos(q, text)`. The fine-tuned model, the vector field, the Elasticsearch index layout, `/search`, the API and the latency are unchanged; only what is stored per paragraph changes. Remove the two keys and the system is the original one.

**Why this, measured before chosen.** The shipped retriever embeds `content` only; the page title is stored but never seen by the model. On Kol-Zchut the title holds the qualifier that separates sibling pages (*הנחה בארנונה לנכים* / *לנכי עבודה*, *דרכון זמני* / *ביומטרי*, a page and its wartime twin). Reading the baseline's failures: of 187 held-out questions not answered at rank 1, 63 had the right page at rank 2–3, mostly such sibling confusions. 64% of paragraphs already begin with their section heading, so the section is embedded and the page is not; the title is the missing piece. A separate title vector carries that signal better than the title diluted inside ~1,200 characters of body.

**Alternatives, rejected on evidence.** *Hybrid BM25:* unanalysed Hebrew prefixes (ו/ה/ב/ל) make lexical matching lower rank-1 precision on this corpus; a Hebrew analyzer is infrastructure work outside scope. *Cross-encoder reranker:* a second 568M model on a GPU to fix ordering that a better representation fixes at the source. *ANN/HNSW:* latency only; retrieval is already ~12× faster than the LLM call (0.5 s vs 6 s measured). *LLM query rewriting:* doubles LLM cost, unmeasurable on single-turn questions.

## 2. Evaluation

**Metrics** (document level, as in Webiks' own evaluator: page scored by its best paragraph, pages ranked). **hit@k**, k = 1, 3, 5, 10: is a gold page among the top k; **hit@3 is the production metric** because the Demo hands `num_of_pages = 3` pages to the LLM. **MRR@10**: mean 1/rank of the first gold page. **nDCG@10**: for the 72 questions with several gold pages.

**Data.** The brief notes the QA set was in the model's training data. I reconstructed the validation split the training code produces (`split_train_eval`, seed 42, 10% of unique questions): 296 questions, 319 gold pages. Comparable, but still optimistic, since public analysis reports these questions also appear in training; absolute numbers are an upper bound, the baseline-vs-change *difference* on identical inputs is the claim. Corpus: all 24,487 paragraphs / 7,007 pages.

**Harness.** `retrieval_eval/eval_retrieval.py` on cached vectors, checked against Webiks' evaluator on a shared 708-paragraph corpus: identical to four decimals on all six metrics. One variant scores in 3 s instead of ~40 min, which is what made the series below affordable.

## 3. Results (same model, corpus and 296 questions)

| representation | hit@1 | **hit@3** | hit@5 | hit@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| baseline: packed paragraphs, content only | 0.368 | 0.581 | 0.652 | 0.760 | 0.496 | 0.519 |
| title in text (step 1) | 0.463 | 0.645 | 0.720 | 0.821 | 0.577 | 0.598 |
| **title in text ⊕ title vector, w=0.3 (shipped)** | **0.527** | **0.709** | **0.757** | **0.848** | **0.636** | **0.648** |
| 256-token chunks, content only | 0.324 | 0.541 | 0.639 | 0.736 | 0.451 | 0.484 |
| 256-token chunks + title | 0.405 | 0.672 | 0.730 | 0.818 | 0.549 | 0.578 |
| LLM-written context + title (gpt-4o-mini, $3.34) ⊕ title vector | 0.490 | 0.703 | 0.740 | 0.828 | 0.605 | 0.623 |

**Baseline → shipped:** hit@1 +15.9pp, hit@3 +12.8pp, MRR +14.0pp; 138 questions improved, 35 worse, 123 unchanged. The weight was chosen by cross-fitting (swept on half the questions, reported on the other half, both ways: hit@1 0.453→0.493 and 0.473→0.527), and the plateau 0.2–0.4 is flat, so 0.3 is shipped. The gain holds under the shipped 50-paragraph candidate pool, and on all 2,951 questions of the QA file (training questions included, so an even more optimistic baseline) it is hit@1 0.412 → 0.529, hit@3 0.664 → 0.753, with 1,177 questions improved and 523 worse. Cost side: most regressions move one or two places between sibling pages.

**Tested and not shipped.** *Smaller chunks* hurt alone (the embedder was fine-tuned on the packed paragraphs) and with the title trade rank-1 for top-3 while doubling the index and handing the LLM half-paragraphs; the right follow-up is small-to-big retrieval. *LLM-written per-paragraph context* lowered every metric against its title-only counterpart: a uniform descriptive preamble pulls vectors toward one register, and the title already supplied the page context. *LLM help on the query side* (rewrite, hypothetical answer, and multi-query: four paraphrases searched separately and fused by rank vote; $0.05 in total): the best, original + four paraphrases with rank fusion, reaches hit@1 0.551 / hit@3 0.733 (71 questions better, 50 worse), but paraphrases can change meaning (a vehicle "טסט" became a driving test), every question costs an LLM call plus five searches, and search itself becomes dependent on OpenAI; not shipped, named as the first query-side method to test if that dependency is acceptable. *Dropping boilerplate paragraphs* and *widening the candidate pool*: no effect.

**Remaining errors (shipped variant).** 52.7% rank 1; 18.2% at rank 2–3 (still inside the three pages the LLM gets, 7.4pp of it sibling pages first); 13.9% at 4–10; 15.2% at >10 or absent. About half of the deep misses are questions that do not identify a page on their own (comments posted on a page, with the page as label); no retriever fixes those.

## 4. Integration

Engine (`Webiks-Hebrew-RAGbot`, PR #1): `document.py` reads the two optional keys (validated; defaults keep the original behaviour); `engine.py` adds `text_to_embed()` and `embed_document()` used by both ingest paths, with the title vector cached per distinct title; 8 new tests on the real package; `pyproject.toml` so a checkout is installable. Demo: `doc-config.json` sets the two keys; `requirements.txt` re-encoded from UTF-16, Windows-only pins restricted with markers, engine dependency pointed at the fork; `retrieval_eval/` holds the harness, `results/` the JSON behind every number. Frontend untouched. The Demo's own suite: 41 pass; the 6 `test_main` errors are identical on upstream (their test patches `builtins.open` during import). Verified: with the Demo config, `Engine.create_paragraphs` yields vectors with cosine 1.000000 against the cached ones the table was computed on.

## 5. Running the backend locally

```bash
git clone -b title-context-embedding https://github.com/arielhalevy123/Webiks-Hebrew-RAGbot-Demo.git && cd Webiks-Hebrew-RAGbot-Demo
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # torch 2.3.1: Python 3.10–3.12
docker run -d --name webiks-es -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" -p 9200:9200 elasticsearch:8.12.2
# model → app/artifacts/Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0/ ; corpus → data/paragraph_corpus.json  (links in the upstream READMEs)
# app/.env from app/.env-example: ES_EMBEDDING_INDEX=embedded_fusion  PATH_TO_ES_INITIAL_VALUES=../../data/paragraph_corpus.json
#   STATIC_DIR=./static  MODEL_LOCATION=../artifacts  DOCUMENT_DEFINITION_CONFIG=./doc-config.json  PORT=5050
#   IS_MOCK_GPT_CLIENT=true  OAI_API_KEY=sk-anything   (the openai client refuses an empty key even in mock mode)
python retrieval_eval/fast_index.py --corpus data/paragraph_corpus.json   # same documents as /initialize_elastic_from_json, minutes instead of hours
cd app/src && python -m uvicorn main:app --host 0.0.0.0 --port 5050      # start from app/src; port 5000 is AirPlay on macOS
curl -s -X POST localhost:5050/search -H 'Content-Type: application/json' -d '{"query":"מי זכאי לקצבת זקנה?","asked_from":"http://localhost:5050/"}'
```
Original behaviour: delete the two keys from `doc-config.json`, re-index, restart. Reproduce §3: `retrieval_eval/README.md`.
