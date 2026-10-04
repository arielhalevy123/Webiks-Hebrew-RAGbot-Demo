# Hebrew RAG Pipeline Enhancement — Retrieval Submission

Ariel Halevy, October 2026. Code: [engine fork, PR #1](https://github.com/arielhalevy123/Webiks-Hebrew-RAGbot/pull/1) and this Demo fork, branch `title-context-embedding`. Full experiment record, error analysis and sources: `APPENDIX.md`.

## 1. The improvement, and why this direction

**Diagnosis.** The retriever embeds each paragraph's `content` only; the page title is stored next to it but never reaches the model. On Kol-Zchut the title holds the qualifier that separates sibling pages (*הוצאת דרכון זמני* / *ביומטרי*, *הנחה בארנונה לנכים* / *לנכי עבודה*). Of 187 held-out questions the original system missed at rank 1, 63 had the right page at rank 2–3, mostly behind such a sibling. 64% of paragraphs already start with their section heading, so the section is embedded and the page is not.

**Shipped change: the page title is part of what gets embedded.** One opt-in key in `app/src/doc-config.json`, `embed_context_fields: ["title"]`, makes the text embedded at indexing time `"<page title>\n<paragraph>"`. Same fine-tuned model, vector field, index layout, `/search`, API and latency; remove the key and re-index to get the original system.

**Optional second stage, off by default: an LLM reorders the top 30 candidate pages by their titles**, through a provider-agnostic `llm_factory` (OpenAI or any OpenAI-compatible server, Anthropic, Gemini, or any model registered in a few lines). Off because it puts an LLM call in the retrieval path (~0.8 s, a key, nothing in the brief's mock mode); §5 turns it on.

**Alternatives.** *Hybrid BM25:* Hebrew prefixes (ו/ה/ב/ל) defeat lexical matching; two public attempts lowered rank-1. *Cross-encoder:* a second 568M model on a GPU; the optional stage gets most of that value from titles behind a switch. *ANN:* latency only; retrieval is already ~12× faster than the answer LLM.

## 2. Evaluation

**Metrics** (document level, as in Webiks' evaluator: a page scores its best paragraph). **hit@k** (k = 1, 3, 5, 10): is a correct page in the top k. **hit@3 is the production metric**, because the Demo hands 3 pages to the LLM. **MRR@10**: mean 1/rank of the first correct page. **nDCG@10** for questions with several correct pages.

**Three question sets**, all against the full corpus (24,487 paragraphs, 7,007 pages):
- **Vendor held-out, 296**: the validation split of Webiks' training code (seed 42). The brief says the QA file was training data, so absolute numbers are optimistic; the claim is the difference on identical inputs.
- **Clean, agent-written, 150**: pages with no row in the QA file, one question each, written by an LLM agent from the page body with the **title hidden**, so the set cannot favour a title change by construction (`retrieval_eval/eval_sets/agent_clean_150/`, method note included).
- **Public generated, 300**: IdoAgai's set from his public fork, questions written from a paragraph. Fetched by `retrieval_eval/fetch_idoagai_set.py`, not republished.

**Harness.** `retrieval_eval/eval_retrieval.py` (cached vectors, 3 s per variant) matches Webiks' evaluator to four decimals; reproduce everything with `retrieval_eval/README.md`.

## 3. Results

hit@1 / hit@3 / MRR@10 on the three sets (vendor hit@5 / hit@10: 0.652 / 0.760 original, 0.720 / 0.821 shipped; every variant in `APPENDIX.md`):

| | vendor 296 | clean 150 | public 300 |
|---|---|---|---|
| original | .368 / .581 / .496 | .533 / .747 / .653 | .553 / .747 / .659 |
| **title in text (shipped)** | **.463 / .645 / .577** | **.600 / .787 / .711** | **.537 / .757 / .663** |
| + title vector w=0.3 (off) | .527 / .709 / .636 | .560 / .773 / .675 | .543 / .730 / .649 |
| shipped + LLM title rerank (off by default)\* | .601 / .753 / .692 | .667 / .840 / .757 | .607 / .773 / .706 |

\*Measured with `retrieval_eval/title_llm.py`, gpt-4o-mini, 30 candidates. Its baseline differs from the evaluator's by 1–2 questions because of near-ties in query encoding; all rows inside that script share query vectors. $0.18 for 1,492 calls, median 0.77 s per call.

**Shipped vs original, per question:** 114 better / 47 worse (vendor), 38 / 19 (clean), 62 / 54 (public; hit@1 −1.6pp, hit@5 +4.3pp, all intervals include zero). On all 2,951 QA questions: hit@1 0.412 → 0.486, 1,027 better / 452 worse. The title helps when a question names the population or sub-case, as real Kol-Zchut questions do, and is neutral when a question is derived from the body. On the 33 clean-set questions that share no word with their title, hit@1 is level (9 better / 8 worse).

**Why the title vector (w=0.3) was switched off.** Shipped until 04.10, it adds +6.4pp hit@1 on the vendor set but gives it back on both independent sets, worse with every increase in weight, and significantly worse on title-poor questions (public set, no title word: hit@3 −12pp, 20 better / 41 worse). Its weight was chosen on questions written with the page in view. It remains an off-by-default engine knob (`embed_title_weight`).

**The optional stage** wins on all three sets (hit@1 gain vs original, 95% CI: [+.19, +.30], [+.07, +.21], [+.00, +.10]); repeated calls gave the same first choice 60/60. On the original index it reaches .520 / .653 / .623, so step 1 also lifts its ceiling (correct page in the 30 candidates: 90% vs 86%).

**Tested and not shipped.** *Smaller chunks:* worse alone (the embedder was fine-tuned on packed paragraphs); with the title they trade rank-1 for top-3, double the index and give the LLM half-paragraphs. *LLM-written context* ($3.34): every metric down. *Query rewriting, HyDE, four paraphrases with rank fusion:* at most +2.4pp, heavy churn. *Rewriting with candidate-title words:* about step 1 + 1–2pp. *Boilerplate filter, wider pool:* no effect.

**Corpus defect found.** Some pages store another page's text under their own title (doc 10075, 11477). A rough title/body check flags up to 271 of 7,007.

**Remaining errors (shipped, vendor):** 46.3% rank 1, 18.2% rank 2–3, 17.6% at 4–10, 17.9% beyond (about half of those name no page at all).

## 4. Integration

**Engine** (PR #1): `document.py` reads `embed_context_fields` and `embed_title_weight` (validated, default off); `engine.py` adds `text_to_embed()` / `embed_document()` for both ingest paths; 8 tests; `pyproject.toml`. **Demo:** `doc-config.json` sets the one key; `llm_factory.py` and `title_rerank.py` add the optional stage, wired into `main.py` with one line (off: the same engine object; on: wraps only `answer_query`, falls back to retrieval order on any LLM failure). `fast_index.py` now follows `doc-config.json` (its old default indexed content only; cos 1.000000 against the cached vectors for both representations). `requirements.txt` re-encoded from UTF-16, Windows-only pins behind markers. Frontend untouched. **Tests:** 60 pass (19 new: parsing, fallbacks, wrapper, switch, factory); the 6 `test_main` errors are identical on upstream.

## 5. Running the backend locally

```bash
git clone -b title-context-embedding https://github.com/arielhalevy123/Webiks-Hebrew-RAGbot-Demo.git && cd Webiks-Hebrew-RAGbot-Demo
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # torch 2.3.1: Python 3.10–3.12
docker run -d --name webiks-es -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" -p 9200:9200 elasticsearch:8.12.2
# model → app/artifacts/Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0/ ; corpus → data/paragraph_corpus.json  (links in the upstream READMEs)
# app/.env from app/.env-example: ES_EMBEDDING_INDEX=embedded_fusion  PATH_TO_ES_INITIAL_VALUES=../../data/paragraph_corpus.json
#   STATIC_DIR=./static  MODEL_LOCATION=../artifacts  DOCUMENT_DEFINITION_CONFIG=./doc-config.json  PORT=5050
#   IS_MOCK_GPT_CLIENT=true  OAI_API_KEY=sk-anything   (must be non-empty even in mock mode)
python retrieval_eval/fast_index.py --corpus data/paragraph_corpus.json   # follows doc-config.json; minutes, not the hours of /initialize_elastic_from_json
cd app/src && python -m uvicorn main:app --host 0.0.0.0 --port 5050      # from app/src; 5000 is AirPlay on macOS
curl -s -X POST localhost:5050/search -H 'Content-Type: application/json' -d '{"query":"דרכון דחוף","asked_from":"http://localhost:5050/"}'
# optional LLM title rerank: TITLE_RERANK_ENABLED=true in app/.env (options: app/.env-example)
# original behaviour: remove embed_context_fields from app/src/doc-config.json, re-index
```
