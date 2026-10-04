# Appendix — full experiment record and notes

Companion to `SUBMISSION.md` (the 2-page summary). Everything here is measured; the per-run JSON is under `results/`.

## 0. Update 05.10.2026: what changed after independent evaluation

Sections 1–7 below are the record up to 04.10, when the shipped configuration was **title in text
+ a separate title vector fused at w = 0.3**; "shipped" there means that configuration. On 05.10 two
independent question sets were added, and the shipped configuration changed as a result.

**New evaluation sets.**
- *Agent-written clean set, 150* (`retrieval_eval/eval_sets/agent_clean_150/`): pages with no row in the
  QA file (5,367 of 7,007; ≥300 characters of non-boilerplate text; Ariel's 60 human-set pages excluded;
  150 sampled with seed 20261005). One question per page, written by an LLM agent from the first 900
  characters of the page body **with the title hidden**; titles attached by script afterwards. The set
  therefore cannot favour a title change by construction (if anything it favours the baseline, whose
  vectors embed exactly the text the questions came from). 33 questions share no 3+-letter word with
  their title (the "strict" subset). Method note, selection and assembly scripts are in that folder.
- *IdoAgai's public generated set, 300*: from his public fork; questions written by an LLM from one
  paragraph each, low lexical overlap by design; 56 of its pages have other questions in the QA file.
  Our baseline reproduces his (0.553 vs 0.550). Fetched by `retrieval_eval/fetch_idoagai_set.py`;
  only aggregate results are committed (`results/clean_eval/idoagai_generated_300_summary.json`).

**Index-side variants on the three sets** (hit@1 / hit@3 / MRR@10; evaluator `eval_retrieval.py`):

| representation | vendor 296 | agent clean 150 | IdoAgai 300 | agent strict 33 | IdoAgai strict 92 |
|---|---|---|---|---|---|
| content (original) | .368/.581/.496 | .533/.747/.653 | .553/.747/.659 | .424/.576/.515 | .348/.587/.479 |
| **title in text (shipped from 05.10)** | **.463/.645/.577** | **.600/.787/.711** | **.537/.757/.663** | .424/.515/.500 | .304/.576/.460 |
| + title vector w=0.2 | .527/.699/.631 | .573/.787/.690 | .553/.757/.661 | .333/.485/.426 | .326/.543/.446 |
| + title vector w=0.3 (shipped until 04.10) | .527/.709/.636 | .560/.773/.675 | .543/.730/.649 | .333/.455/.406 | .293/.467/.409 |
| + title vector w=0.4 | .530/.696/.635 | .540/.747/.655 | .530/.727/.635 | .303/.424/.373 | .228/.467/.362 |

Per question vs content: title in text 114/47 (vendor), 38/19 (agent), 62/54 (Ido); w=0.3 138/35,
32/35, 61/74. Bootstrap 95% CIs on the two independent sets include zero for every hit@1/hit@3/MRR delta
except w=0.3 on the Ido strict subset (hit@3 −.120 [−.217, −.033], MRR −.070 [−.126, −.019]). Head to
head on the agent set, w=0.3 beats title-in-text on 13 questions and loses on 36.

**Reading.** The title helps when the question names the population or sub-case (as the vendor
questions, written with the page in view, and real Kol-Zchut questions do); it is neutral when the
question derives from the body. The separate title vector adds +6.4pp hit@1 on the vendor set, gives it
back on both independent sets, gets worse monotonically with its weight, and hurts where the title says
little about the body (organisation names; pages whose stored text belongs to another page). Its weight
had been cross-fitted on the vendor questions, so it inherited their title-shaped distribution. IdoAgai's
FINDINGS.md reports the same asymmetry for his reranker (title worth +7.8pp on the vendor set, +3.7pp on
his generated set). **Decision: ship title in text only; keep `embed_title_weight` as an off-by-default
engine knob.** On all 2,951 QA questions title in text gives hit@1 0.412 → 0.486, hit@3 0.664 → 0.718,
1,027 better / 452 worse.

**Experiment 7: an LLM picks among the candidate titles (query time).** All 7,007 titles are ~140k
tokens, so instead the base retriever's top 30 pages (gold among them 85–96%) go to gpt-4o-mini with the
question; it returns up to 10 numbers, best first (variant A). Variant B, Ariel's literal idea, has it
rewrite the question with terms from fitting titles and searches again. `retrieval_eval/title_llm.py`,
outputs cached in `results/title_llm/` (2,984 calls in total, $0.36, median 0.77 s, p99 1.9 s).

| | vendor 296 | agent 150 | IdoAgai 300 |
|---|---|---|---|
| content (same query vectors) | .361/.581/.491 | .527/.747/.650 | .553/.743/.659 |
| title in text | .463/.645/.577 | .600/.787/.711 | .537/.757/.663 |
| A on the original index (rerank only) | .520/.713/.630 | .653/.820/.742 | .623/.800/.722 |
| **A on title in text** | **.601/.753/.692** | **.667/.840/.757** | .607/.773/.706 |
| B rewrite, on title in text (alone / RRF with original) | .476 / .480 hit@1 | .613 / .620 | .607 / .590 |

A on title in text vs content, hit@1 95% CI: [+.186, +.297], [+.067, +.213], [+.003, +.103]; better/worse
152/31, 52/20, 79/63. The LLM's first choice is the retriever's own #1 in 70% of calls, #2 13%, #3 5%,
#4–10 10%; rank-1 changes vs title in text +48/−7, +16/−6, +34/−13; 60 repeated calls gave the same first
choice 60/60 (full list identical 53/60). The script's baselines differ from the evaluator's by 1–2
questions (content .361 vs .368) because of float-level near-ties in query encoding; all rows in that
table share query vectors. **Shipped as an optional stage, off by default** (`app/src/title_rerank.py`,
`app/src/llm_factory.py`): it needs an LLM in the retrieval path (+~0.8 s, a key; nothing in the brief's
mock mode), two of the three sets are LLM-written, and it is a second direction on top of the one the
brief asks for. Verified end to end on a second backend instance (`docs/screenshots/06`, `07`).

**Experiment 8 (Ariel's idea): grounded multi-query.** The LLM sees the question and the base
retriever's top pages (30 titles, or 20 titles with a 300-character snippet each), writes 3 rewrites
using their terms, each rewrite is searched, and the original + rewrites are fused by RRF
(`retrieval_eval/grounded_multiquery.py`, $0.47, median 1.5 s per call). hit@1 vs title in text:
vendor .463 → .466–.493, agent .600 → .587–.627, IdoAgai .537 → .540–.573; hit@5/hit@10 rise more
(vendor hit@10 .821 → .868); every hit@1 CI includes zero. Diagnosis on the vendor set: one rewrite's
search alone puts the right page first 45.0% of the time (base 46.3%), the best of the three would
60.8% (≈ A's 60.1%), but the fused result is 46.6%. The three rewrites lean toward different
candidates (in 37% of questions they borrow words from the top wrong sibling's title), so the votes
split and fusion falls back to the original order. The information is there; turning it back into a
query vector and voting loses it, whereas A takes the LLM's judgement directly. Not shipped.

**Integration fix found while switching.** `retrieval_eval/fast_index.py`, which §5 of the submission
used for indexing, defaulted to embedding `content` only and never read `doc-config.json`; following the
04.10 instructions literally produced the original system's vectors. It now derives the representation
from the config (title prepending and, if set, the fused title vector, with the engine's formula);
64-paragraph test indices built both ways match the cached vectors at cosine 1.000000.

**Corpus defect.** At least two pages store another page's text under their own title and link (doc
10075: title about legal advice for Holocaust survivors, text of "מלווה סיעודי לנפגעי פעולת איבה"; doc
11477: title about trauma support centres, text about re-assessing a work-injury disability). A rough
check (rank of a page's own title vector among all 7,007 against its mean content vector) flags up to 271
pages; organisation names also score low on it, so that is an upper bound.


## 1. What was improved, and why this direction

**The change.** The page title is brought into each paragraph's vector, in two steps that
are both switched on in the document config and off by removing two keys:
1. the text handed to the embedder at indexing time becomes `"<page title>\n<paragraph>"`
   instead of `"<paragraph>"` (`embed_context_fields: ["title"]`);
2. the title is also embedded on its own and fused into the stored vector as
   `normalise(0.3·unit(title) + 0.7·unit(text))` (`embed_title_weight: 0.3`).

Because cosine similarity is linear in the query, searching that single fused vector is
the same as scoring `0.3·cos(q, title) + 0.7·cos(q, text)`. So the fine-tuned model, the
vector field, the Elasticsearch index layout, the `/search` code and the latency are all
unchanged; only what is stored per paragraph changes.

**Why this, and not hybrid BM25, a cross-encoder reranker, ANN, or query rewriting.** I
measured the shipped system first and read its failures before choosing a direction.

- The shipped retriever embeds the `content` field only. The page title is stored next to
  it but never seen by the model. On Kol-Zchut the title is exactly where the qualifier
  that separates sibling pages lives: *הנחה בארנונה לנכים* vs *לנכי עבודה*, *הוצאת דרכון
  זמני* vs *ביומטרי*, a page vs its *"בתקופת מלחמת חרבות ברזל"* twin.
- Of the 187 held-out questions the baseline does not answer at rank 1, 63 have the right
  page at rank 2–3, and most of those are sibling confusions of that kind.
- 64% of paragraphs already start with their HTML section heading, so the *section* is in
  the embedding and the *page* is not. Prepending the title adds the one missing piece.
- Two independent sources point the same way: Anthropic's published contextual-retrieval
  results (−35% retrieval failures from prepending document context before embedding), and
  the observation above about where the distinguishing word sits.
- A separate title vector, fused with a weight, turned out to carry more signal than the
  title diluted inside ~1,200 characters of body text (experiment 3 below).
- It costs nothing at query time: no new model, no GPU, no extra dependency, and the
  original behaviour remains the default.

Alternatives, and why not: **hybrid BM25** was rejected because unanalysed Hebrew tokens
(prefixes ו/ה/ב/ל) make lexical matching hurt rank-1 precision on this corpus, and the
right fix (a Hebrew analyzer plugin) is infrastructure work outside this scope; a
**cross-encoder reranker** adds a second 568M-parameter model and GPU latency to fix an
ordering problem that a better representation addresses at the source; **ANN/HNSW** only
improves latency, and retrieval is already ~12× faster than the LLM call in this pipeline
(0.5 s vs 6 s measured end to end); **LLM query rewriting** doubles LLM cost and cannot show
a gain on a single-turn evaluation set.

## 2. Evaluation: metrics, data, and the leakage caveat

**Metrics** are document level, matching Webiks' own `CustomInformationRetrievalEvaluator`
(each page is scored by its best paragraph, then pages are ranked):
- **hit@k**, k = 1, 3, 5, 10: is a gold page among the top k. **hit@3 is the production
  metric**: the Demo hands `num_of_pages = 3` pages to the LLM, so if the right page is not
  among them the answer cannot be right.
- **MRR@10**: mean of 1/rank of the first gold page; separates rank 1 from rank 3.
- **nDCG@10**: for the 72 questions with more than one gold page.

**Evaluation set.** The brief notes the QA dataset was part of the retrieval model's
training data. I reconstructed the validation split the training code itself produces
(`Trainer/utils.split_train_eval`: seed 42, 10% of unique questions): **296 questions,
319 gold pages, 72 questions with 2+ gold pages**. This is the most *comparable* set
available, but it is still optimistic: the vendor's own published numbers are computed on
the same questions, and public analysis reports that they appear in the training CSV. All
absolute numbers below are therefore an upper bound. What this submission claims is the
*difference* between baseline and change, measured on identical inputs.

**Corpus.** All 24,487 paragraphs / 7,007 pages. No subset.

**Harness.** `retrieval_eval/eval_retrieval.py` on cached vectors. Checked against Webiks'
evaluator on a shared 708-paragraph corpus: identical to the fourth decimal on all six
metrics (`retrieval_eval/validate_eval.py`). Their evaluator re-encodes the corpus on every
call (~40 min); the cached harness scores a variant in 3 s, which is what made the series
below affordable. (Their `run.py` also passes `main_score_function='cos_sim'`, which the
`sentence-transformers` release pinned by this Demo rejects; the harness uses `cosine`.)

## 3. Results

Five representations, same model, same corpus, same 296 questions:

| representation | hit@1 | **hit@3** | hit@5 | hit@10 | MRR@10 | nDCG@10 | gold absent from top-200 paragraphs |
|---|---|---|---|---|---|---|---|
| baseline: packed paragraphs, content only | 0.368 | 0.581 | 0.652 | 0.760 | 0.496 | 0.519 | 18 |
| packed, title + content (exp. 1) | 0.463 | 0.645 | 0.720 | 0.821 | 0.577 | 0.598 | 8 |
| 256-token chunks, content only (exp. 2a) | 0.324 | 0.541 | 0.639 | 0.736 | 0.451 | 0.484 | 21 |
| 256-token chunks, title + content (exp. 2b) | 0.405 | 0.672 | 0.730 | 0.818 | 0.549 | 0.578 | 11 |
| **packed, title + content ⊕ title vector, w = 0.3 (exp. 3, shipped)** | **0.527** | **0.709** | **0.757** | **0.848** | **0.636** | **0.648** | **9** |

**Baseline → shipped change:** hit@1 +15.9pp, hit@3 +12.8pp, MRR@10 +14.0pp. Per question:
138 improved (62 reach rank 1), 35 worse (15 lose rank 1), 123 unchanged. Experiment 1
alone (title in the text) accounts for hit@1 +9.5pp with 114 improved / 47 worse; the
fused title vector adds the rest (95 better / 38 worse on top of it). The gains hold under
the shipped candidate pool (top-50 paragraphs, then dedupe to pages), so the production
`/search` path delivers them with no query-side change.

**How the weight was chosen, so the number is not tuned on its own test set.** The weight
w was swept on the even-indexed questions and the result reported on the odd-indexed ones,
and vice versa. Picked on even → odd half: hit@1 0.453 → 0.493, hit@3 0.642 → 0.689. Picked
on odd → even half: hit@1 0.473 → 0.527, hit@3 0.649 → 0.696. The plateau is flat from
w = 0.2 to 0.4 (full-set hit@1 0.527 / 0.527 / 0.530), so 0.3, the middle, is shipped. The
honest headline is therefore hit@1 ≈ 0.49–0.53 and hit@3 ≈ 0.69–0.70; the table row is what
the shipped configuration scores on this set.

**On all 2,951 unique questions** (the 296 held-out plus the 2,655 training questions, which
inflate the baseline further): baseline hit@1 0.412 / hit@3 0.664 / MRR 0.557 → title in text
0.486 / 0.718 / 0.621 → shipped 0.529 / 0.753 / 0.655; 1,177 improved, 523 worse, 1,251
unchanged. Ten times the sample, same direction. Query-side experiments are not re-run on
this set because the model was trained on these questions in their original wording, which
would penalise any rewrite for reasons unrelated to its merit.

**Cost side, stated plainly.** Of the 35 questions that got worse, most moved by one or two
ranks between sibling pages: the title makes siblings distinguishable, not the choice
between them certain. One genuine regression from experiment 1 (*"נפלתי ברחוב..."* ranking a
street/police page first) persists.

**A prediction that was half wrong.** Before experiment 1 I wrote down that the title would
fix near misses (rank 2–3) and barely touch deep misses. Near misses moved as predicted;
deep misses moved too (49 improved, "absent" halved). The title adds topic words that even
vague questions mention. The worklog keeps the original prediction.

**Chunk size (the second experiment) — tested, not shipped.** The corpus packs paragraphs
up to the model's 512-token ceiling. Re-cutting them to 256-token windows with 32-token
overlap (50,008 chunks) *hurts* on its own: every metric down, 123 questions worse vs 60
better. The model was fine-tuned on the packed paragraphs, so they are its training
distribution; half-paragraphs are not, and they lose the section heading on the first
line. With the title on every chunk the damage disappears and hit@3/hit@5 even edge ahead
(+8 questions at hit@3), but rank-1 falls (−17 questions) and MRR with it; against the
shipped variant 89 questions get worse and 50 better. Not shipped for three reasons:
precision loss, index size doubled, and the Demo passes retrieved *paragraph text* to the
LLM, so chunks would hand it half-paragraphs as context. The right way to pursue chunking
is small-to-big retrieval (match on chunks, return the parent paragraph); listed under
next steps.

**Tested and rejected, briefly.** Dropping the 17.4% of paragraphs that are repeated
contact/help boilerplate: no effect (±1.5pp, mixed sign). Widening the 50-paragraph
candidate pool to 100/200/500: no change in any hit@k; only useful with a second stage.

### Where the remaining errors are (shipped variant, n = 296)

| bucket | n | % |
|---|---|---|
| gold page at rank 1 | 156 | 52.7 |
| rank 2–3, a sibling page (≥2 shared title words) is first | 22 | 7.4 |
| rank 2–3, other | 32 | 10.8 |
| rank 4–10 | 41 | 13.9 |
| rank >10 or absent | 45 | 15.2 |

The 18% at ranks 2–3 are still inside the three pages the LLM receives. The real losses are
the 29% at rank 4 or worse (35% in the baseline), and roughly half of those are questions that do not identify a
page on their own (*"עד איזה גיל אישה יכול להוציא נכות"* → gold *גיל פרישה מעבודה*): they
read like comments posted on a page, with the page as the label. No retrieval change fixes
those; they cap any method, including rerankers.

## 4. Integration (modular, default unchanged, nothing else touched)

`Webiks-Hebrew-RAGbot` (fork, branch `title-context-embedding`):
- `document.py`: optional `embed_context_fields` list in the document config, validated
  against `saved_fields`, default `[]` (= original behaviour).
- `document.py`: optional `embed_title_weight` (float in [0, 1), default 0) and
  `title_field` (default `"title"`).
- `engine.py`: `Engine.text_to_embed(doc)` joins the context fields and `field_to_embed`;
  `Engine.embed_document(doc)` returns the stored vector: the plain text vector when the
  weight is 0, otherwise the fused unit vector, with the title vector cached per distinct
  title. Both ingest paths (`update_docs`, `create_paragraphs`) use it. Query embedding,
  vector field name, index layout and `search()` untouched.
- `tests/test_embed_context.py`: 8 tests (default unchanged; title prepended; empty title
  skipped; query path unaffected; unknown field rejected; weight 0 keeps the plain vector;
  fusion maths on orthogonal unit vectors and the title cache; weight range validated). The pre-existing tests import a
  `ragbot` package from a `src/` directory that does not exist in the repository.
- `pyproject.toml` so a checkout is installable (`pip install -e .`).

`Webiks-Hebrew-RAGbot-Demo` (this fork, same branch):
- `app/src/doc-config.json`: `"embed_context_fields": ["title"]` and `"embed_title_weight": 0.3`.
  Remove both keys → original system.
- `requirements.txt`: re-encoded from UTF-16 to UTF-8, Intel-only pins restricted to
  Windows, engine dependency points at the fork branch.
- `retrieval_eval/`: the harness and experiment scripts (README inside). `results/`: the
  JSON behind every number above.
- Frontend and all other backend code untouched.

Verified: with the Demo config on, `Engine.create_paragraphs` produces vectors with cosine
1.000000 against the cached `fused_w0.3` vectors the shipped row was computed on, i.e. the
integrated backend ships the measured gain, not an approximation. The live Demo was
re-indexed from those vectors.

## 5. Running the updated backend locally

Tested on macOS (Apple Silicon) with Python 3.11; Linux should be identical. Windows users
keep the original Intel pins via the environment markers.

```bash
# 1. clone the Demo fork (its requirements pull the engine fork from GitHub)
git clone -b title-context-embedding https://github.com/arielhalevy123/Webiks-Hebrew-RAGbot-Demo.git
cd Webiks-Hebrew-RAGbot-Demo
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch 2.3.1 has no Python 3.13 wheel; use 3.10–3.12

# 2. Elasticsearch (the README's command; data dir inside the checkout)
docker run -d --name webiks-es -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" -p 9200:9200 \
  -v "$PWD/app/data/elastic/data:/usr/share/elasticsearch/data" elasticsearch:8.12.2

# 3. model and data (links in the upstream READMEs)
#    app/artifacts/Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0/   <- unzip the Drive model here
#    data/paragraph_corpus.json                                       <- paragraph corpus
#    data/Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv  <- QA set (evaluation only)

# 4. env: copy app/.env-example to app/.env and set
#    ES_EMBEDDING_INDEX=embedded_fusion   (must not be blank)
#    PATH_TO_ES_INITIAL_VALUES=../../data/paragraph_corpus.json
#    STATIC_DIR=./static   MODEL_LOCATION=../artifacts   DOCUMENT_DEFINITION_CONFIG=./doc-config.json
#    PORT=5050   (port 5000 is taken by AirPlay Receiver on macOS and answers 403)
#    IS_MOCK_GPT_CLIENT=true and OAI_API_KEY=sk-anything   (the openai client refuses an empty key even in mock mode)

# 5. index. Either the Demo's own route (one paragraph per call, ~6.5 h for the full corpus on CPU):
#    curl localhost:5050/initialize_elastic_from_json
#    or the batched loader, same documents, minutes on MPS/GPU:
python retrieval_eval/fast_index.py --corpus data/paragraph_corpus.json   # encodes with the Demo config (title + fused title vector)

# 6. run (must be started from app/src: main.py uses flat imports)
cd app/src && python -m uvicorn main:app --host 0.0.0.0 --port 5050
# frontend: http://localhost:5050   health: curl localhost:5050/health
curl -s -X POST localhost:5050/search -H 'Content-Type: application/json' \
  -d '{"query":"מי זכאי לקצבת זקנה?","asked_from":"http://localhost:5050/"}'
```

To compare with the original behaviour: delete `embed_context_fields` and
`embed_title_weight` from `app/src/doc-config.json`, re-index, restart.

To reproduce the numbers in §3 (no server needed):
```bash
python retrieval_eval/embed_corpus.py --variant content
python retrieval_eval/embed_corpus.py --variant title_content
python retrieval_eval/eval_retrieval.py --variant content
python retrieval_eval/eval_retrieval.py --variant title_content
python retrieval_eval/title_fusion.py          # cross-fitted weight sweep, writes data/emb/fused_w*.npy via the steps in its header
python retrieval_eval/eval_retrieval.py --variant fused_w0.3
python retrieval_eval/compare_runs.py results/content__heldout.json results/fused_w0.3__heldout.json
```

## 6. Next steps I would take

1. Small-to-big retrieval: match on 256-token chunks with the title, return the parent
   paragraph to the LLM. The chunk experiment says the recall is there; the context loss
   is what stopped it.
2. A clean evaluation set written by people, not drawn from the training CSV, to replace
   the upper-bound numbers with real ones.
3. A reranker as a *measured* second stage: done on 05.10 as the optional LLM title rerank (§0);
   next would be testing it on real traffic and against a paragraph-level cross-encoder.
4. Make `/search` return an error status instead of a 200 with an empty body when retrieval
   throws (today the frontend just goes silent).

## 7. Process

`notes/WORKLOG.md` in the assignment folder is the live record: what was asked, what came
back wrong, what was thrown away, and which decisions were mine. Every number above was
produced by a script in `retrieval_eval/` and is stored under `results/`.
