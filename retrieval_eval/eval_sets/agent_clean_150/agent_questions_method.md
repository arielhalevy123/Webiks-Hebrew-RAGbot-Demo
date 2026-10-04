# Agent-written clean eval set: method

Written 05.10.2026 by a Claude Code agent session (the writer is an LLM, not a human; no external
LLM API was called). Purpose: a retrieval eval set on pages the embedder never saw in training,
written in a way that cannot favour the title-fusion change by construction.

## Files
| file | what |
|---|---|
| `select_pages.py` | page selection (seed, filters, wartime cap). Writes `agent_sample_key.json` and a title-free content dump (kept in the session scratchpad, not here) |
| `agent_sample_key.json` | sample index -> doc_id, title, plus the filter counts |
| `agent_questions_raw.tsv` | the 150 questions as written: `index<TAB>question`, no doc_id, no title |
| `build_eval_csv.py` | joins raw questions to the key after writing; computes title-word overlap; writes the two CSVs |
| `agent_questions.csv` | `question, doc_id, title, shared` (150 rows, one gold page each; `shared` = title words found in the question) |
| `agent_questions_strict.csv` | the 33 rows with no shared title word |
| `stats.py` | paired bootstrap CIs, sign test, sensitivity run |
| `results/` | evaluator JSON + per-question CSV for both variants on both sets, `compare__*.txt`, `stats.json` |

## Page selection (seed 20261005)
| step | pages |
|---|---|
| 1. all pages in `paragraph_corpus.json` | 7,007 |
| 2. no row in the QA csv (any doc_id in the QA file removed) | 5,367 |
| 3. minus the 60 pages already given to Ariel for the human set (`pages_key.json`) | 5,307 |
| 4. non-boilerplate content >= 300 chars | 5,243 (of which 53 have "חרבות ברזל" in the title) |
| 5. shuffled with `random.Random(20261005)`, first 150 taken, "חרבות ברזל" titles capped at 15 | 150 (1 wartime page; the cap never bound) |

Boilerplate for step 4 = paragraphs starting with "מוקדים ממשלתיים", "גורמים מסייעים", "מקורות משפטיים
ורשמיים" or "גורם ממשלתי"; exact-duplicate paragraphs inside a page counted once.

Why step 3: it keeps the agent set and the human set independent, and before selection the writer
had seen four titles from `pages_to_question.md`; excluding all 60 removes any chance of writing for
a page whose title was known.

## The no-title rule
The writer only ever read the dump produced by `select_pages.py`: `=== <index>` followed by the page's
non-boilerplate paragraphs, link targets `(/he/...)` and `(https://...)` removed (they often carry page
names), truncated to the first 900 characters. No title, no link, no doc_id. Questions were saved
by index into `agent_questions_raw.tsv` in batches of 15. Titles were attached by `build_eval_csv.py`
only after all 150 were written. One leak: page 150's dump kept four PDF file links
(`/w/he/images/...תקנון_הטרדה_מינית...`) that the regex did not strip; its question shares only
"הטרדה מינית" with the title "נקיטת אמצעי מנע למניעת הטרדה מינית", and that phrase is all over the body.

Style: one question per page, citizen register, mixed keyword-only / spoken / invented personal
details / a few typos ("פיצוים", "פנייתי", "דחיה"). Whole sentences were not copied, but fixed domain
terms were kept when a citizen would use them: 22 of 150 questions contain a verbatim 3-word run from
their page (e.g. "אלרגיה מסכנת חיים", "עיכוב יציאה מהארץ", "השתלת מח עצם").

## Strict subset
A question is "strict" if it shares no content word (3+ Hebrew letters) with its gold title. Matching
is deliberately loose so the subset is conservative: letters only (צה"ל = צהל), final letters
normalised, up to two leading prefix letters (ו ה ב ל מ ש כ) stripped on either side, no stopword list.
33 of 150 questions qualify.

## Things noticed
1. **Two sampled pages have the wrong content in the corpus.** doc 10075 (title "ייעוץ משפטי ראשוני
   לניצולי שואה ונכי המלחמה בנאצים...") stores the text of "מלווה סיעודי לנפגעי פעולת איבה"; doc 11477
   (title "מרכזים לתמיכה נפשית לנפגעי טראומה וחרדה") stores the text of the work-injury disability
   re-evaluation page. Found after the eval, when the per-question regressions were inspected: the
   shipped variant's top-1 was the page whose title matches the text. The questions were written from
   the stored text, as the rule required, so the gold label is "correct" for the corpus but the title is
   wrong for that text, which the title fusion penalises. A rough corpus-wide check (rank of a page's
   own title among all 7,007 title vectors, against its mean content vector) puts 271 pages at rank
   > 100; not all of those are mismatches (organisation names score low too), but the defect is not
   unique to these two. The primary results keep both pages; `stats.py` also reports a run without them.
2. **The strict subset is not a random subset.** By construction it collects pages whose title says
   little about the body: 11 of its 33 pages are organisations with proper-name titles ("בנתיבי אודי",
   "אור לעולם", "מן המיצר", "עמותת אלומה", ...) plus the two mismatched pages above. That is exactly where
   a title vector adds noise, so this subset is close to a worst case for the change.
3. **Sibling pages.** Each question has one gold page. Some near-duplicates exist in the sample and the
   corpus (102/105 rent help for disabled veterans in general vs. while studying; 118/136 income support
   for olim by age vs. by illness; 4 vs. 36 an org list vs. one org on it). A hit on the sibling counts
   as a miss. Not corrected.
4. The writer knew which change was being evaluated. The no-title rule removes the direct route to bias,
   but the set is not blind in the sense a third-party human set would be.
