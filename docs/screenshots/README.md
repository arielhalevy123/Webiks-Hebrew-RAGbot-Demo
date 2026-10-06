# Screenshots

Full corpus (24,487 paragraphs) in Elasticsearch 8.12.2, Demo backend on `localhost`. The shipped
configuration since 05.10.2026 is **title in the embedded text** (`embed_context_fields: ["title"]`);
the optional LLM title rerank stage is off by default. Screenshots 01, 02 and 05 were taken on 04.10 on
the index that then also fused a separate title vector at weight 0.3 (tested, no longer shipped; see
SUBMISSION.md §3). Terminal screenshots are captured output rendered to an image; the raw `.txt` sits
next to each and nothing in them was edited.

| file | what it shows |
|---|---|
| `01_demo_search_arnona_lenechim.png` | 04.10, w=0.3 index. *מי זכאי להנחה בארנונה לנכים?*, a wording typed for the screenshot (not from the QA set); returns the three sibling ארנונה pages with the asked-for one first. No baseline run of this wording was recorded. |
| `02_demo_search_kitzbat_zikna.png` | 04.10, w=0.3 index. *מי זכאי לקצבת זקנה?*: three old-age-pension pages, retrieval 0.13 s. |
| `03_eval_baseline_vs_shipped.png` (+ `.txt`) | `eval_retrieval.py` on the 296 held-out questions for `content` (original) and `title_content` (shipped), then `compare_runs.py`: hit@1 0.368 → 0.463, hit@3 0.581 → 0.645, MRR@10 0.496 → 0.577; 114 improved, 47 worse. |
| `04_test_suites.png` (+ `.txt`) | Demo suite: 60 passed (41 original + 19 new for the rerank stage and LLM factory), the same 6 `test_main.py` errors as on upstream. Engine `test_embed_context.py`: 8 passed. |
| `05_demo_search_darkon_dachuf.png` | 04.10, w=0.3 index. *דרכון דחוף*, a held-out QA question: הוצאת דרכון זמני first (original system: rank 2 behind ביומטרי). Re-checked 05.10 on the shipped title-in-text index: same top three. |
| `06_rerank_off_arnona.png` | 05.10, shipped index, rerank **off** (the default). Held-out question *איזה מסמכים אני צריך לצרף בשביל לקבל הנחה בארנונה בגין נכות כללית* (gold: הנחה בארנונה לנכים): the sibling לנכי עבודה is first, the gold page second. |
| `07_rerank_on_arnona.png` | Same question, same index, a second backend instance with `TITLE_RERANK_ENABLED=true` (gpt-4o-mini, mock answer model, hence the raw mock text in the answer box): הנחה בארנונה לנכים first. Retrieval time includes the LLM call. |
| `08_three_eval_sets.png` (+ `.txt`) | hit@1 / hit@3 / MRR@10 on the three evaluation sets for the original system, the shipped change, rerank only, shipped + rerank, and the not-shipped w=0.3 title vector. |

## Jev title rerank (branch `jev-rerank-experiment`, 07.10)

Taken on the live backend (`localhost:5050`) with `TITLE_RERANK_PROVIDER=typesafe` (Jev Choice over 30 titles,
setup C1) and the real answering model. See `docs/JEV_RERANK.md`.

| file | what it shows |
|---|---|
| `jev_01_ui_urgent_passport.png` | *דרכון דחוף*: הוצאת דרכון זמני first; Jev lifted תעודות זהות, דרכונים ותעודות מעבר from #11 to #2. |
| `jev_02_ui_reserve_duty_school.png` | Held-out *אני במילואים האם ניתן להשהות את בני מבית הספר*: all three pages come from positions 14, 22, 23 of step 1. In this call the gold page was #2 (it was #1 in 7 of 8 live calls). |
| `jev_03_ui_unemployment_workdays.png` | Held-out *סופרים ימי עבודה או ימים רגילים בשביל לקבל אבטלה?*: gold תקופת אכשרה לדמי אבטלה lifted from #18 to #1. |
| `jev_04_before_after.png` | Step-1 top 5 vs top 5 after the Jev rerank for the two held-out questions, with Jev's probabilities (one call each). Rendered HTML of live results. |
| `jev_05_search_response_json.png` | Raw `/search` response for *דרכון דחוף* (content shortened) and the backend log lines showing the rerank ran via TypeSafeJev. Rendered HTML of real output. |
