# Screenshots — the change running, 04.10.2026

All taken on the full corpus (24,487 paragraphs in Elasticsearch 8.12.2) with the shipped
configuration (`embed_context_fields: ["title"]`, `embed_title_weight: 0.3`), Demo backend on
`localhost:5050`, real `gpt-4o-2024-08-06` as the answer model.

| file | what it shows |
|---|---|
| `01_demo_search_arnona_lenechim.png` | The Demo frontend answering *מי זכאי להנחה בארנונה לנכים?* (a free-text check typed for the screenshot, not a question from the QA set). Retrieved pages, in order: **הנחה בארנונה לנכים**, הנחה בארנונה לנכי עבודה, הנחה בארנונה למקבלי קצבת זיקנה לנכה: the three sibling pages, the asked-for one first. No baseline run of this exact wording was recorded, so it shows the shipped behaviour only; the before/after evidence is in `results/` and in screenshot 03. First query after a restart, so the 4.8 s retrieval time includes model warm-up. |
| `02_demo_search_kitzbat_zikna.png` | *מי זכאי לקצבת זקנה?*: קצבת זיקנה (קצבה אזרח ותיק), תנאי זכאות לקצבת זיקנה, גיל הזכאות לקצבת זיקנה. Retrieval 0.13 s on a warm model. |
| `03_eval_baseline_vs_shipped.png` (+ `.txt`) | `retrieval_eval/eval_retrieval.py` on the 296 held-out questions for `content` (baseline) and `fused_w0.3` (shipped), then `compare_runs.py`: hit@1 0.368 → 0.527, hit@3 0.581 → 0.709, MRR@10 0.496 → 0.636; 138 improved / 35 worse / 123 unchanged, with the largest per-question moves in both directions. |
| `04_test_suites.png` (+ `.txt`) | The Demo's own `pytest tests`: 41 passed, 6 errors in `test_main.py` (identical on upstream; the test patches `builtins.open` during import). The engine's new `test_embed_context.py`: 8 passed. |

Files 03 and 04 are the captured terminal output (the `.txt` next to each is the raw text),
rendered to an image so they can be viewed inline; nothing in them was edited.
