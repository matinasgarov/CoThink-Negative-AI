# CoThink

A mentor-led learning platform for Azerbaijani-speaking students: mentors upload
videos, students ask an AI questions about them, and every piece of user text
passes a moderation gate before it reaches the AI or another student.

This repository currently contains the **moderation subsystem**, which is built
and measured. The video and Q&A pipeline is not built yet — see
[Project status](#project-status).

---

## What is here

```
ai-opportunities.md                  Early survey of where AI fits in CoThink
azerbaijani-toxicity-moderation/     The moderation subsystem
  Product_Build_Roadmap_Overview.pdf The 6-phase plan this work follows
  lexicon/                           Bad-word database (see its own README)
  moderation_gate.py                 Combines lexicon + classifier into one decision
  api.py                             HTTP service (/moderate)
  demo_moderated_qa_pipeline.py      End-to-end walkthrough with a stubbed LLM call
```

## The moderation design in one page

Two detectors with opposite failure modes, combined into one gate.

**The lexicon** ([lexicon/](azerbaijani-toxicity-moderation/lexicon/)) is a
database of bad words and their deformations. It is precise and explainable —
when it fires it names the exact word it matched — but it only knows words it
has been told about.

**The classifier** (TF-IDF + logistic regression, trained on ~75k labeled
Azerbaijani comments) generalizes to phrasing no word list covers, at the cost
of a much higher false-positive rate.

Measured on the held-out test split (7,453 rows):

| | Precision | Recall |
|---|---|---|
| Lexicon | 0.948 | 0.171 |
| Classifier | 0.796 | 0.781 |

Neither is good enough alone, and the numbers say why. Auto-blocking on the
classifier's default threshold would wrongly block roughly **one clean comment
in five**. So the gate emits three outcomes rather than a boolean, and only the
signals measured at ~0.96 precision are allowed to block:

| Signal | Precision | Action |
|---|---|---|
| Lexicon severity 3 | 0.957 | **block** |
| Classifier >= 0.95 | 0.958 | **block** |
| Classifier >= 0.50 | 0.796 | review |
| neither | — | allow |

What `review` means depends on who sees the text. A private question to the AI
is answered and logged; a public comment is held for a human. Showing abuse to
another student costs far more than a short delay.

Result on the test split: the block band holds **95.4% precision** across 433
blocked rows.

### Why the lexicon exists at all, given its low recall

Obfuscation. `s3n w3r3fs1z` scores only 0.76 on the classifier — below its block
threshold — but the lexicon blocks it outright, because normalization collapses
leetspeak, spacing, repeated letters and Cyrillic look-alikes onto one key.
**81.7% of lexicon detections survive obfuscation.** This was the roadmap's open
"evasion" risk, previously untested.

## Quick start

```bash
cd azerbaijani-toxicity-moderation

python moderation_gate.py            # see the gate decide on sample text
python demo_moderated_qa_pipeline.py # the full request path, LLM call stubbed
uvicorn api:app --reload             # HTTP service; docs at /docs
```

```bash
curl -X POST localhost:8000/moderate \
     -H "Content-Type: application/json" \
     -d '{"text": "s3n w3r3fs1z", "surface": "public"}'
```

## Tests

```bash
cd azerbaijani-toxicity-moderation
python -m unittest discover -s lexicon -p "test_*.py"   # 45 tests
python -m unittest test_moderation_gate test_api        # 29 tests
```

## Rebuilding from a clone

Model weights and derived datasets are not committed (4.2 GB of regenerable
build output). To rebuild everything from the committed source dataset:

```bash
cd azerbaijani-toxicity-moderation

python fix_label_conflicts.py    # dataset      -> _fixed
python split_dataset.py          # _fixed       -> train / val / test  (seed 42)
python baseline_train.py         # train        -> baseline_*.joblib

cd lexicon
python mine_lexicon.py           # corpus       -> candidates for review
python build_lexicon.py          # roots.csv    -> data/dist/*.csv
python export_sql.py             # dist         -> seed.sql
python build_database.py         # dist         -> cothink_lexicon.db
```

Every step is deterministic, so a rebuild reproduces the numbers above. The
gate degrades to lexicon-only if the `.joblib` artifacts are missing, rather
than failing — so it still blocks known words before you retrain.

The XLM-R fine-tune (`finetune_xlmr.py`) needs a GPU and is not part of the
rebuild.

## Project status

Against the roadmap in `Product_Build_Roadmap_Overview.pdf`:

| Phase | Status |
|---|---|
| 0 — Data foundation | Done |
| 1 — Moderation MVP | Done |
| 2 — Video ingestion | **Not started** — no transcription, chunking, or vector store |
| 3 — RAG Q&A | **Not started** — the demo's LLM call is a printed prompt |
| 4 — Unify moderation + Q&A | Gate, surface policy and audit log done; nothing to gate yet |
| 5 — Strengthen & scale | Per-category thresholds and evasion testing done; GPU fine-tune blocked |

### Known gaps

- **The product does not exist yet.** Moderation is the mature part of a
  platform with no video pipeline behind it.
- **The MySQL and Postgres schemas have never been run on a server.** The
  SQLite build is created and verified; the other two are untested DDL.
- **Rare categories are weak** — `threat` F1 0.218, `severe_toxicity` F1 0.362,
  on 537 and 1,168 training positives. This is data scarcity, not a fixable
  modeling detail. Label repair was tried and measured at +0.0002 macro F1.
- **The classifier over-flags neutral demonyms.** "Mən erməni dilini öyrənirəm"
  (*I am learning Armenian*) scores 0.88 toxic. The lexicon whitelist keeps such
  text out of the block band, but it lands in review — which is the argument for
  keeping review a human step rather than tightening it into an auto-block.
