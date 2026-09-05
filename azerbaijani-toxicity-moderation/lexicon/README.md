# Azerbaijani Bad-Word Lexicon

A database of unethical words and their deformations, plus the pipeline that
builds it from the labeled toxicity corpus in this project.

This is the deterministic half of moderation. The classifier in the parent
directory generalizes but cannot explain itself; this lexicon catches known
words with an audit trail and near-perfect precision. A moderation gate should
run both.

---

## The core idea

A word like `kitab` has hundreds of possible deformations. Storing each one as
a row does not scale, so deformations are split into two kinds.

**Mechanical deformations get no rows at all.** `normalize()` collapses them
onto one key. All of these become `kitab`:

```
kitab   k1tab   K I T A B   k.i.t.a.b   kkitab   kitaaab   k!t@b   кitab
```

It folds Azerbaijani letters (`ə→e ı→i ş→s ç→c ğ→g ö→o ü→u`), un-leets digits
and symbols, maps Cyrillic look-alikes, strips zero-width characters and
separators, and collapses repeated letters.

**Irregular deformations get rows**, because no normalization rule reaches
them: consonant swaps (`kitab → kigap`, `şərəfsiz → werefsiz`) and Azerbaijani
inflection (`peşi → peşiyəm → peşilər → peşisən`).

The result: **146 base words expand to 8,791 variations**, covering far more
input than 8,791 hand-written rows could.

## Tables

```
BadWords       (Id, Word, NormalizedWord, Category, Severity, IsActive, CreatedAt)
WordVariations (Id, BadWordId, Variation, NormalizedVariation, Source, IsActive, CreatedAt)
Whitelist      (Id, Phrase, NormalizedPhrase, Scope, Reason, CreatedAt)
```

**Uniqueness sits on the raw spelling, not the normalized key.** Many spellings
legitimately share one key -- `k1tab`, `kit4b` and `k1t4b` all resolve to
`kitab` -- so `Variation` is `UNIQUE` and `NormalizedVariation` carries a plain
index. The `LexiconLookup` view groups on the key, so a lookup still returns
exactly one row.

Beyond the brief, three additions, each load-bearing:

- **`NormalizedWord` / `NormalizedVariation`** — the column the index actually
  serves. Indexing the raw `Word` would defeat the whole scheme, since user
  input never arrives in canonical spelling.
- **`Severity`** (1-3) — lets the gate auto-block severe terms and route mild
  ones to review, instead of applying one blunt threshold.
- **`Whitelist`** — without it the filter blocks `götürmək` (to take) as a
  deformation of `göt`, and `şikayət` (complaint) as one of `sik`. Both were
  caught during the build; see *False-positive guards*.

`Category` values reuse the corpus label names (`insult`, `obscene`,
`identity_attack`, `sexual_explicit`, `threat`) so lexicon hits and classifier
scores line up in one moderation record.

## Indexing

B-tree indexes on every word column: `UNIQUE` on `BadWords.NormalizedWord` and
on `WordVariations.Variation`, plain indexes on `BadWords.Word` and
`WordVariations.NormalizedVariation` (plain, because many spellings share one
key). The `LexiconLookup` view groups both tables so the application issues one
query per token and gets back one row.

The brief allows "Index or Full-Text Index". **B-tree is the correct choice
here**: lookups are exact equality on a normalized key, answered in O(log n).
FULLTEXT tokenizes on word boundaries and would not match a deformation any
better, while MySQL's default `innodb_ft_min_token_size = 3` would silently
ignore short terms. FULLTEXT earns its place for a different job — re-scanning
a stored comment archive when new terms are added — and `sql/schema_mysql.sql`
documents that case. `sql/schema_postgres.sql` documents `pg_trgm` for fuzzy
"close to a known bad word" review queues.

## Creating the database

```bash
python build_database.py     # creates data/cothink_lexicon.db and verifies it
```

SQLite, because it needs no server -- so the structure is executed and measured
rather than only written down. `sql/schema_mysql.sql` and
`sql/schema_postgres.sql` are the same structure for a real server; load them
with `data/dist/seed.sql`.

The build script verifies three things rather than asserting them:

**Referential integrity.** Deleting a base word cascades to all of its variations
(rolled back afterwards), orphaned variations are 0, a duplicate spelling is
rejected, and duplicate normalized keys are allowed -- which is the design.

**The index is used.** `EXPLAIN QUERY PLAN` reports
`SEARCH WordVariations USING INDEX ... (NormalizedVariation=?)` -- a search, not
a table scan.

**The index earns its place.** 20,000 lookups over the 8,791-row variation
table, against an identical unindexed copy:

| | Total | Per lookup |
|---|---|---|
| With index | 0.076 s | 3.8 us |
| Without index | 6.739 s | 337.0 us |

**88x faster.** That is the brief's *"Axtarış surətini artırmaq üçün Index
tətbiq olunmalıdır"* requirement, measured.

Rerun `build_database.py` after any lexicon change; `test_database.py` fails if
the database drifts from the CSVs.

## Where the words come from

Not hand-authored: **mined from the 60k labeled training comments already in
this project**. `mine_lexicon.py` computes each token's log-odds of appearing
in a toxic versus a clean comment, producing 962 ranked candidates.

Those are *candidates*, not a lexicon. Correlation is not profanity — the
highest-scoring `identity_attack` terms were `erməni`, `türk`, `rus`,
`müsəlman`: **neutral demonyms**. Filtering those would censor ordinary speech
about nationality and religion, so they are whitelisted. `data/roots.csv` is
the reviewed output: 146 roots kept, categorized, with severity assigned.

## Variation sources

`WordVariations.Source` records how each row was obtained, so any entry can be
traced and re-reviewed:

| Source | Rows | Meaning |
|---|---|---|
| `mined` | 489 | Inflected form observed in the corpus - strongest evidence |
| `inflect` | 7,062 | Root + Azerbaijani suffix, generated |
| `leet` | 819 | Digit spelling of the base word (`s3r3fs1z`) |
| `rule` | 413 | Consonant substitution normalization cannot undo |
| `manual` | 8 | Hand-added from `data/manual_variations.csv` |

## False-positive guards

Three layers, because an over-blocking filter is worse than none:

1. **Corpus check.** Every generated variation is tested against the corpus. If
   it appears mostly in clean comments it is a collision, not a deformation,
   and is dropped — 79 were.
2. **Whitelist scope.** Each entry declares how far it reaches: `prefix` shields
   every word built on it (`götür` -> `götürdü`, `götürmək`), `exact` shields
   only itself. Scope is declared rather than inferred, because inferring it
   was wrong in both directions -- see below.
3. **Match-time guard.** Suffix stripping refuses to strip into a whitelisted
   stem, so `sikkəni` (of the coin) does not reduce to a vulgar root.

Why scope is declared rather than inferred: guessing it from word length
silently suppressed 64 real entries and hid two base words entirely.

- `sikkə` (coin) collapses onto the same key as one vulgar form, so treating it
  as a prefix hid the whole `sikərəm` family.
- `lənət` was whitelisted *and* a root — a contradiction that hid all 57 of its
  forms.
- Marking the neutral demonyms as prefixes then hid `rüsvay` (which starts with
  `rus`) and `ermənipərəst`.

Only entries that shield a genuine collision carry `prefix`; everything else is
`exact`, and the build raises on any other value. A reachability check —
every stored form must be findable by the matcher — now guards against this
whole class of bug.

## Results on the held-out test split (7,453 rows)

| | Precision | Recall | False positives |
|---|---|---|---|
| Lexicon | **0.948** | 0.171 | **35 / 3,719** |
| Baseline classifier | 0.796 | 0.781 | 748 / 3,719 |
| Lexicon OR classifier | 0.797 | 0.787 | 750 / 3,719 |

Exactly the expected shape: the lexicon knows only what it has been told, so
recall is low, but when it fires it is almost always right. That precision is
what makes it safe to **auto-block on**, while the classifier's noisier output
routes to human review.

### Evasion (the roadmap's open risk)

Re-running the same rows with leetspeak, inserted dots, doubled letters and
`ş→w` applied:

| | Recall clean | Recall obfuscated | Drop |
|---|---|---|---|
| Lexicon | 0.171 | 0.142 | -17% |
| Baseline classifier | 0.781 | 0.700 | -10% |

**81.7% of lexicon detections survive obfuscation** — the normalization layer
doing its job. This answers the roadmap item *"test the moderation model
against obfuscated variants before launch"*, previously untested.

### Unexpected result: the lexicon finds label errors

Inspecting the 22 "false positives" showed most were not filter errors but
**mislabeled rows** — comments containing `şərəfsiz`, `werefsizlerin`, `oğraş`
and `dığalar` labeled non-toxic. Across the full dataset the lexicon flags
**239 rows labeled non-toxic** across all three splits. `../audit_labels.py`
turns these into a review queue and repairs the train split only, so val and
test stay honest.

Measured effect of that repair: retraining the baseline on 167 corrected rows
moved macro F1 by **+0.0002** — nothing. 167 rows is 0.28% of the training
data, and a sample of 35 non-flagged clean-labeled rows turned up no further
errors, so the clean labels are genuinely in good shape. Label noise is not
what limits this classifier; the sparse categories are (`threat` F1 0.218 on
537 training positives).

## The recall ceiling

Recall is 0.171, and expanding the word list confirmed the ceiling rather
than breaking it. Doubling the lexicon (69 -> 146 roots, including Russian
obscenities common in Azerbaijani chat and vulgar terms the frequency threshold
had excluded) moved recall 0.136 -> 0.171 -- roughly **0.0005 per root**, and
precision slipped 0.959 -> 0.948 as the added terms got more marginal.

Sampling the toxic comments the lexicon misses shows why. The overwhelming
majority are labeled `insult` and contain no profanity at all:

- sarcasm and mockery — *"Allah rəhmət eləməsin"*, *"qaloş geyinib elə bil"*
- ethnic hostility assembled from neutral demonyms (`erməni`, `rus`), each of
  which is whitelisted for good reason
- implied threats — *"Onu sən özün basdırmaq istəyirsən?"*
- generalized contempt — *"Bu ölkə düzələn deyil"*

No word list reaches these. They are the classifier's job, and the division of
labour is the design, not a shortfall: the lexicon owns certainty, the
classifier owns coverage.

## Usage

```bash
python mine_lexicon.py     # corpus -> data/candidates.csv  (review by hand)
python build_lexicon.py    # roots.csv -> data/dist/*.csv
python export_sql.py       # -> data/dist/seed.sql
python -m unittest test_lexicon -v
python evaluate.py         # the metrics above
python match.py            # demo
```

Load into a database:

```bash
mysql cothink < sql/schema_mysql.sql
mysql cothink < data/dist/seed.sql
```

From an application:

```python
from match import LexiconFilter

lexicon = LexiconFilter()
result = lexicon.check("s3n w3r3fs1z")
# {'is_flagged': True, 'max_severity': 3, 'categories': ['insult'], ...}
```

**Any consumer must normalize input the same way before querying.** The stored
keys are normalized, so a raw `LIKE` query against `Word` will miss almost
everything. A PHP or SQL-only consumer needs a port of `normalize.py`; the
generated `inflect` rows exist so such a consumer still matches ordinary
inflections without the match-time suffix fallback.

## Integration

`../moderation_gate.py` combines this lexicon with the trained classifier into
a single three-way decision (block / review / allow), and
`../demo_moderated_qa_pipeline.py` shows it gating the Q&A flow. The gate blocks
on lexicon severity 3 or classifier toxicity >= 0.95 — the two bands measured at
~0.96 precision — and routes everything else weaker to human review.

## Files

```
normalize.py             shared normalization - the heart of the design
mine_lexicon.py          corpus -> ranked candidates for human review
build_lexicon.py         curated roots -> bad words + variations
export_sql.py            dist CSVs -> seed.sql
match.py                 runtime lookup (LexiconFilter)
evaluate.py              precision/recall, clean and obfuscated
test_lexicon.py          24 tests
data/roots.csv           curated base words  <- edit this
data/whitelist.csv       words that must never be flagged  <- edit this
data/manual_variations.csv  hand-added irregular forms  <- edit this
data/candidates.csv      mining output (generated)
data/dist/               build output (generated)
sql/schema_mysql.sql     tables, indexes, view, FULLTEXT notes
sql/schema_postgres.sql  same, plus pg_trgm notes
```

To add a word: append it to `data/roots.csv`, rerun `build_lexicon.py` and
`export_sql.py`, then run the tests.

## Limitations

- **Phrase-level insults are out of scope.** The corpus's most common insult
  pattern targets relatives (`anan…`, `bacını…`), but `ana`, `bacı` and `arvad`
  are innocent words. A word lexicon cannot express this without unacceptable
  false positives, so they are whitelisted and left to the classifier.
- **Threats are mostly phrase-level too** (`öldürəcəm səni`), and `öldürmək` is
  an ordinary verb. Only `gəbər` is included.
- **Azerbaijani only.** Turkish was scoped out; this corpus cannot supply it.
  Adding it means a separate source and a `Lang` column on both tables.
- **Low recall by design.** This is a precise complement to the classifier,
  never a replacement for it.
- **`sox` and `axmaq` have literal senses** (to insert; to flow). They are kept
  because corpus usage is overwhelmingly vulgar, but they remain the likeliest
  source of residual false positives.
- **Severity is a judgment call**, assigned during review rather than measured.
