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
  tune_thresholds.py                 Fits per-label thresholds -> thresholds.json
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
python -m unittest test_moderation_gate test_api        # 35 tests
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
- **Rare categories are weak** — `threat` F1 0.218, `severe_toxicity` F1 0.362.
  Analysed in detail below (in Azerbaijani): it is partly a modeling choice and
  partly a limit of the model class, not only data scarcity.
- **The classifier over-flags neutral demonyms.** "Mən erməni dilini öyrənirəm"
  (*I am learning Armenian*) scores 0.88 toxic. The lexicon whitelist keeps such
  text out of the block band, but it lands in review — which is the argument for
  keeping review a human step rather than tightening it into an auto-block.


---

# Nadir kateqoriyalar niyə zəifdir?

`threat` (F1 0.218) və `severe_toxicity` (F1 0.362) modelin ən zəif
nöqtələridir. Səbəb yalnız "data azdır" deyil — üç ayrı problem üst-üstə düşür.

## 1. Problem dəqiqlikdədir, əhatəlilikdə deyil

Model təhdidlərin təxminən yarısını tapır, lakin siqnal verdiyi hallarda
demək olar ki, həmişə yanılır:

| Etiket | Dəqiqlik (precision) | Əhatəlilik (recall) |
|---|---|---|
| `threat` | **0.141** | 0.475 |
| `severe_toxicity` | **0.243** | 0.710 |
| `insult` | 0.679 | 0.743 |

Bu mənzərə — yüksək əhatəlilik, çökmüş dəqiqlik — təsnifatçıdakı
`class_weight="balanced"` parametrinə işarə edir. Həmin parametr nadir
sinifləri süni şəkildə gücləndirir və qərar sərhədini "müsbət" tərəfə çəkir.
Müsbət nümunələrin cəmi 0.9% olduğu datasetdə bu, həddən artıq aqressivdir.

## 2. Hədd tənzimləməsi problemin bir hissəsini həll edir — TƏTBİQ EDİLİB

Hər etiket üçün ayrıca hədd seçmək real fayda verir. `tune_thresholds.py`
hədləri **yalnız validasiya** bölməsi üzərində seçir; test bölməsi nə hədd
seçmək, nə də hansı etiketin tənzimlənəcəyinə qərar vermək üçün istifadə
olunmur — əks halda aşağıdakı test nəticələri mənasız olardı.

Toxunulmamış test bölməsində nəticə:

| Etiket | Hədd | F1 (0.5) | F1 (tənzimlənmiş) | Fərq |
|---|---|---|---|---|
| `sexual_explicit` | 0.71 | 0.537 | **0.624** | +0.087 |
| `obscene` | 0.65 | 0.553 | **0.593** | +0.040 |
| `identity_attack` | 0.62 | 0.630 | **0.656** | +0.025 |
| `toxicity` | 0.39 | 0.789 | **0.801** | +0.013 |
| `insult` | 0.44 | 0.712 | **0.719** | +0.006 |
| `severe_toxicity` | 0.50 | 0.309 | 0.309 | — |
| `threat` | 0.50 | 0.251 | 0.251 | — |
| **macro F1** | | 0.540 | **0.565** | **+0.024** |

Sonradan təsnifatçının özü də yaxşılaşdırıldı (aşağıya bax) və hədlər yenidən
seçildi. Hazırkı vəziyyət: **macro F1 0.574**.

Müqayisə üçün: etiket təmizləməsi cəmi +0.0002 macro F1 vermişdi. Deməli
zəifliyin bir hissəsi həqiqətən **model qərarıdır, data məhdudiyyəti deyil**.

**Nadir etiketlər isə tənzimlənmir — bilərəkdən.** İki qoruyucu var, hər ikisi
yalnız validasiya məlumatına əsaslanır:

- `threat`: validasiyada cəmi 61 müsbət nümunə (minimum 150 tələb olunur).
  Sınaqda göründüyü kimi, 61 nümunə üzərində seçilmiş hədd test bölməsində
  nəticəni **pisləşdirirdi** (0.218 → 0.198) — yəni səs-küyə uyğunlaşırdı.
- `severe_toxicity`: 155 nümunə ilə həddi keçdi (0.87 seçildi), lakin
  validasiyanın ikiyə bölünmüş digər yarısında 0.5-ə uduzdu və qəbul edilmədi.

Hər ikisi öz-özünə imtina etdi — bu, metodologiyanın işlədiyinin göstəricisidir.

**Gate-də iki ayrı hədd dəsti var.** Bloklama/nəzərdən keçirmə qərarı
**dəqiqliyə** görə seçilir (yalnız ~0.96 dəqiqlikli siqnal bloklaya bilər),
kateqoriyanın qeydə yazılması isə **F1-ə** görə. Bunlar fərqli qərarlardır:
səhv bloklamanın qiyməti ilə səhv kateqoriya etiketinin qiyməti eyni deyil.

Gate səviyyəsində nəticə: **295 yanlış kateqoriya etiketi aradan qalxdı**
(2,867 → 2,572), üstəlik düzgün etiketlərin sayı da artdı (3,463 → 3,516).

## 3. Əsas səbəb: təhdid leksik deyil, qrammatik hadisədir

Datasetdə `threat` kimi etiketlənən cümlələrə baxaq:

> *"Vuran əllərin qurusun inşəallah"*
> *"Mən olsam sənə kicik gizir verərəm"*
> *"Qaçırdacam mən."*

Burada "təhdid lüğəti" yoxdur. Bütün korpusda təhdidə ən xas söz olan
`öldürərəm` cəmi **537 təhdiddən 8-ində** rast gəlinir. Ən güclü 12 marker
birlikdə yalnız **158/537** cümləni əhatə edir və onların çoxu tamamilə adi
sözlərdir: `səni`, `lazımdı`, `tutub`.

Təhdid bir **nitq aktıdır**: şərti quruluş, gələcək zaman və nəzərdə tutulan
hədəf. Model isə simvol n-qramları üzərində qurulmuş loqistik reqressiyadır.
O, *"əgər X, onda sənə Y edəcəyəm"* strukturunu təmsil edə bilmir — sadəcə
səthi fraqmentləri uyğunlaşdırır.

Bu, leksikonun əhatəlilik tavanı ilə eyni tapıntıdır, sadəcə bir səviyyə
yuxarıda: **çətin kateqoriyalar lüğətə deyil, kompozisiyaya əsaslanır.**

## Nəticələr

- ~~**Hədləri yenidən tənzimləmək**~~ — **edilib**: macro F1 +0.024
  (`tune_thresholds.py`, nəticələr `thresholds.json`-da).
- ~~**`class_weight="balanced"`-dan imtina etmək**~~ — **sınanıb və rədd
  edilib**: nəticə xeyli pisləşdi (macro F1 0.586 → 0.526). Parametr qalır.
- **`threat` və `severe_toxicity` üzrə avtomatik bloklamamaq.** 0.141
  dəqiqliklə hər 7 avtomatik blokdan 6-sı səhv olardı. Gate onsuz da bu
  kateqoriyaları yalnız məlumat xarakterli (`advisory`) sayır.
- **Transformer modeli burada həqiqətən kömək edərdi.** Etiket təmizləməsi
  (+0.0002 macro F1) və leksikonun genişləndirilməsi (hər kök üçün ~0.0005
  recall) ölçülüb — təsirləri cüzidir. XLM-R isə söz sırasını modelləşdirir və
  şərti strukturu təmsil edə bilir. GPU ilə fine-tune məhz burada özünü
  doğruldur.


## 4. Model konfiqurasiyası da yoxlanıldı

`experiments.py` altı konfiqurasiyanı təlim → validasiya üzərində müqayisə edir
(hər biri üçün hədlər ayrıca seçilir ki, müqayisə ədalətli olsun):

| Konfiqurasiya | Validasiya macro F1 |
|---|---|
| **char 2-6, 200k xüsusiyyət, C=4** | **0.5941** |
| char 2-5, C=4 | 0.5905 |
| char + söz birləşməsi, C=4 | 0.5883 |
| baza: char 2-5, C=1 | 0.5859 |
| char + söz, LinearSVC | 0.5828 |
| char 2-5, C=1, balanslaşdırma yoxdur | 0.5257 |

Qalib konfiqurasiya tətbiq edildi. Ən böyük fayda `severe_toxicity`-də oldu:
F1 0.310 → **0.378**.

**Vacib yan təsir:** model dəyişdikdə gate-in bloklama həddi öz dəqiqlik
zəmanətini itirdi — 0.95 həddində dəqiqlik 0.958-dən 0.944-ə düşdü. Ona görə
hədd 0.97-yə qaldırıldı (dəqiqlik 0.961). **Bloklama həddi sabit deyil,
zəmanətdir** — hər model dəyişikliyindən sonra yenidən ölçülməlidir.

## 5. Əsl tavan: etiketlərin özü ziddiyyətlidir

Datasetdə **eyni mətn** birdən çox dəfə rast gəlinir (552 qrup). Həmin eyni
mətnlərə annotatorlar nə qədər eyni etiket verib?

| Etiket | Müsbət olan qrup | Ziddiyyət | Uyğunluq |
|---|---|---|---|
| `toxicity` | 169 | 45 | **0.734** |
| `insult` | 126 | 51 | **0.595** |
| `obscene` | 58 | 26 | **0.552** |

**Eyni mətnə `insult` etiketi hallarının 40%-ində fərqli verilib.**

Bu, riyazi bir tavandır. Model `insult` üzrə F1 0.717 göstərir — annotatorların
öz aralarındakı uyğunluq isə ~0.60-dır. Yəni model artıq etiketlərin
razılaşdığı səviyyədədir. Annotatorların özlərinin iki fərqli qərar verdiyi
nümunə üçün heç bir model qayda öyrənə bilməz.

*(Qeyd: `identity_attack` üçün cəmi 10, `severe_toxicity` üçün 5 qrup var —
bu rəqəmlər etibarsızdır. `insult` və `toxicity` isə kifayət qədər böyükdür.)*

## Yekun: bal niyə aşağıdır və nə etmək olar

Ölçülmüş nəticələr:

| Addım | macro F1 | Fərq |
|---|---|---|
| Başlanğıc | 0.540 | — |
| + hədd tənzimləməsi | 0.565 | +0.025 |
| + daha yaxşı model | **0.574** | +0.009 |

Qalan boşluq **modeldə deyil, etiketlərdədir**. Ən çox fayda verəcək iş kod
yazmaq yox, **etiketləmə işidir**: `threat` və `severe_toxicity` üçün aydın
yazılı təlimat hazırlamaq və iki annotatorla yenidən etiketləmək. Transformer
(XLM-R) kompozisiyalı kateqoriyalarda kömək edər, lakin o da eyni tavanla
məhdudlaşır.
