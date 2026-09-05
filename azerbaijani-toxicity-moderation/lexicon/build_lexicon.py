"""Build the distributable lexicon from curated roots.

Three sources feed WordVariations, in descending order of trust:

  mined  -- inflected forms of the root that actually occur in the corpus.
            Azerbaijani is agglutinative, so a single root shows up as
            peşi / peşiyəm / peşilər / peşisən. Taking these from real data
            beats generating them from a suffix table.
  rule   -- irregular consonant substitutions that normalization cannot undo
            (b<->p, k<->g, ş typed as w, final k voiced to y). This is the
            kitab -> kigap class of deformation.
  inflect -- root plus common Azerbaijani suffixes, generated. These make the
            database self-sufficient: a consumer querying only SQL still
            matches ordinary inflections the corpus happened not to contain.
  leet   -- digit and symbol spellings of the base word (k1tab, s3r3fs1z).
            Normalization already collapses these, so these rows add no
            matching power -- they are stored so the deformation an operator
            expects to find is actually in the table, and so the brief's own
            example (kitab -> k1tab) holds literally. Every leet row must
            normalize back to its parent's key, which is asserted at build time.
  manual -- hand-added forms from data/manual_variations.csv.

Purely mechanical deformations (k1tab, k.i.t.a.b, kkitab, Cyrillic homoglyphs)
get NO rows: normalize() already collapses them onto the base key.

Every generated variation is checked against the corpus before it is kept. If a
candidate form appears mostly in clean comments, it is dropped -- that is the
guard that stops 'götürmək' being filed as a deformation of 'göt'.

Run: python build_lexicon.py
"""

import csv
import itertools
import os
from collections import Counter

from normalize import normalize, normalize_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "azerbaijani_toxicity_train.csv")
DATA = os.path.join(HERE, "data")
DIST = os.path.join(DATA, "dist")

MAX_SUFFIX_LEN = 7      # how far past the root an inflected form may run
MIN_MINED_COUNT = 2     # ignore one-off typos in the corpus
CLEAN_DROP_RATIO = 1.0  # drop a form appearing at least this often in clean text

# Substitutions that survive normalization and therefore need their own rows.
# Common Azerbaijani noun/verb suffixes, in folded (post-normalize) form.
# Deliberately excludes 1-character and highly ambiguous endings, which
# over-generate and collide with unrelated words.
SUFFIXES = [
    "lerimiz", "larimiz", "lerinin", "larinin", "leriniz", "lariniz",
    "lerine", "larina", "lerin", "larin", "lerde", "larda", "leri", "lari",
    "ler", "lar", "siniz", "sunuz", "siz", "suz", "sen", "san",
    "dirler", "dirlar", "diler", "dilar", "dir", "dur", "din", "dun",
    "den", "dan", "de", "da", "nin", "nun", "ni", "nu", "yem", "yam",
    "em", "am", "im", "um", "in", "un", "ik", "iq", "li", "lu", "siz",
]

# Digit/symbol stand-ins, mirroring the reverse mapping in normalize.py.
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "g": "9", "b": "8"}

SUBSTITUTIONS = [
    ("b", "p"), ("p", "b"), ("d", "t"), ("t", "d"),
    ("k", "g"), ("g", "k"), ("g", "q"), ("q", "g"),
    ("v", "f"), ("f", "v"), ("z", "s"), ("s", "z"),
    ("s", "w"), ("c", "j"), ("x", "h"), ("h", "x"),
]


def read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def corpus_counts():
    """Normalized token -> (toxic count, clean count) across the train split."""
    toxic, clean = Counter(), Counter()
    with open(TRAIN, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tokens = {norm for _, norm in normalize_tokens(row["comment"])}
            (toxic if row["toxicity"] == "1.0" else clean).update(tokens)
    return toxic, clean


def build_whitelist():
    """Whitelist entries, each carrying how it matches.

    Scope is explicit rather than inferred, because inferring it was wrong in
    both directions: 'götür' must shield its whole inflection family, while
    'sikkə' (coin) collapses to the same key as one vulgar form and must shield
    only itself -- guessing from length silently suppressed 64 real entries.
    """
    entries = []
    for row in read_csv("whitelist.csv"):
        scope = (row.get("Scope") or "exact").strip()
        if scope not in ("exact", "prefix"):
            raise ValueError(f"whitelist.csv: bad Scope {scope!r} for {row['Phrase']!r}")
        entries.append({
            "Phrase": row["Phrase"],
            "NormalizedPhrase": normalize(row["Phrase"]),
            "Scope": scope,
            "Reason": row["Reason"],
        })
    return entries


def is_blocked(candidate, root_norm, guard):
    """True if a candidate variation must not be attached to this root.

    `guard` is (exact_forms, prefix_forms) from split_whitelist().
    """
    exact, prefixes = guard
    if candidate in exact or candidate in prefixes:
        return True
    return any(
        len(p) > len(root_norm) and candidate.startswith(p) for p in prefixes
    )


def split_whitelist(whitelist):
    exact = {w["NormalizedPhrase"] for w in whitelist if w["Scope"] == "exact"}
    prefixes = {w["NormalizedPhrase"] for w in whitelist if w["Scope"] == "prefix"}
    return exact, prefixes


# Combinations of substitutions grow as 2^k, so cap per word. Subsets are
# generated smallest-first, because people substitute a letter or two
# ('s3r3fs1z'), not every letter at once.
MAX_LEET_PER_WORD = 24


def leet_variations(root_norm):
    """Digit spellings of the root, over every combination of substitutable
    letters. Each must still contain a letter, or normalize() reads it as junk.
    """
    letters = [c for c in LEET if c in root_norm]
    forms = []
    for size in range(1, len(letters) + 1):
        for subset in itertools.combinations(letters, size):
            form = root_norm
            for letter in subset:
                form = form.replace(letter, LEET[letter])
            if form != root_norm and any(c.isalpha() for c in form):
                forms.append(form)
            if len(forms) >= MAX_LEET_PER_WORD:
                return forms
    return forms


def rule_variations(root_norm):
    """Apply one consonant substitution at a time to the normalized root.

    Results are re-normalized, since a substitution can produce a doubled
    letter that normalize() collapses.
    """
    seen = []
    for src, dst in SUBSTITUTIONS:
        if src not in root_norm:
            continue
        # Replace every occurrence, and (if there are several) the first only.
        for raw in {root_norm.replace(src, dst),
                    root_norm.replace(src, dst, 1)}:
            form = normalize(raw)
            if form and form != root_norm and form not in seen:
                seen.append(form)
    if root_norm.endswith("k"):
        seen.append(normalize(root_norm[:-1] + "y"))
    return seen


def main():
    roots = read_csv("roots.csv")
    manual = read_csv("manual_variations.csv")
    whitelist = build_whitelist()
    guard = split_whitelist(whitelist)

    toxic, clean = corpus_counts()
    vocabulary = set(toxic) | set(clean)

    bad_words = []
    variations = []
    taken = set()          # every normalized form already claimed
    leet_spellings = set()  # raw leet spellings, which may repeat a taken key
    stats = Counter()

    for index, row in enumerate(roots, start=1):
        word = row["Word"]
        root_norm = normalize(word)
        if root_norm in taken:
            print(f"  ! duplicate root skipped: {word}")
            continue
        taken.add(root_norm)

        bad_words.append({
            "Id": index,
            "Word": word,
            "NormalizedWord": root_norm,
            "Category": row["Category"],
            "Severity": row["Severity"],
        })

        # -- mined: real inflected forms from the corpus -------------------
        for token in vocabulary:
            if token == root_norm or not token.startswith(root_norm):
                continue
            if len(token) - len(root_norm) > MAX_SUFFIX_LEN:
                continue
            if toxic[token] + clean[token] < MIN_MINED_COUNT:
                continue
            if clean[token] >= max(CLEAN_DROP_RATIO, toxic[token]):
                stats["mined_dropped_clean"] += 1
                continue
            if token in taken or is_blocked(token, root_norm, guard):
                continue
            taken.add(token)
            variations.append({
                "BadWordId": index, "Variation": token,
                "NormalizedVariation": token, "Source": "mined",
            })
            stats["mined"] += 1

        # -- inflect: generated Azerbaijani inflections ---------------------
        for suffix in SUFFIXES:
            # Re-normalize: concatenation can create doubled letters that
            # normalize() would collapse, e.g. ogras + siniz -> ograsiniz.
            form = normalize(root_norm + suffix)
            if form in taken or is_blocked(form, root_norm, guard):
                continue
            # If the generated form is a real word that lives in clean text,
            # it is a collision, not an inflection of this root.
            if clean[form] >= max(CLEAN_DROP_RATIO, toxic[form]):
                stats["inflect_dropped_clean"] += 1
                continue
            taken.add(form)
            variations.append({
                "BadWordId": index, "Variation": form,
                "NormalizedVariation": form, "Source": "inflect",
            })
            stats["inflect"] += 1

        # -- leet: digit spellings of this root ----------------------------
        # These deliberately share the root's normalized key, so they are
        # exempt from the `taken` check -- but only for their own parent.
        for form in leet_variations(root_norm):
            if normalize(form) != root_norm:
                raise AssertionError(
                    f"leet form {form!r} normalizes to {normalize(form)!r}, "
                    f"not {root_norm!r}"
                )
            if form in leet_spellings:
                continue
            leet_spellings.add(form)
            variations.append({
                "BadWordId": index, "Variation": form,
                "NormalizedVariation": root_norm, "Source": "leet",
            })
            stats["leet"] += 1

        # -- rule: substitutions normalization cannot undo -----------------
        for form in rule_variations(root_norm):
            if form in taken or is_blocked(form, root_norm, guard):
                continue
            # A generated form that is a common clean word is a false positive.
            if clean[form] >= max(CLEAN_DROP_RATIO, toxic[form]):
                stats["rule_dropped_clean"] += 1
                continue
            taken.add(form)
            variations.append({
                "BadWordId": index, "Variation": form,
                "NormalizedVariation": form, "Source": "rule",
            })
            stats["rule"] += 1

    # -- manual: hand-added irregular forms --------------------------------
    by_norm = {b["NormalizedWord"]: b["Id"] for b in bad_words}
    for row in manual:
        parent = by_norm.get(normalize(row["Word"]))
        if parent is None:
            print(f"  ! manual variation has no root: {row['Word']}")
            continue
        form = normalize(row["Variation"])
        if not form or form in taken:
            continue
        taken.add(form)
        variations.append({
            "BadWordId": parent, "Variation": row["Variation"],
            "NormalizedVariation": form, "Source": "manual",
        })
        stats["manual"] += 1

    for i, v in enumerate(variations, start=1):
        v["Id"] = i

    os.makedirs(DIST, exist_ok=True)
    write(os.path.join(DIST, "bad_words.csv"),
          ["Id", "Word", "NormalizedWord", "Category", "Severity"], bad_words)
    write(os.path.join(DIST, "word_variations.csv"),
          ["Id", "BadWordId", "Variation", "NormalizedVariation", "Source"],
          variations)
    write(os.path.join(DIST, "whitelist.csv"),
          ["Phrase", "NormalizedPhrase", "Scope", "Reason"], whitelist)

    print(f"bad words:  {len(bad_words)}")
    print(f"variations: {len(variations)}"
          f"  (mined {stats['mined']}, inflect {stats['inflect']}, "
          f"rule {stats['rule']}, leet {stats['leet']}, manual {stats['manual']})")
    print(f"whitelist:  {len(whitelist)}")
    dropped = (stats["mined_dropped_clean"] + stats["rule_dropped_clean"]
               + stats["inflect_dropped_clean"])
    print(f"dropped as clean-leaning: {dropped}")
    print(f"written to {DIST}")


def write(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
