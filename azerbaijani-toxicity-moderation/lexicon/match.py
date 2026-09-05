"""Runtime lookup against the lexicon.

This is what a backend calls. It loads the built CSVs into a dict, so a check
is a hash lookup per token; in production the same lookup is a single indexed
query against the LexiconLookup view in sql/.

The lexicon is a deterministic complement to the statistical classifier in this
project, not a replacement. It catches known words and their deformations with
certainty and an audit trail; the classifier catches phrasing the lexicon has
never seen. A moderation gate should run both.
"""

import csv
import os

from normalize import normalize, tokenize

# Suffixes stripped at match time as a fallback, for inflections that are
# neither in the corpus nor in the generated set. Kept to the unambiguous
# multi-character endings; short ones over-strip and cause false positives.
FALLBACK_SUFFIXES = [
    "lerimiz", "larimiz", "lerinin", "larinin", "lerin", "larin",
    "leri", "lari", "ler", "lar", "siniz", "sunuz", "siz", "suz",
    "sen", "san", "dir", "dur", "den", "dan", "nin", "nun",
]
MIN_STEM_LENGTH = 3

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "data", "dist")


class LexiconFilter:
    def __init__(self, dist_dir=DIST):
        self._entries = {}   # normalized form -> match metadata
        self._whitelist = set()
        self._load(dist_dir)

    def _load(self, dist_dir):
        with open(os.path.join(dist_dir, "whitelist.csv"), encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self._whitelist = {r["NormalizedPhrase"] for r in rows}
        self._whitelist_prefixes = {
            r["NormalizedPhrase"] for r in rows if r.get("Scope") == "prefix"
        }

        bad_words = {}
        with open(os.path.join(dist_dir, "bad_words.csv"), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                bad_words[row["Id"]] = row
                self._entries[row["NormalizedWord"]] = {
                    "word": row["Word"],
                    "category": row["Category"],
                    "severity": int(row["Severity"]),
                    "match_type": "base",
                }

        with open(os.path.join(dist_dir, "word_variations.csv"), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                parent = bad_words[row["BadWordId"]]
                self._entries[row["NormalizedVariation"]] = {
                    "word": parent["Word"],
                    "category": parent["Category"],
                    "severity": int(parent["Severity"]),
                    "match_type": "variation",
                }

    def __len__(self):
        return len(self._entries)

    def _is_whitelisted(self, key):
        """True if the key is whitelisted outright, or extends a prefix entry.

        Only entries marked Scope=prefix shield the words built on top of them.
        An exact entry shields itself alone -- 'sikkə' collapses onto the same
        key as a vulgar form, so shielding its whole prefix would suppress the
        entire 'sikərəm' family.
        """
        if key in self._whitelist:
            return True
        return any(key.startswith(safe) for safe in self._whitelist_prefixes)

    def _lookup(self, key):
        """Exact match, falling back to stripping Azerbaijani suffixes."""
        entry = self._entries.get(key)
        if entry:
            return entry, key
        stem = key
        for _ in range(2):  # suffixes stack: serefsiz-ler-in
            for suffix in FALLBACK_SUFFIXES:
                if not stem.endswith(suffix):
                    continue
                candidate = stem[: -len(suffix)]
                if len(candidate) < MIN_STEM_LENGTH:
                    continue
                entry = self._entries.get(candidate)
                if entry:
                    return entry, candidate
                stem = candidate
                break
            else:
                break
        return None, key

    def scan(self, text):
        """Return one entry per flagged token in the text."""
        found = []
        for raw, key in self._candidates(text):
            if not key or self._is_whitelisted(key):
                continue
            entry, matched = self._lookup(key)
            if entry:
                found.append({"token": raw, "normalized": matched, **entry})
        return found

    def _candidates(self, text):
        """Yield (raw, normalized) pairs, including de-spaced letter runs.

        Writing a word one letter at a time ('s e r e f s i z') defeats
        token-by-token matching, so runs of single-character tokens are also
        joined and offered as one candidate.
        """
        tokens = [(raw, normalize(raw)) for raw in tokenize(text)]
        for raw, key in tokens:
            yield raw, key

        run = []
        for raw, key in tokens + [("", "")]:
            if len(key) == 1:
                run.append((raw, key))
                continue
            if len(run) >= 3:
                yield " ".join(r for r, _ in run), "".join(k for _, k in run)
            run = []

    def check(self, text):
        """Summarize a scan into a moderation decision."""
        matches = self.scan(text)
        return {
            "is_flagged": bool(matches),
            "max_severity": max((m["severity"] for m in matches), default=0),
            "categories": sorted({m["category"] for m in matches}),
            "matches": matches,
        }


if __name__ == "__main__":
    lexicon = LexiconFilter()
    print(f"loaded {len(lexicon)} lookup forms\n")
    samples = [
        "Bu dərs çox faydalı idi, təşəkkür edirəm",
        "Kitabı götürmək istəyirəm",       # götürmək must not trip 'göt'
        "sen serefsizsen",
        "s3n w3r3fs1z",                    # obfuscated
        "Ş Ə R Ə F S İ Z",                 # spaced out
        "Mən erməni dilini öyrənirəm",     # demonym, must not flag
        "şikayət etmək istəyirəm",         # collides with 'sik' prefix
    ]
    for text in samples:
        result = lexicon.check(text)
        flag = "FLAG" if result["is_flagged"] else "  ok"
        detail = ""
        if result["is_flagged"]:
            hits = ", ".join(f"{m['token']}->{m['word']}" for m in result["matches"])
            detail = f"  sev={result['max_severity']} {result['categories']} [{hits}]"
        print(f"{flag}  {text!r}{detail}")
