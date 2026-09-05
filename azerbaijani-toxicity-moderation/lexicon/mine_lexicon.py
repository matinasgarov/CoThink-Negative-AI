"""Mine bad-word candidates from the labeled toxicity corpus.

The base word list is not authored by hand -- it is derived from the 60k
labeled training comments already in this project. For every token we compute
the log-odds of appearing in a toxic comment versus a clean one, which surfaces
the vocabulary that actually distinguishes toxic text in this corpus.

The output is a *candidate* list, not a lexicon. Correlation is not profanity:
neutral demonyms and ordinary words co-occur with toxicity and score highly.
A human keeps or drops each row; the kept ones become data/roots.csv.

Run: python mine_lexicon.py
"""

import csv
import math
import os
from collections import Counter, defaultdict

from normalize import normalize_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "azerbaijani_toxicity_train.csv")
OUT = os.path.join(HERE, "data", "candidates.csv")

CATEGORY_COLS = [
    "identity_attack", "insult", "obscene",
    "severe_toxicity", "sexual_explicit", "threat",
]

MIN_TOXIC_COUNT = 10   # ignore terms too rare to judge
MIN_LOG_ODDS = 1.2     # ignore terms that barely lean toxic
MIN_LENGTH = 3         # 1-2 char tokens are too ambiguous to filter on


def load_counts(path):
    toxic, clean = Counter(), Counter()
    per_category = defaultdict(Counter)
    toxic_docs = clean_docs = 0

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tokens = {
                norm for _, norm in normalize_tokens(row["comment"])
                if len(norm) >= MIN_LENGTH
            }
            if row["toxicity"] == "1.0":
                toxic.update(tokens)
                toxic_docs += 1
                for col in CATEGORY_COLS:
                    if row[col] == "1.0":
                        per_category[col].update(tokens)
            else:
                clean.update(tokens)
                clean_docs += 1

    return toxic, clean, per_category, toxic_docs, clean_docs


def main():
    toxic, clean, per_category, toxic_docs, clean_docs = load_counts(TRAIN)
    print(f"toxic docs: {toxic_docs}   clean docs: {clean_docs}")
    print(f"vocabulary: {len(set(toxic) | set(clean))}")

    rows = []
    for word, tox_count in toxic.items():
        if tox_count < MIN_TOXIC_COUNT:
            continue
        clean_count = clean[word]
        # Smoothed log-odds ratio of P(word | toxic) against P(word | clean).
        log_odds = math.log(
            ((tox_count + 0.5) / (toxic_docs + 0.5))
            / ((clean_count + 0.5) / (clean_docs + 0.5))
        )
        if log_odds < MIN_LOG_ODDS:
            continue

        counts = {c: per_category[c][word] for c in CATEGORY_COLS}
        best = max(counts, key=counts.get)
        rows.append({
            "Candidate": word,
            "LogOdds": f"{log_odds:.3f}",
            "ToxicCount": tox_count,
            "CleanCount": clean_count,
            "SuggestedCategory": best if counts[best] else "",
            "CategoryCount": counts[best],
        })

    rows.sort(key=lambda r: float(r["LogOdds"]), reverse=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"candidates written: {len(rows)} -> {OUT}")
    print("\nReview these by hand. High score means 'correlates with toxicity',")
    print("not 'is a bad word' -- demonyms and common verbs rank high too.")


if __name__ == "__main__":
    main()
