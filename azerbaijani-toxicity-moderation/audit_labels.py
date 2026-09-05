"""Find and repair mislabeled rows using the bad-word lexicon.

Rows containing an unambiguous slur but labeled toxicity=0 are label errors.
The lexicon finds them cheaply and, unlike the classifier, names the exact word
it matched -- so every proposed change carries its own evidence and can be
checked by a human in seconds.

This is the same kind of repair as fix_label_conflicts.py, from a different
angle: that script reconciles near-duplicate rows that disagree with each
other, this one catches rows that disagree with the lexicon.

Two deliberate constraints:

  Only train is repaired. Val and test are left untouched, so any measured
  improvement is real rather than the lexicon grading its own work. Candidates
  found in val/test are still reported, flagged as NOT applied.

  Only severity >= 2 is auto-applied. Measured on the test split, the lexicon's
  precision by severity band is 0.900 (sev 1, n=10), 0.958 (sev 2, n=236) and
  0.960 (sev 3, n=250). Severity 1 is both the least precise and the thinnest
  evidence, so it goes to the review queue instead.

Outputs:
  label_review_candidates.csv          every candidate, with evidence
  azerbaijani_toxicity_train_relabeled.csv   train with repairs applied

Run: python audit_labels.py
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lexicon"))

from match import LexiconFilter  # noqa: E402

SPLITS = ["train", "val", "test"]
REPAIRED_SPLIT = "train"
AUTO_APPLY_MIN_SEVERITY = 2

REVIEW_QUEUE = os.path.join(HERE, "label_review_candidates.csv")
RELABELED = os.path.join(HERE, "azerbaijani_toxicity_train_relabeled.csv")

LABEL_COLS = [
    "identity_attack", "insult", "obscene",
    "severe_toxicity", "sexual_explicit", "threat", "toxicity",
]


def split_path(split):
    return os.path.join(HERE, f"azerbaijani_toxicity_{split}.csv")


def find_candidates(lexicon, split):
    """Rows labeled non-toxic that the lexicon flags."""
    with open(split_path(split), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    for index, row in enumerate(rows):
        if row["toxicity"] == "1.0":
            continue
        decision = lexicon.check(row["comment"])
        if not decision["is_flagged"]:
            continue
        candidates.append({
            "row_index": index,
            "row": row,
            "severity": decision["max_severity"],
            "words": sorted({m["word"] for m in decision["matches"]}),
            "categories": decision["categories"],
        })
    return rows, candidates


def main():
    lexicon = LexiconFilter()
    print(f"lexicon: {len(lexicon)} lookup forms\n")

    queue = []
    repaired_rows = None
    applied = 0

    for split in SPLITS:
        rows, candidates = find_candidates(lexicon, split)
        auto = [c for c in candidates
                if c["severity"] >= AUTO_APPLY_MIN_SEVERITY and split == REPAIRED_SPLIT]

        print(f"{split:6s} candidates={len(candidates):4d}  "
              f"auto-applied={len(auto):4d}  "
              f"review-only={len(candidates) - len(auto):4d}")

        for c in candidates:
            will_apply = c in auto
            queue.append({
                "Split": split,
                "RowIndex": c["row_index"],
                "Comment": c["row"]["comment"],
                "MatchedWords": ", ".join(c["words"]),
                "Severity": c["severity"],
                "LexiconCategories": ", ".join(c["categories"]),
                "CurrentToxicity": c["row"]["toxicity"],
                "ProposedToxicity": "1.0",
                "Applied": "yes" if will_apply else "no",
                "WhyNotApplied": (
                    "" if will_apply
                    else "val/test held out" if split != REPAIRED_SPLIT
                    else f"severity {c['severity']} below auto-apply threshold"
                ),
            })

        if split == REPAIRED_SPLIT:
            for c in auto:
                row = rows[c["row_index"]]
                row["toxicity"] = "1.0"
                # Also set the categories the lexicon identified, so the
                # per-category labels stay consistent with the toxicity flag.
                for category in c["categories"]:
                    if category in row:
                        row[category] = "1.0"
                applied += 1
            repaired_rows = rows

    with open(REVIEW_QUEUE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(queue[0].keys()))
        writer.writeheader()
        writer.writerows(queue)

    with open(RELABELED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["comment"] + LABEL_COLS)
        writer.writeheader()
        writer.writerows(repaired_rows)

    print(f"\nreview queue: {len(queue)} rows -> {os.path.basename(REVIEW_QUEUE)}")
    print(f"train repairs applied: {applied} -> {os.path.basename(RELABELED)}")
    print("\nval/test were NOT modified, so evaluation stays honest.")


if __name__ == "__main__":
    main()
