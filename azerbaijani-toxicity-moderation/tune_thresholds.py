"""Fit a per-label decision threshold for the toxicity classifier.

The baseline trains with class_weight="balanced", which counters class
imbalance by pushing the decision boundary toward positive. At a 0.9% base
rate that overshoots badly: `threat` reaches 0.475 recall at 0.141 precision.
A single 0.5 threshold across seven labels of wildly different frequency is
the wrong default.

Method, and why it is shaped this way:

  Thresholds are fitted on the validation split. Test is never consulted --
  not to pick a threshold, and not to decide *whether* to adopt one. Using
  test for either would make the reported test score meaningless.

  Which labels get tuned is therefore decided from validation alone, in two
  gates:

    1. Support. A label needs MIN_SUPPORT positives in validation. `threat`
       has 61, and a threshold fitted on 61 examples is fitting noise -- when
       tried, it degraded test F1 from 0.218 to 0.198.

    2. Generalization within validation. Validation is split in half; the
       threshold is fitted on one half and must beat 0.5 on the other. A
       threshold that cannot survive that is not adopted.

  Only labels clearing both gates get a fitted threshold; the rest keep 0.5.
  Test is scored once at the end, purely as an honest estimate.

Note this optimizes F1, which is the right objective for *reporting* a
category on a moderation record. It is NOT the right objective for deciding
to block -- that needs precision, and moderation_gate.py keeps its own,
higher, precision-driven thresholds for the block decision.

Run: python tune_thresholds.py
"""

import csv
import json
import os
import random

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thresholds.json")

LABEL_COLS = [
    "identity_attack", "insult", "obscene",
    "severe_toxicity", "sexual_explicit", "threat", "toxicity",
]

DEFAULT_THRESHOLD = 0.5
MIN_SUPPORT = 150           # validation positives needed before fitting
GRID = np.arange(0.20, 0.96, 0.01)
SEED = 42


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    texts = [r["comment"] for r in rows]
    labels = np.array([[int(float(r[c])) for c in LABEL_COLS] for r in rows])
    return texts, labels


def f1_at(y_true, scores, threshold):
    return precision_recall_fscore_support(
        y_true, (scores >= threshold).astype(int),
        average="binary", zero_division=0,
    )[2]


def best_threshold(y_true, scores):
    best = (0.0, DEFAULT_THRESHOLD)
    for t in GRID:
        f1 = f1_at(y_true, scores, t)
        if f1 > best[0]:
            best = (f1, round(float(t), 2))
    return best[1]


def main():
    # joblib unpickles; only ever point it at artifacts this project built.
    vectorizer = joblib.load(os.path.join(HERE, "baseline_vectorizer.joblib"))
    classifier = joblib.load(os.path.join(HERE, "baseline_model.joblib"))

    val_texts, val_y = load("azerbaijani_toxicity_val.csv")
    val_scores = classifier.predict_proba(vectorizer.transform(val_texts))

    # Deterministic half-split of validation, for the generalization gate.
    order = list(range(len(val_texts)))
    random.Random(SEED).shuffle(order)
    half = len(order) // 2
    fit_idx, check_idx = np.array(order[:half]), np.array(order[half:])

    decisions = {}
    print(f"{'label':17s} {'val+':>5} {'fitted':>7} {'adopted':>8}  reason")
    for i, label in enumerate(LABEL_COLS):
        y, s = val_y[:, i], val_scores[:, i]
        support = int(y.sum())

        if support < MIN_SUPPORT:
            decisions[label] = (DEFAULT_THRESHOLD, f"only {support} validation positives")
            print(f"{label:17s} {support:5d} {'-':>7} {DEFAULT_THRESHOLD:8.2f}  "
                  f"too few to fit (< {MIN_SUPPORT})")
            continue

        candidate = best_threshold(y[fit_idx], s[fit_idx])
        held = check_idx
        if f1_at(y[held], s[held], candidate) <= f1_at(y[held], s[held], DEFAULT_THRESHOLD):
            decisions[label] = (DEFAULT_THRESHOLD, "did not generalize within validation")
            print(f"{label:17s} {support:5d} {candidate:7.2f} {DEFAULT_THRESHOLD:8.2f}  "
                  f"lost to 0.5 on the held-out half")
            continue

        final = best_threshold(y, s)
        decisions[label] = (final, "fitted on validation")
        print(f"{label:17s} {support:5d} {candidate:7.2f} {final:8.2f}  adopted")

    # Test is touched only now, and only to report.
    test_texts, test_y = load("azerbaijani_toxicity_test.csv")
    test_scores = classifier.predict_proba(vectorizer.transform(test_texts))

    print(f"\nHonest estimate on the untouched test split:")
    print(f"{'label':17s} {'F1 @0.5':>8} {'F1 @tuned':>10} {'delta':>8}")
    total_before = total_after = 0.0
    for i, label in enumerate(LABEL_COLS):
        threshold = decisions[label][0]
        before = f1_at(test_y[:, i], test_scores[:, i], DEFAULT_THRESHOLD)
        after = f1_at(test_y[:, i], test_scores[:, i], threshold)
        total_before += before
        total_after += after
        mark = "  <--" if abs(after - before) >= 0.005 else ""
        print(f"{label:17s} {before:8.3f} {after:10.3f} {after - before:+8.3f}{mark}")
    n = len(LABEL_COLS)
    print(f"{'macro F1':17s} {total_before / n:8.3f} {total_after / n:10.3f} "
          f"{(total_after - total_before) / n:+8.3f}")

    payload = {
        "_comment": "Per-label thresholds for reporting a category. Fitted on "
                    "validation only; see tune_thresholds.py. The block/review "
                    "decision uses moderation_gate.py's own precision-driven "
                    "thresholds, not these.",
        "default": DEFAULT_THRESHOLD,
        "thresholds": {k: v[0] for k, v in decisions.items()},
        "reasons": {k: v[1] for k, v in decisions.items()},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nwritten to {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
