"""Does repairing the labels actually improve the classifier?

Trains the same baseline twice -- once on the original train split, once on the
lexicon-repaired one -- and evaluates both on the UNTOUCHED validation split.
Because val was never modified, any difference is a real effect rather than the
lexicon marking its own homework.

Expect a small effect: 150 repaired rows out of 59,847 is 0.25% of the training
data. The point is to measure it rather than assume it.

Run: python compare_relabel.py
"""

import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL_COLS = [
    "identity_attack", "insult", "obscene",
    "severe_toxicity", "sexual_explicit", "threat", "toxicity",
]


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8", newline="") as f:
        texts, labels = [], []
        for row in csv.DictReader(f):
            texts.append(row["comment"])
            labels.append([int(float(row[c])) for c in LABEL_COLS])
    return texts, labels


def train_and_score(train_file, val_texts, val_labels):
    train_texts, train_labels = load(train_file)
    positives = sum(row[LABEL_COLS.index("toxicity")] for row in train_labels)
    print(f"  training on {train_file}  ({len(train_texts)} rows, "
          f"{positives} toxic)")

    # Identical hyperparameters to baseline_train.py, so the only thing that
    # differs between the two runs is the labels.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=100_000, sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_texts)
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
        n_jobs=-1,
    )
    clf.fit(x_train, train_labels)

    predictions = clf.predict(vectorizer.transform(val_texts))
    scores = {}
    for i, col in enumerate(LABEL_COLS):
        y_true = [row[i] for row in val_labels]
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, predictions[:, i], average="binary", zero_division=0
        )
        scores[col] = (p, r, f1)
    return scores


def main():
    val_texts, val_labels = load("azerbaijani_toxicity_val.csv")
    print(f"validation rows: {len(val_texts)} (untouched)\n")

    print("BEFORE (original labels)")
    before = train_and_score("azerbaijani_toxicity_train.csv", val_texts, val_labels)
    print("\nAFTER (lexicon-repaired labels)")
    after = train_and_score("azerbaijani_toxicity_train_relabeled.csv",
                            val_texts, val_labels)

    print("\n" + "=" * 72)
    print(f"{'label':17s} {'F1 before':>10} {'F1 after':>10} {'delta':>8}")
    print("=" * 72)
    for col in LABEL_COLS:
        f1_before, f1_after = before[col][2], after[col][2]
        delta = f1_after - f1_before
        marker = "  <-- " if abs(delta) >= 0.005 else ""
        print(f"{col:17s} {f1_before:10.4f} {f1_after:10.4f} "
              f"{delta:+8.4f}{marker}")

    macro_before = sum(v[2] for v in before.values()) / len(before)
    macro_after = sum(v[2] for v in after.values()) / len(after)
    print("-" * 72)
    print(f"{'macro F1':17s} {macro_before:10.4f} {macro_after:10.4f} "
          f"{macro_after - macro_before:+8.4f}")

    print("\ntoxicity detail:")
    for name, scores in (("before", before), ("after", after)):
        p, r, f1 = scores["toxicity"]
        print(f"  {name:6s} precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}")


if __name__ == "__main__":
    main()
