"""Search for a better classifier than the baseline, measured honestly.

The baseline is char_wb 2-5 TF-IDF + logistic regression at C=1 with
class_weight="balanced" -- reasonable defaults that were never revisited. This
tries the changes most likely to matter on short, morphologically rich,
misspelled text.

Method: every configuration is fitted on train and scored on validation, with
per-label thresholds fitted on validation too (otherwise a model that happens
to suit a 0.5 cutoff wins for the wrong reason). The winner is chosen on
validation. Test is scored once, at the end, for the winner only.

Run: python experiments.py
"""

import csv
import os
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_COLS = [
    "identity_attack", "insult", "obscene",
    "severe_toxicity", "sexual_explicit", "threat", "toxicity",
]
GRID = np.arange(0.20, 0.96, 0.02)


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return ([r["comment"] for r in rows],
            np.array([[int(float(r[c])) for c in LABEL_COLS] for r in rows]))


def char_features(max_features=100_000, ngram=(2, 5)):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2,
                           max_features=max_features, sublinear_tf=True)


def word_features(max_features=60_000, ngram=(1, 2)):
    return TfidfVectorizer(analyzer="word", ngram_range=ngram, min_df=2,
                           max_features=max_features, sublinear_tf=True)


def union():
    return FeatureUnion([("char", char_features()), ("word", word_features())])


def logreg(C=1.0, balanced=True):
    return OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=C,
                           class_weight="balanced" if balanced else None),
        n_jobs=-1)


def linear_svc(C=0.5, balanced=True):
    # LinearSVC has no predict_proba; calibration supplies one so the same
    # threshold search applies to every configuration.
    return OneVsRestClassifier(
        CalibratedClassifierCV(
            LinearSVC(C=C, class_weight="balanced" if balanced else None),
            cv=3, method="sigmoid"),
        n_jobs=-1)


CONFIGS = [
    ("baseline: char 2-5, LogReg C=1",        char_features, lambda: logreg(1.0)),
    ("char 2-5, LogReg C=4",                  char_features, lambda: logreg(4.0)),
    ("char 2-5, LogReg C=1, no balancing",    char_features, lambda: logreg(1.0, False)),
    ("char 2-6, 200k feats, LogReg C=4",
     lambda: char_features(200_000, (2, 6)),  lambda: logreg(4.0)),
    ("char + word union, LogReg C=4",         union,        lambda: logreg(4.0)),
    ("char + word union, LinearSVC C=0.5",    union,        lambda: linear_svc(0.5)),
]


def tuned_macro_f1(y_true, scores):
    """Macro F1 with a per-label threshold fitted on this same split.

    Fitting thresholds here would flatter every model equally, so it is fair
    for ranking. The winner's honest score is measured on test afterwards.
    """
    per_label, thresholds = [], []
    for i in range(len(LABEL_COLS)):
        best = (0.0, 0.5)
        for t in GRID:
            f1 = precision_recall_fscore_support(
                y_true[:, i], (scores[:, i] >= t).astype(int),
                average="binary", zero_division=0)[2]
            if f1 > best[0]:
                best = (f1, float(t))
        per_label.append(best[0])
        thresholds.append(round(best[1], 2))
    return float(np.mean(per_label)), per_label, thresholds


def main():
    train_texts, train_y = load("azerbaijani_toxicity_train.csv")
    val_texts, val_y = load("azerbaijani_toxicity_val.csv")
    print(f"train {len(train_texts)}, val {len(val_texts)}\n")

    results = []
    for name, make_features, make_model in CONFIGS:
        started = time.perf_counter()
        features = make_features()
        x_train = features.fit_transform(train_texts)
        model = make_model()
        model.fit(x_train, train_y)
        scores = model.predict_proba(features.transform(val_texts))
        macro, per_label, thresholds = tuned_macro_f1(val_y, scores)
        results.append((macro, name, features, model, per_label, thresholds))
        print(f"{macro:.4f}  {name:38s} ({x_train.shape[1]:,} feats, "
              f"{time.perf_counter() - started:.0f}s)")

    results.sort(key=lambda r: -r[0])
    macro, name, features, model, per_label, thresholds = results[0]
    print(f"\nbest on validation: {name}  (macro F1 {macro:.4f})")

    test_texts, test_y = load("azerbaijani_toxicity_test.csv")
    scores = model.predict_proba(features.transform(test_texts))
    print("\nTest split, using the validation-fitted thresholds:")
    print(f"{'label':17s} {'thr':>5} {'prec':>6} {'recall':>7} {'f1':>6}")
    f1s = []
    for i, label in enumerate(LABEL_COLS):
        p, r, f1, _ = precision_recall_fscore_support(
            test_y[:, i], (scores[:, i] >= thresholds[i]).astype(int),
            average="binary", zero_division=0)
        f1s.append(f1)
        print(f"{label:17s} {thresholds[i]:5.2f} {p:6.3f} {r:7.3f} {f1:6.3f}")
    print(f"{'macro F1':17s} {'':5s} {'':6s} {'':7s} {np.mean(f1s):6.3f}")
    print("\nBaseline for comparison: macro F1 0.565 on the same split.")


if __name__ == "__main__":
    main()
