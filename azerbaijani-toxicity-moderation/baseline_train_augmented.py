import csv

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier

LABEL_COLS = [
    "identity_attack",
    "insult",
    "obscene",
    "severe_toxicity",
    "sexual_explicit",
    "threat",
    "toxicity",
]


def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        texts, labels = [], []
        for row in reader:
            texts.append(row["comment"])
            labels.append([int(float(row[c])) for c in LABEL_COLS])
    return texts, labels


train_texts, train_labels = load("azerbaijani_toxicity_train_augmented.csv")
val_texts, val_labels = load("azerbaijani_toxicity_val.csv")  # untouched, same as baseline eval

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=2,
    max_features=100_000,
    sublinear_tf=True,
)
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

clf = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
    n_jobs=-1,
)
clf.fit(X_train, train_labels)

val_pred = clf.predict(X_val)

print("=== Per-label results (threshold=0.5), trained on AUGMENTED data ===\n")
for i, col in enumerate(LABEL_COLS):
    y_true = [row[i] for row in val_labels]
    y_pred = val_pred[:, i]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    pos_rate = sum(y_true) / len(y_true)
    print(f"{col:16s} pos_rate={pos_rate:.3f}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")

print("\n=== Detailed report: toxicity ===")
tox_i = LABEL_COLS.index("toxicity")
y_true = [row[tox_i] for row in val_labels]
y_pred = val_pred[:, tox_i]
print(classification_report(y_true, y_pred, target_names=["not_toxic", "toxic"], zero_division=0))

joblib.dump(vectorizer, "baseline_vectorizer_augmented.joblib")
joblib.dump(clf, "baseline_model_augmented.joblib")
print("Saved model artifacts: baseline_vectorizer_augmented.joblib, baseline_model_augmented.joblib")
