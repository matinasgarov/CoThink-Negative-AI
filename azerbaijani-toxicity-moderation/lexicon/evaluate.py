"""Measure the lexicon on the held-out test split, and against obfuscation.

Two questions:

  1. How does a deterministic lexicon compare to the statistical baseline?
     Expected shape: much higher precision, much lower recall. The lexicon only
     knows words it has been told about; the classifier generalizes. They are
     complements, not competitors.

  2. What happens under obfuscation? The roadmap lists evasion as an open risk
     ("test the moderation model against obfuscated variants before launch").
     This is that test.

Run: python evaluate.py
"""

import csv
import os
import random
import sys

from match import LexiconFilter

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "..", "azerbaijani_toxicity_test.csv")

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}
random.seed(42)


def obfuscate(text):
    """Apply the evasion tricks a motivated user actually uses."""
    out = []
    for word in text.split(" "):
        roll = random.random()
        if roll < 0.30:
            word = "".join(LEET.get(c.lower(), c) for c in word)
        elif roll < 0.50 and len(word) > 3:
            i = random.randrange(1, len(word))
            word = word[:i] + "." + word[i:]
        elif roll < 0.65 and len(word) > 3:
            i = random.randrange(1, len(word))
            word = word[:i] + word[i] + word[i:]
        word = word.replace("ş", "w").replace("Ş", "W")
        out.append(word)
    return " ".join(out)


def score(name, predictions, truths):
    tp = sum(1 for p, t in zip(predictions, truths) if p and t)
    fp = sum(1 for p, t in zip(predictions, truths) if p and not t)
    fn = sum(1 for p, t in zip(predictions, truths) if not p and t)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    clean_total = sum(1 for t in truths if not t)
    print(f"{name:34s} precision={precision:.3f}  recall={recall:.3f}  "
          f"f1={f1:.3f}  false positives={fp}/{clean_total}")
    return precision, recall


def main():
    with open(TEST, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    texts = [r["comment"] for r in rows]
    truths = [r["toxicity"] == "1.0" for r in rows]
    print(f"test rows: {len(rows)}  (toxic: {sum(truths)})\n")

    lexicon = LexiconFilter()
    print(f"lexicon: {len(lexicon)} lookup forms\n")

    print("=== Clean text ===")
    lex_pred = [lexicon.check(t)["is_flagged"] for t in texts]
    score("lexicon", lex_pred, truths)

    baseline = None
    try:
        import joblib
        vec = joblib.load(os.path.join(HERE, "..", "baseline_vectorizer.joblib"))
        clf = joblib.load(os.path.join(HERE, "..", "baseline_model.joblib"))
        label_cols = ["identity_attack", "insult", "obscene", "severe_toxicity",
                      "sexual_explicit", "threat", "toxicity"]
        tox_i = label_cols.index("toxicity")
        base_pred = [bool(p) for p in clf.predict(vec.transform(texts))[:, tox_i]]
        score("baseline classifier", base_pred, truths)
        combined = [a or b for a, b in zip(lex_pred, base_pred)]
        score("lexicon OR classifier", combined, truths)
        baseline = (vec, clf, tox_i)
    except Exception as exc:  # noqa: BLE001 - model artifacts are optional
        print(f"(baseline classifier unavailable: {exc})")

    print("\n=== Obfuscated text (same rows, evasion applied) ===")
    obf = [obfuscate(t) for t in texts]
    lex_obf = [lexicon.check(t)["is_flagged"] for t in obf]
    _, lex_recall_obf = score("lexicon", lex_obf, truths)

    if baseline:
        vec, clf, tox_i = baseline
        base_obf = [bool(p) for p in clf.predict(vec.transform(obf))[:, tox_i]]
        _, base_recall_obf = score("baseline classifier", base_obf, truths)
        combined_obf = [a or b for a, b in zip(lex_obf, base_obf)]
        score("lexicon OR classifier", combined_obf, truths)

    toxic_only = [t for t, y in zip(texts, truths) if y]
    kept = sum(
        1 for t in toxic_only
        if lexicon.check(t)["is_flagged"] and lexicon.check(obfuscate(t))["is_flagged"]
    )
    caught = sum(1 for t in toxic_only if lexicon.check(t)["is_flagged"])
    if caught:
        print(f"\nlexicon detections surviving obfuscation: "
              f"{kept}/{caught} ({kept / caught * 100:.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
