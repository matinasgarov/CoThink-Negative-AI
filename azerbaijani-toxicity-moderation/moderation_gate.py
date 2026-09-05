"""The moderation gate: one decision from two complementary signals.

The lexicon and the classifier fail in opposite directions, and the gate is
where that becomes useful rather than merely true:

  lexicon     precision 0.948, recall 0.171 -- only knows words it was told,
              but is almost always right when it fires, and says which word.
  classifier  precision 0.796, recall 0.781 -- generalizes to phrasing no
              lexicon covers, at the cost of ~750 false positives per 3.7k
              clean comments.

Auto-blocking on the classifier's default 0.5 threshold would wrongly block
roughly one clean comment in five. So the gate emits three outcomes, not two,
and only the high-precision signals are allowed to block outright.

Two different threshold sets, for two different jobs. The block/review routing
below optimizes precision; the per-category thresholds in thresholds.json
optimize F1, because naming a category on a record is a different decision from
refusing to publish something.

Routing thresholds, measured on the held-out test split, not guessed:

    threshold   precision   recall
      0.50        0.796      0.781   <- too imprecise to block on
      0.80        0.918      0.344
      0.95        0.958      0.067   <- matches lexicon severity-3 precision
    lexicon sev 3 0.957      n/a

Re-measure with evaluate.py and compare_relabel.py after changing the lexicon.

Usage:
    gate = ModerationGate.load()
    decision = gate.moderate(text)
    if decision["action"] == "block":
        ...
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lexicon"))

from match import LexiconFilter  # noqa: E402

LABEL_COLS = [
    "identity_attack", "insult", "obscene",
    "severe_toxicity", "sexual_explicit", "threat", "toxicity",
]
TOXICITY_IDX = LABEL_COLS.index("toxicity")

# Routing thresholds for the block/review decision. These are chosen for
# PRECISION -- only signals measured at ~0.96 may block -- and are deliberately
# not the F1-optimal values, because the cost of a wrong block is not symmetric
# with the cost of a miss.
CLASSIFIER_BLOCK_THRESHOLD = 0.95
CLASSIFIER_REVIEW_THRESHOLD = 0.50
LEXICON_BLOCK_SEVERITY = 3

# Per-label thresholds for *reporting* a category on the record, which is a
# different objective: there F1 is right, and one 0.5 cutoff across seven
# labels of very different frequency is not. Fitted by tune_thresholds.py on
# validation only; labels with too few examples keep 0.5 on purpose.
CATEGORY_THRESHOLDS_FILE = os.path.join(HERE, "thresholds.json")


def _load_category_thresholds():
    try:
        with open(CATEGORY_THRESHOLDS_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        # Missing or unreadable: fall back to one cutoff for every label.
        return {label: CLASSIFIER_REVIEW_THRESHOLD for label in LABEL_COLS}
    default = payload.get("default", CLASSIFIER_REVIEW_THRESHOLD)
    fitted = payload.get("thresholds", {})
    return {label: float(fitted.get(label, default)) for label in LABEL_COLS}


CATEGORY_THRESHOLDS = _load_category_thresholds()

# Categories the roadmap flags as too sparse to act on alone (537 threat and
# 1,168 severe_toxicity positives in training). They annotate a decision; they
# never drive one.
ADVISORY_ONLY_CATEGORIES = {"threat", "severe_toxicity"}

BLOCK, REVIEW, ALLOW = "block", "review", "allow"

# Surfaces differ in who sees the text, which changes what `review` should do.
PRIVATE, PUBLIC = "private", "public"


def apply_policy(decision, surface):
    """Map a gate decision onto product behaviour for a given surface.

    Kept separate from the gate itself: the thresholds are evidence-based and
    belong to the model, while this mapping is a product choice.

      private (a question to the AI, seen by nobody else)
          -> answer it, but log the flag. Refusing on a 0.796-precision signal
             would turn away legitimate questions for no safety gain.
      public (comments, chat, anything other students read)
          -> hold for a human. Showing abuse to another student costs far more
             than a short delay.

    Returns (proceed, outcome).
    """
    if decision["action"] == BLOCK:
        return False, "blocked"
    if decision["action"] == REVIEW:
        if surface == PUBLIC:
            return False, "held_for_review"
        return True, "allowed_flagged"
    return True, "allowed"


class ModerationGate:
    """Combines the lexicon and the toxicity classifier into one decision."""

    def __init__(self, lexicon=None, vectorizer=None, classifier=None):
        self.lexicon = lexicon
        self.vectorizer = vectorizer
        self.classifier = classifier

    @classmethod
    def load(cls, base_dir=HERE, require_classifier=False):
        """Load the lexicon and, if its artifacts exist, the classifier.

        The gate degrades to lexicon-only rather than failing, so a deployment
        missing the model artifacts still blocks known bad words. Pass
        require_classifier=True to make that a hard error instead.
        """
        lexicon = LexiconFilter()
        vectorizer = classifier = None
        try:
            # joblib unpickles, which executes code, so it must only ever be
            # pointed at artifacts this project produced itself (written by
            # baseline_train.py). Never load a .joblib from an outside source.
            import joblib
            vectorizer = joblib.load(os.path.join(base_dir, "baseline_vectorizer.joblib"))
            classifier = joblib.load(os.path.join(base_dir, "baseline_model.joblib"))
        except Exception as exc:  # noqa: BLE001 - artifacts are optional
            if require_classifier:
                raise
            print(f"warning: classifier unavailable, running lexicon-only ({exc})",
                  file=sys.stderr)
        return cls(lexicon, vectorizer, classifier)

    @property
    def has_classifier(self):
        return self.vectorizer is not None and self.classifier is not None

    def _classify(self, text):
        """Per-label probabilities, or None when running lexicon-only."""
        if not self.has_classifier:
            return None
        probabilities = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        return {label: float(probabilities[i]) for i, label in enumerate(LABEL_COLS)}

    def moderate(self, text):
        """Return a moderation decision for a single piece of user text."""
        matches = self.lexicon.scan(text or "")
        max_severity = max((m["severity"] for m in matches), default=0)
        scores = self._classify(text or "")
        toxicity = scores["toxicity"] if scores else 0.0

        flagged_categories = {}
        if scores:
            flagged_categories = {
                label: round(score, 3)
                for label, score in scores.items()
                if label != "toxicity" and score >= CATEGORY_THRESHOLDS[label]
            }

        # Highest-precision signals first; the first match wins.
        if max_severity >= LEXICON_BLOCK_SEVERITY:
            words = sorted({m["word"] for m in matches if m["severity"] >= LEXICON_BLOCK_SEVERITY})
            action, reason = BLOCK, f"lexicon severity {max_severity}: {', '.join(words)}"
        elif toxicity >= CLASSIFIER_BLOCK_THRESHOLD:
            action, reason = BLOCK, f"classifier toxicity {toxicity:.2f} >= {CLASSIFIER_BLOCK_THRESHOLD}"
        elif matches:
            words = sorted({m["word"] for m in matches})
            action, reason = REVIEW, f"lexicon severity {max_severity}: {', '.join(words)}"
        elif toxicity >= CLASSIFIER_REVIEW_THRESHOLD:
            action, reason = REVIEW, f"classifier toxicity {toxicity:.2f} >= {CLASSIFIER_REVIEW_THRESHOLD}"
        else:
            action, reason = ALLOW, "no signal"

        advisory = sorted(set(flagged_categories) & ADVISORY_ONLY_CATEGORIES)

        return {
            "action": action,
            "reason": reason,
            "allowed": action == ALLOW,
            "toxicity_score": round(toxicity, 3),
            "lexicon_severity": max_severity,
            "lexicon_matches": [
                {"token": m["token"], "word": m["word"],
                 "category": m["category"], "severity": m["severity"]}
                for m in matches
            ],
            "flagged_categories": flagged_categories,
            # Present for a human reviewer; deliberately not acted on.
            "advisory_categories": advisory,
        }


if __name__ == "__main__":
    gate = ModerationGate.load()
    print(f"lexicon forms: {len(gate.lexicon)}   classifier: {gate.has_classifier}\n")

    samples = [
        "Binary search necə işləyir?",
        "Bu dərs çox faydalı idi, təşəkkür edirəm",
        "Mən erməni dilini öyrənirəm",
        "Kitabı götürmək istəyirəm",
        "sen serefsizsen",
        "s3n w3r3fs1z",
        "Ş Ə R Ə F S İ Z",
        "Sen kimsen? Pay atonan!!!",
    ]
    width = max(len(s) for s in samples)
    for text in samples:
        d = gate.moderate(text)
        print(f"{d['action'].upper():6s} {text:<{width}s}  tox={d['toxicity_score']:.2f}  {d['reason']}")
