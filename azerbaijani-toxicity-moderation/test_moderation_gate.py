"""Tests for the moderation gate and the surface policy.

Run: python -m unittest test_moderation_gate -v
"""

import unittest

from moderation_gate import (
    ALLOW, BLOCK, CATEGORY_THRESHOLDS, LABEL_COLS, PRIVATE, PUBLIC, REVIEW,
    ModerationGate, apply_policy,
)


class FakeClassifier:
    """Stands in for the trained model so policy is tested without artifacts."""

    def __init__(self, toxicity, **per_label):
        self.scores = {label: 0.0 for label in LABEL_COLS}
        self.scores["toxicity"] = toxicity
        self.scores.update(per_label)

    def predict_proba(self, _x):
        # Column order must match LABEL_COLS; toxicity is last.
        return [[self.scores[label] for label in LABEL_COLS]]


class FakeVectorizer:
    def transform(self, texts):
        return texts


def gate_with(toxicity, **per_label):
    real = ModerationGate.load()
    return ModerationGate(real.lexicon, FakeVectorizer(),
                          FakeClassifier(toxicity, **per_label))


class TestGateDecisions(unittest.TestCase):
    def test_lexicon_severity_3_blocks_regardless_of_classifier(self):
        decision = gate_with(0.01).moderate("sen serefsiz")
        self.assertEqual(decision["action"], BLOCK)
        self.assertIn("şərəfsiz", decision["reason"])

    def test_obfuscated_text_blocks_when_classifier_is_unsure(self):
        # The case the lexicon exists for: classifier below its block
        # threshold, lexicon certain.
        decision = gate_with(0.60).moderate("s3n w3r3fs1z")
        self.assertEqual(decision["action"], BLOCK)

    def test_high_confidence_classifier_blocks_without_lexicon_hit(self):
        decision = gate_with(0.99).moderate("neytral mətn")
        self.assertEqual(decision["action"], BLOCK)
        self.assertIn("classifier", decision["reason"])

    def test_mid_confidence_classifier_reviews_not_blocks(self):
        # 0.796 precision is not good enough to auto-block on.
        decision = gate_with(0.80).moderate("neytral mətn")
        self.assertEqual(decision["action"], REVIEW)

    def test_clean_text_is_allowed(self):
        decision = gate_with(0.05).moderate("Bu dərs çox faydalı idi")
        self.assertEqual(decision["action"], ALLOW)
        self.assertTrue(decision["allowed"])

    def test_whitelisted_word_does_not_block(self):
        decision = gate_with(0.05).moderate("Kitabı götürmək istəyirəm")
        self.assertEqual(decision["action"], ALLOW)

    def test_sparse_categories_are_advisory_only(self):
        # threat / severe_toxicity annotate but must never drive a block.
        decision = gate_with(0.10).moderate("neytral mətn")
        self.assertEqual(decision["action"], ALLOW)
        self.assertEqual(decision["advisory_categories"], [])

    def test_empty_input_is_allowed(self):
        for text in ["", None, "   "]:
            self.assertEqual(gate_with(0.0).moderate(text)["action"], ALLOW)

    def test_decision_reports_matched_word(self):
        decision = gate_with(0.01).moderate("sen serefsiz")
        words = [m["word"] for m in decision["lexicon_matches"]]
        self.assertIn("şərəfsiz", words)


class TestCategoryThresholds(unittest.TestCase):
    """Categories are reported at their own fitted threshold, not a flat 0.5."""

    def test_every_label_has_a_threshold(self):
        for label in LABEL_COLS:
            self.assertIn(label, CATEGORY_THRESHOLDS)
            self.assertGreater(CATEGORY_THRESHOLDS[label], 0.0)
            self.assertLessEqual(CATEGORY_THRESHOLDS[label], 1.0)

    def test_rare_labels_kept_the_default(self):
        # threat had 61 validation positives and severe_toxicity failed the
        # generalization check, so both must stay at 0.5 rather than be fitted.
        self.assertEqual(CATEGORY_THRESHOLDS["threat"], 0.5)
        self.assertEqual(CATEGORY_THRESHOLDS["severe_toxicity"], 0.5)

    def test_score_below_a_raised_threshold_is_not_reported(self):
        # sexual_explicit sits at 0.71, so 0.60 must no longer flag it -- this
        # is the over-reporting the retune exists to fix.
        self.assertGreater(CATEGORY_THRESHOLDS["sexual_explicit"], 0.60)
        decision = gate_with(0.10, sexual_explicit=0.60).moderate("neytral mətn")
        self.assertNotIn("sexual_explicit", decision["flagged_categories"])

    def test_score_above_a_raised_threshold_is_reported(self):
        decision = gate_with(0.10, sexual_explicit=0.90).moderate("neytral mətn")
        self.assertIn("sexual_explicit", decision["flagged_categories"])

    def test_score_above_a_lowered_threshold_is_reported(self):
        # insult was fitted down to 0.44, so 0.46 should now flag.
        self.assertLess(CATEGORY_THRESHOLDS["insult"], 0.5)
        decision = gate_with(0.10, insult=0.46).moderate("neytral mətn")
        self.assertIn("insult", decision["flagged_categories"])

    def test_thresholds_do_not_change_the_block_decision(self):
        # Routing is precision-driven and must be unaffected by F1 tuning.
        self.assertEqual(gate_with(0.99).moderate("neytral mətn")["action"], BLOCK)
        self.assertEqual(gate_with(0.80).moderate("neytral mətn")["action"], REVIEW)


class TestLexiconOnlyFallback(unittest.TestCase):
    """A deployment without model artifacts must still block known words."""

    def setUp(self):
        real = ModerationGate.load()
        self.gate = ModerationGate(real.lexicon, None, None)

    def test_runs_without_classifier(self):
        self.assertFalse(self.gate.has_classifier)
        self.assertEqual(self.gate.moderate("sen serefsiz")["action"], BLOCK)

    def test_allows_clean_text_without_classifier(self):
        self.assertEqual(self.gate.moderate("Bu dərs faydalı idi")["action"], ALLOW)


class TestSurfacePolicy(unittest.TestCase):
    def test_block_never_proceeds_on_either_surface(self):
        decision = {"action": BLOCK}
        for surface in (PRIVATE, PUBLIC):
            proceed, outcome = apply_policy(decision, surface)
            self.assertFalse(proceed)
            self.assertEqual(outcome, "blocked")

    def test_review_is_answered_privately_but_held_publicly(self):
        decision = {"action": REVIEW}
        proceed, outcome = apply_policy(decision, PRIVATE)
        self.assertTrue(proceed)
        self.assertEqual(outcome, "allowed_flagged")

        proceed, outcome = apply_policy(decision, PUBLIC)
        self.assertFalse(proceed)
        self.assertEqual(outcome, "held_for_review")

    def test_allow_proceeds_everywhere(self):
        for surface in (PRIVATE, PUBLIC):
            proceed, outcome = apply_policy({"action": ALLOW}, surface)
            self.assertTrue(proceed)
            self.assertEqual(outcome, "allowed")


if __name__ == "__main__":
    unittest.main()
