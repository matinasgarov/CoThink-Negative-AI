"""Tests for the bad-word lexicon.

Run: python -m unittest test_lexicon -v
"""

import unittest

from match import LexiconFilter
from normalize import normalize


class TestNormalize(unittest.TestCase):
    def test_collapses_mechanical_deformations_onto_one_key(self):
        # The whole point of normalization: these need no rows of their own.
        for variant in ["kitab", "k1tab", "K I T A B", "k.i.t.a.b",
                        "kkitab", "kitaaab", "k!tab", "k1t@b"]:
            self.assertEqual(normalize(variant), "kitab", variant)

    def test_folds_azerbaijani_letters(self):
        self.assertEqual(normalize("şərəfsiz"), "serefsiz")
        self.assertEqual(normalize("ÇÖĞÜŞI"), "cogusi")

    def test_folds_cyrillic_homoglyphs(self):
        self.assertEqual(normalize("ѕikim"), normalize("sikim"))
        self.assertEqual(normalize("кitab"), "kitab")

    def test_strips_zero_width_characters(self):
        self.assertEqual(normalize("ki\u200btab"), "kitab")

    def test_distinct_words_stay_distinct(self):
        self.assertNotEqual(normalize("göt"), normalize("götür"))

    def test_empty_input(self):
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize(None), "")
        self.assertEqual(normalize("!!!"), "")


class TestLexiconFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon = LexiconFilter()

    def assertFlagged(self, text):
        result = self.lexicon.check(text)
        self.assertTrue(result["is_flagged"], f"expected flag: {text!r}")
        return result

    def assertClean(self, text):
        result = self.lexicon.check(text)
        self.assertFalse(
            result["is_flagged"],
            f"false positive on {text!r}: {result['matches']}",
        )

    def test_lexicon_loaded(self):
        self.assertGreater(len(self.lexicon), 500)

    def test_flags_base_word(self):
        result = self.assertFlagged("sen serefsiz adamsan")
        self.assertEqual(result["max_severity"], 3)
        self.assertIn("insult", result["categories"])

    def test_flags_leetspeak(self):
        self.assertFlagged("s3r3fs1z")

    def test_flags_spaced_letters(self):
        self.assertFlagged("s e r e f s i z")

    def test_flags_repeated_letters(self):
        self.assertFlagged("şşşərəfsiiiz")

    def test_flags_cyrillic_mix(self):
        self.assertFlagged("ѕerefsiz")

    def test_flags_consonant_substitution(self):
        # ş typed as w -- the kitab -> kigap class of deformation.
        self.assertFlagged("werefsiz")

    def test_flags_unseen_inflection_via_suffix_fallback(self):
        self.assertFlagged("serefsizsen")

    def test_reports_category_and_severity(self):
        result = self.assertFlagged("qaraçı")
        self.assertIn("identity_attack", result["categories"])

    # -- false positives: the cases that make a filter unusable ------------

    def test_does_not_flag_ordinary_text(self):
        self.assertClean("Bu dərs çox faydalı idi, təşəkkür edirəm")

    def test_does_not_flag_prefix_collision(self):
        self.assertClean("Kitabı götürmək istəyirəm")
        self.assertClean("Şikayət etmək istəyirəm")

    def test_does_not_flag_neutral_demonyms(self):
        for text in ["Mən erməni dilini öyrənirəm",
                     "Türk dili dərsi",
                     "Rus ədəbiyyatı haqqında sual"]:
            self.assertClean(text)

    def test_does_not_flag_relational_nouns(self):
        # ana/bacı appear inside insult phrases but are innocent words.
        self.assertClean("Anam müəllimdir")

    def test_empty_and_whitespace(self):
        self.assertClean("")
        self.assertClean("   \n\t ")


class TestLexiconIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import csv
        import os
        dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dist")
        with open(os.path.join(dist, "bad_words.csv"), encoding="utf-8", newline="") as f:
            cls.bad_words = list(csv.DictReader(f))
        with open(os.path.join(dist, "word_variations.csv"), encoding="utf-8", newline="") as f:
            cls.variations = list(csv.DictReader(f))

    def test_stored_spellings_are_unique(self):
        spellings = [v["Variation"] for v in self.variations]
        duplicates = {f for f in spellings if spellings.count(f) > 1}
        self.assertEqual(duplicates, set(), f"duplicate spellings: {duplicates}")

    def test_normalized_keys_are_unique_outside_leet(self):
        """Leet rows deliberately share their parent's key; nothing else may.

        Two different base words resolving to one key would make a lookup
        ambiguous, which is the failure this guards against.
        """
        forms = [b["NormalizedWord"] for b in self.bad_words]
        forms += [v["NormalizedVariation"] for v in self.variations
                  if v["Source"] != "leet"]
        duplicates = {f for f in forms if forms.count(f) > 1}
        self.assertEqual(duplicates, set(), f"duplicate keys: {duplicates}")

    def test_leet_rows_resolve_to_their_own_base_word(self):
        by_id = {b["Id"]: b for b in self.bad_words}
        leet = [v for v in self.variations if v["Source"] == "leet"]
        self.assertGreater(len(leet), 100)
        for row in leet:
            parent = by_id[row["BadWordId"]]
            self.assertEqual(normalize(row["Variation"]), parent["NormalizedWord"],
                             f"{row['Variation']} does not resolve to {parent['Word']}")

    def test_every_variation_has_a_parent(self):
        ids = {b["Id"] for b in self.bad_words}
        orphans = [v for v in self.variations if v["BadWordId"] not in ids]
        self.assertEqual(orphans, [])

    def test_normalized_columns_are_actually_normalized(self):
        for row in self.bad_words:
            self.assertEqual(normalize(row["Word"]), row["NormalizedWord"])
        for row in self.variations:
            self.assertEqual(
                normalize(row["Variation"]), row["NormalizedVariation"]
            )

    def test_severity_in_range(self):
        for row in self.bad_words:
            self.assertIn(row["Severity"], {"1", "2", "3"})

    def test_every_stored_form_is_reachable_by_the_matcher(self):
        """A row the matcher can never return is dead weight, not protection.

        This catches whitelist entries that over-reach: marking 'sikkə' or a
        demonym as a prefix silently suppressed 64 stored forms and two whole
        base words before this test existed.
        """
        lexicon = LexiconFilter()
        unreachable = [
            row["NormalizedWord"] for row in self.bad_words
            if not lexicon.check(row["NormalizedWord"])["is_flagged"]
        ] + [
            row["NormalizedVariation"] for row in self.variations
            if not lexicon.check(row["NormalizedVariation"])["is_flagged"]
        ]
        self.assertEqual(
            unreachable[:10], [],
            f"{len(unreachable)} stored forms cannot be matched; "
            f"check whitelist Scope values",
        )


if __name__ == "__main__":
    unittest.main()
