"""Tests for the lexicon database schema.

Builds the schema into an in-memory database and asserts the constraints hold,
so a broken schema fails here rather than in production. The row-count tests
run against the built database file when it exists.

Run: python -m unittest test_database -v
"""

import csv
import os
import sqlite3
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "sql", "schema_sqlite.sql")
DB_PATH = os.path.join(HERE, "data", "cothink_lexicon.db")
DIST = os.path.join(HERE, "data", "dist")


def fresh_schema():
    """An empty in-memory database with the schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    with open(SCHEMA, encoding="utf-8") as f:
        # WAL and DROP statements are harmless but meaningless in memory.
        conn.executescript(f.read())
    return conn


def add_word(conn, word="test", normalized="test", category="insult", severity=2):
    cursor = conn.execute(
        "INSERT INTO BadWords (Word, NormalizedWord, Category, Severity) "
        "VALUES (?, ?, ?, ?)", (word, normalized, category, severity)
    )
    return cursor.lastrowid


class TestSchemaConstraints(unittest.TestCase):
    def setUp(self):
        self.conn = fresh_schema()

    def tearDown(self):
        self.conn.close()

    def test_tables_and_view_exist(self):
        names = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )}
        for expected in ["BadWords", "WordVariations", "Whitelist", "LexiconLookup"]:
            self.assertIn(expected, names)

    def test_normalized_word_must_be_unique(self):
        add_word(self.conn, normalized="serefsiz")
        with self.assertRaises(sqlite3.IntegrityError):
            add_word(self.conn, word="other", normalized="serefsiz")

    def test_stored_spelling_must_be_unique(self):
        word_id = add_word(self.conn)
        self.conn.execute(
            "INSERT INTO WordVariations (BadWordId, Variation, NormalizedVariation) "
            "VALUES (?, 'k1tab', 'kitab')", (word_id,))
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO WordVariations (BadWordId, Variation, NormalizedVariation) "
                "VALUES (?, 'k1tab', 'kitab')", (word_id,))

    def test_many_spellings_may_share_one_normalized_key(self):
        # This is the design: k1tab, kit4b and k1t4b all resolve to kitab.
        word_id = add_word(self.conn)
        for spelling in ("k1tab", "kit4b", "k1t4b"):
            self.conn.execute(
                "INSERT INTO WordVariations "
                "(BadWordId, Variation, NormalizedVariation, Source) "
                "VALUES (?, ?, 'kitab', 'leet')", (word_id, spelling))
        count = self.conn.execute(
            "SELECT COUNT(*) FROM WordVariations WHERE NormalizedVariation = 'kitab'"
        ).fetchone()[0]
        self.assertEqual(count, 3)

    def test_view_returns_one_row_per_normalized_key(self):
        word_id = add_word(self.conn, word="kitab", normalized="kitab")
        for spelling in ("k1tab", "kit4b"):
            self.conn.execute(
                "INSERT INTO WordVariations "
                "(BadWordId, Variation, NormalizedVariation, Source) "
                "VALUES (?, ?, 'kitab', 'leet')", (word_id, spelling))
        rows = self.conn.execute(
            "SELECT COUNT(*) FROM LexiconLookup WHERE NormalizedForm = 'kitab'"
        ).fetchone()[0]
        self.assertEqual(rows, 1)

    def test_variation_requires_an_existing_base_word(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO WordVariations (BadWordId, Variation, NormalizedVariation) "
                "VALUES (9999, 'x', 'x')")

    def test_deleting_a_base_word_cascades_to_its_variations(self):
        word_id = add_word(self.conn)
        self.conn.execute(
            "INSERT INTO WordVariations (BadWordId, Variation, NormalizedVariation) "
            "VALUES (?, 'x', 'x')", (word_id,))
        self.conn.execute("DELETE FROM BadWords WHERE Id = ?", (word_id,))
        remaining = self.conn.execute(
            "SELECT COUNT(*) FROM WordVariations").fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_severity_is_constrained_to_its_range(self):
        for bad in (0, 4, -1):
            with self.assertRaises(sqlite3.IntegrityError, msg=f"severity {bad}"):
                add_word(self.conn, normalized=f"n{bad}", severity=bad)

    def test_source_is_constrained_to_known_values(self):
        word_id = add_word(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO WordVariations "
                "(BadWordId, Variation, NormalizedVariation, Source) "
                "VALUES (?, 'x', 'x', 'invented')", (word_id,))

    def test_whitelist_scope_is_constrained(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO Whitelist (Phrase, NormalizedPhrase, Scope, Reason) "
                "VALUES ('a', 'a', 'everywhere', 'r')")

    def test_whitelist_requires_a_reason(self):
        # Reason is NOT NULL on purpose: an unexplained whitelist entry is
        # unauditable, and this table is where bias mistakes get fixed.
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO Whitelist (Phrase, NormalizedPhrase) VALUES ('a', 'a')")

    def test_view_returns_both_base_words_and_variations(self):
        word_id = add_word(self.conn, word="şərəfsiz", normalized="serefsiz", severity=3)
        self.conn.execute(
            "INSERT INTO WordVariations (BadWordId, Variation, NormalizedVariation) "
            "VALUES (?, 'werefsiz', 'werefsiz')", (word_id,))
        rows = {r[0]: r[1] for r in self.conn.execute(
            "SELECT NormalizedForm, MatchType FROM LexiconLookup")}
        self.assertEqual(rows, {"serefsiz": "base", "werefsiz": "variation"})

    def test_view_hides_deactivated_rows(self):
        word_id = add_word(self.conn, normalized="serefsiz")
        self.conn.execute("UPDATE BadWords SET IsActive = 0 WHERE Id = ?", (word_id,))
        count = self.conn.execute("SELECT COUNT(*) FROM LexiconLookup").fetchone()[0]
        self.assertEqual(count, 0)

    def test_lookup_uses_an_index_not_a_scan(self):
        add_word(self.conn, normalized="serefsiz")
        plan = " ".join(r[3] for r in self.conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM BadWords WHERE NormalizedWord = 'x'"))
        self.assertIn("SEARCH", plan)
        self.assertNotIn("SCAN BadWords", plan)


@unittest.skipUnless(os.path.exists(DB_PATH), "run build_database.py first")
class TestBuiltDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(DB_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def csv_rows(self, name):
        with open(os.path.join(DIST, name), encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_row_counts_match_the_source_csvs(self):
        for table, filename in [
            ("BadWords", "bad_words.csv"),
            ("WordVariations", "word_variations.csv"),
            ("Whitelist", "whitelist.csv"),
        ]:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            self.assertEqual(count, len(self.csv_rows(filename)), table)

    def test_no_orphaned_variations(self):
        orphans = self.conn.execute(
            "SELECT COUNT(*) FROM WordVariations v "
            "LEFT JOIN BadWords b ON b.Id = v.BadWordId WHERE b.Id IS NULL"
        ).fetchone()[0]
        self.assertEqual(orphans, 0)

    def test_known_deformation_resolves_to_its_base_word(self):
        row = self.conn.execute(
            "SELECT Word, Category, Severity FROM LexiconLookup "
            "WHERE NormalizedForm = 'werefsiz'").fetchone()
        self.assertEqual(row, ("şərəfsiz", "insult", 3))

    def test_whitelisted_word_is_not_in_the_lookup_view(self):
        # götürmək must never resolve as a deformation of göt.
        row = self.conn.execute(
            "SELECT COUNT(*) FROM LexiconLookup WHERE NormalizedForm = 'gotur'"
        ).fetchone()[0]
        self.assertEqual(row, 0)


if __name__ == "__main__":
    unittest.main()
