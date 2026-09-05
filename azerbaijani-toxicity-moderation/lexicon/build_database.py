"""Create the lexicon database and prove the structure works.

Everything before this script produced files: schemas, CSVs, a seed script.
This one actually creates a database, loads it, and then verifies the three
claims the design rests on:

  1. Referential integrity holds  -- a variation cannot outlive its base word.
  2. The index is used            -- EXPLAIN QUERY PLAN says SEARCH, not SCAN.
  3. The index earns its place    -- timed lookups against an unindexed copy.

SQLite is the engine here because it needs no server, so the structure can be
executed and measured rather than only written down. The MySQL and Postgres
schemas in sql/ are the same structure for a real server.

Run: python build_database.py
"""

import csv
import os
import sqlite3
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "data", "dist")
SCHEMA = os.path.join(HERE, "sql", "schema_sqlite.sql")
DB_PATH = os.path.join(HERE, "data", "cothink_lexicon.db")

BENCHMARK_LOOKUPS = 20_000


def read_csv(name):
    with open(os.path.join(DIST, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def connect(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite needs this per connection
    return conn


def create(conn):
    try:
        with open(SCHEMA, encoding="utf-8") as f:
            conn.executescript(f.read())
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc):
            raise
        raise SystemExit(
            f"Cannot rebuild: another program holds a write lock on "
            f"{os.path.basename(DB_PATH)}.\n"
            f"Finish or roll back its transaction, or close it, then run again."
        )
    conn.commit()


def load(conn):
    bad_words = read_csv("bad_words.csv")
    variations = read_csv("word_variations.csv")
    whitelist = read_csv("whitelist.csv")

    conn.executemany(
        "INSERT INTO BadWords (Id, Word, NormalizedWord, Category, Severity) "
        "VALUES (?, ?, ?, ?, ?)",
        [(int(r["Id"]), r["Word"], r["NormalizedWord"], r["Category"],
          int(r["Severity"])) for r in bad_words],
    )
    conn.executemany(
        "INSERT INTO WordVariations "
        "(Id, BadWordId, Variation, NormalizedVariation, Source) "
        "VALUES (?, ?, ?, ?, ?)",
        [(int(r["Id"]), int(r["BadWordId"]), r["Variation"],
          r["NormalizedVariation"], r["Source"]) for r in variations],
    )
    conn.executemany(
        "INSERT INTO Whitelist (Phrase, NormalizedPhrase, Scope, Reason) "
        "VALUES (?, ?, ?, ?)",
        [(r["Phrase"], r["NormalizedPhrase"], r["Scope"], r["Reason"])
         for r in whitelist],
    )
    conn.commit()
    return len(bad_words), len(variations), len(whitelist)


def verify_integrity(conn):
    """A variation must not be able to outlive the base word it belongs to."""
    print("\n2. Referential integrity")

    orphans = conn.execute(
        "SELECT COUNT(*) FROM WordVariations v "
        "LEFT JOIN BadWords b ON b.Id = v.BadWordId WHERE b.Id IS NULL"
    ).fetchone()[0]
    print(f"   orphaned variations: {orphans}")

    # Prove ON DELETE CASCADE in a transaction we roll back.
    word_id, word = conn.execute(
        "SELECT b.Id, b.Word FROM BadWords b "
        "JOIN WordVariations v ON v.BadWordId = b.Id "
        "GROUP BY b.Id ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    before = conn.execute(
        "SELECT COUNT(*) FROM WordVariations WHERE BadWordId = ?", (word_id,)
    ).fetchone()[0]
    conn.execute("DELETE FROM BadWords WHERE Id = ?", (word_id,))
    after = conn.execute(
        "SELECT COUNT(*) FROM WordVariations WHERE BadWordId = ?", (word_id,)
    ).fetchone()[0]
    conn.rollback()
    restored = conn.execute(
        "SELECT COUNT(*) FROM WordVariations WHERE BadWordId = ?", (word_id,)
    ).fetchone()[0]
    print(f"   deleting '{word}' cascaded {before} variations -> {after} "
          f"(rolled back, now {restored})")

    # A duplicate *spelling* must be rejected; a duplicate normalized key must
    # not be, since many spellings legitimately share one key.
    spelling, key = conn.execute(
        "SELECT Variation, NormalizedVariation FROM WordVariations LIMIT 1"
    ).fetchone()
    try:
        conn.execute(
            "INSERT INTO WordVariations (BadWordId, Variation, NormalizedVariation) "
            "VALUES (1, ?, 'zzz')", (spelling,)
        )
        print("   duplicate spelling: NOT rejected  <-- constraint missing")
    except sqlite3.IntegrityError:
        print(f"   duplicate spelling '{spelling}': rejected as expected")
    finally:
        conn.rollback()

    sharing = conn.execute(
        "SELECT COUNT(*) FROM WordVariations WHERE NormalizedVariation = ?", (key,)
    ).fetchone()[0]
    print(f"   spellings sharing the key '{key}': {sharing} (many is by design)")


def verify_index_used(conn):
    """The planner must SEARCH via the index, not SCAN the table."""
    print("\n3. Query plan")
    for sql, label in [
        ("SELECT * FROM BadWords WHERE NormalizedWord = 'serefsiz'", "base word lookup"),
        ("SELECT * FROM WordVariations WHERE NormalizedVariation = 'werefsiz'",
         "variation lookup"),
    ]:
        plan = " ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql))
        verdict = "USES INDEX" if "SEARCH" in plan else "FULL SCAN"
        print(f"   {label:18s} {verdict}")
        print(f"      {plan}")


def benchmark(conn):
    """Time indexed lookups against an identical table with no index."""
    print("\n4. Does the index actually earn its place?")

    keys = [r[0] for r in conn.execute(
        "SELECT NormalizedVariation FROM WordVariations"
    ).fetchall()]

    # An unindexed copy of the same rows, as the control.
    conn.execute("CREATE TEMP TABLE Unindexed AS SELECT * FROM WordVariations")

    def time_lookups(table, column):
        cursor = conn.cursor()
        sql = f"SELECT Id FROM {table} WHERE {column} = ?"
        started = time.perf_counter()
        for i in range(BENCHMARK_LOOKUPS):
            cursor.execute(sql, (keys[i % len(keys)],)).fetchone()
        return time.perf_counter() - started

    indexed = time_lookups("WordVariations", "NormalizedVariation")
    unindexed = time_lookups("Unindexed", "NormalizedVariation")
    conn.execute("DROP TABLE Unindexed")

    print(f"   {BENCHMARK_LOOKUPS:,} lookups over {len(keys):,} rows")
    print(f"   with index:    {indexed:6.3f}s  "
          f"({indexed / BENCHMARK_LOOKUPS * 1e6:7.1f} us/lookup)")
    print(f"   without index: {unindexed:6.3f}s  "
          f"({unindexed / BENCHMARK_LOOKUPS * 1e6:7.1f} us/lookup)")
    print(f"   speedup:       {unindexed / indexed:.1f}x")


def sample_queries(conn):
    print("\n5. The queries an application actually runs")

    row = conn.execute(
        "SELECT Word, Category, Severity, MatchType FROM LexiconLookup "
        "WHERE NormalizedForm = ?", ("werefsiz",)
    ).fetchone()
    print(f"   lookup 'werefsiz' via LexiconLookup -> {row}")

    print("   rows per category:")
    for category, words, variations in conn.execute(
        "SELECT b.Category, COUNT(DISTINCT b.Id), COUNT(v.Id) "
        "FROM BadWords b LEFT JOIN WordVariations v ON v.BadWordId = b.Id "
        "GROUP BY b.Category ORDER BY COUNT(v.Id) DESC"
    ):
        print(f"      {category:18s} {words:3d} words, {variations:5d} variations")

    print("   variations by source:")
    for source, count in conn.execute(
        "SELECT Source, COUNT(*) FROM WordVariations GROUP BY Source "
        "ORDER BY COUNT(*) DESC"
    ):
        print(f"      {source:10s} {count:5d}")


def main():
    # Rebuild in place rather than deleting the file. Windows refuses to delete
    # a file another process has open, so delete-and-recreate failed whenever a
    # sqlite3 prompt or a database viewer was pointed at it -- which is most of
    # the time. The schema drops and recreates its own objects, and the VACUUM
    # at the end reclaims the freed pages.
    conn = connect(DB_PATH)
    try:
        print("1. Creating database")
        create(conn)
        words, variations, whitelist = load(conn)
        print(f"   {DB_PATH}")
        print(f"   loaded {words} bad words, {variations} variations, "
              f"{whitelist} whitelist entries")

        verify_integrity(conn)
        verify_index_used(conn)
        benchmark(conn)
        sample_queries(conn)

        # Fold the write-ahead log back into the main file, so the database is
        # a single self-contained artifact and the size reported is the real one.
        conn.commit()  # VACUUM cannot run inside a transaction
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.close()
        print(f"\nDatabase created and verified "
              f"({os.path.getsize(DB_PATH) / 1024:.0f} KB on disk).")
    finally:
        try:
            conn.close()
        except sqlite3.ProgrammingError:
            pass  # already closed above


if __name__ == "__main__":
    main()
