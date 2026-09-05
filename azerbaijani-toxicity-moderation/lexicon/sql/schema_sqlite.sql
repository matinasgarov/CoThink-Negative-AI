-- Bad-word lexicon schema (SQLite).
--
-- Same structure as schema_mysql.sql and schema_postgres.sql; see the MySQL
-- file for the full design notes. SQLite is here because it needs no server,
-- so the database can actually be created and verified rather than only
-- written down. Everything below maps one-to-one onto the other two engines.
--
-- Note: SQLite does not enforce foreign keys unless the connection runs
--   PRAGMA foreign_keys = ON;
-- build_database.py sets it.

PRAGMA journal_mode = WAL;

DROP VIEW  IF EXISTS LexiconLookup;
DROP TABLE IF EXISTS WordVariations;
DROP TABLE IF EXISTS BadWords;
DROP TABLE IF EXISTS Whitelist;

CREATE TABLE BadWords (
    Id             INTEGER PRIMARY KEY AUTOINCREMENT,
    Word           TEXT    NOT NULL,                     -- canonical Azerbaijani spelling
    NormalizedWord TEXT    NOT NULL UNIQUE,              -- normalize() output; the lookup key
    Category       TEXT    NOT NULL,                     -- matches the toxicity dataset labels
    Severity       INTEGER NOT NULL DEFAULT 2 CHECK (Severity BETWEEN 1 AND 3),
    IsActive       INTEGER NOT NULL DEFAULT 1 CHECK (IsActive IN (0, 1)),
    CreatedAt      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- The UNIQUE constraint above already creates the index that serves lookups.
-- These two are for the other access paths: admin search by spelling, and
-- "show me every active insult".
CREATE INDEX ix_badwords_word            ON BadWords (Word);
CREATE INDEX ix_badwords_category_active ON BadWords (Category, IsActive);

CREATE TABLE WordVariations (
    Id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    BadWordId           INTEGER NOT NULL,
    Variation           TEXT    NOT NULL UNIQUE,         -- the deformation as written
    NormalizedVariation TEXT    NOT NULL,                -- normalize() output; the lookup key
    Source              TEXT    NOT NULL DEFAULT 'manual'
                        CHECK (Source IN ('mined', 'inflect', 'rule', 'leet', 'manual')),
    IsActive            INTEGER NOT NULL DEFAULT 1 CHECK (IsActive IN (0, 1)),
    CreatedAt           TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (BadWordId) REFERENCES BadWords (Id) ON DELETE CASCADE
);

CREATE INDEX ix_variations_badword     ON WordVariations (BadWordId);
-- Not unique: many spellings share one normalized key, which is the point.
-- k1tab, kit4b and k1t4b all resolve to kitab.
CREATE INDEX ix_variations_normalized  ON WordVariations (NormalizedVariation);

-- Words that must never be flagged even when they collide with a bad word's
-- prefix or normalized form: götürmək (to take) vs göt, şikayət vs sik, and
-- the neutral demonyms.
CREATE TABLE Whitelist (
    Id               INTEGER PRIMARY KEY AUTOINCREMENT,
    Phrase           TEXT NOT NULL,
    NormalizedPhrase TEXT NOT NULL UNIQUE,
    Scope            TEXT NOT NULL DEFAULT 'exact' CHECK (Scope IN ('exact', 'prefix')),
    Reason           TEXT NOT NULL,                      -- required, so entries stay auditable
    CreatedAt        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One place to resolve any normalized token, base word or deformation alike,
-- so the application issues a single query per token instead of two.
-- Grouped so a normalized key resolves to exactly one row, even though several
-- stored spellings may share it. MIN() prefers 'base' over 'variation'.
CREATE VIEW LexiconLookup AS
SELECT NormalizedForm, BadWordId, Word, Category, Severity, MIN(MatchType) AS MatchType
FROM (
    SELECT b.NormalizedWord AS NormalizedForm, b.Id AS BadWordId, b.Word,
           b.Category, b.Severity, 'base' AS MatchType
    FROM BadWords b
    WHERE b.IsActive = 1
    UNION ALL
    SELECT v.NormalizedVariation, b.Id, b.Word, b.Category, b.Severity, 'variation'
    FROM WordVariations v
    JOIN BadWords b ON b.Id = v.BadWordId
    WHERE v.IsActive = 1 AND b.IsActive = 1
)
GROUP BY NormalizedForm;
