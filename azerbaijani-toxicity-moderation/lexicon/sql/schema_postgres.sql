-- Bad-word lexicon schema (PostgreSQL).
-- Same model as schema_mysql.sql; see that file for the design notes.

CREATE TABLE BadWords (
    Id             SERIAL PRIMARY KEY,
    Word           VARCHAR(64) NOT NULL,
    NormalizedWord VARCHAR(64) NOT NULL UNIQUE,
    Category       VARCHAR(32) NOT NULL,
    Severity       SMALLINT    NOT NULL DEFAULT 2 CHECK (Severity BETWEEN 1 AND 3),
    IsActive       BOOLEAN     NOT NULL DEFAULT TRUE,
    CreatedAt      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_badwords_word ON BadWords (Word);
CREATE INDEX ix_badwords_category_active ON BadWords (Category, IsActive);

CREATE TABLE WordVariations (
    Id                  SERIAL PRIMARY KEY,
    BadWordId           INTEGER     NOT NULL REFERENCES BadWords (Id) ON DELETE CASCADE,
    Variation           VARCHAR(64) NOT NULL UNIQUE,
    NormalizedVariation VARCHAR(64) NOT NULL,
    Source              VARCHAR(8)  NOT NULL DEFAULT 'manual'
                        CHECK (Source IN ('mined', 'inflect', 'rule', 'leet', 'manual')),
    IsActive            BOOLEAN     NOT NULL DEFAULT TRUE,
    CreatedAt           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_variations_badword    ON WordVariations (BadWordId);
-- Not unique: many spellings share one normalized key (k1tab, kit4b -> kitab).
CREATE INDEX ix_variations_normalized ON WordVariations (NormalizedVariation);

CREATE TABLE Whitelist (
    Id               SERIAL PRIMARY KEY,
    Phrase           VARCHAR(128) NOT NULL,
    NormalizedPhrase VARCHAR(128) NOT NULL UNIQUE,
    Scope            VARCHAR(6)   NOT NULL DEFAULT 'exact'
                     CHECK (Scope IN ('exact', 'prefix')),
    Reason           VARCHAR(255) NOT NULL,
    CreatedAt        TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE VIEW LexiconLookup AS
SELECT NormalizedForm, BadWordId, Word, Category, Severity, MIN(MatchType) AS MatchType
FROM (
    SELECT b.NormalizedWord AS NormalizedForm, b.Id AS BadWordId, b.Word,
           b.Category, b.Severity, 'base' AS MatchType
    FROM BadWords b WHERE b.IsActive
    UNION ALL
    SELECT v.NormalizedVariation, b.Id, b.Word, b.Category, b.Severity, 'variation'
    FROM WordVariations v JOIN BadWords b ON b.Id = v.BadWordId
    WHERE v.IsActive AND b.IsActive
) t
GROUP BY NormalizedForm, BadWordId, Word, Category, Severity;

-- Optional: fuzzy matching for forms the lexicon has not seen yet.
-- Postgres is the one engine here that makes this cheap, via trigram indexes.
-- Use it as a review queue ("close to a known bad word"), never to auto-block.
--
--   CREATE EXTENSION IF NOT EXISTS pg_trgm;
--   CREATE INDEX ix_badwords_trgm ON BadWords USING gin (NormalizedWord gin_trgm_ops);
--   SELECT Word, similarity(NormalizedWord, 'serefsz') AS score
--     FROM BadWords WHERE NormalizedWord % 'serefsz' ORDER BY score DESC;
