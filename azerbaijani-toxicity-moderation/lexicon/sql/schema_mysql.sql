-- Bad-word lexicon schema (MySQL / MariaDB, utf8mb4).
--
-- Lookup model: incoming text is normalized in the application, then each
-- token is matched for EQUALITY against NormalizedWord / NormalizedVariation.
-- That is why the operative indexes are B-tree, not FULLTEXT -- see the note
-- at the bottom before reaching for FULLTEXT.

CREATE TABLE BadWords (
    Id              INT UNSIGNED NOT NULL AUTO_INCREMENT,
    Word            VARCHAR(64)  NOT NULL COMMENT 'Canonical form, correct Azerbaijani spelling',
    NormalizedWord  VARCHAR(64)  NOT NULL COMMENT 'Output of normalize(); the actual lookup key',
    Category        VARCHAR(32)  NOT NULL COMMENT 'Matches the toxicity dataset label names',
    Severity        TINYINT UNSIGNED NOT NULL DEFAULT 2 COMMENT '1 mild, 2 moderate, 3 severe',
    IsActive        TINYINT(1)   NOT NULL DEFAULT 1,
    CreatedAt       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (Id),
    UNIQUE KEY uq_badwords_normalized (NormalizedWord),
    KEY ix_badwords_word (Word),
    KEY ix_badwords_category_active (Category, IsActive)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

CREATE TABLE WordVariations (
    Id                  INT UNSIGNED NOT NULL AUTO_INCREMENT,
    BadWordId           INT UNSIGNED NOT NULL,
    Variation           VARCHAR(64)  NOT NULL COMMENT 'Deformation as written',
    NormalizedVariation VARCHAR(64)  NOT NULL COMMENT 'Output of normalize(); the lookup key',
    Source              ENUM('mined','inflect','rule','leet','manual') NOT NULL DEFAULT 'manual'
                        COMMENT 'mined = seen in corpus, inflect+rule = generated, manual = curated',
    IsActive            TINYINT(1)   NOT NULL DEFAULT 1,
    CreatedAt           DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (Id),
    UNIQUE KEY uq_variations_variation (Variation),
    -- Not unique: many spellings share one normalized key (k1tab, kit4b -> kitab).
    KEY ix_variations_normalized (NormalizedVariation),
    KEY ix_variations_badword (BadWordId),
    CONSTRAINT fk_variations_badword
        FOREIGN KEY (BadWordId) REFERENCES BadWords (Id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- Words that must never be flagged, even when they collide with a bad word's
-- prefix or normalized form. Without this table a filter blocks 'götürmək'
-- (to take) as a deformation of 'göt', and neutral demonyms as identity slurs.
CREATE TABLE Whitelist (
    Id               INT UNSIGNED NOT NULL AUTO_INCREMENT,
    Phrase           VARCHAR(128) NOT NULL,
    NormalizedPhrase VARCHAR(128) NOT NULL,
    Scope            ENUM('exact','prefix') NOT NULL DEFAULT 'exact'
                     COMMENT 'prefix shields derived words too; exact shields only this form',
    Reason           VARCHAR(255) NOT NULL COMMENT 'Why this is safe - required, so entries stay auditable',
    CreatedAt        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (Id),
    UNIQUE KEY uq_whitelist_normalized (NormalizedPhrase)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- One place to resolve any normalized token, base word or deformation alike,
-- so the application issues a single query per token instead of two.
CREATE OR REPLACE VIEW LexiconLookup AS
SELECT b.Id AS BadWordId, b.Word, b.NormalizedWord AS NormalizedForm,
       b.Category, b.Severity, 'base' AS MatchType
FROM BadWords b
WHERE b.IsActive = 1
UNION ALL
SELECT b.Id, b.Word, v.NormalizedVariation, b.Category, b.Severity, 'variation'
FROM WordVariations v
JOIN BadWords b ON b.Id = v.BadWordId
WHERE v.IsActive = 1 AND b.IsActive = 1;

-- FULLTEXT is deliberately NOT the primary index here.
--
-- The brief allows "Index or Full-Text Index". For this workload the plain
-- B-tree unique indexes above are the correct choice: lookups are exact
-- equality on a normalized key, which a B-tree answers in O(log n), while
-- FULLTEXT tokenizes on word boundaries and would not match a deformation any
-- better. FULLTEXT earns its place only for the different job of scanning
-- stored user content for known bad words after the fact -- e.g. re-scanning a
-- comment archive when new terms are added. Enable it there:
--
--   ALTER TABLE Comments ADD FULLTEXT KEY ft_comments_body (Body);
--   SELECT Id FROM Comments WHERE MATCH(Body) AGAINST ('serefsiz' IN BOOLEAN MODE);
--
-- Note that MySQL's default minimum token length (innodb_ft_min_token_size = 3)
-- would silently ignore shorter terms.
