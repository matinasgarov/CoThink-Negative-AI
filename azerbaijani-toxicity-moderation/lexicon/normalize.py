"""Shared text normalization for the bad-word lexicon.

The whole design rests on this module: instead of storing every possible
deformation of a word as its own row, we collapse deformations that follow a
*mechanical* pattern (leetspeak, spacing, diacritics, repeated letters,
Cyrillic look-alikes) into a single canonical key. Both the lexicon and the
incoming user text pass through normalize(), so `s1k!m`, `s i k i m` and
`ssikkim` all land on the same key with zero extra rows.

Only deformations that normalization *cannot* undo -- consonant swaps like
kitab -> kigap, and Azerbaijani inflection -- need rows in WordVariations.
"""

import re
import unicodedata

# Cyrillic characters that render identically (or near enough) to Latin ones.
# Mixing scripts is a cheap, common way to slip a word past a filter.
_CYRILLIC_HOMOGLYPHS = {
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ѕ": "s",
    "ј": "j", "ԁ": "d", "ԛ": "q", "ԝ": "w", "г": "r", "п": "n", "л": "l",
}

# Digit/symbol substitutions. Applied before punctuation is stripped, so the
# symbol forms (@ $ !) still exist at this point.
_LEET = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "6": "b", "7": "t",
    "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "|": "i", "£": "e",
    "€": "e", "+": "t", "(": "c", "*": "", "&": "",
}

# Azerbaijani-specific letters folded to their ASCII skeleton. Users type these
# inconsistently anyway (ə as e or a, ı as i, ş as s or w), so folding removes a
# large class of accidental *and* deliberate variation at once.
_FOLD = {
    "ə": "e", "ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ö": "o", "ü": "u",
    "İ": "i", "I": "i", "Ə": "e",
}

# Zero-width and directionality characters, invisible but filter-breaking.
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x00AD], None
)

_REPEAT_RE = re.compile(r"(.)\1+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _fold_chars(text):
    out = []
    for ch in text:
        if ch in _FOLD:
            out.append(_FOLD[ch])
        elif ch in _CYRILLIC_HOMOGLYPHS:
            out.append(_CYRILLIC_HOMOGLYPHS[ch])
        elif ch in _LEET:
            out.append(_LEET[ch])
        else:
            out.append(ch)
    return "".join(out)


def normalize(text):
    """Collapse a word (or phrase) to its canonical lookup key.

    Returns lowercase [a-z]+ with repeated letters collapsed. Separators
    inside a word are dropped, so 'g o t' and 'g.o.t' both become 'got'.
    """
    if not text:
        return ""
    text = text.translate(_INVISIBLE)
    # Leet substitution only makes sense for a token that is trying to be a
    # word. Without this guard '!!!' would normalize to 'i'.
    if not any(ch.isalpha() for ch in text):
        return ""
    text = _fold_chars(text)
    text = text.lower()
    # Lowercasing can expose more foldable characters (e.g. 'Ş' -> 'ş').
    text = _fold_chars(text).lower()
    # Strip any remaining combining marks (é -> e).
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(c)
    )
    text = _NON_ALNUM_RE.sub("", text)
    text = _REPEAT_RE.sub(r"\1", text)
    return text


# Token splitting keeps separators that are plausibly word boundaries, so a
# sentence is normalized token-by-token rather than into one giant blob.
_TOKEN_SPLIT_RE = re.compile(r"[\s,;:.!?()\[\]{}\"'\u2019/\|<>+=~`^]+")


def tokenize(text):
    """Split text into raw tokens, before normalization."""
    return [t for t in _TOKEN_SPLIT_RE.split(text or "") if t]


def normalize_tokens(text):
    """Yield (raw_token, normalized_token) pairs, skipping empties."""
    for raw in tokenize(text):
        norm = normalize(raw)
        if norm:
            yield raw, norm
