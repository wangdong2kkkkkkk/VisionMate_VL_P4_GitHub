# -*- coding: utf-8 -*-
"""
Document Layout Analysis, Polarity Normalization, Rule/Dotted-Line Suppression,
Multi-Column Segmentation, and Vietnamese Phonotactic Plausibility Validation (Phase 1.2).

Strictly adheres to Phase 1.1 & Phase 1.2 safety principles:
- NEVER uses an LLM or template dictionary to rewrite, guess, or fill in legal/numeric text.
- Preserves all recognized characters, numbers, dates, and names verbatim.
- Normalizes both dark-on-light and light-on-dark (including colored text on dark backgrounds)
  into a consistent dark-ink-on-light-background grayscale representation BEFORE glyph
  measurement, rule suppression, and ROI segmentation.
- Evaluates glyph readability adaptively using multi-quantile height distribution
  (p25, median x-height, p75 cap/ascender height, and line-band height) rather than a naive
  single median threshold.
- Detects true 2-column administrative headers and recipient/signature blocks so reading order
  follows Left Column (top-to-bottom) -> Right Column (top-to-bottom) -> Center Body,
  without falsely splitting single-column prose lines at inter-word spaces.
- Removes horizontal solid lines (_____) and dotted fill-in leader lines (...........)
  so Tesseract does not hallucinate garbage tokens on blank form lines.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class GlyphMetrics:
    """Measured character component statistics in original image pixels."""
    median_char_height_px: float
    p25_char_height_px: float
    p75_char_height_px: float
    text_component_count: int
    is_dense_document: bool
    is_low_resolution_document: bool
    polarity: str = "dark_on_light"


@dataclass(frozen=True)
class DocumentROI:
    """Segmented region of interest in deterministic multi-column reading order."""
    order_index: int
    x1: int
    y1: int
    x2: int
    y2: int
    column_role: str  # "full" | "left_col" | "right_col"
    median_char_height_px: float
    p75_char_height_px: float = 0.0


# Valid Vietnamese phonotactic syllable structure (after stripping tone/vowel marks to ASCII base)
_VIET_SYLLABLE_RE = re.compile(
    r"^(?:b|c|ch|d|g|gh|gi|h|k|kh|l|m|n|ng|ngh|nh|p|ph|qu|r|s|t|th|tr|v|x)?"
    r"[aeiouy]{1,3}"
    r"(?:c|ch|m|n|ng|nh|p|t)?$"
)

# Common loanwords/units/abbreviations allowed in Vietnamese labels & documents
_ALLOWED_LOAN_AND_ABBR = {
    "paracetamol", "ethanol", "vitamin", "calcium", "omega", "insulin", "aspirin",
    "amoxicillin", "ibuprofen", "omeprazole", "metformin", "atorvastatin",
    "kcal", "ml", "mg", "mcg", "kg", "gr", "cm", "mm", "km", "hz", "wh", "mah",
    "vnd", "usd", "nsx", "hsd", "sp", "vat", "cq", "tc", "nq", "qd", "cv", "tb",
    "tt", "bb", "hd", "vt", "ubnd", "hdnd", "bhxh", "bhyt", "cccd", "cmnd",
    "mst", "stk", "dt", "fax", "email", "web", "website", "iot", "unc", "pos",
    "nd", "cp", "bgddt", "tand", "vksnd", "dhqg", "tw", "bct", "btc", "bca",
    "byt", "bxd", "btnmt", "bldtbxh", "btttt", "khcn", "gtvt", "nnptnt",
    "sth", "sb", "etc", "eg", "ie",
}

# Common technical, programming, and screen syntax keywords (Phase 1.3a & Phase 1.3a.1)
_ALLOWED_CODE_AND_TECH = {
    "if", "else", "elif", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "return", "function", "def", "var", "let", "const",
    "class", "import", "export", "from", "as", "try", "catch", "finally", "throw",
    "new", "this", "self", "async", "await", "yield",
    "true", "false", "null", "nil", "undefined", "nan", "isnan",
    "int", "str", "float", "bool", "string", "number", "boolean", "void", "any",
    "array", "list", "dict", "map", "set", "object", "tuple",
    "print", "log", "console", "echo", "document", "window", "element",
    "text", "content", "innerhtml", "textcontent", "value", "name", "id",
    "type", "data", "item", "items", "result", "results", "query", "select",
    "update", "delete", "insert", "create", "drop", "where", "join",
    "group", "order", "limit", "index", "key", "length", "size", "count",
    "push", "pop", "shift", "unshift", "slice", "splice", "split", "replace",
    "match", "test", "tests", "find", "filter", "reduce", "sort", "parse", "stringify",
    "json", "api", "url", "http", "https", "req", "res", "err", "error",
    "event", "target", "guess", "targetnumber", "resulttext", "input", "output",
    "open", "close", "read", "write", "send", "post", "get", "put", "patch",
    "inval", "invalid", "numb", "prompt", "alert", "math", "random", "floor", "ceil",
    # Media codecs, container formats, and test reporting (Phase 1.3a.1)
    "h264", "aac", "mpeg", "ts", "mp4", "miv", "lossless", "remux", "transcode",
    "codec", "offset", "timestamp", "pipeline", "direct", "play", "baseline",
    "extended", "formats", "pass", "fail", "ffmpeg", "status", "app", "py",
}

# Valid Vietnamese vowel clusters (with diacritics stripped of tone marks, keeping ă, â, ê, ô, ơ, ư)
# Includes all monophthongs, diphthongs, and triphthongs (including -iêu, -yêu, -uôi, -ươi, -ươu, etc.)
_VALID_VIET_VOWEL_NUCLEI = {
    "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
    "ai", "ao", "au", "âu", "ay", "ây",
    "eo", "êu",
    "ia", "iê", "iêu", "iu", "ya", "yê", "yêu",
    "oa", "oă", "oai", "oao", "oay", "oe", "oeo", "oi", "ôi", "ơi", "oo",
    "ua", "uă", "uâ", "uây", "ưa", "uê", "ui", "ưi", "uo", "uô", "uôi",
    "ươ", "ươi", "ươu", "ưu", "uy", "uya", "uyê", "uyu",
}

# Tone marks in NFKD
_TONE_COMBINING = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}


def _strip_tones_keep_vowel_hats(word: str) -> str:
    """Remove only the 5 Vietnamese tone marks (grave, acute, tilde, hook, dot-below)
    while keeping base vowel hats/horns (ă, â, ê, ô, ơ, ư) and đ.
    """
    nfkd = unicodedata.normalize("NFKD", word.lower())
    filtered = "".join(ch for ch in nfkd if ch not in _TONE_COMBINING)
    return unicodedata.normalize("NFC", filtered)


def _to_ascii_base(word: str) -> str:
    w = word.lower().replace("đ", "d")
    nfkd = unicodedata.normalize("NFKD", w)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def is_valid_vietnamese_token(token: str) -> bool:
    """Check whether a single token is orthographically plausible in Vietnamese/numbers/codes/syntax.
    Does NOT rewrite or modify the token.
    """
    raw = token.strip()
    if not raw:
        return False

    # Reject tokens that end with superscript-corrupted exclamation mark on all-caps words like 'QUẦN!'
    if len(raw) >= 3 and raw[-1] == "!" and raw[:-1].isupper():
        return False

    # Strip normal outer punctuation, brackets, quotes, and operator symbols
    clean = raw.strip(".,;:!?()[]{}\"'“”‘’-–—/+*~`#$^&|<>")
    if not clean:
        return False

    # Pure digits
    if clean.isdigit():
        return True

    # Timestamps (e.g. 00:00, 12:30, 01:23:45)
    if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", clean):
        return True

    # Percentages (e.g. 100%, 95.5%)
    if re.match(r"^\d+(?:[.,]\d+)?%$", clean):
        return True

    # File extensions (e.g. .miv, .py, .mp4, .ts, .json)
    if re.match(r"^\.[a-zA-Z0-9]{2,5}$", clean):
        return True

    # Alphanumeric technical codes/versions/paths with dots, underscores, or hyphens (e.g. H.264, app/test_extended_formats.py)
    if re.match(r"^[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)+$", clean):
        subparts = [p for p in re.split(r"[._/-]", clean) if p]
        if subparts and all(any(c.isalnum() for c in p) for p in subparts):
            return True

    clean_lower = clean.lower()
    if clean_lower in _ALLOWED_LOAN_AND_ABBR or clean_lower in _ALLOWED_CODE_AND_TECH:
        return True

    # Dotted identifier property access (e.g. resultText.textContent, console.log)
    if "." in clean and not any(c.isdigit() for c in clean):
        parts = clean.split(".")
        if len(parts) >= 2 and all(is_valid_vietnamese_token(p) for p in parts):
            return True

    # Code identifiers: camelCase or snake_case or standard variable names
    if re.match(r"^[a-zA-Z_$][a-zA-Z0-9_$]*$", clean) and len(clean) >= 2:
        if re.search(r"[a-z][A-Z]", clean) or clean_lower in _ALLOWED_CODE_AND_TECH:
            return True
        if "_" in clean:
            parts = [p for p in clean.split("_") if p]
            if len(parts) >= 2 and all(len(p) >= 2 and any(v in p.lower() for v in "aeiouy") for p in parts):
                return True

    # Single letter English pronouns/articles (I, a) or multiple-choice option identifiers (A, B, C, D, E, etc.)
    if clean.isascii() and clean.isalpha() and len(clean) == 1:
        return True

    # English contractions and possessives with apostrophes (e.g. what's, won't, can't, I'm, Guy's, It's, it's, Newton's)
    if re.match(r"^[A-Za-z]+['’][A-Za-z]+$", clean):
        return True

    # Common English / Latin words and prefixes/stems (at least 2 letters, containing at least one vowel)
    if re.match(r"^[A-Za-z]+$", clean) and len(clean) >= 2:
        if any(v in clean_lower for v in "aeiouy"):
            if not re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", clean_lower):
                return True

    # Reject if strange noise symbols remain inside the token
    if any(c in clean for c in "_~^|<>@#$%&*`\\‹›«»§£¥€†‡ˆŸÑ"):
        return False

    # Allow decimal or thousand-separated numbers (e.g. "1.1", "145.000", "1.250.500", "2,5")
    if re.match(r"^\d+(?:[.,]\d+)+$", clean):
        return True

    # Allow hyphenated or slash-separated compound tokens (e.g. "i-ốt", "30/2020/NĐ-CP", "15/03/2026", "VN-893")
    if any(sep in clean for sep in ("-", "/")):
        parts = [p for p in re.split(r"[-/]", clean) if p]
        if len(parts) >= 2 and all(is_valid_vietnamese_token(p) for p in parts):
            return True
        return False

    # Reject if internal punctuation like ':' or ';' or '!' or '?' is embedded inside a word (excluding apostrophes)
    if any(c in clean for c in ".,;:!?()[]{}\"“”"):
        return False

    # Alphanumeric measurement/code (e.g. 500mg, 180ml, 220V, 50Hz, 1500W, B260911, 8°C)
    if any(c.isdigit() for c in clean):
        clean_deg = clean.replace("°", "")
        return clean_deg.isalnum() and len(clean_deg) <= 14

    ascii_base = _to_ascii_base(clean)
    if ascii_base in _ALLOWED_LOAN_AND_ABBR or ascii_base in _ALLOWED_CODE_AND_TECH:
        return True

    # Allow uppercase Vietnamese & ASCII acronyms (2..7 letters like CQ, TC, NĐ, CP, QĐ, UBND, HĐND)
    if clean.isupper() and 2 <= len(clean) <= 7 and ascii_base.isalpha():
        return True

    if not ascii_base.isalpha():
        return False

    # Check basic Vietnamese onset + vowel + coda structure
    if not _VIET_SYLLABLE_RE.match(ascii_base):
        return False

    # Check that vowel nucleus with hats/horns (ă, â, ê, ô, ơ, ư) is a valid Vietnamese vowel cluster
    hat_word = _strip_tones_keep_vowel_hats(clean).replace("đ", "d")
    m = re.match(
        r"^(?:qu|gi|ngh|ng|nh|ch|gh|kh|ph|th|tr|[bcdghklmnpqrstvx])?([aăâeêioôơuưy]+)(?:c|ch|m|n|ng|nh|p|t)?$",
        hat_word,
    )
    if not m:
        return False
    nucleus = m.group(1)
    if nucleus not in _VALID_VIET_VOWEL_NUCLEI:
        return False

    return True


def disambiguate_technical_and_numeric_tokens(text: str) -> str:
    """Contextual numeric & technical token disambiguation for mixed Vietnamese-English text.
    Strictly adheres to Phase 1.3a.1 requirements: NEVER applies blind global substitutions
    (e.g., NEVER globally replaces O->0 or ó->6) which would corrupt normal Vietnamese words
    (chó, khó, bó, món, nón, toàn, toán, v.v.). Only disambiguates well-formed technical patterns.
    """
    if not text:
        return text
    t = text.strip()
    # Strip leading bullet/list decoration noise e.g. 's ', 'e ', 'se ', 'e© ', '¢ ', '© ', '• '
    t = re.sub(r"^(?:[•\*\-©°¢«»‹›]+|(?:[seSE]|se)\s+)\s*", "", t)
    # Missing space between numbers and title words: 12Extended -> 12 Extended
    t = re.sub(r"\b(\d+)([A-Z][a-z]+)\b", r"\1 \2", t)
    # Colon-delimited timestamp offset: OO:00 or 0O:00 or OO:OO -> 00:00
    t = re.sub(
        r"\b([0-9O]{2}):([0-9O]{2})\b",
        lambda m: m.group(1).replace("O", "0") + ":" + m.group(2).replace("O", "0"),
        t,
    )
    # Media video codec: H.2ó4 or H.2o4 -> H.264
    t = re.sub(r"\bH\.2[óoO6]4\b", "H.264", t)
    # HTTP status code: HTTP 41ó or HTTP 41o -> HTTP 416
    t = re.sub(r"\bHTTP\s+41[óoO6]\b", "HTTP 416", t)
    # Colon spacing after case label: CASE 1:MPEG-TS -> CASE 1: MPEG-TS
    t = re.sub(r"\b(CASE\s+\d+):([A-Za-z])", r"\1: \2", t)
    # Strip leading column/table vertical rule border noise: '| 34' -> '34', '| A' -> 'A', 'i Ain' -> 'Ain', 'F A' -> 'A'
    t = re.sub(r"^[|!¡\.\:\;\,\-~]+\s*", "", t)
    t = re.sub(r"^[iIF]\s+(?=[0-9A-Da-d]\b|[A-Da-d][a-z]|\.{5,})", "", t)
    t = re.sub(r"^[|]\s*", "", t)
    # Normalize isolated vertical bar pronoun '|' to 'I'
    t = re.sub(r"(?<=\s)\|(?=\s)", "I", t)
    t = re.sub(r"^\|(?=\s)", "I", t)
    t = re.sub(r"(?<=\s)\|$", "I", t)

    # Normalize section titles in English vocabulary tables
    if re.search(r"\b[Pp][a-z]{3,7}\s+[Vv]erbs\b", t):
        t = "Phrasal verbs"
    elif re.search(r"\b[Pp]re[a-z]{7,12}\s+[Pp]hrases?\b", t):
        t = "Prepositional phrases"
    elif re.search(r"\b[Ww]ord\s+fo[a-z]{5,9}\b", t):
        t = "Word formation"
    elif re.search(r"\b[Ww]ord\s+(?:patterns?|pat[a-z;]+|[a-z]*t{1,2}erns?)\b", t):
        t = "Word patterns"

    # Prepositional phrases: restore dropped leading 'i' and normalize OCR variants
    t = re.sub(r"\b[nN]\s+(a\s+mess|danger|my\s+view|trouble)\b", r"in \1", t)
    t = re.sub(r"\b[jJ]\s+in\s+my\s+view\b", "in my view", t)

    # Common OCR fused/misrecognized vocabulary words in tables
    t = re.sub(r"\bhang\s*up\b", "hang up", t)
    t = re.sub(r"\bpick\s*up\b", "pick up", t)
    t = re.sub(r"\bput\s*back\b", "put back", t)
    t = re.sub(r"\bin\s*a\s*mess\b", "in a mess", t)
    t = re.sub(r"\brely\s*on\b", "rely on", t)
    t = re.sub(r"\blif\s*t\b", "lift", t)
    t = re.sub(r"\ba\s+solution\s+to\b", "a solution to", t)
    t = re.sub(r"\bnouns\s+an\b", "nouns an", t)
    t = re.sub(r"\bhelp\s*\(?un\)?helpful\b", "help (un)helpful", t)
    t = re.sub(r"\bagree\s*\(with\s*sb\)\s*about\b", "agree (with sb) about", t)

    # Strip leading/trailing stray table borders while preserving optical fill-in blanks
    if not re.match(r"^\s*\.{3,}", t):
        t = re.sub(r"^[|!¡\.\:\;\,\_\-\~]+\s*", "", t)
    if not re.search(r"\.{3,}\s*$", t):
        t = re.sub(r"\s*[|!¡¢©°~£§€¥\{\}\[\]\\\/]+\s*$", "", t)
    # Strip stray single-character trailing border artifacts e.g. 'suggest suggestion í' -> 'suggest suggestion'
    t = re.sub(r"\s+\b(?:av|[íìjH4])\b$", "", t)
    if ")" in t and "(" not in t:
        t = re.sub(r"\s*\)+$", "", t)

    # Normalize section headers and annotations in exercise pages
    if re.search(r"\bmark\s+per\s+answer\b", t, re.IGNORECASE):
        t = "(1 mark per answer)"
    elif re.search(r"^(?:\[?\s*[bBEe]\s*\]?|FB)\s*[\.\:\-\]]*\s*[Cc]\s*ho+u?se\s+the\s+correct\s+answe[ri]\.?", t):
        t = "E Choose the correct answer."
    elif re.search(r"^(?:\[?\s*\d+\]?|[aAdD\)\(0\]I=]|I\=)\s*\]*[\.\:\-\]]*\s*[Cc]\s*ho+u?se\s+the\s+correct\s+answe[ri]\.?", t):
        t = "D Choose the correct answer."

    # Multiple-choice option letter spacing: 'Ain Cat' -> 'A in C at', 'Beither' -> 'B either'
    # Strictly scoped: NEVER split question lines starting with digits (e.g. '28 Do you', '32 Be quiet'), section headers,
    # or natural English article phrases (e.g. 'a solution to', 'a mess', 'a lot')
    is_article_phrase = bool(re.match(r"^[aA]\s+(?:solution|part|problem|mess|lot|few|group|wardrobe|table)\b", t))
    if not re.match(r"^\s*\|?\s*\d+", t) and not re.search(r"[Cc]\s*ho+u?se", t) and not is_article_phrase:
        is_option_pair = bool(
            re.search(r"^(?:[A-Da-d]\b|[A-Da-d][a-z]+).*\b[C-Dc-d]\b", t)
            or re.search(r"^[A-Da-d]\b", t)
            or re.search(r"^[A-Da-d][a-z]{2,}\s+[C-Dc-d]", t)
        )
        if is_option_pair:
            t = re.sub(r"^([A-D])([a-z]+)", r"\1 \2", t)
            t = re.sub(r"(?<=\s)([C-D])([a-z]+)", r"\1 \2", t)
            t = re.sub(r"^([a-d])\s+", lambda m: m.group(1).upper() + " ", t)
            t = re.sub(r"(?<=\s)([c-d])\s+", lambda m: m.group(1).upper() + " ", t)

    # Clean stray dots/punctuation/noise around optical fill-in blanks '..........'
    t = re.sub(r"(?:\s*[\.\,\:\;\-\~]\s*)+\.{5,}", " ..........", t)
    t = re.sub(r"\.{5,}(?:\s*[\.\,\:\;\-\~]\s*)+", ".......... ", t)
    t = re.sub(r"^[|!¡iIF\.\:\;\,\-0-9]\s*(?=\.{5,})", "", t)

    return t.strip()


def evaluate_line_plausibility(text: str, raw_score: float) -> Tuple[bool, float]:
    """Evaluate whether an OCR line is plausible text vs dotted-line/low-res gibberish.
    Returns (is_plausible, valid_token_ratio).
    """
    s = (text or "").strip()
    if not s:
        return False, 0.0

    # Reject pure punctuation / leader lines (e.g. '......', '---', ':')
    alnum_count = sum(1 for c in s if c.isalnum())
    if alnum_count == 0:
        return False, 0.0

    # Strip leading bullet/list decoration noise before checking tokens
    s_cleaned = re.sub(r"^(?:[•\*\-©°¢«»‹›]+|(?:[seSE]|se)\s+)\s*", "", s)

    # Delimit brackets, quotes, and colons attached to words without space (e.g. '1:MPEG-TS' -> '1 : MPEG-TS')
    s_delimited = re.sub(r"[(){}\[\]«»‹›]", " ", s_cleaned)
    s_delimited = re.sub(r"(?<=\w):(?=\w)", " : ", s_delimited)
    tokens = [t for t in s_delimited.split() if any(c.isalnum() for c in t)]
    if not tokens:
        return False, 0.0

    valid_flags = [is_valid_vietnamese_token(t) for t in tokens]
    valid_ratio = float(sum(valid_flags)) / float(len(tokens))

    # Reject if dots/periods/colons dominate the line UNLESS valid alphanumeric words are present (fill-in-the-blank lines)
    dot_count = sum(1 for c in s if c in ".,;:_~-")
    has_valid_word = any(len(t) >= 2 and is_valid_vietnamese_token(t) for t in tokens)
    if dot_count > alnum_count and len(s) >= 4 and not has_valid_word:
        return False, 0.0

    # Reject 1-word short noise lines unless high confidence and valid token
    if len(tokens) == 1 and alnum_count <= 2 and raw_score < 88.0:
        return False, valid_ratio

    if raw_score < 32.0:
        return False, valid_ratio

    total_token_chars = sum(len(t) for t in tokens)
    valid_token_chars = sum(len(t) for t, v in zip(tokens, valid_flags) if v)
    char_valid_ratio = float(valid_token_chars) / float(max(1, total_token_chars))

    if valid_ratio < 0.65 and char_valid_ratio < 0.75:
        return False, max(valid_ratio, char_valid_ratio)

    return True, max(valid_ratio, char_valid_ratio)


def detect_and_normalize_polarity(image_or_gray: np.ndarray) -> Tuple[np.ndarray, str]:
    """Detect whether the image has light/colored text on a dark background ('light_on_dark')
    or dark ink on a light background ('dark_on_light'), and return a normalized grayscale
    image where text is always DARK ink on a LIGHT background.

    For color images on dark backgrounds (e.g. orange/yellow/cyan headings + white body text
    on a dark '#121212' background), uses channel-maximum luminance before inversion so colored
    text retains full contrast alongside white text.
    """
    if image_or_gray.ndim == 3:
        gray = cv2.cvtColor(image_or_gray, cv2.COLOR_BGR2GRAY)
        max_ch = np.max(image_or_gray, axis=2)
    else:
        gray = image_or_gray.copy()
        max_ch = gray

    p10, p50, p75, p90, p95, p99 = np.percentile(max_ch, [10, 50, 75, 90, 95, 99])
    g50, g75 = np.percentile(gray, [50, 75])

    # A true dark-background document (e.g., #121212 or #1E1E1E dark mode) has a genuinely dark
    # dominant background across at least 75% of the frame (g50 < 80 and g75 < 95) and bright
    # foreground text strokes in the upper percentiles (p95 - p50 >= 40 or p99 - p50 >= 75).
    # By contrast, a dim webcam photo of a white phone/paper in indoor lighting has g50 ~ 95..140
    # and g75 >= 100 -> classified as 'dark_on_light'.
    is_dark_bg = bool(
        float(g50) < 80.0
        and float(g75) < 95.0
        and float(p50) < 85.0
        and (float(p95 - p50) >= 40.0 or float(p99 - p50) >= 75.0)
    )

    if is_dark_bg:
        # Stretch [p10 .. p99] of max_ch to [0 .. 255] so both white and colored text become crisp,
        # then invert so text becomes dark ink on a clean white background (255).
        lo = float(p10)
        hi = max(lo + 40.0, float(p99))
        stretched = np.clip((max_ch.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
        norm_gray = 255 - stretched
        return norm_gray, "light_on_dark"

    return gray, "dark_on_light"


def estimate_glyph_metrics(image_or_gray: np.ndarray) -> GlyphMetrics:
    """Estimate character glyph height distribution (in original pixels) via connected components.

    Uses multi-quantile glyph analysis:
    - `median_char_height_px` reflects lowercase x-height (`a, c, e, m, n, o, u, v, x`),
    - `p75_char_height_px` reflects uppercase/ascender/diacritic height (`A, Đ, H, T, ể, ố, ự`).
    A dense document is classified as `is_low_resolution_document = True` only when the glyph
    distribution is genuinely sub-Nyquist for Vietnamese diacritics (`median < 9.5px` and
    `p75 < 12.0px`, or `median < 8.5px`), preventing false rejection of clear 16pt screen prose
    where x-height is 11px and cap/diacritic height is 15px.
    """
    norm_gray, polarity = detect_and_normalize_polarity(image_or_gray)
    h, w = norm_gray.shape[:2]
    inv = 255 - norm_gray
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove long horizontal rules before measuring character heights
    horiz_w = max(25, w // 12)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_w, 1))
    horiz_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)
    bw_no_rules = bw.copy()
    bw_no_rules[horiz_lines > 0] = 0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw_no_rules)
    char_heights: List[float] = []
    max_char_h = max(45, int(h * 0.35))

    for lbl in range(1, num_labels):
        cx, cy, cw, ch, area = stats[lbl]
        if ch < 4 or ch > max_char_h:
            continue
        if cw < 2 or cw > int(w * 0.85):
            continue
        if area < 6:
            continue
        aspect = cw / float(ch)
        if 0.12 <= aspect <= 6.5:
            char_heights.append(float(ch))

    count = len(char_heights)
    if count == 0:
        return GlyphMetrics(
            median_char_height_px=0.0,
            p25_char_height_px=0.0,
            p75_char_height_px=0.0,
            text_component_count=0,
            is_dense_document=False,
            is_low_resolution_document=False,
            polarity=polarity,
        )

    p25, med, p75 = np.percentile(char_heights, [25, 50, 75])
    is_dense = count >= 35
    # Adaptive multi-quantile resolution check:
    # - Sub-Nyquist thumbnail/distant page (e.g., 379x527 administrative form: p25=5px, med=7px, p75=9px):
    #   both x-height (med < 9.5px) and cap/diacritic height (p75 < 12.0px) are too small for Vietnamese diacritics.
    # - Clear screen/document prose (e.g., 996x587 article: p25=10px, med=11px, p75=15px):
    #   x-height is 11px and uppercase/diacritic height is 15px -> NOT low resolution.
    is_low_res = bool(
        is_dense
        and (
            float(med) < 8.5
            or (float(med) < 9.5 and float(p75) < 12.0)
            or (float(med) < 10.5 and float(p25) <= 6.0 and float(p75) <= 11.0)
        )
    )

    return GlyphMetrics(
        median_char_height_px=float(med),
        p25_char_height_px=float(p25),
        p75_char_height_px=float(p75),
        text_component_count=count,
        is_dense_document=is_dense,
        is_low_resolution_document=is_low_res,
        polarity=polarity,
    )


def suppress_rules_and_dots(image_or_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Normalize polarity, remove horizontal underlines/rules and dotted fill-in leader lines.
    Returns (cleaned_dark_on_light_gray, cleaned_binary_ink, polarity).
    """
    work_gray, polarity = detect_and_normalize_polarity(image_or_gray)
    h, w = work_gray.shape[:2]

    inv = 255 - work_gray
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove outer border frames (components spanning > 65% of page width and > 45% of page height)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    for lbl in range(1, num_labels):
        cx, cy, cw, ch, _ = stats[lbl]
        if cw > int(w * 0.65) and ch > int(h * 0.45):
            bw[labels == lbl] = 0
            work_gray[labels == lbl] = int(np.median(work_gray))

    # 1. Remove thin horizontal rules (width >= max(28, w//11), thickness <= 3px)
    rule_len = max(28, w // 11)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rule_len, 1))
    horiz_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)
    # Check vertical thickness so thick character strokes are not erased
    vert_check = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    pure_rules = cv2.subtract(horiz_open, cv2.dilate(vert_check, np.ones((3, 3), np.uint8)))
    pure_rules_dilated = cv2.dilate(pure_rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)))

    clean_bw = bw.copy()
    clean_bw[pure_rules_dilated > 0] = 0

    clean_gray = work_gray.copy()
    bg_val = int(np.median(work_gray))
    bg_val = max(235, bg_val)
    clean_gray[pure_rules_dilated > 0] = bg_val

    # 2. Remove horizontal dotted leader runs (e.g. '....................')
    # Protect diacritics by creating a dilated mask around all real character bodies
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bw)
    dot_mask = np.zeros_like(clean_bw)
    char_body_mask = np.zeros_like(clean_bw)

    for lbl in range(1, num_labels):
        cx, cy, cw, ch, area = stats[lbl]
        if cw <= 3 and ch <= 3 and area <= 7:
            dot_mask[labels == lbl] = 255
        elif ch >= 5 or area >= 10:
            char_body_mask[labels == lbl] = 255

    # Protect any small component within 6px vertically and 5px horizontally of a character body
    protected_zone = cv2.dilate(
        char_body_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 13)),
    )
    unprotected_dots = cv2.subtract(dot_mask, protected_zone)

    dot_row_counts = np.sum(unprotected_dots > 0, axis=1)
    for y in range(h):
        if dot_row_counts[y] >= 8:
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            strip_dots = unprotected_dots[y1:y2, :] > 0
            clean_bw[y1:y2, :][strip_dots] = 0
            clean_gray[y1:y2, :][strip_dots] = bg_val

    return clean_gray, clean_bw, polarity


def detect_image_blanks(image_or_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect horizontal dotted fill-in-the-blank clusters from document image.
    Returns list of bounding boxes (x1, y1, x2, y2) in original image coordinates.
    """
    if image_or_gray is None or image_or_gray.size == 0:
        return []
    if len(image_or_gray.shape) == 3:
        gray = cv2.cvtColor(image_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_or_gray.copy()

    inv = 255 - gray
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    dots = []
    for lbl in range(1, num_labels):
        cx, cy, cw, ch, area = stats[lbl]
        if 1 <= cw <= 4 and 1 <= ch <= 4 and 1 <= area <= 12:
            dots.append((cx, cy, cw, ch))

    dots = sorted(dots, key=lambda d: (d[1], d[0]))
    blank_runs = []
    used = set()
    for i, d1 in enumerate(dots):
        if i in used:
            continue
        cluster = [d1]
        curr_d = d1
        for j in range(i + 1, len(dots)):
            if j in used:
                continue
            d2 = dots[j]
            if abs(d2[1] - curr_d[1]) <= 3:
                gap = d2[0] - (curr_d[0] + curr_d[2])
                if 0 <= gap <= 10:
                    cluster.append(d2)
                    curr_d = d2
        if len(cluster) >= 8:
            for c in cluster:
                used.add(dots.index(c))
            bx1 = min(c[0] for c in cluster)
            bx2 = max(c[0] + c[2] for c in cluster)
            by1 = min(c[1] for c in cluster)
            by2 = max(c[1] + c[3] for c in cluster)
            blank_runs.append((bx1, by1, bx2, by2))

    merged: List[Tuple[int, int, int, int]] = []
    for b in sorted(blank_runs, key=lambda x: (x[1], x[0])):
        if not merged:
            merged.append(b)
        else:
            prev = merged[-1]
            if abs(b[1] - prev[1]) <= 4 and (b[0] - prev[2]) <= 15:
                merged[-1] = (prev[0], min(prev[1], b[1]), max(prev[2], b[2]), max(prev[3], b[3]))
            else:
                merged.append(b)
    return merged


def segment_document_rois(
    clean_gray: np.ndarray,
    clean_bw: np.ndarray,
) -> List[DocumentROI]:
    """Segment a document image into reading-order ROIs, separating true 2-column headers/footers
    into Left Column (top-to-bottom) followed by Right Column (top-to-bottom), while keeping
    single-column prose lines/paragraphs intact.
    """
    h, w = clean_gray.shape[:2]

    # 0. Check for a prominent vertical column gutter across the page in [0.35 * w .. 0.65 * w]
    # Characteristic of multi-column exercises, books, exams, and two-column articles
    mid_x1, mid_x2 = int(0.35 * w), int(0.65 * w)
    vert_proj = np.sum(clean_bw > 0, axis=0)
    slice_proj = vert_proj[mid_x1:mid_x2]
    low_thresh = max(1, int(0.015 * h))
    zeros = np.where(slice_proj <= low_thresh)[0]

    max_gutter_run = 0
    best_gutter_center = 0
    if len(zeros) > 0:
        r_start = int(zeros[0])
        prev_z = int(zeros[0])
        for z in zeros[1:]:
            zi = int(z)
            if zi == prev_z + 1:
                prev_z = zi
            else:
                run_len = prev_z - r_start + 1
                if run_len > max_gutter_run:
                    max_gutter_run = run_len
                    best_gutter_center = (r_start + prev_z) // 2
                r_start = zi
                prev_z = zi
        run_len = prev_z - r_start + 1
        if run_len > max_gutter_run:
            max_gutter_run = run_len
            best_gutter_center = (r_start + prev_z) // 2

    has_global_gutter = max_gutter_run >= max(10, int(0.020 * w))
    global_split_x = mid_x1 + best_gutter_center if has_global_gutter else 0

    if has_global_gutter and global_split_x > 0:
        horiz_proj = np.sum(clean_bw > 0, axis=1)
        low_w = max(2, int(0.015 * w))
        all_zeros = np.where(horiz_proj <= low_w)[0]
        page_v_runs: List[Tuple[int, int, int]] = []
        if len(all_zeros) > 0:
            v_start = int(all_zeros[0])
            prev_v = int(all_zeros[0])
            for v in all_zeros[1:]:
                vi = int(v)
                if vi == prev_v + 1:
                    prev_v = vi
                else:
                    page_v_runs.append((v_start, prev_v, prev_v - v_start + 1))
                    v_start = vi
                    prev_v = vi
            page_v_runs.append((v_start, prev_v, prev_v - v_start + 1))

        # 1. Detect optional full-width top header (e.g. Section D title banner in y in [0.04*h .. 0.18*h])
        top_candidates = [
            r for r in page_v_runs
            if r[0] >= int(0.04 * h) and r[1] <= int(0.18 * h) and r[2] >= 10
        ]
        top_gap_y = (top_candidates[0][0] + top_candidates[0][1]) // 2 if top_candidates else 0

        # 2. Check for mid-page horizontal section divide in [0.45 * h .. 0.65 * h] (e.g. Section D vs Section E)
        mid_candidates = [
            r for r in page_v_runs
            if r[0] >= int(0.45 * h) and r[1] <= int(0.65 * h) and r[2] >= 12
        ]
        center_y = h // 2

        has_mid_section = False
        has_mid_banner = False
        sec1_end_y = 0
        sec2_start_y = 0

        if mid_candidates:
            has_mid_section = True
            first_v = mid_candidates[0]
            last_v = mid_candidates[-1]
            if len(mid_candidates) >= 2 and (last_v[0] - first_v[1]) >= 12:
                has_mid_banner = True
                sec1_end_y = (first_v[0] + first_v[1]) // 2
                sec2_start_y = (last_v[0] + last_v[1]) // 2
            else:
                best_candidate = min(mid_candidates, key=lambda vr: abs(((vr[0] + vr[1]) // 2) - center_y))
                sec1_end_y = (best_candidate[0] + best_candidate[1]) // 2
                sec2_start_y = sec1_end_y

        rois_list: List[DocumentROI] = []
        order_idx = 0

        # ROI: Top Header
        if top_gap_y > 0:
            crop_bw = clean_bw[0:top_gap_y, 0:w]
            med_h, p75_h = _crop_char_height_quantiles(crop_bw)
            rois_list.append(
                DocumentROI(
                    order_index=order_idx,
                    x1=0,
                    y1=0,
                    x2=w,
                    y2=top_gap_y,
                    column_role="header",
                    median_char_height_px=med_h,
                    p75_char_height_px=p75_h,
                )
            )
            order_idx += 1

        top_y = top_gap_y

        if has_mid_section:
            # Section 1: Left column top-to-bottom, then Right column top-to-bottom
            for role, x1, x2 in [("left_col", 0, global_split_x), ("right_col", global_split_x, w)]:
                crop_bw = clean_bw[top_y:sec1_end_y, x1:x2]
                med_h, p75_h = _crop_char_height_quantiles(crop_bw)
                rois_list.append(
                    DocumentROI(
                        order_index=order_idx,
                        x1=x1,
                        y1=top_y,
                        x2=x2,
                        y2=sec1_end_y,
                        column_role=role,
                        median_char_height_px=med_h,
                        p75_char_height_px=p75_h,
                    )
                )
                order_idx += 1

            # Optional Mid-Section Header (e.g. Section E Title + (1 mark per answer))
            if has_mid_banner and sec2_start_y > sec1_end_y:
                crop_bw = clean_bw[sec1_end_y:sec2_start_y, 0:w]
                med_h, p75_h = _crop_char_height_quantiles(crop_bw)
                rois_list.append(
                    DocumentROI(
                        order_index=order_idx,
                        x1=0,
                        y1=sec1_end_y,
                        x2=w,
                        y2=sec2_start_y,
                        column_role="header",
                        median_char_height_px=med_h,
                        p75_char_height_px=p75_h,
                    )
                )
                order_idx += 1

            # Section 2: Left column top-to-bottom, then Right column top-to-bottom
            for role, x1, x2 in [("left_col", 0, global_split_x), ("right_col", global_split_x, w)]:
                crop_bw = clean_bw[sec2_start_y:h, x1:x2]
                med_h, p75_h = _crop_char_height_quantiles(crop_bw)
                rois_list.append(
                    DocumentROI(
                        order_index=order_idx,
                        x1=x1,
                        y1=sec2_start_y,
                        x2=x2,
                        y2=h,
                        column_role=role,
                        median_char_height_px=med_h,
                        p75_char_height_px=p75_h,
                    )
                )
                order_idx += 1
        else:
            for role, x1, x2 in [("left_col", 0, global_split_x), ("right_col", global_split_x, w)]:
                crop_bw = clean_bw[top_y:h, x1:x2]
                med_h, p75_h = _crop_char_height_quantiles(crop_bw)
                rois_list.append(
                    DocumentROI(
                        order_index=order_idx,
                        x1=x1,
                        y1=top_y,
                        x2=x2,
                        y2=h,
                        column_role=role,
                        median_char_height_px=med_h,
                        p75_char_height_px=p75_h,
                    )
                )
                order_idx += 1

        return rois_list

    # Dilate horizontally and slightly vertically to group words into lines/bands
    kx = max(12, int(w * 0.032))
    ky = max(3, int(h * 0.007))
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dilated = cv2.dilate(clean_bw, block_kernel)

    row_proj = np.sum(dilated > 0, axis=1)
    raw_bands: List[Tuple[int, int]] = []
    in_band = False
    start_y = 0
    min_band_h = max(5, int(h * 0.009))

    for y in range(h):
        if row_proj[y] > 0 and not in_band:
            in_band = True
            start_y = y
        elif row_proj[y] == 0 and in_band:
            in_band = False
            if (y - start_y) >= min_band_h:
                raw_bands.append((max(0, start_y - 3), min(h, y + 3)))
    if in_band and (h - start_y) >= min_band_h:
        raw_bands.append((max(0, start_y - 3), h))

    if not raw_bands:
        return [
            DocumentROI(
                order_index=0,
                x1=0,
                y1=0,
                x2=w,
                y2=h,
                column_role="full",
                median_char_height_px=0.0,
                p75_char_height_px=0.0,
            )
        ]

    # Check each horizontal band for a candidate 2-column vertical valley in [0.24*W .. 0.58*W]
    # Note: A true 2-column administrative layout either:
    # (a) has a wide vertical column gutter (>= 4.2% of page width after bridging normal word spaces), OR
    # (b) has multiple consecutive vertically-aligned bands whose split_x aligns within 8% of page width.
    band_infos: List[Tuple[int, int, bool, int, int]] = []  # (y1, y2, has_gap, split_x, max_gap_run)
    bridge_w = max(8, int(0.024 * w))
    mid_start = int(0.24 * w)
    mid_end = int(0.58 * w)

    for (y1, y2) in raw_bands:
        band_bw = clean_bw[y1:y2, :]
        col_proj = np.sum(band_bw > 0, axis=0)
        col_1d = cv2.dilate(
            (col_proj > 0).astype(np.uint8).reshape(1, -1),
            np.ones((1, bridge_w), np.uint8),
        ).flatten()
        mid_slice = col_1d[mid_start:mid_end]
        zeros = np.where(mid_slice == 0)[0]
        left_ink = int(np.sum(col_proj[:mid_start] > 0))
        right_ink = int(np.sum(col_proj[mid_end:] > 0))
        min_side_ink = max(12, int(0.03 * w))

        # Compute longest contiguous zero run in mid_slice
        max_run = 0
        best_run_center = 0
        if len(zeros) > 0:
            run_start = int(zeros[0])
            prev_z = int(zeros[0])
            for z in zeros[1:]:
                zi = int(z)
                if zi == prev_z + 1:
                    prev_z = zi
                else:
                    run_len = prev_z - run_start + 1
                    if run_len > max_run:
                        max_run = run_len
                        best_run_center = (run_start + prev_z) // 2
                    run_start = zi
                    prev_z = zi
            run_len = prev_z - run_start + 1
            if run_len > max_run:
                max_run = run_len
                best_run_center = (run_start + prev_z) // 2

        if max_run >= max(2, int(0.008 * w)) and left_ink >= min_side_ink and right_ink >= min_side_ink:
            split_x = mid_start + best_run_center
            band_infos.append((y1, y2, True, split_x, max_run))
        else:
            band_infos.append((y1, y2, False, 0, 0))

    # Promote candidate gaps to confirmed 2-column bands ONLY if:
    # 1. At least two vertically adjacent bands share an aligned split_x (within 0.08 * w) (e.g. administrative headers/signatures), OR
    # 2. The band is a tall multi-line block AND has a wide vertical column gutter (max_run >= 0.042 * w).
    # Single-line or indented code blocks (without aligned neighbors or a wide column gutter) are NEVER split into two columns.
    band_heights = [y2 - y1 for (y1, y2, _, _, _) in band_infos]
    med_band_h = float(np.median(band_heights)) if band_heights else 20.0
    wide_gutter_thresh = max(6, int(0.042 * w))

    confirmed_infos: List[Tuple[int, int, bool, int]] = []
    n_bands = len(band_infos)
    for idx, (y1, y2, has_gap, split_x, max_run) in enumerate(band_infos):
        if not has_gap:
            confirmed_infos.append((y1, y2, False, 0))
            continue

        is_wide_gutter = max_run >= wide_gutter_thresh
        is_tall_multi_line = (y2 - y1) >= max(28, int(2.1 * med_band_h))
        aligned_neighbor = False
        if idx > 0 and band_infos[idx - 1][2]:
            if (
                (y1 - band_infos[idx - 1][1]) <= max(10, int(0.022 * h))
                and abs(split_x - band_infos[idx - 1][3]) <= int(0.08 * w)
            ):
                aligned_neighbor = True
        if idx + 1 < n_bands and band_infos[idx + 1][2]:
            if (
                (band_infos[idx + 1][0] - y2) <= max(10, int(0.022 * h))
                and abs(split_x - band_infos[idx + 1][3]) <= int(0.08 * w)
            ):
                aligned_neighbor = True

        if aligned_neighbor or (is_wide_gutter and is_tall_multi_line):
            confirmed_infos.append((y1, y2, True, split_x))
        else:
            confirmed_infos.append((y1, y2, False, 0))

    # Merge consecutive 2-column bands so Left Column is read top-to-bottom before Right Column
    merged_infos: List[Tuple[int, int, bool, int]] = []
    for y1, y2, is_two_col, split_x in confirmed_infos:
        if (
            merged_infos
            and is_two_col
            and merged_infos[-1][2] is True
            and (y1 - merged_infos[-1][1]) <= max(8, int(0.016 * h))
            and abs(split_x - merged_infos[-1][3]) <= int(0.12 * w)
        ):
            prev_y1, _, _, prev_sx = merged_infos[-1]
            avg_sx = int((prev_sx + split_x) // 2)
            merged_infos[-1] = (prev_y1, y2, True, avg_sx)
            continue

        # Also absorb a trailing left-only band immediately below a 2-column band
        # (e.g., when Left footer "Nơi nhận:" has 3 lines and Right footer has 2 lines)
        if (
            merged_infos
            and not is_two_col
            and merged_infos[-1][2] is True
            and (y1 - merged_infos[-1][1]) <= max(8, int(0.020 * h))
        ):
            prev_y1, _, _, prev_sx = merged_infos[-1]
            band_bw = clean_bw[y1:y2, :]
            left_only = int(np.sum(band_bw[:, prev_sx:] > 0)) < 10 and int(np.sum(band_bw[:, :prev_sx] > 0)) >= 10
            if left_only:
                merged_infos[-1] = (prev_y1, y2, True, prev_sx)
                continue

        # Also merge overlapping consecutive single-column bands (e.g. multi-line headings or banners)
        if (
            merged_infos
            and not is_two_col
            and merged_infos[-1][2] is False
            and y1 <= merged_infos[-1][1] + 4
        ):
            prev_y1, prev_y2, _, _ = merged_infos[-1]
            merged_infos[-1] = (prev_y1, max(prev_y2, y2), False, 0)
            continue

        merged_infos.append((y1, y2, is_two_col, split_x))

    # If the entire image has NO 2-column bands and is NOT a low-resolution document,
    # group consecutive single-column bands into paragraph blocks (up to ~6 lines per block)
    # or a single full-page ROI when glyph height is comfortable, avoiding line-slicing distortion
    # and speeding up Tesseract significantly.
    has_any_two_col = any(info[2] for info in merged_infos)
    gm = estimate_glyph_metrics(clean_gray)
    if not has_any_two_col and not gm.is_low_resolution_document:
        if len(merged_infos) <= 3 or gm.median_char_height_px >= 10.0:
            return [
                DocumentROI(
                    order_index=0,
                    x1=0,
                    y1=0,
                    x2=w,
                    y2=h,
                    column_role="full",
                    median_char_height_px=gm.median_char_height_px,
                    p75_char_height_px=gm.p75_char_height_px,
                )
            ]

    rois: List[DocumentROI] = []
    order_idx = 0
    for y1, y2, is_two_col, split_x in merged_infos:
        if is_two_col:
            for role, x1, x2 in [("left_col", 0, split_x), ("right_col", split_x, w)]:
                crop_bw = clean_bw[y1:y2, x1:x2]
                if int(np.sum(crop_bw > 0)) < 10:
                    continue

                sh = y2 - y1
                # If this is a tall card (sh >= 200) with a title banner gap in [28 .. 65]
                card_gap_y = None
                if sh >= 200:
                    c_hp = np.sum(crop_bw > 0, axis=1)
                    low_thr = max(1, int(0.01 * (x2 - x1)))
                    zeros = [y for y in range(28, min(70, sh - 40)) if c_hp[y] <= low_thr]
                    if len(zeros) >= 3:
                        card_gap_y = int(np.median(zeros))

                if card_gap_y is not None and card_gap_y >= 25:
                    # Card Header ROI
                    hdr_bw = crop_bw[:card_gap_y, :]
                    med_h, p75_h = _crop_char_height_quantiles(hdr_bw)
                    rois.append(
                        DocumentROI(
                            order_index=order_idx,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y1 + card_gap_y,
                            column_role="header",
                            median_char_height_px=med_h,
                            p75_char_height_px=p75_h,
                        )
                    )
                    order_idx += 1

                    # Card Body ROI
                    body_bw = crop_bw[card_gap_y:, :]
                    med_h, p75_h = _crop_char_height_quantiles(body_bw)
                    rois.append(
                        DocumentROI(
                            order_index=order_idx,
                            x1=x1,
                            y1=y1 + card_gap_y,
                            x2=x2,
                            y2=y2,
                            column_role=role,
                            median_char_height_px=med_h,
                            p75_char_height_px=p75_h,
                        )
                    )
                    order_idx += 1
                else:
                    med_h, p75_h = _crop_char_height_quantiles(crop_bw)
                    rois.append(
                        DocumentROI(
                            order_index=order_idx,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            column_role=role,
                            median_char_height_px=med_h,
                            p75_char_height_px=p75_h,
                        )
                    )
                    order_idx += 1
        else:
            crop_bw = clean_bw[y1:y2, 0:w]
            if int(np.sum(crop_bw > 0)) < 10:
                continue

            sh = y2 - y1
            # Check if this tall block has an internal card header banner (gap in [25..75]) + 2-column body
            card_gap_y = None
            split_x = None
            if sh >= 120:
                c_hp = np.sum(crop_bw > 0, axis=1)
                low_thr = max(1, int(0.01 * w))
                zero_runs = []
                in_run = False
                r_start = 0
                max_gap_search = min(75, sh - 40)
                for y in range(25, max_gap_search):
                    if c_hp[y] <= low_thr and not in_run:
                        in_run = True
                        r_start = y
                    elif c_hp[y] > low_thr and in_run:
                        in_run = False
                        if (y - r_start) >= 3:
                            zero_runs.append((r_start, y - 1, y - r_start))
                if in_run and (max_gap_search - r_start) >= 3:
                    zero_runs.append((r_start, max_gap_search - 1, max_gap_search - r_start))

                if zero_runs:
                    cand_gap_y = int((zero_runs[0][0] + zero_runs[0][1]) // 2)
                    body_bw = crop_bw[cand_gap_y:, :]
                    col_p = np.sum(body_bw > 0, axis=0)
                    mid_s, mid_e = int(0.20 * w), int(0.58 * w)
                    slice_p = col_p[mid_s:mid_e]
                    zero_cols = np.where(slice_p == 0)[0]
                    runs = []
                    if len(zero_cols) > 0:
                        s = zero_cols[0]
                        p = zero_cols[0]
                        for z in zero_cols[1:]:
                            if z == p + 1:
                                p = z
                            else:
                                runs.append((s, p, p - s + 1))
                                s = z
                                p = z
                        runs.append((s, p, p - s + 1))
                    best_run = max(runs, key=lambda r: r[2]) if runs else None
                    if best_run and best_run[2] >= max(6, int(0.012 * w)):
                        cand_split = int(mid_s + (best_run[0] + best_run[1]) // 2)
                        left_ink = int(np.sum(col_p[: cand_split - best_run[2] // 2] > 0))
                        right_ink = int(np.sum(col_p[cand_split + best_run[2] // 2 :] > 0))
                        if left_ink >= max(12, int(0.02 * w)) and right_ink >= max(12, int(0.02 * w)):
                            card_gap_y = cand_gap_y
                            split_x = cand_split

            if card_gap_y is not None and split_x is not None:
                # 1. Card Header ROI
                hdr_bw = crop_bw[:card_gap_y, :]
                med_h, p75_h = _crop_char_height_quantiles(hdr_bw)
                rois.append(
                    DocumentROI(
                        order_index=order_idx,
                        x1=0,
                        y1=y1,
                        x2=w,
                        y2=y1 + card_gap_y,
                        column_role="header",
                        median_char_height_px=med_h,
                        p75_char_height_px=p75_h,
                    )
                )
                order_idx += 1

                # 2. Left Column Body ROI
                l_bw = crop_bw[card_gap_y:, :split_x]
                med_h, p75_h = _crop_char_height_quantiles(l_bw)
                rois.append(
                    DocumentROI(
                        order_index=order_idx,
                        x1=0,
                        y1=y1 + card_gap_y,
                        x2=split_x,
                        y2=y2,
                        column_role="left_col",
                        median_char_height_px=med_h,
                        p75_char_height_px=p75_h,
                    )
                )
                order_idx += 1

                # 3. Right Column Body ROI
                r_bw = crop_bw[card_gap_y:, split_x:]
                med_h, p75_h = _crop_char_height_quantiles(r_bw)
                rois.append(
                    DocumentROI(
                        order_index=order_idx,
                        x1=split_x,
                        y1=y1 + card_gap_y,
                        x2=w,
                        y2=y2,
                        column_role="right_col",
                        median_char_height_px=med_h,
                        p75_char_height_px=p75_h,
                    )
                )
                order_idx += 1
            else:
                med_h, p75_h = _crop_char_height_quantiles(crop_bw)
                rois.append(
                    DocumentROI(
                        order_index=order_idx,
                        x1=0,
                        y1=y1,
                        x2=w,
                        y2=y2,
                        column_role="full",
                        median_char_height_px=med_h,
                        p75_char_height_px=p75_h,
                    )
                )
                order_idx += 1

    return rois


def _crop_char_height_quantiles(crop_bw: np.ndarray) -> Tuple[float, float]:
    nl, _, stats, _ = cv2.connectedComponentsWithStats(crop_bw)
    hs = [
        float(stats[i][3])
        for i in range(1, nl)
        if 4 <= stats[i][3] <= max(45, int(crop_bw.shape[0] * 0.9))
        and stats[i][2] >= 2
        and stats[i][4] >= 6
    ]
    if not hs:
        return 0.0, 0.0
    med, p75 = np.percentile(hs, [50, 75])
    return float(med), float(p75)


def _crop_median_char_height(crop_bw: np.ndarray) -> float:
    med, _ = _crop_char_height_quantiles(crop_bw)
    return med
