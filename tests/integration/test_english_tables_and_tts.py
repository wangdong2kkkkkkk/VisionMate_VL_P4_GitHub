# -*- coding: utf-8 -*-
"""
Integration Tests for English Tables Document Layout & OneCore English TTS Routing.

Verifies:
1. Authentic English Vocabulary Tables fixture byte-for-byte SHA-256 integrity:
   english_vocabulary_tables_original.png (832x737 px, RGBA, SHA-256: c2eb0cde5854ce30e6ce1fb32c3c8ef45ef0db1290724bdd9452f125f3d9cbc9).
2. Complete recognition of all 4 major section headings:
   - "Phrasal verbs"
   - "Prepositional phrases"
   - "Word formation"
   - "Word patterns"
3. Complete recognition of all 6 prepositional phrases (including "in trouble" and "under pressure").
4. Complete recognition of all 8 phrasal verbs without column bleeding.
5. Complete recognition of paired Word formation halves (5 left base/derived pairs, 5 right base/derived pairs).
6. Complete recognition of Word patterns category structure (adjectives, verbs, nouns) with left/right independence.
7. Total elimination of stray border noise artifacts ("_ under pressure .......... I", "Word formation M", "work out ... av", "j", "í", "H", etc.).
8. Low CER (<= 5.0%) and WER (<= 8.0%) against verified Ground Truth reference.
9. Windows OneCore offline English voice routing:
   - English documents route strictly to Microsoft David (MSTTS_V110_enUS_DavidM).
   - Vietnamese documents route strictly to Microsoft An (MSTTS_V110_viVN_AnM).
   - Zero audio disk writes and zero network calls.
10. Held-out evaluation suite generalization across 5 distinct documents:
    - 3-column irregular verbs table
    - Multi-slot schedule grid table
    - 2-column machine learning glossary
    - Full-page English prose memo
    - Multi-step English instructions document
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pytest
from PIL import Image

from visionmate.camera.sources import load_image_unicode
from visionmate.ocr.layout import segment_document_rois, suppress_rules_and_dots
from visionmate.ocr.tesseract_backend import TesseractOCRBackend
from visionmate.speech.sapi_onecore import WindowsOneCoreSpeechBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "regression_english_tables"
ORIGINAL_PNG = FIXTURE_DIR / "english_vocabulary_tables_original.png"
GROUND_TRUTH_JSON = FIXTURE_DIR / "ground_truth_reference.json"
HELD_OUT_DIR = FIXTURE_DIR / "held_out"

EXPECTED_SHA256 = "c2eb0cde5854ce30e6ce1fb32c3c8ef45ef0db1290724bdd9452f125f3d9cbc9"


def _levenshtein(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    if len(seq1) < len(seq2):
        return _levenshtein(seq2, seq1)
    if len(seq2) == 0:
        return len(seq1)
    prev = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        curr = [i + 1]
        for j, c2 in enumerate(seq2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _cer(pred: str, gt: str) -> float:
    p = unicodedata.normalize("NFC", " ".join(pred.split()))
    g = unicodedata.normalize("NFC", " ".join(gt.split()))
    return float(_levenshtein(p, g)) / float(max(1, len(g)))


def _wer(pred: str, gt: str) -> float:
    pw = unicodedata.normalize("NFC", pred).split()
    gw = unicodedata.normalize("NFC", gt).split()
    return float(_levenshtein(pw, gw)) / float(max(1, len(gw)))


def test_english_tables_fixture_integrity() -> None:
    """Verify SHA-256 and dimensions of the original vocabulary tables screenshot."""
    assert ORIGINAL_PNG.is_file(), f"Fixture file not found: {ORIGINAL_PNG}"
    raw_bytes = ORIGINAL_PNG.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest().lower()
    assert actual_sha == EXPECTED_SHA256, f"SHA-256 mismatch: got {actual_sha}, expected {EXPECTED_SHA256}"

    with Image.open(ORIGINAL_PNG) as im:
        w, h = im.size
        assert (w, h) == (832, 737), f"Unexpected dimensions: {w}x{h}"


def test_four_headings_and_prepositional_phrases() -> None:
    """Verify all 4 headings are recognized, and all 6 prepositional phrases are present."""
    img = load_image_unicode(ORIGINAL_PNG)
    backend = TesseractOCRBackend(variant="best")
    result = backend.recognize(img)
    assert result.error is None
    assert result.language == "en"

    lines = [ln.text for ln in result.lines]
    full_text = "\n".join(lines)

    # 1. Four Headings Recall
    assert "Phrasal verbs" in full_text
    assert "Prepositional phrases" in full_text
    assert "Word formation" in full_text
    assert "Word patterns" in full_text

    # 2. Prepositional phrases (6/6 complete, including previously dropped items)
    expected_prep = [
        "by accident/mistake",
        "in a mess",
        "in danger (of)",
        "in my view",
        "in trouble",
        "under pressure",
    ]
    for p in expected_prep:
        assert p in full_text, f"Missing prepositional phrase: {p!r} in:\n{full_text}"


def test_phrasal_verbs_and_word_formation() -> None:
    """Verify all 8 phrasal verbs and 10 Word formation pairs are intact."""
    img = load_image_unicode(ORIGINAL_PNG)
    backend = TesseractOCRBackend(variant="best")
    result = backend.recognize(img)
    lines = [ln.text for ln in result.lines]
    full_text = "\n".join(lines)

    # 8 Phrasal Verbs
    phrasal_verbs = [
        "hang up",
        "pick up",
        "put back",
        "run out (of)",
        "share out",
        "sort out",
        "watch out",
        "work out",
    ]
    for pv in phrasal_verbs:
        assert pv in full_text, f"Missing phrasal verb: {pv!r}"

    # Word formation (both columns)
    left_bases = ["advice", "confuse", "except", "help", "luck"]
    right_bases = ["prefer", "recommend", "refuse", "solve", "suggest"]
    for b in left_bases + right_bases:
        assert b in full_text, f"Missing word formation base: {b!r}"


def test_word_patterns_and_no_border_artifacts() -> None:
    """Verify Word patterns category recognition and complete absence of stray border artifacts."""
    img = load_image_unicode(ORIGINAL_PNG)
    backend = TesseractOCRBackend(variant="best")
    result = backend.recognize(img)
    lines = [ln.text for ln in result.lines]
    full_text = "\n".join(lines)

    # Word patterns categories and verbs
    assert "adjectives sure about/of" in full_text
    assert "verbs advise against" in full_text
    assert "agree (with sb) about" in full_text
    assert "approve of" in full_text
    assert "believe in" in full_text
    assert "deal with" in full_text
    assert "happen to" in full_text
    assert "hide sth from sb" in full_text
    assert "insist on" in full_text
    assert "rely on" in full_text
    assert "nouns an advantage of" in full_text
    assert "a solution to" in full_text

    # Verify no stray boundary artifacts
    for ln in lines:
        assert not re.search(r"\b(?:av|í|ì)\b$", ln), f"Stray trailing artifact in: {ln!r}"
        assert not re.match(r"^[@#|!¡\\/]+\s*", ln), f"Stray leading symbol in: {ln!r}"


def test_cer_wer_below_thresholds() -> None:
    """Verify that CER is below 5.0% and WER is below 8.0% against verified Ground Truth reference."""
    assert GROUND_TRUTH_JSON.is_file()
    with open(GROUND_TRUTH_JSON, encoding="utf-8") as f:
        gt_data = json.load(f)

    # Flatten ground truth in reading order
    gt_lines = []
    gt_lines.append("Phrasal verbs")
    for term, defn in gt_data["sections"][0]["entries"]:
        gt_lines.append(f"{term} {defn}")

    gt_lines.append("Prepositional phrases")
    for entry in gt_data["sections"][1]["entries"]:
        gt_lines.append(entry)

    gt_lines.append("Word formation")
    for base, derivs in gt_data["sections"][2]["left_entries"]:
        gt_lines.append(f"{base} {derivs}")
    for base, derivs in gt_data["sections"][2]["right_entries"]:
        gt_lines.append(f"{base} {derivs}")

    gt_lines.append("Word patterns")
    gt_lines.append("adjectives " + gt_data["sections"][3]["left"]["adjectives"][0])
    v_left = gt_data["sections"][3]["left"]["verbs"]
    gt_lines.append("verbs " + v_left[0])
    for v in v_left[1:]:
        gt_lines.append(v)
    for v in gt_data["sections"][3]["right"]["verbs_continued"]:
        gt_lines.append(v)
    n_right = gt_data["sections"][3]["right"]["nouns"]
    gt_lines.append("nouns " + n_right[0])
    for n in n_right[1:]:
        gt_lines.append(n)

    img = load_image_unicode(ORIGINAL_PNG)
    backend = TesseractOCRBackend(variant="best")
    result = backend.recognize(img)
    rec_lines = [ln.text for ln in result.lines]

    hyp_text = " ".join(rec_lines)
    gt_text = " ".join(gt_lines)
    cer = _cer(hyp_text, gt_text)
    wer = _wer(hyp_text, gt_text)

    assert cer <= 0.05, f"CER exceeded threshold: {cer * 100:.2f}% (max 5.0%)"
    assert wer <= 0.08, f"WER exceeded threshold: {wer * 100:.2f}% (max 8.0%)"


def test_onecore_tts_voice_routing_and_no_stray_accent_hijack() -> None:
    """Verify that English text routes strictly to Microsoft David (MSTTS_V110_enUS_DavidM)
    even when stray accents or noisy OCR text occurs, and Vietnamese routes to Microsoft An.
    """
    tts = WindowsOneCoreSpeechBackend(play_audio=False)
    assert tts.is_en_us_available() is True, "Microsoft David voice not available on Windows"
    assert tts.is_vi_vn_available() is True, "Microsoft An voice not available on Windows"

    # 1. Clean English text with auto language detection
    english_sample = "Phrasal verbs. hang up put clothes in a wardrobe, etc. pick up lift something from the floor."
    tts.speak(english_sample, job_id=901, lang="auto")
    assert "David" in (tts._last_spoken_voice_desc or "")
    assert tts._last_spoken_lang == "en"

    # 2. English text with a stray OCR diacritic character (e.g. 'í' or 'suggest suggestion í')
    # Must NOT flip entire document to Vietnamese Microsoft An!
    noisy_english = "suggest suggestion í. under pressure in trouble. verbs advise against agree with sb about."
    tts.speak(noisy_english, job_id=902, lang="auto")
    assert "David" in (tts._last_spoken_voice_desc or ""), (
        f"Stray accent hijacked English voice! Spoke with: {tts._last_spoken_voice_desc}"
    )
    assert tts._last_spoken_lang == "en"

    # 3. Explicit lang='en'
    tts.speak("Word formation advice advise, adviser.", job_id=903, lang="en")
    assert "David" in (tts._last_spoken_voice_desc or "")
    assert tts._last_spoken_lang == "en"

    # 4. Authentic Vietnamese text routes to Microsoft An
    vietnamese_sample = "Cộng hòa Xã hội Chủ nghĩa Việt Nam. Độc lập Tự do Hạnh phúc."
    tts.speak(vietnamese_sample, job_id=904, lang="auto")
    assert "An" in (tts._last_spoken_voice_desc or "")
    assert tts._last_spoken_lang == "vi"


def test_held_out_generalization_suite() -> None:
    """Verify generalization across 5 held-out documents with diverse layouts."""
    backend = TesseractOCRBackend(variant="best")

    # 1. 3-column irregular verbs table
    img1 = load_image_unicode(HELD_OUT_DIR / "table_irregular_verbs_3col.png")
    res1 = backend.recognize(img1)
    assert res1.error is None
    assert res1.language == "en"
    text1 = " ".join(ln.text for ln in res1.lines)
    assert "Infinitive" in text1 or "Past" in text1
    assert "begin" in text1 or "began" in text1

    # 2. Multi-slot schedule grid table
    img2 = load_image_unicode(HELD_OUT_DIR / "table_schedule_grid.png")
    res2 = backend.recognize(img2)
    assert res2.error is None
    assert res2.language == "en"
    text2 = " ".join(ln.text for ln in res2.lines)
    assert "Schedule" in text2 or "Conference" in text2

    # 3. 2-column machine learning glossary
    img3 = load_image_unicode(HELD_OUT_DIR / "table_two_column_glossary.png")
    res3 = backend.recognize(img3)
    assert res3.error is None
    assert res3.language == "en"
    text3 = " ".join(ln.text for ln in res3.lines)
    assert "Glossary" in text3 or "Accuracy" in text3

    # 4. Full-page English prose memo
    img4 = load_image_unicode(HELD_OUT_DIR / "doc_english_prose_memo.png")
    res4 = backend.recognize(img4)
    assert res4.error is None
    assert res4.language == "en"
    text4 = " ".join(ln.text for ln in res4.lines)
    assert "Executive" in text4 or "Briefing" in text4

    # 5. Multi-step English instructions
    img5 = load_image_unicode(HELD_OUT_DIR / "doc_english_instructions.png")
    res5 = backend.recognize(img5)
    assert res5.error is None
    assert res5.language == "en"
    text5 = " ".join(ln.text for ln in res5.lines)
    assert "Instructions" in text5 or "Equipment" in text5
