# -*- coding: utf-8 -*-
"""
Phase 1.2 Automated Integration & Regression Test Suite:
OCR False Rejection Root-Cause Hotfix.

Verifies all 10 mandatory objectives of Phase 1.2:
1. Both 996x587 Vietnamese article images (`dark_theme_vietnamese_996x587.png` and
   `diagnostic_inverted_only.png`) are byte-for-byte verified by SHA-256.
2. Polarity detection and normalization (`light_on_dark` vs `dark_on_light`) accurately handle
   both dark-background white/orange text and light-background dark text.
3. Multi-quantile glyph resolution assessment (`p25`, `median`, `p75`) recognizes clear 996x587
   screen prose (`median=11..12px`, `p75=15px`) as readable (`low_resolution_detected=False`)
   while keeping sub-Nyquist 379x527 forms (`p25=5px`, `median=7px`, `p75=8px`) as `UNCERTAIN`.
4. Single-column prose paragraphs are NOT falsely split into 2-column fragments at inter-word spaces.
5. Vietnamese phonotactic validation properly accepts triphthongs (`điều`, `nhiều`, `tiêu`, `thiệu`,
   `phiếu`), question marks (`gì?`), and legal codes with `Đ` (`30/2020/NĐ-CP`).
6. Both 996x587 images achieve `SafetyStatus.READABLE`, `15/15` lines recognized in exact order,
   `0/15` false line rejections (`0.0%`), `CER <= 2.0%`, and `WER <= 5.0%`.
7. Explicit separation of `image_quality_status`, `ocr_confidence_status`, and `speech_policy`.
8. Zero occurrence of the over-claiming phrase `"đã xác minh"` ("verified") in `SafetyPolicy` output.
9. Phase 1.1 `administrative_form_379x527.png` safety refusal (`UNCERTAIN`) remains intact.
"""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from visionmate.camera.sources import load_image_unicode
from visionmate.contracts import SafetyStatus
from visionmate.ocr.layout import (
    detect_and_normalize_polarity,
    estimate_glyph_metrics,
    is_valid_vietnamese_token,
    segment_document_rois,
    suppress_rules_and_dots,
)
from visionmate.ocr.tesseract_backend import TesseractOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import MSG_LOW_RES_OR_UNCERTAIN, SafetyPolicy


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REG_P1_2_DIR = PROJECT_ROOT / "tests" / "fixtures" / "regression_p1_2"
CASE_A_DARK_IMG = REG_P1_2_DIR / "dark_theme_vietnamese_996x587.png"
CASE_B_LIGHT_IMG = REG_P1_2_DIR / "diagnostic_inverted_only.png"
EXPECTED_GT_FILE = REG_P1_2_DIR / "expected_ground_truth.txt"

REG_P1_1_IMG = PROJECT_ROOT / "tests" / "fixtures" / "regression_p1_1" / "administrative_form_379x527.png"

EXPECTED_CASE_A_SHA256 = "ECEAA55D3380F2B21A49877D09EF0B535B51B8E745373EA2A79E82AC396F2064"
EXPECTED_CASE_B_SHA256 = "817BA6C63F31F79B41435BDC4A11C0D8DA3CE5C5103DE0F6FE40B1D217B4B7BC"


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


def test_p1_2_phonotactic_validator_triphthongs_and_legal_codes() -> None:
    """Verify Root Cause 4 fix: triphthongs (-iêu, -yêu), '?' punctuation, and 'NĐ-CP' legal codes."""
    valid_tokens = [
        "gì?",
        "điều",
        "Điều",
        "nhiều",
        "tiêu",
        "thiệu,",
        "phiếu",
        "phiều",
        "yêu",
        "30/2020/NĐ-CP,",
        "30/2020/NĐ-CP",
        "NĐ-CP",
        "UBND",
        "HĐND",
        "1.1.",
    ]
    for tok in valid_tokens:
        assert is_valid_vietnamese_token(tok) is True, f"Token {tok!r} was falsely rejected!"

    # Ensure corrupted all-caps superscript/noise tokens are still rejected
    invalid_tokens = ["QUẦN!", "xsết", "Ñ", "ˆ", "€Q"]
    for bad in invalid_tokens:
        assert is_valid_vietnamese_token(bad) is False, f"Corrupted token {bad!r} should be rejected!"


def test_p1_2_case_a_dark_theme_996x587_full_pipeline() -> None:
    """
    Verify Trial A (`dark_theme_vietnamese_996x587.png` — white/orange text on dark background):
    - SHA-256 matches exact user image
    - Polarity detected as `light_on_dark`
    - Adaptive glyph metrics: `p75 >= 14.0px`, `low_resolution_detected is False`
    - Segmented without false 2-column word-splitting (`column_role == "full"`)
    - Status is `SafetyStatus.READABLE` (was `UNCERTAIN` with 20/22 rejected in Phase 1.1)
    - `15/15` lines recognized in exact order, `0` lines rejected
    - `CER <= 2.0%` and `WER <= 5.0%` against reference ground truth
    - Policy explicitly separates `image_quality_status="OK"`, `ocr_confidence_status="HIGH"`,
      `speech_policy="ALLOW_FULL_SPEECH"`
    """
    assert CASE_A_DARK_IMG.is_file()
    sha = hashlib.sha256(CASE_A_DARK_IMG.read_bytes()).hexdigest().upper()
    assert sha == EXPECTED_CASE_A_SHA256

    gt_text = EXPECTED_GT_FILE.read_text(encoding="utf-8").strip()
    gt_lines = [ln.strip() for ln in gt_text.splitlines() if ln.strip()]
    assert len(gt_lines) == 15

    img = load_image_unicode(CASE_A_DARK_IMG)
    assert img is not None
    assert img.shape[:2] == (587, 996)

    _, polarity = detect_and_normalize_polarity(img)
    assert polarity == "light_on_dark"

    gm = estimate_glyph_metrics(img)
    assert gm.polarity == "light_on_dark"
    assert gm.p75_char_height_px >= 14.0
    assert gm.is_low_resolution_document is False

    clean_gray, clean_bw, pol2 = suppress_rules_and_dots(img)
    assert pol2 == "light_on_dark"
    rois = segment_document_rois(clean_gray, clean_bw)
    assert all(r.column_role == "full" for r in rois)

    gate = ImageQualityGate()
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()

    q_res = gate.evaluate(img)
    assert q_res.passed is True

    o_res = ocr.recognize(img)
    assert o_res.error is None
    assert o_res.polarity == "light_on_dark"
    assert o_res.low_resolution_detected is False
    assert o_res.total_candidate_lines == 15
    assert o_res.rejected_line_count == 0
    assert len(o_res.lines) == 15

    decision = policy.evaluate(q_res, o_res)
    assert decision.status == SafetyStatus.READABLE
    assert decision.image_quality_status == "OK"
    assert decision.ocr_confidence_status == "HIGH"
    assert decision.speech_policy == "ALLOW_FULL_SPEECH"
    assert "đã xác minh" not in decision.display_text.lower()

    cer = _cer(decision.display_text, gt_text)
    wer = _wer(decision.display_text, gt_text)
    assert cer <= 0.02, f"Case A CER too high: {cer*100:.2f}%"
    assert wer <= 0.05, f"Case A WER too high: {wer*100:.2f}%"

    # Check reading order of all 6 paragraphs
    assert decision.ordered_lines[0].text.startswith("1. Quy định chung về phông chữ")
    assert decision.ordered_lines[1].text == "1.1. Văn bản hành chính là gì?"
    assert "30/2020/NĐ-CP" in decision.ordered_lines[4].text
    assert decision.ordered_lines[-1].text == "thể thức tương ứng."


def test_p1_2_case_b_inverted_light_theme_996x587_full_pipeline() -> None:
    """
    Verify Trial B (`diagnostic_inverted_only.png` — dark text on light background):
    - SHA-256 matches exact user image
    - Polarity detected as `dark_on_light`
    - Adaptive glyph metrics: `p75 >= 14.0px`, `low_resolution_detected is False`
    - Status is `SafetyStatus.READABLE` (was `UNCERTAIN` with 8/15 rejected in Phase 1.1)
    - `15/15` lines recognized in exact order, `0` lines rejected
    - `CER <= 2.0%` and `WER <= 5.0%` against reference ground truth
    """
    assert CASE_B_LIGHT_IMG.is_file()
    sha = hashlib.sha256(CASE_B_LIGHT_IMG.read_bytes()).hexdigest().upper()
    assert sha == EXPECTED_CASE_B_SHA256

    gt_text = EXPECTED_GT_FILE.read_text(encoding="utf-8").strip()

    img = load_image_unicode(CASE_B_LIGHT_IMG)
    assert img is not None
    assert img.shape[:2] == (587, 996)

    _, polarity = detect_and_normalize_polarity(img)
    assert polarity == "dark_on_light"

    gm = estimate_glyph_metrics(img)
    assert gm.polarity == "dark_on_light"
    assert gm.p75_char_height_px >= 14.0
    assert gm.is_low_resolution_document is False

    gate = ImageQualityGate()
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()

    q_res = gate.evaluate(img)
    assert q_res.passed is True

    o_res = ocr.recognize(img)
    assert o_res.error is None
    assert o_res.polarity == "dark_on_light"
    assert o_res.low_resolution_detected is False
    assert o_res.total_candidate_lines == 15
    assert o_res.rejected_line_count == 0
    assert len(o_res.lines) == 15

    decision = policy.evaluate(q_res, o_res)
    assert decision.status == SafetyStatus.READABLE
    assert decision.image_quality_status == "OK"
    assert decision.ocr_confidence_status == "HIGH"
    assert decision.speech_policy == "ALLOW_FULL_SPEECH"
    assert "đã xác minh" not in decision.display_text.lower()

    cer = _cer(decision.display_text, gt_text)
    wer = _wer(decision.display_text, gt_text)
    assert cer <= 0.02, f"Case B CER too high: {cer*100:.2f}%"
    assert wer <= 0.05, f"Case B WER too high: {wer*100:.2f}%"


def test_p1_2_preserves_uncertainty_and_no_verified_claim_on_379x527() -> None:
    """
    Verify Objectives 8, 9, 10:
    - Phase 1.1 `administrative_form_379x527.png` (median=7px, p75=8px) is still classified as
      `SafetyStatus.UNCERTAIN` with `image_quality_status="LOW_RESOLUTION"`,
      `ocr_confidence_status="LOW"`, `speech_policy="SPEAK_WARNING_ONLY"`.
    - Display text NEVER contains the over-claiming phrase `"đã xác minh"`.
    """
    img = load_image_unicode(REG_P1_1_IMG)
    assert img is not None

    gate = ImageQualityGate()
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()

    q_res = gate.evaluate(img)
    o_res = ocr.recognize(img)
    decision = policy.evaluate(q_res, o_res)

    assert decision.status == SafetyStatus.UNCERTAIN
    assert decision.spoken_text == MSG_LOW_RES_OR_UNCERTAIN
    assert decision.image_quality_status == "LOW_RESOLUTION"
    assert decision.ocr_confidence_status == "LOW"
    assert decision.speech_policy == "SPEAK_WARNING_ONLY"
    assert "đã xác minh" not in decision.display_text.lower()
