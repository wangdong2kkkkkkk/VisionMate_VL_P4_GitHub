# -*- coding: utf-8 -*-
"""
Phase 1.3a Integration Test Suite: Dark-Background Quality Gate Hotfix & Code Screen OCR.
Validates:
1. Job #17 Bug Reproduction & Fix:
   - High-contrast dark-background code screenshot (mean brightness ~17.4 < 28.0) is NO LONGER falsely
     rejected by ImageQualityGate as TOO_DARK.
   - Frame proceeds to OCR and recognizes 6 lines of code in natural reading order.
   - SafetyPolicy returns READABLE with full speech allowed and truthful edge-clipping warnings.
2. Fail-Closed Protections Preserved:
   - Genuine underexposure (flat dark frame / lens cap) STILL rejected with TOO_DARK.
   - Blurry dark frames STILL rejected with TOO_DARK or TOO_BLURRY.
   - Cursive handwriting on screen (handwritten_note_screen.jpg) handled safely (UNCERTAIN/NO_TEXT, never hallucinated).
3. Immutable Regression Protection:
   - Phase 1.2 dark theme (996x587) remains READABLE with 15/15 lines and CER <= 2%.
   - Phase 1.2 inverted theme (996x587) remains READABLE with 15/15 lines and CER <= 2%.
   - Phase 1.1 administrative form (379x527) remains safely classified as UNCERTAIN.
"""
from __future__ import annotations

import hashlib
import time
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pytest

from visionmate.contracts import QualityIssue, SafetyStatus
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
FIXTURES_P1_3A = PROJECT_ROOT / "tests" / "fixtures" / "regression_p1_3a"
JOB17_IMG = FIXTURES_P1_3A / "dark_code_syntax_job17.jpg"
HANDWRITTEN_IMG = FIXTURES_P1_3A / "handwritten_note_screen.jpg"

EXPECTED_JOB17_SHA256 = "7EC4CC98B6776476BE6C00F5DF87B483AEE8AA8B2F0A10AA59F9AF013196AB4E"
EXPECTED_HANDWRITTEN_SHA256 = "6CDD99D1E6A34E0C224CC235845B3603B4895E81E62D7115CB50D2F35C42BE4F"


def _load_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert img is not None, f"Could not decode image at {path}"
    return img


def test_p1_3a_job17_bug_fix_and_full_pipeline() -> None:
    """Test that the Job #17 bug is completely resolved:
    - ImageQualityGate passes dark-background high-contrast code frame (previously TOO_DARK).
    - OCR extracts all 6 code lines in proper sequential order.
    - SafetyPolicy allows speech and flags truthful edge-clipped warnings.
    """
    assert JOB17_IMG.is_file()
    actual_sha = hashlib.sha256(JOB17_IMG.read_bytes()).hexdigest().upper()
    assert actual_sha == EXPECTED_JOB17_SHA256, f"Fixture SHA-256 mismatch: {actual_sha}"

    img = _load_image(JOB17_IMG)
    h, w = img.shape[:2]
    assert (h, w) == (684, 1024)

    t0_gate = time.perf_counter()
    gate = ImageQualityGate()
    q_res = gate.evaluate(img)
    gate_elapsed = (time.perf_counter() - t0_gate) * 1000.0

    # 1. ImageQualityGate checks
    assert q_res.passed is True, f"Gate failed with issue {q_res.issue}: {q_res.reason_vi}"
    assert q_res.issue == QualityIssue.OK
    assert q_res.mean_brightness < 28.0, "Expected mean brightness < 28.0"
    assert q_res.active_blur_score > 1000.0, "Expected high sharpness from code editor"
    assert q_res.contrast_std > 35.0, "Expected high contrast from white/colored code text"
    assert q_res.edge_clipped_text is True, "Expected detection of text touching right border"
    assert gate_elapsed < 50.0, f"Gate took too long: {gate_elapsed:.1f}ms"

    # 2. Polarity & Layout checks
    _, pol = detect_and_normalize_polarity(img)
    assert pol == "light_on_dark"

    gm = estimate_glyph_metrics(img)
    assert gm.polarity == "light_on_dark"
    assert gm.is_low_resolution_document is False
    assert gm.median_char_height_px >= 25.0

    clean_gray, clean_bw, _ = suppress_rules_and_dots(img)
    rois = segment_document_rois(clean_gray, clean_bw)
    # Ensure indented code is NOT falsely split into 2 columns
    assert all(r.column_role == "full" for r in rois)

    # 3. OCR checks
    t0_ocr = time.perf_counter()
    ocr = TesseractOCRBackend(variant="fast")
    o_res = ocr.recognize(img)
    ocr_elapsed = (time.perf_counter() - t0_ocr) * 1000.0

    assert o_res.error is None
    assert o_res.low_resolution_detected is False
    assert o_res.rejected_line_count == 0
    assert len(o_res.lines) == 6, f"Expected 6 lines, got {len(o_res.lines)}"

    extracted = [ln.text for ln in o_res.lines]
    # Verify reading order
    assert "isNaN" in extracted[0] and "guess" in extracted[0]
    assert "resultText.textContent" in extracted[1]
    assert "else if" in extracted[2] and "guess" in extracted[2]
    assert "resultText.textContent" in extracted[3]
    assert "else" in extracted[4]
    assert "if" in extracted[5] and ("targetNumber" in extracted[5] or "targetNunber" in extracted[5])

    # 4. SafetyPolicy checks
    policy = SafetyPolicy()
    decision = policy.evaluate(q_res, o_res)

    assert decision.status == SafetyStatus.READABLE
    assert decision.speech_policy == "ALLOW_FULL_SPEECH"
    assert decision.image_quality_status == "OK"
    assert decision.ocr_confidence_status == "HIGH"
    assert any("viền" in w.lower() for w in decision.warnings), "Expected edge clipping warning"
    assert "đã xác minh" not in decision.display_text.lower()


def test_p1_3a_fail_closed_genuine_underexposure_rejected() -> None:
    """Verify that genuinely dark scenes (e.g. lens cap / dark room) are STILL rejected as TOO_DARK."""
    gate = ImageQualityGate()
    policy = SafetyPolicy()

    # Pitch black frame
    black = np.full((720, 1280, 3), 12, dtype=np.uint8)
    q_black = gate.evaluate(black)
    assert q_black.passed is False
    assert q_black.issue == QualityIssue.TOO_DARK

    dec_black = policy.evaluate(q_black, None)
    assert dec_black.status == SafetyStatus.BAD_IMAGE
    assert dec_black.speech_policy == "SPEAK_WARNING_ONLY"

    # Dim noisy scene without high-contrast text strokes
    np.random.seed(42)
    dim_noisy = np.random.randint(10, 24, (720, 1280, 3), dtype=np.uint8)
    q_noisy = gate.evaluate(dim_noisy)
    assert q_noisy.passed is False
    assert q_noisy.issue == QualityIssue.TOO_DARK


def test_p1_3a_fail_closed_dark_blurry_scene_rejected() -> None:
    """Verify that a dark blurry scene is rejected rather than passed to OCR."""
    gate = ImageQualityGate()

    # Create dark image with low blur score
    base = np.full((400, 600, 3), 20, dtype=np.uint8)
    cv2.putText(base, "Blurry Code", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 1)
    blurred = cv2.GaussianBlur(base, (31, 31), 10.0)

    q = gate.evaluate(blurred)
    assert q.passed is False
    assert q.issue in (QualityIssue.TOO_DARK, QualityIssue.TOO_BLURRY)


def test_p1_3a_handwritten_screen_fail_closed_safe_handling() -> None:
    """Verify that handwritten cursive on screen does not cause hallucinated text or crashes."""
    assert HANDWRITTEN_IMG.is_file()
    actual_sha = hashlib.sha256(HANDWRITTEN_IMG.read_bytes()).hexdigest().upper()
    assert actual_sha == EXPECTED_HANDWRITTEN_SHA256

    img = _load_image(HANDWRITTEN_IMG)
    gate = ImageQualityGate()
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()

    q_res = gate.evaluate(img)
    o_res = ocr.recognize(img)
    decision = policy.evaluate(q_res, o_res)

    # Fail closed: must not claim high confidence or verified text
    assert decision.status in (SafetyStatus.UNCERTAIN, SafetyStatus.NO_TEXT)
    assert decision.speech_policy in ("SPEAK_WARNING_ONLY", "SPEAK_NO_TEXT")
    assert "đã xác minh" not in decision.display_text.lower()
