# -*- coding: utf-8 -*-
"""Unit tests for ReadingOrder and SafetyPolicy (FR-04, A04, A05, Section 6.1)."""
from __future__ import annotations

from visionmate.contracts import (
    OCRLine,
    OCRResult,
    QualityCheckResult,
    QualityIssue,
    SafetyStatus,
)
from visionmate.reading.order import ReadingOrder
from visionmate.reading.policy import (
    MSG_ENGINE_ERROR,
    MSG_NO_TEXT,
    MSG_SENSITIVE_CAUTION,
    MSG_UNCERTAIN,
    SafetyPolicy,
)


def _ok_quality() -> QualityCheckResult:
    return QualityCheckResult(
        passed=True,
        issue=QualityIssue.OK,
        blur_score=180.0,
        active_blur_score=320.0,
        mean_brightness=210.0,
        glare_ratio=0.1,
        contrast_std=45.0,
        elapsed_ms=2.0,
    )


def test_reading_order_top_to_bottom_left_to_right() -> None:
    lines = [
        OCRLine(text="Dòng 2 phải", bbox=(300, 120, 120, 30), raw_score=90.0),
        OCRLine(text="Dòng 1", bbox=(40, 30, 200, 30), raw_score=92.0),
        OCRLine(text="Dòng 2 trái", bbox=(40, 118, 120, 30), raw_score=91.0),
    ]
    ordered = ReadingOrder.sort_lines(lines)
    assert [ln.text for ln in ordered] == ["Dòng 1", "Dòng 2 trái", "Dòng 2 phải"]


def test_digits_dates_dosages_preserved_verbatim() -> None:
    raw_text = "Paracetamol 500mg - NSX: 15/03/2026 - HSD: 15/03/2027 - Lô: B260911"
    ocr_res = OCRResult(
        lines=[OCRLine(text=raw_text, bbox=(10, 10, 500, 32), raw_score=93.0)],
        engine_name="fake",
        elapsed_ms=10.0,
    )
    policy = SafetyPolicy()
    decision = policy.evaluate(_ok_quality(), ocr_res)

    assert decision.status == SafetyStatus.READABLE
    assert decision.display_text == raw_text
    assert raw_text in decision.spoken_text
    assert decision.contains_sensitive_content is True
    assert any(MSG_SENSITIVE_CAUTION in w for w in decision.warnings)
    assert MSG_SENSITIVE_CAUTION in decision.spoken_text


def test_no_text_policy_never_invents_description() -> None:
    ocr_res = OCRResult(lines=[], engine_name="fake", elapsed_ms=10.0)
    policy = SafetyPolicy()
    decision = policy.evaluate(_ok_quality(), ocr_res)

    assert decision.status == SafetyStatus.NO_TEXT
    assert decision.spoken_text == MSG_NO_TEXT
    assert decision.ordered_lines == []


def test_uncertain_low_score_policy() -> None:
    ocr_res = OCRResult(
        lines=[OCRLine(text="Sữa tươi", bbox=(10, 10, 100, 25), raw_score=25.0)],
        engine_name="fake",
        elapsed_ms=10.0,
    )
    policy = SafetyPolicy(min_ocr_raw_score=42.0)
    decision = policy.evaluate(_ok_quality(), ocr_res)

    assert decision.status == SafetyStatus.UNCERTAIN
    assert decision.spoken_text == MSG_UNCERTAIN


def test_engine_error_policy() -> None:
    ocr_res = OCRResult(lines=[], engine_name="fake", elapsed_ms=5.0, error="Timeout")
    policy = SafetyPolicy()
    decision = policy.evaluate(_ok_quality(), ocr_res)

    assert decision.status == SafetyStatus.ENGINE_ERROR
    assert decision.spoken_text == MSG_ENGINE_ERROR
