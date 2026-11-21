# -*- coding: utf-8 -*-
"""Unit tests for ImageQualityGate (FR-03, Section 4.1, Section 6.1)."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from visionmate.config import QualityThresholds
from visionmate.contracts import QualityIssue
from visionmate.quality.gate import (
    MSG_BLURRY,
    MSG_CORRUPT,
    MSG_DARK,
    MSG_GLARE,
    ImageQualityGate,
)


def test_corrupt_or_empty_frames_rejected() -> None:
    gate = ImageQualityGate()

    res_none = gate.evaluate(None)
    assert not res_none.passed
    assert res_none.issue == QualityIssue.CORRUPT_OR_EMPTY
    assert res_none.reason_vi == MSG_CORRUPT

    res_empty = gate.evaluate(np.array([], dtype=np.uint8))
    assert not res_empty.passed
    assert res_empty.issue == QualityIssue.CORRUPT_OR_EMPTY

    res_tiny = gate.evaluate(np.zeros((10, 10, 3), dtype=np.uint8))
    assert not res_tiny.passed
    assert res_tiny.issue == QualityIssue.CORRUPT_OR_EMPTY


def test_too_dark_frame_rejected() -> None:
    gate = ImageQualityGate()
    dark = np.full((360, 640, 3), 12, dtype=np.uint8)
    res = gate.evaluate(dark)
    assert not res.passed
    assert res.issue == QualityIssue.TOO_DARK
    assert res.reason_vi == MSG_DARK


def test_too_bright_glare_frame_rejected() -> None:
    gate = ImageQualityGate()
    glare = np.full((360, 640, 3), 252, dtype=np.uint8)
    res = gate.evaluate(glare)
    assert not res.passed
    assert res.issue == QualityIssue.TOO_BRIGHT_OR_GLARE
    assert res.reason_vi == MSG_GLARE


def test_blurry_frame_rejected() -> None:
    gate = ImageQualityGate()
    img = np.full((360, 640, 3), 230, dtype=np.uint8)
    cv2.rectangle(img, (120, 80), (520, 280), (40, 40, 40), -1)
    blurred = cv2.GaussianBlur(img, (51, 51), 18.0)

    res = gate.evaluate(blurred)
    assert not res.passed
    assert res.issue == QualityIssue.TOO_BLURRY
    assert res.reason_vi == MSG_BLURRY


def test_sharp_printed_frame_passes() -> None:
    gate = ImageQualityGate()
    pil_img = Image.new("RGB", (640, 240), color=(250, 250, 250))
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([20, 20, 620, 220], outline=(20, 20, 20), width=3)
    draw.text((50, 90), "Sữa tươi không đường 180ml", fill=(10, 10, 10))
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    res = gate.evaluate(bgr)
    assert res.passed
    assert res.issue == QualityIssue.OK
    assert res.reason_vi == ""


def test_custom_thresholds_respected() -> None:
    strict = QualityThresholds(min_brightness=100.0)
    gate = ImageQualityGate(thresholds=strict)
    dim = np.full((240, 320, 3), 80, dtype=np.uint8)
    res = gate.evaluate(dim)
    assert not res.passed
    assert res.issue == QualityIssue.TOO_DARK
