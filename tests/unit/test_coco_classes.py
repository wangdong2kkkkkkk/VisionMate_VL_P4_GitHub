# -*- coding: utf-8 -*-
"""
Unit tests for COCO 80-Class Labels and Vietnamese Phrasing Synthesis.
"""
from __future__ import annotations

import pytest

from visionmate.contracts import DetectedObject
from visionmate.detection.coco_classes import (
    COCO_CLASSES_EN,
    format_vietnamese_noun_phrase,
    format_vietnamese_object_summary,
    get_class_name_vi,
    get_class_name_vi_by_id,
)


def test_coco_classes_count_and_mapping() -> None:
    """Verify all 80 COCO classes are mapped with valid Vietnamese translations."""
    assert len(COCO_CLASSES_EN) == 80
    assert COCO_CLASSES_EN[0] == "person"
    assert COCO_CLASSES_EN[39] == "bottle"
    assert COCO_CLASSES_EN[41] == "cup"
    assert COCO_CLASSES_EN[67] == "cell phone"

    for i, en_name in enumerate(COCO_CLASSES_EN):
        vi_name = get_class_name_vi(en_name)
        assert vi_name and vi_name != "", f"Missing Vietnamese translation for {en_name}"
        assert get_class_name_vi_by_id(i) == vi_name


def test_vietnamese_noun_phrases() -> None:
    """Verify singular and plural grammatical noun phrasing."""
    assert format_vietnamese_noun_phrase("bottle", 1) == "một chai nước"
    assert format_vietnamese_noun_phrase("cup", 1) == "một chiếc cốc"
    assert format_vietnamese_noun_phrase("cell phone", 1) == "một điện thoại"
    assert format_vietnamese_noun_phrase("chair", 1) == "một chiếc ghế"
    assert format_vietnamese_noun_phrase("chair", 2) == "2 chiếc ghế"
    assert format_vietnamese_noun_phrase("book", 3) == "3 quyển sách"
    assert format_vietnamese_noun_phrase("person", 1) == "một người"
    assert format_vietnamese_noun_phrase("person", 4) == "4 người"


def test_format_vietnamese_object_summary_examples() -> None:
    """Verify natural phrasing synthesis for 0, 1, 2, and 3+ objects."""
    # 0 objects
    assert format_vietnamese_object_summary([]) == "Chưa xác định được đồ vật."

    # 1 object
    assert format_vietnamese_object_summary(["bottle"]) == "Phát hiện một chai nước."
    assert format_vietnamese_object_summary(["cup"]) == "Phát hiện một chiếc cốc."

    # 2 objects
    assert (
        format_vietnamese_object_summary(["bottle", "cup"])
        == "Phát hiện một chai nước và một chiếc cốc."
    )

    # 3 objects (matching user specification exactly)
    assert (
        format_vietnamese_object_summary(["bottle", "cup", "cell phone"])
        == "Phát hiện một chai nước, một chiếc cốc và một điện thoại."
    )

    # Grouped multiple instances
    assert (
        format_vietnamese_object_summary(["chair", "chair", "dining table"])
        == "Phát hiện 2 chiếc ghế và một chiếc bàn."
    )
    assert (
        format_vietnamese_object_summary(["horse", "horse", "horse"])
        == "Phát hiện 3 con ngựa."
    )
