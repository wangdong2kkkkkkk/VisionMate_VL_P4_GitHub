# -*- coding: utf-8 -*-
"""
Unit tests for YOLOXObjectDetector and FakeObjectDetector.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from visionmate.contracts import DetectedObject
from visionmate.detection.annotator import draw_detection_annotations
from visionmate.detection.detector import (
    DEFAULT_MODEL_PATH,
    FakeObjectDetector,
    YOLOXObjectDetector,
    _yolox_grid_postprocess,
)


def test_yolox_detector_initialization() -> None:
    """Verify YOLOX detector initializes with default model path."""
    detector = YOLOXObjectDetector()
    assert detector.is_ready() is True
    assert detector.model_name == "yolox_nano"
    assert detector.conf_threshold == 0.40
    assert detector.nms_threshold == 0.45


def test_yolox_missing_model_handled_gracefully() -> None:
    """Verify missing model path does not throw unhandled exceptions."""
    detector = YOLOXObjectDetector(model_path="non_existent_model.onnx")
    assert detector.is_ready() is False
    fake_img = np.zeros((416, 416, 3), dtype=np.uint8)
    res = detector.detect(fake_img)
    assert res.error is not None
    assert "Không tìm thấy file mô hình" in res.error
    assert res.spoken_text == "Chưa xác định được đồ vật."
    assert res.confidence_status == "NONE"


def test_yolox_empty_or_corrupt_image_handled_safely() -> None:
    """Verify empty or degenerate image arrays are handled safely."""
    detector = YOLOXObjectDetector()
    res1 = detector.detect(None)  # type: ignore[arg-type]
    assert res1.objects == []
    assert res1.spoken_text == "Chưa xác định được đồ vật."

    res2 = detector.detect(np.array([], dtype=np.uint8))
    assert res2.objects == []
    assert res2.spoken_text == "Chưa xác định được đồ vật."

    res3 = detector.detect(np.zeros((5, 5, 3), dtype=np.uint8))
    assert res3.objects == []
    assert res3.spoken_text == "Chưa xác định được đồ vật."


def test_yolox_cancel_token_stops_inference() -> None:
    """Verify pre-set cancel token aborts detection immediately."""
    detector = YOLOXObjectDetector()
    token = threading.Event()
    token.set()
    fake_img = np.zeros((416, 416, 3), dtype=np.uint8)
    res = detector.detect(fake_img, cancel_token=token)
    assert res.error == "CANCELLED"
    assert res.objects == []


def test_yolox_grid_postprocess_dimensions() -> None:
    """Verify grid postprocess outputs match 3549 anchor points."""
    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    decoded = _yolox_grid_postprocess(raw, (416, 416))
    assert decoded.shape == (1, 3549, 85)
    # Check that width and height were exponentiated and multiplied by strides
    assert np.all(decoded[..., 2:4] > 0)


def test_draw_detection_annotations() -> None:
    """Verify annotator draws bounding boxes without mutating original frame."""
    orig = np.zeros((480, 640, 3), dtype=np.uint8)
    objs = [
        DetectedObject(
            class_id=39,
            class_name_en="bottle",
            class_name_vi="chai nước",
            confidence=0.85,
            box_xywh=(100, 100, 80, 200),
        )
    ]
    annotated = draw_detection_annotations(orig, objs)
    assert annotated.shape == orig.shape
    # Original frame must remain pure black
    assert np.all(orig == 0)
    # Annotated frame must have non-zero pixels from bounding box and text
    assert np.any(annotated > 0)


def test_fake_object_detector() -> None:
    """Verify FakeObjectDetector mock behavior and cancellation."""
    mock_objs = [
        DetectedObject(
            class_id=39,
            class_name_en="bottle",
            class_name_vi="chai nước",
            confidence=0.90,
            box_xywh=(50, 50, 60, 120),
        )
    ]
    fake = FakeObjectDetector(mock_objects=mock_objs, latency_ms=1.0)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    res = fake.detect(img)
    assert len(res.objects) == 1
    assert res.spoken_text == "Phát hiện một chai nước."
    assert res.confidence_status == "HIGH"
    assert res.error is None
    assert fake.detect_call_count == 1
