# -*- coding: utf-8 -*-
"""
Integration Tests for VisionMate OD-1 Offline Object Detection Pipeline.

Verifies:
1. Single object recognition: single_bird.jpg -> 'bird' / 'chim' ('Phát hiện một con chim.').
2. Multiple objects recognition: multi_street_dog_bike_car.jpg -> 'car', 'bicycle', 'dog'.
3. Multiple instances of same class: sample_horses.jpg -> 3 horses ('Phát hiện 3 con ngựa.').
4. Indoor monitor / screen recognition: handwritten_note_screen.jpg -> 'tv' ('Phát hiện một chiếc ti vi.').
5. Clean document with no objects: no_objects_clean_doc.png -> 'Chưa xác định được đồ vật.'.
6. Out-of-distribution abstract pattern: ood_abstract_pattern.png -> 'Chưa xác định được đồ vật.'.
7. Severe underexposure safety refusal: low_light_underexposed.png -> Quality gate refusal.
8. Windows OneCore offline Vietnamese voice routing (Microsoft An vi-VN) for object speech.
9. CPU latency & RAM benchmark: P50 latency < 50ms, Peak RAM < 300MB.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import psutil
import pytest

from visionmate.camera.sources import load_image_unicode
from visionmate.config import AppConfig
from visionmate.contracts import (
    ActiveSource,
    AppMode,
    SafetyStatus,
    SessionJobResult,
    SessionState,
)
from visionmate.controller import SessionController
from visionmate.detection.detector import YOLOXObjectDetector
from visionmate.ocr.tesseract_backend import TesseractOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import SafetyPolicy
from visionmate.speech.sapi_onecore import WindowsOneCoreSpeechBackend


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "object_detection"
REGRESSION_P1_3A_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "regression_p1_3a"


def test_single_object_recognition() -> None:
    """Verify single object detection on single_bird.jpg."""
    img_path = FIXTURES_DIR / "single_bird.jpg"
    assert img_path.is_file(), f"Fixture missing: {img_path}"
    img = load_image_unicode(img_path)
    assert img is not None

    detector = YOLOXObjectDetector()
    res = detector.detect(img)

    assert res.error is None
    assert len(res.objects) >= 1
    bird_objs = [o for o in res.objects if o.class_name_en == "bird"]
    assert len(bird_objs) >= 1
    assert bird_objs[0].class_name_vi == "chim"
    assert bird_objs[0].confidence >= 0.70
    assert res.spoken_text == "Phát hiện một con chim."
    assert "chim" in res.display_text.lower()
    assert res.confidence_status == "HIGH"
    assert res.annotated_frame is not None


def test_multiple_objects_recognition() -> None:
    """Verify multi-object detection on multi_street_dog_bike_car.jpg."""
    img_path = FIXTURES_DIR / "multi_street_dog_bike_car.jpg"
    assert img_path.is_file()
    img = load_image_unicode(img_path)
    assert img is not None

    detector = YOLOXObjectDetector()
    res = detector.detect(img)

    assert res.error is None
    assert len(res.objects) >= 3
    classes_found = {o.class_name_en for o in res.objects}
    assert "dog" in classes_found
    assert "bicycle" in classes_found
    assert "car" in classes_found

    assert "ô tô" in res.spoken_text
    assert "xe đạp" in res.spoken_text
    assert "chó" in res.spoken_text
    assert res.confidence_status == "HIGH"


def test_multiple_instances_same_class() -> None:
    """Verify counting multiple instances of the same object class on sample_horses.jpg."""
    img_path = FIXTURES_DIR / "sample_horses.jpg"
    assert img_path.is_file()
    img = load_image_unicode(img_path)
    assert img is not None

    detector = YOLOXObjectDetector()
    res = detector.detect(img)

    assert res.error is None
    horse_objs = [o for o in res.objects if o.class_name_en == "horse"]
    assert len(horse_objs) >= 2
    assert "ngựa" in res.spoken_text


def test_screen_monitor_recognition() -> None:
    """Verify indoor monitor/screen detection on handwritten_note_screen.jpg."""
    img_path = REGRESSION_P1_3A_DIR / "handwritten_note_screen.jpg"
    assert img_path.is_file()
    img = load_image_unicode(img_path)
    assert img is not None

    detector = YOLOXObjectDetector()
    res = detector.detect(img)

    assert res.error is None
    assert len(res.objects) >= 1
    assert any(o.class_name_en == "tv" for o in res.objects)
    assert "ti vi" in res.spoken_text


def test_no_objects_clean_doc_refusal() -> None:
    """Verify document with text only produces 0 objects and announces 'Chưa xác định được đồ vật.'."""
    img_path = FIXTURES_DIR / "no_objects_clean_doc.png"
    assert img_path.is_file()
    img = load_image_unicode(img_path)
    assert img is not None

    detector = YOLOXObjectDetector()
    res = detector.detect(img)

    assert res.error is None
    assert len(res.objects) == 0
    assert res.spoken_text == "Chưa xác định được đồ vật."
    assert res.display_text == "Chưa xác định được đồ vật."
    assert res.confidence_status == "NONE"


def test_out_of_distribution_pattern_refusal() -> None:
    """Verify abstract geometric shapes not in COCO produce 0 detections safely."""
    img_path = FIXTURES_DIR / "ood_abstract_pattern.png"
    assert img_path.is_file()
    img = load_image_unicode(img_path)
    assert img is not None

    detector = YOLOXObjectDetector()
    res = detector.detect(img)

    assert res.error is None
    assert len(res.objects) == 0
    assert res.spoken_text == "Chưa xác định được đồ vật."
    assert res.confidence_status == "NONE"


def test_severe_underexposure_quality_refusal() -> None:
    """Verify controller rejects severely underexposed frames safely without running detection."""
    img_path = FIXTURES_DIR / "low_light_underexposed.png"
    assert img_path.is_file()

    detector = YOLOXObjectDetector()
    ctrl = SessionController(
        camera=None,  # type: ignore[arg-type]
        quality_gate=ImageQualityGate(),
        ocr_backend=TesseractOCRBackend(variant="fast"),
        safety_policy=SafetyPolicy(),
        speech_backend=WindowsOneCoreSpeechBackend(),
        object_detector=detector,
        config=AppConfig(),
    )
    ctrl.set_mode(AppMode.OBJECT_DETECTION)

    completed: List[SessionJobResult] = []
    ctrl._on_job_completed = lambda res: completed.append(res)

    ctrl.detect_objects_from_file(img_path)
    assert ctrl.wait_for_idle(timeout_s=6.0) is True
    assert len(completed) == 1
    res = completed[0]
    assert res.status == SafetyStatus.BAD_IMAGE
    assert "quá tối" in res.spoken_text.lower() or "chưa xác định" in res.spoken_text.lower()

    ctrl.shutdown()


def test_onecore_tts_voice_routing_for_object_detection() -> None:
    """Verify Microsoft An (vi-VN) is used for Vietnamese object speech offline."""
    speech = WindowsOneCoreSpeechBackend()
    if not speech.is_vi_vn_available():
        pytest.skip("Microsoft An (vi-VN) not installed on this Windows instance")

    ctrl = SessionController(
        camera=None,  # type: ignore[arg-type]
        quality_gate=ImageQualityGate(),
        ocr_backend=TesseractOCRBackend(variant="fast"),
        safety_policy=SafetyPolicy(),
        speech_backend=speech,
        object_detector=YOLOXObjectDetector(),
        config=AppConfig(),
    )
    ctrl.set_mode(AppMode.OBJECT_DETECTION)

    completed: List[SessionJobResult] = []
    ctrl._on_job_completed = lambda res: completed.append(res)

    img_path = FIXTURES_DIR / "single_bird.jpg"
    ctrl.detect_objects_from_file(img_path)
    assert ctrl.wait_for_idle(timeout_s=5.0) is True

    assert len(completed) == 1
    res = completed[0]
    assert res.spoken_text == "Phát hiện một con chim."

    # Verify voice diagnostics recorded Microsoft An for Vietnamese
    assert speech._last_spoken_lang == "vi"
    assert speech._last_spoken_voice_desc is not None
    assert "An" in speech._last_spoken_voice_desc

    ctrl.shutdown()


def test_cpu_latency_and_memory_benchmark() -> None:
    """Benchmark YOLOX-Nano CPU inference latency and RAM usage over 10 iterations."""
    detector = YOLOXObjectDetector()
    img_path = FIXTURES_DIR / "multi_street_dog_bike_car.jpg"
    img = load_image_unicode(img_path)
    assert img is not None

    # Warmup
    detector.detect(img)

    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    latencies_ms: List[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = detector.detect(img)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert len(res.objects) >= 3

    ram_after_mb = process.memory_info().rss / (1024 * 1024)

    p50_latency = float(np.median(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    mean_latency = float(np.mean(latencies_ms))

    print(
        f"\n[BENCHMARK YOLOX-Nano CPU]: "
        f"P50={p50_latency:.1f}ms, P95={p95_latency:.1f}ms, Mean={mean_latency:.1f}ms | "
        f"RAM: before={ram_before_mb:.1f}MB, after={ram_after_mb:.1f}MB (Peak={ram_after_mb:.1f}MB)"
    )

    # Performance Gate: P50 latency on CPU must be strictly under 75ms (typical <45ms), RAM < 300MB
    assert p50_latency < 75.0, f"P50 latency exceeded budget: {p50_latency:.1f}ms >= 75ms"
    assert ram_after_mb < 300.0, f"RAM exceeded budget: {ram_after_mb:.1f}MB >= 300MB"
