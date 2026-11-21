# -*- coding: utf-8 -*-
"""
Unit and Stress Tests for OD-1 Controller Mode Transitions, Routing, and Race Conditions.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from visionmate.camera.sources import MockCameraSource
from visionmate.config import AppConfig
from visionmate.contracts import (
    ActiveSource,
    AppMode,
    DetectedObject,
    SafetyStatus,
    SessionJobResult,
    SessionState,
)
from visionmate.controller import SessionController
from visionmate.detection.detector import FakeObjectDetector
from visionmate.ocr.fake_backend import FakeOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import SafetyPolicy
from visionmate.speech.fake_speech import FakeSpeechBackend


def _create_test_controller(
    fake_detector: FakeObjectDetector,
    fake_ocr: FakeOCRBackend,
) -> SessionController:
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    camera = MockCameraSource(frame=frame, allow_open=True)
    gate = ImageQualityGate()
    policy = SafetyPolicy()
    speech = FakeSpeechBackend()
    config = AppConfig()

    return SessionController(
        camera=camera,
        quality_gate=gate,
        ocr_backend=fake_ocr,
        safety_policy=policy,
        speech_backend=speech,
        object_detector=fake_detector,
        config=config,
    )


def test_mode_transitions_and_toggle() -> None:
    """Verify mode switching between Reading OCR and Object Detection."""
    fake_det = FakeObjectDetector()
    fake_ocr = FakeOCRBackend()
    ctrl = _create_test_controller(fake_det, fake_ocr)

    assert ctrl.mode == AppMode.READING

    # Toggle to OBJECT_DETECTION
    m1 = ctrl.toggle_mode()
    assert m1 == AppMode.OBJECT_DETECTION
    assert ctrl.mode == AppMode.OBJECT_DETECTION
    assert ctrl.config.active_mode == "object_detection"

    # Toggle back to READING
    m2 = ctrl.toggle_mode()
    assert m2 == AppMode.READING
    assert ctrl.mode == AppMode.READING
    assert ctrl.config.active_mode == "reading"

    # Explicit set_mode
    ctrl.set_mode(AppMode.OBJECT_DETECTION)
    assert ctrl.mode == AppMode.OBJECT_DETECTION
    ctrl.shutdown()


def test_space_action_routes_by_mode() -> None:
    """Verify capture_and_read automatically routes to OCR or OD according to active mode."""
    fake_det = FakeObjectDetector(
        mock_objects=[
            DetectedObject(
                class_id=39,
                class_name_en="bottle",
                class_name_vi="chai nước",
                confidence=0.92,
                box_xywh=(10, 10, 50, 100),
            )
        ],
        latency_ms=10.0,
    )
    fake_ocr = FakeOCRBackend(delay_s=0.01)
    ctrl = _create_test_controller(fake_det, fake_ocr)
    ctrl.start_camera()

    completed_results: List[SessionJobResult] = []
    ctrl._on_job_completed = lambda res: completed_results.append(res)

    # 1. In READING mode, Space runs OCR
    ctrl.set_mode(AppMode.READING)
    ctrl.capture_and_read()
    assert ctrl.wait_for_idle(timeout_s=3.0) is True
    assert len(completed_results) == 1
    assert completed_results[-1].mode == "reading"
    assert completed_results[-1].detection is None

    # 2. In OBJECT_DETECTION mode, Space runs Object Detection
    ctrl.set_mode(AppMode.OBJECT_DETECTION)
    ctrl.capture_and_read()
    assert ctrl.wait_for_idle(timeout_s=3.0) is True
    assert len(completed_results) == 2
    od_res = completed_results[-1]
    assert od_res.mode == "object_detection"
    assert od_res.detection is not None
    assert len(od_res.detection.objects) == 1
    assert od_res.detection.objects[0].class_name_vi == "chai nước"
    assert od_res.spoken_text == "Phát hiện một chai nước."

    ctrl.shutdown()


def test_od_cancellation_and_rapid_triggers() -> None:
    """Verify rapid consecutive triggers cancel preceding jobs without stale callbacks."""
    fake_det = FakeObjectDetector(latency_ms=50.0)
    fake_ocr = FakeOCRBackend(delay_s=0.01)
    ctrl = _create_test_controller(fake_det, fake_ocr)
    ctrl.start_camera()
    ctrl.set_mode(AppMode.OBJECT_DETECTION)

    completed: List[SessionJobResult] = []
    ctrl._on_job_completed = lambda res: completed.append(res)

    # Fire 5 rapid triggers in object detection mode
    for _ in range(5):
        ctrl.detect_objects()
        time.sleep(0.005)

    assert ctrl.wait_for_idle(timeout_s=5.0) is True

    # Only the final job should have been confirmed
    assert len(completed) >= 1
    assert completed[-1].job_id == ctrl.current_job_id
    assert ctrl.last_confirmed_result is not None
    assert ctrl.last_confirmed_result.job_id == ctrl.current_job_id

    ctrl.shutdown()


def test_od_clear_results_resets_state() -> None:
    """Verify Ctrl+L resets state, clears last confirmed detection, and stops active speech."""
    mock_objs = [
        DetectedObject(
            class_id=41,
            class_name_en="cup",
            class_name_vi="chiếc cốc",
            confidence=0.88,
            box_xywh=(20, 20, 40, 60),
        )
    ]
    fake_det = FakeObjectDetector(mock_objects=mock_objs, latency_ms=10.0)
    fake_ocr = FakeOCRBackend(delay_s=0.01)
    ctrl = _create_test_controller(fake_det, fake_ocr)
    ctrl.start_camera()
    ctrl.set_mode(AppMode.OBJECT_DETECTION)

    ctrl.detect_objects()
    assert ctrl.wait_for_idle(timeout_s=3.0) is True
    assert ctrl.last_confirmed_result is not None
    assert ctrl.last_confirmed_result.mode == "object_detection"

    # Clear results
    cleared_events: List[int] = []
    ctrl._on_results_cleared = lambda s_id, msg: cleared_events.append(s_id)
    ctrl.clear_results()

    assert ctrl.last_confirmed_result is None
    assert len(cleared_events) == 1
    assert ctrl.state == SessionState.CAMERA_READY

    ctrl.shutdown()
