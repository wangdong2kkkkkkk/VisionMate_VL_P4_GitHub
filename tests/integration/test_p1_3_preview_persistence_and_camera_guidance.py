# -*- coding: utf-8 -*-
"""
Integration tests for VisionMate Glasses Phase 1.3:
File Preview Persistence & Camera Reading Reliability Hotfix.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import visionmate
from visionmate import PHASE_LABEL
from visionmate.camera.sources import (
    CameraSource,
    MockCameraSource,
    load_image_unicode,
)
from visionmate.config import AppConfig
from visionmate.contracts import (
    ActiveSource,
    QualityCheckResult,
    SafetyStatus,
    SessionJobResult,
    SessionState,
)
from visionmate.controller import SessionController
from visionmate.ocr.tesseract_backend import TesseractOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import SafetyPolicy
from visionmate.speech.fake_speech import FakeSpeechBackend
from visionmate.ui.main_window import AccessibleMainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_p1_3_version_and_window_title(qapp):
    """Verify version numbers and dynamic window title in Phase 1.3/1.3a."""
    assert visionmate.__version__ in ("0.1.3", "0.1.3a", "0.1.3a.1")
    assert PHASE_LABEL in ("Phase 1.3", "Phase 1.3a", "Phase 1.3a.1")

    cfg = AppConfig()
    cam = MockCameraSource(allow_open=True)
    gate = ImageQualityGate(cfg.quality)
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()
    speech = FakeSpeechBackend(vi_available=True)

    ctrl = SessionController(
        camera=cam,
        quality_gate=gate,
        ocr_backend=ocr,
        safety_policy=policy,
        speech_backend=speech,
        config=cfg,
    )
    win = AccessibleMainWindow(ctrl, cfg, auto_start_camera=False)

    title = win.windowTitle()
    assert "Phase 1.3" in title
    assert "Phase 1.1" not in title
    assert "VisionMate Glasses" in title

    win.close()


def test_p1_3_preview_persistence_across_ocr_and_ticks(qapp):
    """Verify that opening an image with 'O' retains the image preview indefinitely until Space or Ctrl+L."""
    cfg = AppConfig()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cam = MockCameraSource(frame=mock_frame, allow_open=True)
    gate = ImageQualityGate(cfg.quality)
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()
    speech = FakeSpeechBackend(vi_available=True)

    ctrl = SessionController(
        camera=cam,
        quality_gate=gate,
        ocr_backend=ocr,
        safety_policy=policy,
        speech_backend=speech,
        config=cfg,
    )
    ctrl.start_camera()
    win = AccessibleMainWindow(ctrl, cfg, auto_start_camera=False)

    # Initial state: ActiveSource.CAMERA
    assert win._active_source == ActiveSource.CAMERA
    assert win._selected_image_frame is None
    assert "Camera trực tiếp" in win.preview_group.title()

    # Open test image
    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
    test_img_path = str(fixtures_dir / "regression_p1_2" / "dark_theme_vietnamese_996x587.png")
    assert os.path.exists(test_img_path)

    win.action_open_image(test_img_path)

    # Should switch to ActiveSource.IMAGE immediately
    assert win._active_source == ActiveSource.IMAGE
    assert win._selected_image_frame is not None
    assert win._selected_image_name == "dark_theme_vietnamese_996x587.png"
    assert "dark_theme_vietnamese_996x587.png" in win.preview_group.title()

    # Simulate 50 preview timer ticks spanning beyond the old 15-second timer
    for _ in range(50):
        win._update_preview_tick()

    # Verify preview is STILL the image, NOT overwritten by camera
    assert win._active_source == ActiveSource.IMAGE
    assert win._selected_image_frame is not None
    assert "dark_theme_vietnamese_996x587.png" in win.preview_group.title()

    # Repeat ('R') and Stop ('Esc') must retain image preview
    win.action_repeat()
    win._update_preview_tick()
    assert win._active_source == ActiveSource.IMAGE
    assert win._selected_image_frame is not None

    win.action_stop()
    win._update_preview_tick()
    assert win._active_source == ActiveSource.IMAGE
    assert win._selected_image_frame is not None

    # Pressing Space ('action_capture_and_read') must preempt image mode and restore camera
    win.action_capture_and_read()
    assert win._active_source == ActiveSource.CAMERA
    assert win._selected_image_frame is None
    assert "Camera trực tiếp" in win.preview_group.title()

    # Open image again and test Ctrl+L ('action_clear_results')
    win.action_open_image(test_img_path)
    assert win._active_source == ActiveSource.IMAGE
    win.action_clear_results()
    assert win._active_source == ActiveSource.CAMERA
    assert win._selected_image_frame is None
    assert "Camera trực tiếp" in win.preview_group.title()

    win.close()


def test_p1_3_20_source_switches_stress_race(qapp):
    """Stress test: 20 rapid switches between Image mode and Camera mode."""
    cfg = AppConfig()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cam = MockCameraSource(frame=mock_frame, allow_open=True)
    gate = ImageQualityGate(cfg.quality)
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()
    speech = FakeSpeechBackend(vi_available=True)

    ctrl = SessionController(
        camera=cam,
        quality_gate=gate,
        ocr_backend=ocr,
        safety_policy=policy,
        speech_backend=speech,
        config=cfg,
    )
    ctrl.start_camera()
    win = AccessibleMainWindow(ctrl, cfg, auto_start_camera=False)

    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
    test_img_path = str(fixtures_dir / "regression_p1_2" / "dark_theme_vietnamese_996x587.png")

    for i in range(20):
        if i % 2 == 0:
            win.action_open_image(test_img_path)
            assert win._active_source == ActiveSource.IMAGE
        else:
            win.action_capture_and_read()
            assert win._active_source == ActiveSource.CAMERA
        win._update_preview_tick()

    win.close()


def test_p1_3_webcam_phone_screen_diagnostics_and_guidance():
    """Verify diagnostic detection and Vietnamese guidance on the user's phone screenshot."""
    packet_img = Path(__file__).resolve().parent.parent / "fixtures" / "regression_p1_3" / "webcam_uncertain_job10_1920x1080.png"
    assert packet_img.exists()

    bgr = load_image_unicode(str(packet_img))
    assert bgr is not None
    assert bgr.shape == (1080, 1920, 3)

    cfg = AppConfig()
    gate = ImageQualityGate(cfg.quality)
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()

    metrics = gate.evaluate(bgr)

    # 1. Edge clipping and guidance detection
    assert metrics.edge_clipped_text is True
    assert len(metrics.guidance_hints_vi) >= 1
    assert any("giữa khung hình" in h for h in metrics.guidance_hints_vi)

    # 2. Reading Safety Policy behavior: must return UNCERTAIN, never false-claim verified/readable
    ocr_res = ocr.recognize(bgr)
    verdict = policy.evaluate(metrics, ocr_res)

    assert verdict.status in (SafetyStatus.UNCERTAIN, SafetyStatus.BAD_IMAGE)
    assert verdict.status != SafetyStatus.READABLE

    # Check that guidance hints are propagated into warnings for speech and banner display
    warnings_text = " ".join(verdict.warnings)
    assert any("giữa khung hình" in w for w in verdict.warnings)


def test_p1_3_camera_burst_sharpness_selector():
    """Verify CameraSource burst sharpness selection without disk writes."""
    cam = MockCameraSource(allow_open=True)
    sharp_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(sharp_img, "VISIONMATE SHARP TEST", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    blurry_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)

    cam.set_burst_frames([blurry_img, sharp_img, blurry_img])
    cam.open()

    ok, selected_frame = cam.read_frame()
    assert ok is True
    assert selected_frame is not None
    assert selected_frame.shape == (480, 640, 3)

    sharpness_selected = CameraSource._frame_sharpness(selected_frame)
    sharpness_blurry = CameraSource._frame_sharpness(blurry_img)
    assert sharpness_selected > sharpness_blurry
    cam.close()


def test_p1_3_regression_phase1_2_images_still_readable():
    """Verify that Phase 1.2 test images (Dark theme & Inverted) remain READABLE with high accuracy."""
    fixtures_p1_2 = Path(__file__).resolve().parent.parent / "fixtures" / "regression_p1_2"
    dark_path = fixtures_p1_2 / "dark_theme_vietnamese_996x587.png"
    inv_path = fixtures_p1_2 / "diagnostic_inverted_only.png"

    cfg = AppConfig()
    gate = ImageQualityGate(cfg.quality)
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()

    for img_path in (dark_path, inv_path):
        bgr = load_image_unicode(str(img_path))
        assert bgr is not None
        metrics = gate.evaluate(bgr)
        ocr_res = ocr.recognize(bgr)
        verdict = policy.evaluate(metrics, ocr_res)

        assert verdict.status == SafetyStatus.READABLE
        assert len(ocr_res.lines) >= 12
        full_text = " ".join(ln.text for ln in ocr_res.lines).lower()
        assert "nghị định 30/2020" in full_text
        assert "văn bản hành chính" in full_text
