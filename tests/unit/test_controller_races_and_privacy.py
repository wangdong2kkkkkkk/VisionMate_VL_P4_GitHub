# -*- coding: utf-8 -*-
"""Unit tests for SessionController concurrency, cancellation, races, and privacy (FR-07, FR-08, A09, A11)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw

from visionmate.camera.sources import MockCameraSource
from visionmate.config import AppConfig
from visionmate.contracts import OCRLine, SafetyStatus, SessionJobResult, SessionState
from visionmate.controller import MSG_CAMERA_MISSING, MSG_TTS_BLOCKED, SessionController
from visionmate.ocr.fake_backend import FakeOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import SafetyPolicy
from visionmate.speech.fake_speech import FakeSpeechBackend


def _make_sharp_frame() -> np.ndarray:
    pil_img = Image.new("RGB", (640, 220), color=(250, 250, 250))
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([20, 20, 620, 200], outline=(10, 10, 10), width=3)
    draw.text((40, 80), "Sữa tươi không đường 180ml", fill=(10, 10, 10))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def test_20_cancels_and_20_superseding_races_no_stale_output(tmp_path: Path) -> None:
    frame = _make_sharp_frame()
    cam = MockCameraSource(frame=frame, allow_open=True)
    gate = ImageQualityGate()
    ocr = FakeOCRBackend(
        lines=[OCRLine(text="Sữa tươi không đường", bbox=(20, 40, 300, 30), raw_score=95.0)],
        delay_s=0.03,
    )
    policy = SafetyPolicy()
    speech = FakeSpeechBackend(vi_available=True, speak_duration_s=0.04)

    results: List[SessionJobResult] = []
    ctrl = SessionController(
        camera=cam,
        quality_gate=gate,
        ocr_backend=ocr,
        safety_policy=policy,
        speech_backend=speech,
        on_job_completed=lambda r: results.append(r),
        debug_dir=tmp_path / "debug_captures",
    )
    assert ctrl.start_camera() is True
    assert ctrl.state == SessionState.CAMERA_READY

    # 20 rapid cancels while worker is busy
    for _ in range(20):
        ctrl.capture_and_read()
        time.sleep(0.01)
        ctrl.stop_current()
    assert ctrl.wait_for_idle(timeout_s=3.0)
    assert len(results) == 0

    # 20 rapid superseding requests: only the final job_id is allowed to complete
    results.clear()
    speech.history.clear()
    last_job_id = -1
    for _ in range(20):
        last_job_id = ctrl.capture_and_read()
        time.sleep(0.005)

    assert ctrl.wait_for_idle(timeout_s=3.0)
    assert len(results) == 1
    assert results[0].job_id == last_job_id
    stale_speeches = [s for s in speech.history if s.completed and s.job_id != last_job_id]
    assert stale_speeches == []

    ctrl.shutdown()


def test_missing_camera_and_corrupt_image_handling(tmp_path: Path) -> None:
    cam = MockCameraSource(frame=None, allow_open=False)
    gate = ImageQualityGate()
    ocr = FakeOCRBackend(lines=[OCRLine(text="Test", bbox=(10, 10, 80, 20), raw_score=90.0)])
    policy = SafetyPolicy()
    speech = FakeSpeechBackend(vi_available=True, speak_duration_s=0.01)

    results: List[SessionJobResult] = []
    ctrl = SessionController(
        camera=cam,
        quality_gate=gate,
        ocr_backend=ocr,
        safety_policy=policy,
        speech_backend=speech,
        on_job_completed=lambda r: results.append(r),
        debug_dir=tmp_path / "debug",
    )
    assert ctrl.start_camera() is False
    assert ctrl.state == SessionState.ERROR

    # Try capture with missing camera -> graceful ENGINE_ERROR message, no crash
    ctrl.capture_and_read()
    assert ctrl.wait_for_idle(timeout_s=2.0)
    assert len(results) == 1
    assert results[0].status == SafetyStatus.ENGINE_ERROR
    assert MSG_CAMERA_MISSING in results[0].spoken_text

    # Corrupt file open -> graceful BAD_IMAGE message, no crash
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"not_an_image_at_all")
    ctrl.read_image_file(corrupt_file)
    assert ctrl.wait_for_idle(timeout_s=2.0)
    assert len(results) == 2
    assert results[1].status == SafetyStatus.BAD_IMAGE

    ctrl.shutdown()


def test_tts_blocked_mode_reports_truthfully(tmp_path: Path) -> None:
    frame = _make_sharp_frame()
    cam = MockCameraSource(frame=frame, allow_open=True)
    gate = ImageQualityGate()
    ocr = FakeOCRBackend(
        lines=[OCRLine(text="Trà xanh túi lọc", bbox=(10, 10, 200, 30), raw_score=92.0)]
    )
    policy = SafetyPolicy()
    speech = FakeSpeechBackend(vi_available=False)

    results: List[SessionJobResult] = []
    ctrl = SessionController(
        camera=cam,
        quality_gate=gate,
        ocr_backend=ocr,
        safety_policy=policy,
        speech_backend=speech,
        on_job_completed=lambda r: results.append(r),
        debug_dir=tmp_path / "debug",
    )
    ctrl.start_camera()
    ctrl.capture_and_read()
    assert ctrl.wait_for_idle(timeout_s=2.0)

    assert len(results) == 1
    assert results[0].tts_blocked is True
    assert results[0].lines[0].text == "Trà xanh túi lọc"
    assert any(MSG_TTS_BLOCKED in w for w in results[0].warnings)
    assert speech.history == []
    assert ctrl.state == SessionState.ERROR

    ctrl.shutdown()


def test_privacy_by_default_writes_no_files(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg = AppConfig(debug_opt_in_runtime=True)
    cfg.save(cfg_path)

    # Reloaded config must always reset debug_opt_in_runtime to False
    reloaded = AppConfig.load(cfg_path)
    assert reloaded.debug_opt_in_runtime is False

    debug_dir = tmp_path / "debug_captures"
    frame = _make_sharp_frame()
    ctrl = SessionController(
        camera=MockCameraSource(frame=frame, allow_open=True),
        quality_gate=ImageQualityGate(),
        ocr_backend=FakeOCRBackend(lines=[OCRLine(text="Nước khoáng", bbox=(10, 10, 100, 25), raw_score=90.0)]),
        safety_policy=SafetyPolicy(),
        speech_backend=FakeSpeechBackend(vi_available=True, speak_duration_s=0.01),
        config=reloaded,
        debug_dir=debug_dir,
    )
    ctrl.start_camera()
    ctrl.capture_and_read()
    assert ctrl.wait_for_idle(timeout_s=2.0)
    assert not debug_dir.exists()

    # Now opt-in at runtime and verify purge works
    reloaded.debug_opt_in_runtime = True
    ctrl.capture_and_read()
    assert ctrl.wait_for_idle(timeout_s=2.0)
    assert debug_dir.exists() and len(list(debug_dir.glob("*.json"))) == 1

    purged = ctrl.purge_debug_logs()
    assert purged == 1
    assert not debug_dir.exists()
    ctrl.shutdown()
