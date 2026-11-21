# -*- coding: utf-8 -*-
"""
Integration tests for VisionMate Glasses Phase 1 (Section 8 Step 4, Section 9 A01..A11):
- Real Tesseract OCR backend (`tessdata_fast` and `tessdata_best`) on Vietnamese text & digits
- Real Windows OneCore TTS backend (`MSTTS_V110_viVN_An`) availability and RAM synthesis
- Offline runtime enforcement (socket.connect blocked at runtime -> zero network calls)
- Real PySide6 AccessibleMainWindow keyboard shortcuts (Space, R, Esc, F1) and accessible names
"""
from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from visionmate.camera.sources import MockCameraSource, load_image_unicode
from visionmate.config import AppConfig
from visionmate.contracts import SafetyStatus, SessionState
from visionmate.controller import SessionController
from visionmate.ocr.paddle_backend import PaddleONNXOCRBackend
from visionmate.ocr.tesseract_backend import TesseractOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.order import ReadingOrder
from visionmate.reading.policy import SafetyPolicy
from visionmate.speech.fake_speech import FakeSpeechBackend
from visionmate.speech.sapi_onecore import WindowsOneCoreSpeechBackend
from visionmate.ui.main_window import AccessibleMainWindow


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "tests" / "fixtures" / "dataset"
MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "manifest.json"


def test_manifest_has_at_least_70_authorized_images() -> None:
    assert MANIFEST_PATH.is_file()
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["synthetic"] is True
    assert manifest["consent_safe"] is True
    items = manifest["items"]
    assert len(items) >= 70

    cats = {}
    for it in items:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
        assert (DATASET_DIR / it["filename"]).is_file()

    assert cats.get("clear_vi", 0) >= 30
    assert cats.get("challenging_vi", 0) >= 20
    assert cats.get("no_text", 0) >= 10
    assert cats.get("digits_dates", 0) >= 10


def test_real_tesseract_fast_and_best_vietnamese_ocr() -> None:
    img = load_image_unicode(DATASET_DIR / "clear_01.png")
    assert img is not None

    fast_ocr = TesseractOCRBackend(variant="fast")
    best_ocr = TesseractOCRBackend(variant="best")
    assert fast_ocr.is_available() is True
    assert best_ocr.is_available() is True

    res_fast = fast_ocr.recognize(img)
    res_best = best_ocr.recognize(img)
    assert res_fast.error is None
    assert res_best.error is None
    assert ReadingOrder.join_for_display(res_fast.lines) == "Sữa tươi tiệt trùng không đường"
    assert ReadingOrder.join_for_display(res_best.lines) == "Sữa tươi tiệt trùng không đường"


def test_real_paddle_onnx_backend_pluggable() -> None:
    img = load_image_unicode(DATASET_DIR / "clear_01.png")
    assert img is not None
    paddle = PaddleONNXOCRBackend()
    assert paddle.is_available() is True
    res = paddle.recognize(img)
    assert res.error is None
    assert len(res.lines) >= 1


def test_real_windows_onecore_vietnamese_tts_offline() -> None:
    tts = WindowsOneCoreSpeechBackend(rate=0, volume=80, play_audio=False)
    voices = tts.list_voices()
    assert len(voices) >= 1
    assert tts.is_vi_vn_available() is True

    started = []
    ok = tts.speak(
        text="Xin chào, thử nghiệm đọc tiếng Việt ngoại tuyến.",
        job_id=101,
        on_speech_started=lambda jid, ms: started.append((jid, ms)),
    )
    assert ok is True
    assert len(started) == 1
    assert started[0][0] == 101
    tts.shutdown()


def test_strict_offline_runtime_no_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify A01 & A11: Full pipeline executes with socket & urllib completely blocked."""
    def _block_network(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Network access is strictly forbidden at runtime!")

    monkeypatch.setattr(socket.socket, "connect", _block_network)
    monkeypatch.setattr(urllib.request, "urlopen", _block_network)

    img_path = DATASET_DIR / "digits_01.png"
    ctrl = SessionController(
        camera=MockCameraSource(allow_open=False),
        quality_gate=ImageQualityGate(),
        ocr_backend=TesseractOCRBackend(variant="fast"),
        safety_policy=SafetyPolicy(),
        speech_backend=WindowsOneCoreSpeechBackend(play_audio=False),
        config=AppConfig(),
    )
    ctrl.read_image_file(img_path)
    assert ctrl.wait_for_idle(timeout_s=8.0)

    res = ctrl.last_confirmed_result
    assert res is not None
    assert res.status == SafetyStatus.READABLE
    assert "15/03/2026" in res.spoken_text
    assert "15/03/2027" in res.spoken_text
    ctrl.shutdown()


def test_accessible_desktop_ui_shortcuts_and_labels() -> None:
    app = QApplication.instance() or QApplication([])
    img = load_image_unicode(DATASET_DIR / "clear_02.png")
    assert img is not None

    ctrl = SessionController(
        camera=MockCameraSource(frame=img, allow_open=True),
        quality_gate=ImageQualityGate(),
        ocr_backend=TesseractOCRBackend(variant="fast"),
        safety_policy=SafetyPolicy(),
        speech_backend=FakeSpeechBackend(vi_available=True, speak_duration_s=0.02),
        config=AppConfig(),
    )
    win = AccessibleMainWindow(controller=ctrl, config=AppConfig(), auto_start_camera=False)
    win.show()
    ctrl.start_camera()
    app.processEvents()

    # Check accessible names exist on all primary controls (A10)
    assert win.btn_read.accessibleName() != ""
    assert win.btn_repeat.accessibleName() != ""
    assert win.btn_stop.accessibleName() != ""
    assert win.btn_open_image.accessibleName() != ""
    assert win.btn_help.accessibleName() != ""
    assert win.txt_output.accessibleName() != ""
    assert win.cmb_camera.accessibleName() != ""
    assert win.cmb_ocr.accessibleName() != ""

    # Trigger Space key via eventFilter -> Capture & Read
    space_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    win.eventFilter(win, space_event)
    assert ctrl.wait_for_idle(timeout_s=5.0)
    app.processEvents()
    assert "Nước khoáng thiên nhiên đóng chai" in win.txt_output.toPlainText()

    # Trigger F1 key via eventFilter -> Help
    f1_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_F1, Qt.KeyboardModifier.NoModifier)
    win.eventFilter(win, f1_event)
    assert ctrl.wait_for_idle(timeout_s=2.0)

    # Trigger Esc key via eventFilter -> Stop
    esc_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    win.eventFilter(win, esc_event)
    app.processEvents()
    assert ctrl.state in (SessionState.CAMERA_READY, SessionState.CANCELLED)

    win.close()
    app.processEvents()
