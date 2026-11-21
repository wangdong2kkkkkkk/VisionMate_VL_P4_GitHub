# -*- coding: utf-8 -*-
"""
Phase 1.1 Regression, Real-Document Layout, Multi-Column Reading Order, Uncertainty Refusal,
and UI Clear Results (`Ctrl+L`) Automated Test Suite.

Covers all mandatory requirements in `ANTIGRAVITY_TASK.md`:
1. Untouched primary regression image (`administrative_form_379x527.png`) SHA-256 verification,
   low-resolution `UNCERTAIN` refusal with exact Vietnamese spoken message, zero hallucinated
   garbage strings from `observed_ocr.txt`, and verified headings in proper 2-column order.
2. Multi-column administrative document reading order (Left Column top-to-bottom -> Right Column
   top-to-bottom -> Body -> Left Footer -> Right Footer) and dotted-line/rule suppression.
3. `clear_results()` (`Ctrl+L` / button `"Xóa kết quả (Ctrl+L)"`):
   - Clears recognized text and spoken content immediately
   - Stops TTS and cancels active async OCR/TTS jobs
   - Resets `repeat_last()` (`R`) so old content is not replayed (`MSG_NO_REPEAT`)
   - Keeps live webcam preview open (`CAMERA_READY`)
   - Allows immediate new capture (`Space`) or file open (`O`)
4. Stress test:
   - 20 `clear_results()` calls during active OCR/TTS
   - 20 rapid source switches (`file` <-> `camera`) + shutdown
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from visionmate.camera.sources import MockCameraSource, load_image_unicode
from visionmate.config import AppConfig
from visionmate.contracts import OCRLine, SafetyStatus, SessionJobResult, SessionState
from visionmate.controller import MSG_NO_REPEAT, MSG_RESULTS_CLEARED, SessionController
from visionmate.ocr.fake_backend import FakeOCRBackend
from visionmate.ocr.tesseract_backend import TesseractOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import MSG_LOW_RES_OR_UNCERTAIN, SafetyPolicy
from visionmate.speech.fake_speech import FakeSpeechBackend
from visionmate.ui.main_window import AccessibleMainWindow


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGRESSION_DIR = PROJECT_ROOT / "tests" / "fixtures" / "regression_p1_1"
REGRESSION_IMG = REGRESSION_DIR / "administrative_form_379x527.png"
REAL_DOCS_DIR = PROJECT_ROOT / "tests" / "fixtures" / "real_docs_p1_1"
MANIFEST_P1_1 = REAL_DOCS_DIR / "manifest_p1_1.json"
DATASET_DIR = PROJECT_ROOT / "tests" / "fixtures" / "dataset"

EXPECTED_REGRESSION_SHA256 = "7CADF447D955D1E74647723D2709338818B434D42A506EADBEE232FB2D44CDB7"


def test_p1_1_regression_sample_untouched_and_handled_safely() -> None:
    """
    Verify that `administrative_form_379x527.png` is byte-for-byte identical to the user's
    original regression packet and that Phase 1.1:
    - Detects its low-resolution (5-7px footnotes) and classifies full-page reading as `UNCERTAIN`
    - Speaks the exact Vietnamese low-resolution warning instead of reading out garbled text
    - Eliminates the garbage strings from `observed_ocr.txt`
    - Extracts the readable large headings in proper 2-column order
    """
    assert REGRESSION_IMG.is_file()
    actual_sha = hashlib.sha256(REGRESSION_IMG.read_bytes()).hexdigest().upper()
    assert actual_sha == EXPECTED_REGRESSION_SHA256

    img = load_image_unicode(REGRESSION_IMG)
    assert img is not None
    assert img.shape[:2] == (527, 379)

    gate = ImageQualityGate()
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()

    q_res = gate.evaluate(img)
    assert q_res.passed is True

    o_res = ocr.recognize(img)
    assert o_res.error is None
    assert o_res.low_resolution_detected is True
    assert o_res.median_char_height_px <= 9.0
    assert o_res.rejected_line_count > 0

    decision = policy.evaluate(q_res, o_res)
    assert decision.status == SafetyStatus.UNCERTAIN
    assert decision.spoken_text == MSG_LOW_RES_OR_UNCERTAIN

    # Verify the verified headings are extracted in proper reading order
    extracted_texts = [ln.text for ln in decision.ordered_lines]
    joined_headings = "\n".join(extracted_texts)
    assert "Mẫu 1.1 - Nghị quyết (cá biệt)" in joined_headings
    assert "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" in joined_headings
    assert "Độc lập - Tự do - Hạnh phúc" in joined_headings
    assert "NGHỊ QUYẾT" in joined_headings
    assert "QUYẾT NGHỊ:" in joined_headings
    assert "Nơi nhận:" in joined_headings
    assert "QUYỀN HẠN, CHỨC VỤ CỦA NGƯỜI KÝ" in joined_headings

    # Verify 2-column footer order: Left column ("Nơi nhận:") comes BEFORE Right column ("QUYỀN HẠN...")
    idx_noi_nhan = next(i for i, t in enumerate(extracted_texts) if "Nơi nhận" in t)
    idx_quyen_han = next(i for i, t in enumerate(extracted_texts) if "QUYỀN HẠN" in t)
    assert idx_noi_nhan < idx_quyen_han

    # Verify none of the Phase 1 garbage strings from observed_ocr.txt appear
    forbidden_garbage = [
        "TÊN €Q,",
        "Số ẢNG.2",
        "ng. sâm",
        "(Chữ b š ca cơ quan",
        "Ghi ehú:",
        "của cơ quan, 1ổ chức",
    ]
    for bad in forbidden_garbage:
        assert bad not in joined_headings
        assert bad not in decision.display_text


def test_p1_1_real_docs_manifest_and_multicol_reading_order() -> None:
    """Verify the 40-image Phase 1.1 evaluation suite and 2-column administrative reading order."""
    assert MANIFEST_P1_1.is_file()
    with MANIFEST_P1_1.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    counts = manifest["counts"]
    assert counts["total"] >= 40
    assert counts["readable_doc_single"] >= 10
    assert counts["readable_doc_multicol"] >= 10
    assert counts["low_quality_unreadable_doc"] >= 10
    assert counts["product_packaging_receipt"] >= 10

    # Test a 2-column administrative document (doc_multicol_01.png)
    img = load_image_unicode(REAL_DOCS_DIR / "doc_multicol_01.png")
    assert img is not None
    gate = ImageQualityGate()
    ocr = TesseractOCRBackend(variant="fast")
    policy = SafetyPolicy()

    q_res = gate.evaluate(img)
    o_res = ocr.recognize(img)
    decision = policy.evaluate(q_res, o_res)

    assert decision.status == SafetyStatus.READABLE
    lines = [ln.text for ln in decision.ordered_lines]

    # Top Left column ("ỦY BAN NHÂN", "QUẬN HOÀN KIẾM", "Số: 12/NQ-UBND") must precede Top Right ("CỘNG HÒA...")
    idx_ubnd = next(i for i, t in enumerate(lines) if "ỦY BAN NHÂN" in t)
    idx_so = next(i for i, t in enumerate(lines) if "12/NQ-UBND" in t)
    idx_chxh = next(i for i, t in enumerate(lines) if "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" in t)
    idx_doc_lap = next(i for i, t in enumerate(lines) if "Độc lập - Tự do - Hạnh phúc" in t)
    idx_nghi_quyet = next(i for i, t in enumerate(lines) if t.strip() == "NGHỊ QUYẾT")
    idx_noi_nhan = next(i for i, t in enumerate(lines) if "Nơi nhận" in t)
    idx_chu_tich = next(i for i, t in enumerate(lines) if "CHỦ TỊCH" in t)

    assert idx_ubnd < idx_so < idx_chxh < idx_doc_lap < idx_nghi_quyet < idx_noi_nhan < idx_chu_tich


def test_ui_clear_results_ctrl_l_and_button_behavior() -> None:
    """
    Verify Objective 1 & 2 (`ANTIGRAVITY_TASK.md`):
    - Button `"Xóa kết quả (Ctrl+L)"` and shortcut `Ctrl+L` clear text immediately
    - Stops TTS and resets `repeat_last()` (`R`) so old content is not replayed
    - Keeps camera preview open (`CAMERA_READY`)
    - New jobs (`Space` or `O`) replace old results immediately
    """
    app = QApplication.instance() or QApplication([])
    cam_img = load_image_unicode(DATASET_DIR / "clear_01.png")
    file_img_path = DATASET_DIR / "clear_02.png"
    assert cam_img is not None

    speech = FakeSpeechBackend(vi_available=True, speak_duration_s=0.02)
    ctrl = SessionController(
        camera=MockCameraSource(frame=cam_img, allow_open=True),
        quality_gate=ImageQualityGate(),
        ocr_backend=TesseractOCRBackend(variant="fast"),
        safety_policy=SafetyPolicy(),
        speech_backend=speech,
        config=AppConfig(),
    )
    win = AccessibleMainWindow(controller=ctrl, config=AppConfig(), auto_start_camera=False)
    win.show()
    ctrl.start_camera()
    app.processEvents()

    assert win.btn_clear.accessibleName() != ""
    assert "Ctrl+L" in win.btn_clear.text()
    assert win.cmb_roi.accessibleName() != ""

    # 1. Open test image (O) and wait for result
    win.action_open_image(str(file_img_path))
    assert ctrl.wait_for_idle(timeout_s=5.0)
    app.processEvents()
    assert "Nước khoáng thiên nhiên đóng chai" in win.txt_output.toPlainText()
    assert ctrl.last_confirmed_result is not None

    # 2. Press Ctrl+L via eventFilter -> Clear results immediately
    ctrl_l_event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_L,
        Qt.KeyboardModifier.ControlModifier,
    )
    win.eventFilter(win, ctrl_l_event)
    app.processEvents()

    assert win.txt_output.toPlainText() == ""
    assert ctrl.last_confirmed_result is None
    assert ctrl.camera.is_open() is True
    assert ctrl.state == SessionState.CAMERA_READY
    assert MSG_RESULTS_CLEARED in win.lbl_warning_banner.text()

    # 3. Press R after clear_results -> must NOT replay old OCR text; must speak MSG_NO_REPEAT
    speech.history.clear()
    r_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier)
    win.eventFilter(win, r_event)
    assert ctrl.wait_for_idle(timeout_s=2.0)
    app.processEvents()
    assert len(speech.history) == 1
    assert speech.history[-1].text == MSG_NO_REPEAT

    # 4. Press Space after clear_results -> captures from live camera and displays new result
    space_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    win.eventFilter(win, space_event)
    assert ctrl.wait_for_idle(timeout_s=5.0)
    app.processEvents()
    assert "Sữa tươi tiệt trùng không đường" in win.txt_output.toPlainText()

    # 5. Click btn_clear directly -> clears again
    win.btn_clear.click()
    app.processEvents()
    assert win.txt_output.toPlainText() == ""
    assert ctrl.last_confirmed_result is None

    win.close()
    app.processEvents()


def test_20_clear_results_and_20_source_switches_race_stress(tmp_path: Path) -> None:
    """
    Verify Objective 2 (`ANTIGRAVITY_TASK.md`):
    - At least 20 `clear_results()` calls during active OCR/TTS
    - At least 20 rapid switches between file open (`O`) and camera capture (`Space`)
    - Close application while job is running
    - Zero stale results repopulating UI, zero old speech continuing, zero crashes
    """
    cam_img = load_image_unicode(DATASET_DIR / "clear_01.png")
    file_img_path = DATASET_DIR / "clear_02.png"
    assert cam_img is not None

    cam = MockCameraSource(frame=cam_img, allow_open=True)
    gate = ImageQualityGate()
    ocr = FakeOCRBackend(
        lines=[OCRLine(text="Văn bản thử nghiệm bất đồng bộ", bbox=(10, 10, 260, 35), raw_score=95.0)],
        delay_s=0.035,
    )
    policy = SafetyPolicy()
    speech = FakeSpeechBackend(vi_available=True, speak_duration_s=0.04)

    completed_results: List[SessionJobResult] = []
    cleared_events: List[int] = []

    ctrl = SessionController(
        camera=cam,
        quality_gate=gate,
        ocr_backend=ocr,
        safety_policy=policy,
        speech_backend=speech,
        on_job_completed=lambda r: completed_results.append(r),
        on_results_cleared=lambda sid, _msg: cleared_events.append(sid),
        debug_dir=tmp_path / "debug",
    )
    ctrl.start_camera()

    # Part A: 20 clear_results() calls while OCR/TTS is actively running
    for i in range(20):
        if i % 2 == 0:
            ctrl.capture_and_read()
        else:
            ctrl.read_image_file(file_img_path)
        time.sleep(0.012)
        ctrl.clear_results()

    assert ctrl.wait_for_idle(timeout_s=3.0)
    # Zero stale jobs should have completed after clear_results()
    assert len(completed_results) == 0
    assert len(cleared_events) == 20
    assert ctrl.last_confirmed_result is None
    assert ctrl.camera.is_open() is True
    assert ctrl.state == SessionState.CAMERA_READY

    # Part B: 20 rapid switches between file and camera capture
    completed_results.clear()
    speech.history.clear()
    last_job_id = -1
    for i in range(20):
        if i % 2 == 0:
            last_job_id = ctrl.read_image_file(file_img_path)
        else:
            last_job_id = ctrl.capture_and_read()
        time.sleep(0.006)

    assert ctrl.wait_for_idle(timeout_s=3.0)
    assert len(completed_results) == 1
    assert completed_results[0].job_id == last_job_id
    stale_speeches = [s for s in speech.history if s.completed and s.job_id != last_job_id]
    assert stale_speeches == []

    # Part C: Shutdown while job is active
    ctrl.capture_and_read()
    time.sleep(0.01)
    ctrl.shutdown()
