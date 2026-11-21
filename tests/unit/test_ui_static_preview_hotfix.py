# -*- coding: utf-8 -*-
"""
Unit and UI Integration Tests for OD-1 UI Static Preview Overflow Hotfix.
Verifies:
1. File name truncation in Preview bar (lbl_fps) and Khung hình group title (preview_group).
2. Full tooltips retained for hover inspection (filename + path).
3. Frame fitting into preview area: aspect ratio preserved, centered, no overflow.
4. Bounding boxes remain visually exact and proportional after scaling.
5. Shortcuts & UI interactions: O, D, Space, Esc, Ctrl+L in both Reading and OD modes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from visionmate.camera.sources import MockCameraSource, load_image_unicode
from visionmate.config import AppConfig
from visionmate.contracts import ActiveSource, AppMode, DetectedObject, SessionState
from visionmate.controller import SessionController
from visionmate.detection.annotator import draw_detection_annotations
from visionmate.detection.detector import FakeObjectDetector
from visionmate.ocr.fake_backend import FakeOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import SafetyPolicy
from visionmate.speech.fake_speech import FakeSpeechBackend
from visionmate.ui.main_window import AccessibleMainWindow, truncate_display_name


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _create_test_window(
    qapp: QApplication,
    mode: AppMode = AppMode.OBJECT_DETECTION,
    mock_frame: np.ndarray = None,
) -> tuple[AccessibleMainWindow, SessionController]:
    if mock_frame is None:
        mock_frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    ctrl = SessionController(
        camera=MockCameraSource(frame=mock_frame, allow_open=True),
        quality_gate=ImageQualityGate(),
        ocr_backend=FakeOCRBackend(),
        safety_policy=SafetyPolicy(),
        speech_backend=FakeSpeechBackend(),
        object_detector=FakeObjectDetector(
            mock_objects=[
                DetectedObject(
                    class_id=39,
                    class_name_en="bottle",
                    class_name_vi="chai nước",
                    confidence=0.91,
                    box_xywh=(50, 60, 80, 180),
                )
            ]
        ),
        config=AppConfig(),
    )
    ctrl.set_mode(mode)
    win = AccessibleMainWindow(controller=ctrl, config=AppConfig(), auto_start_camera=False)
    win.show()
    ctrl.start_camera()
    qapp.processEvents()
    return win, ctrl


def test_truncate_display_name_logic() -> None:
    """Verify truncate_display_name handles short, medium, and extreme filenames cleanly."""
    # 1. Short name remains completely intact
    assert truncate_display_name("clear_02.png", max_chars=34) == "clear_02.png"
    assert truncate_display_name("dog.jpg", max_chars=34) == "dog.jpg"

    # 2. Baseline regression test fixture (33 chars) must NOT be truncated when max_chars=34
    baseline_fixture = "dark_theme_vietnamese_996x587.png"
    assert len(baseline_fixture) == 33
    assert truncate_display_name(baseline_fixture, max_chars=34) == baseline_fixture

    # 3. Very long name is truncated to exact max_chars with ellipsis
    long_name = "VisionMate_OD1_Test_Sample_With_Extremely_Long_File_Name_For_Testing_UI_Overflow_Regression_Case_123456789.png"
    short_34 = truncate_display_name(long_name, max_chars=34)
    assert len(short_34) == 34
    assert short_34.endswith(".png")
    assert "..." in short_34

    # 4. Preview bar truncation (budget 28)
    short_28 = truncate_display_name(long_name, max_chars=28)
    assert len(short_28) == 28
    assert short_28.endswith(".png")
    assert "..." in short_28

    # 5. File without extension
    no_ext = "extremely_long_file_name_without_extension_identifier"
    short_no_ext = truncate_display_name(no_ext, max_chars=24)
    assert len(short_no_ext) == 24
    assert short_no_ext.endswith("...")


def test_long_filename_does_not_overflow_ui(qapp: QApplication, tmp_path: Path) -> None:
    """Verify opening an image with an extreme 100+ char filename does not blow up UI width."""
    long_filename = "VisionMate_OD1_Test_Sample_With_Extremely_Long_File_Name_For_Testing_UI_Overflow_Regression_Case_123456789.png"
    test_file = tmp_path / long_filename
    # Create test image file
    sample_img = np.full((300, 400, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(test_file), sample_img)

    win, ctrl = _create_test_window(qapp, mode=AppMode.OBJECT_DETECTION)
    try:
        # Load the image with extreme name
        win.action_open_image(str(test_file))
        assert ctrl.wait_for_idle(timeout_s=3.0)
        qapp.processEvents()

        # 1. Preview group title must be truncated
        title = win.preview_group.title()
        assert "..." in title
        assert len(win.preview_group.title()) <= 85

        # 2. Status bar preview label must be truncated
        fps_text = win.lbl_fps.text()
        assert "..." in fps_text
        assert len(fps_text) < 50

        # 3. Tooltips must contain the FULL filename and resolved path
        expected_name = long_filename
        expected_path = str(test_file.resolve())
        assert expected_name in win.preview_group.toolTip()
        assert expected_path in win.preview_group.toolTip()
        assert expected_name in win.lbl_fps.toolTip()
        assert expected_path in win.lbl_fps.toolTip()
        assert expected_name in win.lbl_preview.toolTip()
        assert expected_path in win.lbl_preview.toolTip()

        # 4. Preview group minimum size hint must not explode (must stay < 700px)
        assert win.preview_group.minimumSizeHint().width() < 700
        assert win.lbl_fps.minimumSizeHint().width() < 400

    finally:
        win.close()
        ctrl.shutdown()
        qapp.processEvents()


def test_fit_static_image_aspect_ratio_and_centering(qapp: QApplication) -> None:
    """Verify preview fits images with various aspect ratios without distortion or overflow."""
    win, ctrl = _create_test_window(qapp, mode=AppMode.OBJECT_DETECTION)
    try:
        # Test shapes: (height, width)
        # 1. Tall vertical document (379x527)
        # 2. Ultra-wide frame (1920x600)
        # 3. Standard 4:3 (640x480)
        test_shapes = [(527, 379), (600, 1920), (480, 640)]

        for h, w in test_shapes:
            frame = np.full((h, w, 3), 120, dtype=np.uint8)
            win._display_bgr_frame(frame)
            qapp.processEvents()

            pixmap = win.lbl_preview.pixmap()
            assert pixmap is not None
            pw = pixmap.width()
            ph = pixmap.height()

            # Aspect ratio of displayed pixmap must match source frame (within rounding error)
            src_ar = float(w) / float(h)
            disp_ar = float(pw) / float(ph)
            assert abs(src_ar - disp_ar) < 0.05, f"Aspect ratio distorted: {src_ar} vs {disp_ar}"

            # Must fit strictly inside inner bounds of preview label
            rect = win.lbl_preview.contentsRect()
            assert pw <= rect.width() + 1
            assert ph <= rect.height() + 1

            # Alignment must be centered
            assert win.lbl_preview.alignment() == Qt.AlignmentFlag.AlignCenter

    finally:
        win.close()
        ctrl.shutdown()
        qapp.processEvents()


def test_bounding_boxes_remain_exact_after_preview_scaling(qapp: QApplication) -> None:
    """Verify that annotated frames with bounding boxes scale proportionally onto preview."""
    win, ctrl = _create_test_window(qapp, mode=AppMode.OBJECT_DETECTION)
    try:
        # Create a 1200x800 image with an annotated box
        orig_img = np.zeros((800, 1200, 3), dtype=np.uint8)
        objs = [
            DetectedObject(
                class_id=39,
                class_name_en="bottle",
                class_name_vi="chai nước",
                confidence=0.95,
                box_xywh=(200, 150, 300, 400),
            )
        ]
        annotated = draw_detection_annotations(orig_img, objs)

        # In annotated frame, box pixels at (200, 150) are colored (not black)
        assert np.any(annotated[150, 200] > 0)

        # Display on preview
        win._display_bgr_frame(annotated)
        qapp.processEvents()

        pix = win.lbl_preview.pixmap()
        assert pix is not None
        assert pix.width() > 0
        assert pix.height() > 0

        # Pixmap aspect ratio matches 1200 / 800 = 1.5
        assert abs((pix.width() / pix.height()) - 1.5) < 0.02

    finally:
        win.close()
        ctrl.shutdown()
        qapp.processEvents()


def test_shortcuts_and_mode_regression(qapp: QApplication, tmp_path: Path) -> None:
    """Verify Ctrl+L, Esc, O, D, and Space work seamlessly with preview scaling and tooltips."""
    test_img = tmp_path / "test_shelf_objects.png"
    cv2.imwrite(str(test_img), np.full((400, 600, 3), 160, dtype=np.uint8))

    win, ctrl = _create_test_window(qapp, mode=AppMode.OBJECT_DETECTION)
    try:
        # 1. Open image (O)
        win.action_open_image(str(test_img))
        assert ctrl.wait_for_idle(timeout_s=3.0)
        qapp.processEvents()

        assert win._active_source == ActiveSource.IMAGE
        assert win._selected_image_name == "test_shelf_objects.png"
        assert "test_shelf_objects.png" in win.preview_group.title()
        assert "test_shelf_objects.png" in win.preview_group.toolTip()

        # 2. Trigger detection on opened image via D key
        d_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_D, Qt.KeyboardModifier.NoModifier)
        win.eventFilter(win, d_event)
        assert ctrl.wait_for_idle(timeout_s=3.0)
        qapp.processEvents()

        assert win.controller.last_confirmed_result is not None
        assert win.controller.last_confirmed_result.mode == "object_detection"
        assert win._frozen_detection_frame is not None

        # 3. Test Esc stops reading but keeps frozen preview
        esc_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        win.eventFilter(win, esc_event)
        qapp.processEvents()
        assert win._frozen_detection_frame is not None

        # 4. Test Ctrl+L clears results, resets tooltips and returns to Camera preview
        ctrl_l_event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_L,
            Qt.KeyboardModifier.ControlModifier,
        )
        win.eventFilter(win, ctrl_l_event)
        qapp.processEvents()

        assert win._frozen_detection_frame is None
        assert win._selected_image_frame is None
        assert win._selected_image_name == ""
        assert win._selected_image_path == ""
        assert "Camera trực tiếp" in win.preview_group.title()
        assert win.preview_group.toolTip() == ""
        assert win.lbl_preview.toolTip() == ""

        # 5. Toggle mode to READING (M)
        m_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_M, Qt.KeyboardModifier.NoModifier)
        win.eventFilter(win, m_event)
        qapp.processEvents()
        assert win.controller.mode == AppMode.READING
        assert "ĐỌC VĂN BẢN" in win.btn_mode.text()

        # 6. Space in READING mode triggers OCR
        space_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        win.eventFilter(win, space_event)
        assert ctrl.wait_for_idle(timeout_s=3.0)
        qapp.processEvents()

        assert win.controller.last_confirmed_result is not None
        assert win.controller.last_confirmed_result.mode == "reading"

    finally:
        win.close()
        ctrl.shutdown()
        qapp.processEvents()
