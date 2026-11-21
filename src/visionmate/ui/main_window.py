# -*- coding: utf-8 -*-
"""
AccessibleDesktopUI (Section 4.1, Section 6.2, FR-01..FR-11, A08, A10).
High-contrast, keyboard-first PySide6 interface for VisionMate Glasses Phase 1.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QFont,
    QImage,
    QKeyEvent,
    QKeySequence,
    QPixmap,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from visionmate import PHASE_LABEL
from visionmate.camera.sources import (
    CameraSource,
    enumerate_cameras,
    load_image_unicode,
)
from visionmate.config import AppConfig
from visionmate.contracts import ActiveSource, AppMode, SafetyStatus, SessionJobResult, SessionState
from visionmate.controller import HELP_MESSAGE_VI, MSG_RESULTS_CLEARED, SessionController
from visionmate.ocr.base import OCRBackend
from visionmate.ocr.paddle_backend import PaddleONNXOCRBackend
from visionmate.ocr.tesseract_backend import TesseractOCRBackend


STATE_LABELS_VI = {
    SessionState.IDLE: "CHỜ (IDLE)",
    SessionState.CAMERA_READY: "CAMERA SẴN SÀNG",
    SessionState.CAPTURING: "ĐANG CHỤP ẢNH...",
    SessionState.QUALITY_CHECK: "ĐANG KIỂM TRA ĐỘ RÕ...",
    SessionState.RECOGNIZING: "ĐANG NHẬN DẠNG CHỮ (OCR)...",
    SessionState.READY_TO_SPEAK: "CHUẨN BỊ ĐỌC...",
    SessionState.SPEAKING: "ĐANG ĐỌC TIẾNG VIỆT...",
    SessionState.CANCELLED: "ĐÃ DỪNG (CANCELLED)",
    SessionState.ERROR: "CẢNH BÁO / LỖI",
}


class _BridgeSignals(QObject):
    """Thread-safe Qt signal bridge from SessionController worker threads to UI thread."""
    state_changed = Signal(str, int, str)
    job_completed = Signal(object)
    results_cleared = Signal(int, str)


def truncate_display_name(name: str, max_chars: int = 34) -> str:
    """Truncate filename for UI display with ellipsis, preserving extension and key tokens."""
    if not name or len(name) <= max_chars:
        return name
    p = Path(name)
    ext = p.suffix
    stem = p.stem
    if ext and len(ext) <= 7 and len(stem) > 6:
        budget = max_chars - len(ext) - 3  # 3 for "..."
        if budget >= 4:
            pre_len = budget // 2 + budget % 2
            suf_len = budget // 2
            return f"{stem[:pre_len]}...{stem[-suf_len:]}{ext}"
    return f"{name[:max_chars - 3]}..."


class AccessibleMainWindow(QMainWindow):
    """Accessible Windows desktop window for VisionMate Glasses Phase 1.1."""

    def __init__(
        self,
        controller: SessionController,
        config: AppConfig,
        auto_start_camera: bool = True,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.config = config
        self._closing = False
        self._fps_timestamps: Deque[float] = deque(maxlen=30)
        self._current_fps: float = 0.0
        self._active_source: ActiveSource = ActiveSource.CAMERA
        self._selected_image_frame: Optional[np.ndarray] = None
        self._selected_image_name: str = ""
        self._selected_image_path: str = ""
        self._frozen_detection_frame: Optional[np.ndarray] = None

        self._signals = _BridgeSignals(self)
        self._signals.state_changed.connect(self._on_ui_state_changed)
        self._signals.job_completed.connect(self._on_ui_job_completed)
        self._signals.results_cleared.connect(self._on_ui_results_cleared)

        self.controller._on_state_changed = self._marshal_state_changed
        self.controller._on_job_completed = self._marshal_job_completed
        self.controller._on_results_cleared = self._marshal_results_cleared

        self.setWindowTitle(f"VisionMate Glasses — {PHASE_LABEL} (Đọc chữ & Nhận diện đồ vật Ngoại tuyến)")
        self.resize(1160, 780)
        self._apply_high_contrast_stylesheet()
        self._build_ui()
        self._install_keyboard_filter()

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)  # ~30 FPS target
        self._preview_timer.timeout.connect(self._update_preview_tick)
        self._preview_timer.start()

        if auto_start_camera:
            QTimer.singleShot(50, self._initialize_camera)

    def _apply_high_contrast_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #121417;
                color: #F5F7FA;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 15px;
            }
            QGroupBox {
                border: 2px solid #3A4149;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #90CAF9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #1E293B;
                color: #FFFFFF;
                border: 2px solid #64748B;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #94A3B8;
            }
            QPushButton#PrimaryReadButton {
                background-color: #1565C0;
                border: 2px solid #64B5F6;
                font-size: 18px;
            }
            QPushButton#StopButton {
                background-color: #B71C1C;
                border: 2px solid #EF9A9A;
            }
            QPushButton#ClearButton {
                background-color: #4A148C;
                border: 2px solid #CE93D8;
            }
            QPushButton:focus, QComboBox:focus, QSlider:focus, QPlainTextEdit:focus, QCheckBox:focus {
                border: 3px solid #FFD600;
                outline: none;
            }
            QPlainTextEdit {
                background-color: #0B0D10;
                color: #FFFDE7;
                border: 2px solid #455A64;
                border-radius: 6px;
                font-size: 20px;
                padding: 8px;
            }
            QComboBox {
                background-color: #1E293B;
                color: #FFFFFF;
                border: 2px solid #64748B;
                border-radius: 5px;
                padding: 5px 10px;
            }
            """
        )

    def _get_image_tooltip(self) -> str:
        """Return full file path and name for accessible mouse hover tooltip."""
        if not self._selected_image_name:
            return ""
        if self._selected_image_path and self._selected_image_path != self._selected_image_name:
            return f"Tên tệp: {self._selected_image_name}\nĐường dẫn: {self._selected_image_path}"
        return f"Tên tệp: {self._selected_image_name}"

    def _update_preview_title_and_tooltips(self) -> None:
        """Update group title, status label, and tooltips with truncated name and full metadata."""
        if self._active_source == ActiveSource.IMAGE and self._selected_image_name:
            short_title = truncate_display_name(self._selected_image_name, 34)
            short_preview = truncate_display_name(self._selected_image_name, 28)
            self.preview_group.setTitle(f"Khung hình: Ảnh đã chọn [{short_title}] (Chỉ xử lý trên RAM)")
            tooltip = self._get_image_tooltip()
            self.preview_group.setToolTip(tooltip)
            self.lbl_preview.setToolTip(tooltip)
            cam = self.controller.camera
            if isinstance(cam, CameraSource) and cam.is_open():
                self.lbl_fps.setText(f"Preview: Ảnh [{short_preview}] (Camera {self._current_fps:.1f} FPS)")
            else:
                self.lbl_fps.setText(f"Preview: Ảnh [{short_preview}]")
            self.lbl_fps.setToolTip(tooltip)
        else:
            self.preview_group.setTitle("Khung hình Camera trực tiếp (Chỉ xử lý trên RAM)")
            self.preview_group.setToolTip("")
            self.lbl_preview.setToolTip("")
            self.lbl_fps.setToolTip("Tốc độ khung hình camera trực tiếp")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        # Top status bar banner
        status_frame = QFrame(self)
        status_frame.setStyleSheet(
            "background-color: #1E252D; border: 2px solid #455A64; border-radius: 8px; padding: 6px;"
        )
        status_layout = QHBoxLayout(status_frame)

        self.lbl_state_badge = QLabel("TRẠNG THÁI: CHỜ (IDLE)")
        self.lbl_state_badge.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self.lbl_state_badge.setAccessibleName("Trạng thái hệ thống")
        self.lbl_state_badge.setStyleSheet("color: #FFD54F;")

        self.lbl_tts_status = QLabel()
        self.lbl_tts_status.setAccessibleName("Trạng thái giọng đọc tiếng Việt")
        self._refresh_tts_badge()

        self.lbl_fps = QLabel("Preview: -- FPS")
        self.lbl_fps.setAccessibleName("Tốc độ khung hình camera")
        self.lbl_fps.setStyleSheet("color: #A5D6A7; font-weight: bold;")

        status_layout.addWidget(self.lbl_state_badge, stretch=2)
        status_layout.addWidget(self.lbl_tts_status, stretch=2)
        status_layout.addWidget(self.lbl_fps, stretch=1)
        root_layout.addWidget(status_frame)

        # Primary action buttons bar (Mode / Space / R / Esc / Ctrl+L / O / F1)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_mode = QPushButton("Chế độ: ĐỌC VĂN BẢN (M)")
        self.btn_mode.setObjectName("ModeButton")
        self.btn_mode.setAccessibleName("Chuyển đổi giữa chế độ đọc văn bản và nhận diện đồ vật, phím tắt M")
        self.btn_mode.setStyleSheet(
            "background-color: #0D47A1; color: #FFFFFF; font-weight: bold; padding: 10px; border-radius: 6px;"
        )
        self.btn_mode.clicked.connect(self.action_toggle_mode)

        self.btn_read = QPushButton("Chụp && Đọc chữ (Phím Cách)")
        self.btn_read.setObjectName("PrimaryReadButton")
        self.btn_read.setAccessibleName("Chụp và đọc chữ, phím tắt Phím Cách")
        self.btn_read.setAccessibleDescription("Chụp một khung hình từ camera và đọc văn bản tiếng Việt")
        self.btn_read.clicked.connect(self.action_capture_and_read)

        self.btn_repeat = QPushButton("Đọc lại (R)")
        self.btn_repeat.setAccessibleName("Đọc lại kết quả vừa nhận, phím tắt R")
        self.btn_repeat.clicked.connect(self.action_repeat)

        self.btn_stop = QPushButton("Dừng đọc (Esc)")
        self.btn_stop.setObjectName("StopButton")
        self.btn_stop.setAccessibleName("Dừng đọc ngay lập tức, phím tắt Escape")
        self.btn_stop.clicked.connect(self.action_stop)

        self.btn_clear = QPushButton("Xóa kết quả (Ctrl+L)")
        self.btn_clear.setObjectName("ClearButton")
        self.btn_clear.setAccessibleName("Xóa kết quả nhận dạng hiện tại, phím tắt Ctrl+L")
        self.btn_clear.setAccessibleDescription(
            "Xóa toàn bộ kết quả OCR cũ, dừng giọng đọc ngay lập tức và sẵn sàng nhận dạng mới"
        )
        self.btn_clear.clicked.connect(self.action_clear_results)

        self.btn_open_image = QPushButton("Mở ảnh thử nghiệm (O)")
        self.btn_open_image.setAccessibleName("Mở ảnh thử nghiệm từ máy tính, phím tắt O")
        self.btn_open_image.clicked.connect(self.action_open_image)

        self.btn_help = QPushButton("Hướng dẫn (F1)")
        self.btn_help.setAccessibleName("Nghe hướng dẫn phím tắt, phím tắt F1")
        self.btn_help.clicked.connect(self.action_help)

        btn_layout.addWidget(self.btn_mode, stretch=2)
        btn_layout.addWidget(self.btn_read, stretch=2)
        btn_layout.addWidget(self.btn_repeat, stretch=1)
        btn_layout.addWidget(self.btn_stop, stretch=1)
        btn_layout.addWidget(self.btn_clear, stretch=1)
        btn_layout.addWidget(self.btn_open_image, stretch=1)
        btn_layout.addWidget(self.btn_help, stretch=1)
        root_layout.addLayout(btn_layout)

        # Main middle area: Left = Camera preview, Right = Recognized text & warnings
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(12)

        # Left: Camera preview group
        self.preview_group = QGroupBox("Khung hình Camera trực tiếp (Chỉ xử lý trên RAM)")
        preview_layout = QVBoxLayout(self.preview_group)
        self.lbl_preview = QLabel("Đang khởi tạo camera...")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumSize(480, 320)
        self.lbl_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.lbl_preview.setStyleSheet("background-color: #08090C; border: 1px solid #334155; color: #94A3B8;")
        self.lbl_preview.setAccessibleName("Khung hiển thị hình ảnh camera")
        preview_layout.addWidget(self.lbl_preview, stretch=1)

        self.lbl_metrics = QLabel("Thời gian xử lý: Chưa có lượt đọc")
        self.lbl_metrics.setAccessibleName("Thông số thời gian xử lý và chất lượng ảnh")
        self.lbl_metrics.setStyleSheet("color: #90CAF9; font-size: 13px;")
        preview_layout.addWidget(self.lbl_metrics)
        mid_layout.addWidget(self.preview_group, stretch=5)

        # Right: OCR Text & Safety notifications
        text_group = QGroupBox("Nội dung Nhận dạng && Phản hồi Âm thanh")
        text_layout = QVBoxLayout(text_group)

        self.lbl_warning_banner = QLabel("Sẵn sàng hỗ trợ đọc chữ in tiếng Việt trong nhà (Offline).")
        self.lbl_warning_banner.setWordWrap(True)
        self.lbl_warning_banner.setMinimumHeight(54)
        self.lbl_warning_banner.setAccessibleName("Thông báo hướng dẫn và cảnh báo an toàn")
        self.lbl_warning_banner.setStyleSheet(
            "background-color: #1E293B; color: #FFE082; padding: 8px; border-radius: 6px; font-weight: bold;"
        )
        text_layout.addWidget(self.lbl_warning_banner)

        self.txt_output = QPlainTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setPlaceholderText(
            "Văn bản tiếng Việt nhận dạng được sẽ hiển thị nguyên văn tại đây.\n"
            "Nhấn Phím Cách (Space) để chụp và đọc chữ, hoặc nhấn O để mở ảnh thử nghiệm."
        )
        self.txt_output.setAccessibleName("Văn bản nhận dạng được")
        text_layout.addWidget(self.txt_output, stretch=1)
        mid_layout.addWidget(text_group, stretch=5)

        root_layout.addLayout(mid_layout, stretch=1)

        # Bottom settings & developer controls group
        settings_group = QGroupBox("Cài đặt Thiết bị, Giọng đọc && Quyền riêng tư")
        settings_grid = QGridLayout(settings_group)

        # Camera selector
        lbl_cam = QLabel("Thiết bị Camera:")
        self.cmb_camera = QComboBox()
        self.cmb_camera.setAccessibleName("Chọn thiết bị camera")
        self.cmb_camera.addItem(f"Camera {self.config.camera_index} (Mặc định 720p)", userData=self.config.camera_index)
        self.cmb_camera.currentIndexChanged.connect(self._on_camera_selection_changed)

        self.btn_rescan_cam = QPushButton("Kết nối lại Camera")
        self.btn_rescan_cam.setAccessibleName("Quét và kết nối lại camera")
        self.btn_rescan_cam.clicked.connect(self._on_rescan_cameras)

        # OCR engine selector
        lbl_ocr = QLabel("Bộ nhận dạng OCR:")
        self.cmb_ocr = QComboBox()
        self.cmb_ocr.setAccessibleName("Chọn bộ nhận dạng chữ OCR")
        self.cmb_ocr.addItem("Tesseract Fast (vie) — Khuyên dùng (CPU nhanh)", userData="tesseract_fast")
        self.cmb_ocr.addItem("Tesseract Best (vie) — Độ chính xác cao", userData="tesseract_best")
        self.cmb_ocr.addItem("PaddleOCR ONNX (latin/vi) — Đối chiếu thực nghiệm", userData="paddleocr_vi")
        idx_ocr = self.cmb_ocr.findData(self.config.ocr_backend)
        if idx_ocr >= 0:
            self.cmb_ocr.setCurrentIndex(idx_ocr)
        self.cmb_ocr.currentIndexChanged.connect(self._on_ocr_backend_changed)

        # Speech rate slider
        self.lbl_rate = QLabel(f"Tốc độ đọc ({self.config.speech_rate}):")
        self.sld_rate = QSlider(Qt.Orientation.Horizontal)
        self.sld_rate.setRange(-8, 8)
        self.sld_rate.setValue(self.config.speech_rate)
        self.sld_rate.setAccessibleName("Thanh trượt điều chỉnh tốc độ giọng đọc")
        self.sld_rate.valueChanged.connect(self._on_rate_changed)

        # Speech volume slider
        self.lbl_volume = QLabel(f"Âm lượng ({self.config.speech_volume}%):")
        self.sld_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_volume.setRange(10, 100)
        self.sld_volume.setValue(self.config.speech_volume)
        self.sld_volume.setAccessibleName("Thanh trượt điều chỉnh âm lượng giọng đọc")
        self.sld_volume.valueChanged.connect(self._on_volume_changed)

        # Developer opt-in debug log checkbox + purge button (FR-11) + ROI selector (Objective 5)
        self.chk_debug = QCheckBox("Chế độ Debug (Chỉ dành cho Developer — Opt-in)")
        self.chk_debug.setChecked(False)
        self.chk_debug.setAccessibleName("Bật chế độ ghi log debug cho nhà phát triển")
        self.chk_debug.toggled.connect(self._on_debug_toggled)

        self.btn_purge_debug = QPushButton("Xóa log debug")
        self.btn_purge_debug.setAccessibleName("Xóa toàn bộ file log debug đã ghi")
        self.btn_purge_debug.clicked.connect(self._on_purge_debug)

        lbl_roi = QLabel("Vùng đọc tài liệu (ROI):")
        self.cmb_roi = QComboBox()
        self.cmb_roi.setAccessibleName("Chọn vùng đọc tài liệu trên khung hình")
        self.cmb_roi.addItem("Toàn trang (Tự động tách cột & phân vùng)", userData=None)
        self.cmb_roi.addItem("Vùng tiêu đề trên (Top 35%)", userData=(0.0, 0.0, 1.0, 0.35))
        self.cmb_roi.addItem("Vùng nội dung chính (Middle 55%)", userData=(0.0, 0.22, 1.0, 0.77))
        self.cmb_roi.addItem("Vùng chân trang / chữ ký (Bottom 38%)", userData=(0.0, 0.62, 1.0, 1.0))

        settings_grid.addWidget(lbl_cam, 0, 0)
        settings_grid.addWidget(self.cmb_camera, 0, 1)
        settings_grid.addWidget(self.btn_rescan_cam, 0, 2)
        settings_grid.addWidget(lbl_ocr, 0, 3)
        settings_grid.addWidget(self.cmb_ocr, 0, 4)

        settings_grid.addWidget(self.lbl_rate, 1, 0)
        settings_grid.addWidget(self.sld_rate, 1, 1)
        settings_grid.addWidget(self.lbl_volume, 1, 2)
        settings_grid.addWidget(self.sld_volume, 1, 3, 1, 2)

        settings_grid.addWidget(lbl_roi, 2, 0)
        settings_grid.addWidget(self.cmb_roi, 2, 1)
        settings_grid.addWidget(self.chk_debug, 2, 2, 1, 2)
        settings_grid.addWidget(self.btn_purge_debug, 2, 4)

        root_layout.addWidget(settings_group)

        # Set explicit tab order for keyboard navigation (Section 6.2, A10)
        QWidget.setTabOrder(self.btn_read, self.btn_repeat)
        QWidget.setTabOrder(self.btn_repeat, self.btn_stop)
        QWidget.setTabOrder(self.btn_stop, self.btn_clear)
        QWidget.setTabOrder(self.btn_clear, self.btn_open_image)
        QWidget.setTabOrder(self.btn_open_image, self.btn_help)
        QWidget.setTabOrder(self.btn_help, self.txt_output)
        QWidget.setTabOrder(self.txt_output, self.cmb_camera)
        QWidget.setTabOrder(self.cmb_camera, self.btn_rescan_cam)
        QWidget.setTabOrder(self.btn_rescan_cam, self.cmb_ocr)
        QWidget.setTabOrder(self.cmb_ocr, self.cmb_roi)
        QWidget.setTabOrder(self.cmb_roi, self.sld_rate)
        QWidget.setTabOrder(self.sld_rate, self.sld_volume)
        QWidget.setTabOrder(self.sld_volume, self.chk_debug)
        QWidget.setTabOrder(self.chk_debug, self.btn_purge_debug)

        self.btn_read.setFocus()

    def _install_keyboard_filter(self) -> None:
        """Ensure Space, R, Esc, Ctrl+L, O, and F1 work reliably regardless of focused child widget."""
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)
        self._shortcut_clear = QShortcut(QKeySequence("Ctrl+L"), self)
        self._shortcut_clear.activated.connect(self.action_clear_results)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            key = event.key()
            modifiers = event.modifiers()
            if (modifiers & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_L:
                self.action_clear_results()
                return True
            if modifiers == Qt.KeyboardModifier.NoModifier:
                if key == Qt.Key.Key_Space:
                    self.action_capture_and_read()
                    return True
                if key == Qt.Key.Key_M:
                    self.action_toggle_mode()
                    return True
                if key == Qt.Key.Key_D:
                    self.action_detect_objects()
                    return True
                if key == Qt.Key.Key_R:
                    self.action_repeat()
                    return True
                if key == Qt.Key.Key_Escape:
                    self.action_stop()
                    return True
                if key == Qt.Key.Key_O:
                    self.action_open_image()
                    return True
                if key == Qt.Key.Key_F1:
                    self.action_help()
                    return True
        return super().eventFilter(watched, event)

    def _refresh_tts_badge(self) -> None:
        if self.controller.speech_backend.is_vi_vn_available():
            self.lbl_tts_status.setText("TTS Tiếng Việt: SẴN SÀNG (Offline vi-VN)")
            self.lbl_tts_status.setStyleSheet("color: #81C784; font-weight: bold;")
        else:
            self.lbl_tts_status.setText("TTS Tiếng Việt: CHƯA SẴN SÀNG (TTS_BLOCKED)")
            self.lbl_tts_status.setStyleSheet("color: #FF8A65; font-weight: bold;")

    def _populate_camera_list(self) -> None:
        self.cmb_camera.blockSignals(True)
        self.cmb_camera.clear()
        devices = enumerate_cameras(max_index=2)
        if devices:
            for dev in devices:
                self.cmb_camera.addItem(dev.label, userData=dev.index)
        else:
            self.cmb_camera.addItem("Camera 0 (Mặc định)", userData=0)
        self.cmb_camera.blockSignals(False)

    def _initialize_camera(self) -> None:
        if self._closing:
            return
        self.controller.start_camera()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._frozen_detection_frame is not None:
            self._display_bgr_frame(self._frozen_detection_frame)
        elif self._active_source == ActiveSource.IMAGE and self._selected_image_frame is not None:
            self._display_bgr_frame(self._selected_image_frame)

    def _update_preview_tick(self) -> None:
        if self._closing:
            return

        now = time.perf_counter()
        cam = self.controller.camera

        # Frozen Detection Frame: Retain annotated detection boxes until next capture or clear
        if self._frozen_detection_frame is not None:
            self._display_bgr_frame(self._frozen_detection_frame)
            if self._active_source == ActiveSource.IMAGE and self._selected_image_name:
                short_preview = truncate_display_name(self._selected_image_name, 28)
                self.lbl_fps.setText(f"Preview: Ảnh [{short_preview}]")
                self.lbl_fps.setToolTip(self._get_image_tooltip())
            return

        # Persistent Image Preview Mode: If user opened an image via 'O', retain it permanently
        # until the user explicitly initiates a camera capture ('Space') or clears results ('Ctrl+L').
        if self._active_source == ActiveSource.IMAGE and self._selected_image_frame is not None:
            short_preview = truncate_display_name(self._selected_image_name, 28)
            tooltip = self._get_image_tooltip()
            if isinstance(cam, CameraSource) and cam.is_open():
                self.lbl_fps.setText(
                    f"Preview: Ảnh [{short_preview}] (Camera {self._current_fps:.1f} FPS)"
                )
            else:
                self.lbl_fps.setText(f"Preview: Ảnh [{short_preview}]")
            self.lbl_fps.setToolTip(tooltip)
            self._display_bgr_frame(self._selected_image_frame)
            return

        # Camera Preview Mode: Stream live video from camera
        if isinstance(cam, CameraSource) and cam.is_open():
            ok, frame = cam.read_preview_frame()
            if ok and frame is not None:
                self._fps_timestamps.append(now)
                if len(self._fps_timestamps) >= 2:
                    span = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    if span > 0:
                        self._current_fps = (len(self._fps_timestamps) - 1) / span
                        self.lbl_fps.setText(f"Preview: {self._current_fps:.1f} FPS")
                self._display_bgr_frame(frame)
        else:
            self.lbl_fps.setText("Preview: Không có camera (Chế độ ảnh thử nghiệm)")
            self.lbl_fps.setToolTip("")

    def _display_bgr_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        if h <= 0 or w <= 0:
            return
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale into available inner area without pushing layout boundaries
        rect = self.lbl_preview.contentsRect()
        target_w = rect.width()
        target_h = rect.height()
        if target_w <= 10 or target_h <= 10:
            target_w = max(480, self.lbl_preview.width())
            target_h = max(320, self.lbl_preview.height())

        pix = QPixmap.fromImage(qimg).scaled(
            QSize(target_w, target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_preview.setPixmap(pix)

    def _selected_roi_norm(self) -> Optional[Tuple[float, float, float, float]]:
        data = self.cmb_roi.currentData()
        if isinstance(data, tuple) and len(data) == 4:
            return data
        return None

    # --- Actions (Mode / Space / D / R / Esc / Ctrl+L / O / F1) ---
    def action_toggle_mode(self) -> None:
        new_mode = self.controller.toggle_mode()
        self._frozen_detection_frame = None
        self._update_mode_ui(new_mode)
        if new_mode == AppMode.OBJECT_DETECTION:
            announce = "Đã chuyển sang chế độ nhận diện đồ vật ngoại tuyến. Nhấn Phím Cách hoặc D để nhận diện."
        else:
            announce = "Đã chuyển sang chế độ đọc văn bản ngoại tuyến. Nhấn Phím Cách để chụp và đọc chữ."
        self.lbl_warning_banner.setText(announce)
        self.lbl_warning_banner.setStyleSheet(
            "background-color: #004D40; color: #E0F2F1; padding: 8px; border-radius: 6px; font-weight: bold;"
            if new_mode == AppMode.OBJECT_DETECTION
            else "background-color: #0D47A1; color: #E3F2FD; padding: 8px; border-radius: 6px; font-weight: bold;"
        )
        self.controller.speak_announcement(announce)

    def _update_mode_ui(self, mode: AppMode) -> None:
        if mode == AppMode.OBJECT_DETECTION:
            self.btn_mode.setText("Chế độ: ĐỒ VẬT (M)")
            self.btn_mode.setStyleSheet(
                "background-color: #00695C; color: #E0F2F1; font-weight: bold; padding: 10px; border-radius: 6px;"
            )
            self.btn_read.setText("Nhận diện đồ vật (Cách / D)")
            self.txt_output.setPlaceholderText(
                "Danh sách đồ vật nhận diện được sẽ hiển thị tại đây.\n"
                "Nhấn Phím Cách hoặc D để nhận diện đồ vật, hoặc nhấn O để mở ảnh."
            )
        else:
            self.btn_mode.setText("Chế độ: ĐỌC VĂN BẢN (M)")
            self.btn_mode.setStyleSheet(
                "background-color: #0D47A1; color: #FFFFFF; font-weight: bold; padding: 10px; border-radius: 6px;"
            )
            self.btn_read.setText("Chụp && Đọc chữ (Phím Cách)")
            self.txt_output.setPlaceholderText(
                "Văn bản tiếng Việt nhận dạng được sẽ hiển thị nguyên văn tại đây.\n"
                "Nhấn Phím Cách (Space) để chụp và đọc chữ, hoặc nhấn O để mở ảnh thử nghiệm."
            )

    def action_detect_objects(self) -> None:
        self._frozen_detection_frame = None
        self.txt_output.clear()
        if self.controller.mode != AppMode.OBJECT_DETECTION:
            self.controller.set_mode(AppMode.OBJECT_DETECTION)
            self._update_mode_ui(AppMode.OBJECT_DETECTION)
        if self._active_source == ActiveSource.IMAGE and self._selected_image_frame is not None:
            target_file = self._selected_image_path or self._selected_image_name
            self.controller.detect_objects_from_file(target_file, roi_norm=self._selected_roi_norm())
        else:
            self.controller.detect_objects(roi_norm=self._selected_roi_norm())

    def action_capture_and_read(self) -> None:
        self._frozen_detection_frame = None
        self._active_source = ActiveSource.CAMERA
        self._selected_image_frame = None
        self._selected_image_name = ""
        self._selected_image_path = ""
        self._update_preview_title_and_tooltips()
        self.txt_output.clear()
        self.controller.capture_and_read(roi_norm=self._selected_roi_norm())

    def action_repeat(self) -> None:
        self.controller.repeat_last()

    def action_stop(self) -> None:
        self.controller.stop_current()

    def action_clear_results(self) -> None:
        self._frozen_detection_frame = None
        self._active_source = ActiveSource.CAMERA
        self._selected_image_frame = None
        self._selected_image_name = ""
        self._selected_image_path = ""
        self._update_preview_title_and_tooltips()
        self.txt_output.clear()
        self.lbl_metrics.setText("Thời gian xử lý: Đã xóa kết quả trước đó")
        self.lbl_warning_banner.setText(MSG_RESULTS_CLEARED)
        self.lbl_warning_banner.setStyleSheet(
            "background-color: #1E293B; color: #FFE082; padding: 8px; border-radius: 6px; font-weight: bold;"
        )
        self.controller.clear_results()

    def action_help(self) -> None:
        self.lbl_warning_banner.setText(HELP_MESSAGE_VI)
        self.controller.speak_help()

    def action_open_image(self, file_path: Optional[str] = None) -> None:
        self._frozen_detection_frame = None
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Chọn ảnh thử nghiệm (Dev/Test Mode)",
                str(Path.cwd()),
                "Image Files (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
            )
        if not file_path:
            return
        self.txt_output.clear()
        img = load_image_unicode(file_path)
        if img is not None:
            self._active_source = ActiveSource.IMAGE
            self._selected_image_frame = img
            self._selected_image_path = str(Path(file_path).resolve())
            self._selected_image_name = Path(file_path).name
            self._update_preview_title_and_tooltips()
            self._display_bgr_frame(img)
        self.controller.read_image_file(file_path, roi_norm=self._selected_roi_norm())

    # --- Settings callbacks ---
    def _on_camera_selection_changed(self, _idx: int) -> None:
        cam_idx = self.cmb_camera.currentData()
        if cam_idx is None:
            return
        self._active_source = ActiveSource.CAMERA
        self._selected_image_frame = None
        self._selected_image_name = ""
        self._selected_image_path = ""
        self._frozen_detection_frame = None
        self._update_preview_title_and_tooltips()
        self.txt_output.clear()
        self.config.camera_index = int(cam_idx)
        self.config.save()
        new_cam = CameraSource(
            device_index=self.config.camera_index,
            width=self.config.camera_width,
            height=self.config.camera_height,
        )
        self.controller.switch_camera(new_cam)

    def _on_rescan_cameras(self) -> None:
        self._populate_camera_list()
        self._on_camera_selection_changed(self.cmb_camera.currentIndex())

    def _on_ocr_backend_changed(self, _idx: int) -> None:
        backend_id = str(self.cmb_ocr.currentData() or "tesseract_fast")
        self.config.ocr_backend = backend_id
        self.config.save()
        new_ocr: OCRBackend
        if backend_id == "tesseract_best":
            new_ocr = TesseractOCRBackend(variant="best")
        elif backend_id == "paddleocr_vi":
            new_ocr = PaddleONNXOCRBackend()
        else:
            new_ocr = TesseractOCRBackend(variant="fast")
        self.controller.switch_ocr_backend(new_ocr)

    def _on_rate_changed(self, val: int) -> None:
        self.config.speech_rate = int(val)
        self.lbl_rate.setText(f"Tốc độ đọc ({val}):")
        self.controller.speech_backend.set_rate(val)
        self.config.save()

    def _on_volume_changed(self, val: int) -> None:
        self.config.speech_volume = int(val)
        self.lbl_volume.setText(f"Âm lượng ({val}%):")
        self.controller.speech_backend.set_volume(val)
        self.config.save()

    def _on_debug_toggled(self, checked: bool) -> None:
        if checked:
            reply = QMessageBox.warning(
                self,
                "Cảnh báo Quyền riêng tư (Developer Debug Mode)",
                "Khi bật chế độ Debug, kết quả JSON của các lượt nhận dạng trong phiên này "
                "sẽ được ghi vào thư mục debug_captures/.\n\n"
                "Không sử dụng chế độ này với tài liệu cá nhân hoặc nhạy cảm.\n"
                "Bạn có chắc chắn muốn bật trong phiên làm việc hiện tại không?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.chk_debug.blockSignals(True)
                self.chk_debug.setChecked(False)
                self.chk_debug.blockSignals(False)
                self.config.debug_opt_in_runtime = False
                return
            self.config.debug_opt_in_runtime = True
        else:
            self.config.debug_opt_in_runtime = False

    def _on_purge_debug(self) -> None:
        removed = self.controller.purge_debug_logs()
        QMessageBox.information(
            self,
            "Đã xóa dữ liệu Debug",
            f"Đã xóa sạch thư mục debug_captures ({removed} mục).",
        )

    # --- Thread-safe UI slot updates ---
    def _marshal_state_changed(self, state: SessionState, job_id: int, message: str) -> None:
        if not self._closing:
            self._signals.state_changed.emit(state.value, job_id, message)

    def _marshal_job_completed(self, result: SessionJobResult) -> None:
        if not self._closing:
            self._signals.job_completed.emit(result)

    def _marshal_results_cleared(self, session_id: int, message: str) -> None:
        if not self._closing:
            self._signals.results_cleared.emit(session_id, message)

    def _on_ui_results_cleared(self, _session_id: int, message: str) -> None:
        if self._closing:
            return
        self._active_source = ActiveSource.CAMERA
        self._selected_image_frame = None
        self._selected_image_name = ""
        self._selected_image_path = ""
        self._frozen_detection_frame = None
        self._update_preview_title_and_tooltips()
        self.txt_output.clear()
        self.lbl_metrics.setText("Thời gian xử lý: Đã xóa kết quả trước đó")
        self.lbl_warning_banner.setText(message)
        self.lbl_warning_banner.setStyleSheet(
            "background-color: #1E293B; color: #FFE082; padding: 8px; border-radius: 6px; font-weight: bold;"
        )

    def _on_ui_state_changed(self, state_val: str, job_id: int, message: str) -> None:
        if self._closing:
            return
        try:
            st = SessionState(state_val)
        except ValueError:
            st = SessionState.IDLE
        label_vi = STATE_LABELS_VI.get(st, state_val)
        self.lbl_state_badge.setText(f"TRẠNG THÁI: {label_vi} [Job #{job_id}]")
        if message and st not in (SessionState.READY_TO_SPEAK, SessionState.SPEAKING):
            self.lbl_warning_banner.setText(message)

    def _on_ui_job_completed(self, result_obj: object) -> None:
        if self._closing or not isinstance(result_obj, SessionJobResult):
            return
        res: SessionJobResult = result_obj
        if res.session_id and res.session_id != self.controller.session_id:
            return
        if res.job_id != self.controller.current_job_id:
            return

        if res.active_source == ActiveSource.IMAGE and res.selected_image_name:
            self._active_source = ActiveSource.IMAGE
            self._selected_image_name = res.selected_image_name
            self._update_preview_title_and_tooltips()

        if res.display_text:
            self.txt_output.setPlainText(res.display_text)
        elif res.lines:
            display_txt = "\n".join(ln.text for ln in res.lines)
            self.txt_output.setPlainText(display_txt)
        else:
            self.txt_output.setPlainText(res.spoken_text)

        if getattr(res, "mode", "reading") == "object_detection":
            if res.annotated_frame is not None:
                self._frozen_detection_frame = res.annotated_frame
                self._display_bgr_frame(res.annotated_frame)

            self.txt_output.setPlainText(res.display_text or res.spoken_text)

            if res.warnings:
                self.lbl_warning_banner.setText(" | ".join(res.warnings))
                self.lbl_warning_banner.setStyleSheet(
                    "background-color: #3E2723; color: #FFCC80; padding: 8px; border-radius: 6px; font-weight: bold;"
                )
            elif res.detection and res.detection.objects:
                self.lbl_warning_banner.setText(res.spoken_text)
                self.lbl_warning_banner.setStyleSheet(
                    "background-color: #1B5E20; color: #E8F5E9; padding: 8px; border-radius: 6px; font-weight: bold;"
                )
            else:
                self.lbl_warning_banner.setText("Chưa xác định được đồ vật.")
                self.lbl_warning_banner.setStyleSheet(
                    "background-color: #1E293B; color: #FFE082; padding: 8px; border-radius: 6px; font-weight: bold;"
                )

            t = res.timings_ms
            det_engine = res.detection.model_name if res.detection else "YOLOX-Nano"
            self.lbl_metrics.setText(
                f"Job #{res.job_id} ({res.source}, Nhận diện đồ vật) — "
                f"Capture: {t.get('capture', 0)}ms, Detection ({det_engine}): {t.get('detection', 0)}ms, "
                f"TTS Start: {t.get('tts_start', 0)}ms"
            )
            return

        if res.warnings:
            self.lbl_warning_banner.setText(" | ".join(res.warnings))
            self.lbl_warning_banner.setStyleSheet(
                "background-color: #3E2723; color: #FFCC80; padding: 8px; border-radius: 6px; font-weight: bold;"
            )
        elif res.status == SafetyStatus.READABLE:
            self.lbl_warning_banner.setText("Đã nhận dạng rõ văn bản tiếng Việt.")
            self.lbl_warning_banner.setStyleSheet(
                "background-color: #1B5E20; color: #E8F5E9; padding: 8px; border-radius: 6px; font-weight: bold;"
            )
        else:
            self.lbl_warning_banner.setText(res.spoken_text)
            self.lbl_warning_banner.setStyleSheet(
                "background-color: #1E293B; color: #FFE082; padding: 8px; border-radius: 6px; font-weight: bold;"
            )

        t = res.timings_ms
        q_info = ""
        if res.quality is not None:
            q_info = (
                f" | Độ nét: {res.quality.active_blur_score:.1f}, "
                f"Độ sáng: {res.quality.mean_brightness:.1f}"
            )
        self.lbl_metrics.setText(
            f"Job #{res.job_id} ({res.source}, {res.status.value}) — "
            f"Capture: {t.get('capture', 0)}ms, Gate: {t.get('quality', 0)}ms, "
            f"OCR ({res.ocr_engine or 'n/a'}): {t.get('ocr', 0)}ms, "
            f"TTS Start: {t.get('tts_start', 0)}ms{q_info}"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        self._preview_timer.stop()
        self.controller.shutdown()
        super().closeEvent(event)
