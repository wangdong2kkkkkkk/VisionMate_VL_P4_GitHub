# -*- coding: utf-8 -*-
"""
Application entry point for VisionMate Glasses Phase 1.
Wires CameraSource, ImageQualityGate, OCRBackend, SafetyPolicy,
WindowsOneCoreSpeechBackend, SessionController, and AccessibleMainWindow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import QApplication

from visionmate.camera.sources import CameraSource
from visionmate.config import AppConfig
from visionmate.controller import SessionController
from visionmate.ocr.base import OCRBackend
from visionmate.ocr.paddle_backend import PaddleONNXOCRBackend
from visionmate.ocr.tesseract_backend import TesseractOCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import SafetyPolicy
from visionmate.speech.sapi_onecore import WindowsOneCoreSpeechBackend
from visionmate.ui.main_window import AccessibleMainWindow


def create_ocr_backend(backend_name: str) -> OCRBackend:
    if backend_name == "tesseract_best":
        return TesseractOCRBackend(variant="best")
    if backend_name == "paddleocr_vi":
        return PaddleONNXOCRBackend()
    return TesseractOCRBackend(variant="fast")


def build_controller(config: AppConfig, play_audio: bool = True) -> SessionController:
    camera = CameraSource(
        device_index=config.camera_index,
        width=config.camera_width,
        height=config.camera_height,
    )
    quality_gate = ImageQualityGate(thresholds=config.quality)
    ocr_backend = create_ocr_backend(config.ocr_backend)
    safety_policy = SafetyPolicy(min_ocr_raw_score=config.min_ocr_raw_score)
    speech_backend = WindowsOneCoreSpeechBackend(
        rate=config.speech_rate,
        volume=config.speech_volume,
        play_audio=play_audio,
    )
    return SessionController(
        camera=camera,
        quality_gate=quality_gate,
        ocr_backend=ocr_backend,
        safety_policy=safety_policy,
        speech_backend=speech_backend,
        config=config,
    )


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="VisionMate Glasses Phase 1 — Windows Offline Vietnamese Reading Assistant"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional local test image path to open immediately on startup (Dev/Test mode).",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Start in Dev/Test mode without opening the webcam automatically.",
    )
    parser.add_argument(
        "--auto-close-ms",
        type=int,
        default=0,
        help="Optional timer (ms) to close window automatically during automated GUI smoke tests.",
    )
    args = parser.parse_args(argv)

    config = AppConfig.load()
    controller = build_controller(config)

    app = QApplication.instance() or QApplication(sys.argv)
    window = AccessibleMainWindow(
        controller=controller,
        config=config,
        auto_start_camera=(not args.no_camera and not args.image),
    )
    window.show()

    if args.image:
        from PySide6.QtCore import QTimer

        img_path = str(Path(args.image).resolve())
        QTimer.singleShot(100, lambda: window.action_open_image(img_path))

    if args.auto_close_ms and args.auto_close_ms > 0:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(args.auto_close_ms, window.close)

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
