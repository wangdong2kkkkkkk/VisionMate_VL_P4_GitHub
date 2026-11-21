# -*- coding: utf-8 -*-
"""
Local, non-sensitive configuration for VisionMate Glasses Phase 1.
Privacy by default (FR-08, FR-10, FR-11):
- Never persists recognized text, images, or audio.
- Debug logging is strictly opt-in per session and resets to False on startup.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.json"


@dataclass
class QualityThresholds:
    """Calibrated thresholds for ImageQualityGate (FR-03).
    Note: No single heuristic guarantees OCR accuracy; these reject clearly unusable frames.
    """
    min_active_laplacian_var: float = 35.0
    min_global_laplacian_var: float = 12.0
    min_brightness: float = 28.0
    max_brightness: float = 246.0
    max_glare_ratio: float = 0.88
    min_contrast_std: float = 6.5


@dataclass
class AppConfig:
    """Application settings stored locally (non-sensitive only)."""
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    ocr_backend: str = "tesseract_fast"  # "tesseract_fast" | "tesseract_best" | "paddleocr_vi"
    speech_rate: int = 0                 # SAPI rate: -10 to +10
    speech_volume: int = 90              # SAPI volume: 0 to 100
    min_ocr_raw_score: float = 42.0      # Minimum raw score threshold before UNCERTAIN classification
    quality: QualityThresholds = None  # type: ignore[assignment]
    # Object Detection (OD-1) settings
    od_enabled: bool = True
    od_model_path: str = "models/object_detection/yolox_nano.onnx"
    od_conf_threshold: float = 0.40
    od_nms_threshold: float = 0.45
    active_mode: str = "reading"         # "reading" | "object_detection"
    # Runtime-only developer flag; never persisted as True across restarts
    debug_opt_in_runtime: bool = False

    def __post_init__(self) -> None:
        if self.quality is None:
            self.quality = QualityThresholds()
        self.speech_rate = max(-10, min(10, int(self.speech_rate)))
        self.speech_volume = max(0, min(100, int(self.speech_volume)))

    def to_persisted_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Never persist debug_opt_in_runtime as True (FR-08, FR-11 privacy by default)
        data.pop("debug_opt_in_runtime", None)
        return data

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_persisted_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "AppConfig":
        if not path.is_file():
            return cls()
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            q_raw = raw.get("quality", {})
            quality = QualityThresholds(
                min_active_laplacian_var=float(q_raw.get("min_active_laplacian_var", 35.0)),
                min_global_laplacian_var=float(q_raw.get("min_global_laplacian_var", 12.0)),
                min_brightness=float(q_raw.get("min_brightness", 28.0)),
                max_brightness=float(q_raw.get("max_brightness", 246.0)),
                max_glare_ratio=float(q_raw.get("max_glare_ratio", 0.88)),
                min_contrast_std=float(q_raw.get("min_contrast_std", 6.5)),
            )
            return cls(
                camera_index=int(raw.get("camera_index", 0)),
                camera_width=int(raw.get("camera_width", 1280)),
                camera_height=int(raw.get("camera_height", 720)),
                ocr_backend=str(raw.get("ocr_backend", "tesseract_fast")),
                speech_rate=int(raw.get("speech_rate", 0)),
                speech_volume=int(raw.get("speech_volume", 90)),
                min_ocr_raw_score=float(raw.get("min_ocr_raw_score", 42.0)),
                quality=quality,
                od_enabled=bool(raw.get("od_enabled", True)),
                od_model_path=str(raw.get("od_model_path", "models/object_detection/yolox_nano.onnx")),
                od_conf_threshold=float(raw.get("od_conf_threshold", 0.40)),
                od_nms_threshold=float(raw.get("od_nms_threshold", 0.45)),
                active_mode=str(raw.get("active_mode", "reading")),
                debug_opt_in_runtime=False,
            )
        except Exception:
            return cls()
