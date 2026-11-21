# -*- coding: utf-8 -*-
"""
Data contracts, enums, and state machine definitions for VisionMate Glasses Phase 1 & 1.1.
Follows Section 4.1, 4.2, and 4.3 of VisionMate_Glasses_Product_Design_V0_1.md and ANTIGRAVITY_TASK.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SessionState(str, Enum):
    """State machine states for SessionController (Section 4.2)."""
    IDLE = "IDLE"
    CAMERA_READY = "CAMERA_READY"
    CAPTURING = "CAPTURING"
    QUALITY_CHECK = "QUALITY_CHECK"
    RECOGNIZING = "RECOGNIZING"
    READY_TO_SPEAK = "READY_TO_SPEAK"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class SafetyStatus(str, Enum):
    """Safety & readability classification (Section 4.1)."""
    READABLE = "READABLE"
    UNCERTAIN = "UNCERTAIN"
    NO_TEXT = "NO_TEXT"
    BAD_IMAGE = "BAD_IMAGE"
    ENGINE_ERROR = "ENGINE_ERROR"


class QualityIssue(str, Enum):
    """Image quality gate classification."""
    OK = "OK"
    CORRUPT_OR_EMPTY = "CORRUPT_OR_EMPTY"
    TOO_BLURRY = "TOO_BLURRY"
    TOO_DARK = "TOO_DARK"
    TOO_BRIGHT_OR_GLARE = "TOO_BRIGHT_OR_GLARE"
    TEXT_TOO_SMALL = "TEXT_TOO_SMALL"


class ActiveSource(str, Enum):
    """Explicit active input/preview source mode (Phase 1.3 Section A.3)."""
    CAMERA = "CAMERA"
    IMAGE = "IMAGE"


class AppMode(str, Enum):
    """Application operating mode: Document Reading (OCR) vs Object Detection (OD-1)."""
    READING = "reading"
    OBJECT_DETECTION = "object_detection"


@dataclass(frozen=True)
class QualityCheckResult:
    """Result from ImageQualityGate evaluation."""
    passed: bool
    issue: QualityIssue
    blur_score: float
    active_blur_score: float
    mean_brightness: float
    glare_ratio: float
    contrast_std: float
    elapsed_ms: float
    reason_vi: str = ""
    median_char_height_px: float = 0.0
    reflective_band_detected: bool = False
    edge_clipped_text: bool = False
    guidance_hints_vi: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OCRLine:
    """Single recognized line of text with optional bounding box, ROI order, and raw score.
    Note: raw_score is an uncalibrated engine score, NOT a probability of correctness.
    """
    text: str
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    raw_score: Optional[float] = None
    order_index: Optional[int] = None
    column_role: str = "full"

    def to_contract_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "raw_score": round(self.raw_score, 4) if self.raw_score is not None else None,
        }


@dataclass(frozen=True)
class OCRResult:
    """Output from an OCRBackend."""
    lines: List[OCRLine]
    engine_name: str
    elapsed_ms: float
    error: Optional[str] = None
    low_resolution_detected: bool = False
    median_char_height_px: float = 0.0
    p25_char_height_px: float = 0.0
    p75_char_height_px: float = 0.0
    polarity: str = "dark_on_light"
    roi_count: int = 0
    rejected_line_count: int = 0
    total_candidate_lines: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    language: str = "vi"


@dataclass(frozen=True)
class PolicyDecision:
    """Output from SafetyPolicy & MessagePolicy.
    Explicitly separates:
    1. image_quality_status ("OK" | "LOW_RESOLUTION" | QualityIssue value)
    2. ocr_confidence_status ("HIGH" | "MEDIUM" | "LOW" | "NONE")
    3. speech_policy ("ALLOW_FULL_SPEECH" | "SPEAK_WARNING_ONLY" | "SPEAK_NO_TEXT" | "SPEAK_ERROR")
    """
    status: SafetyStatus
    ordered_lines: List[OCRLine]
    spoken_text: str
    display_text: str
    warnings: List[str] = field(default_factory=list)
    contains_sensitive_content: bool = False
    image_quality_status: str = "OK"
    ocr_confidence_status: str = "HIGH"
    speech_policy: str = "ALLOW_FULL_SPEECH"
    language: str = "vi"


@dataclass(frozen=True)
class DetectedObject:
    """Single detected object instance."""
    class_id: int
    class_name_en: str
    class_name_vi: str
    confidence: float
    box_xywh: Tuple[int, int, int, int]  # (x, y, w, h) in pixel coordinates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name_en": self.class_name_en,
            "class_name_vi": self.class_name_vi,
            "confidence": round(float(self.confidence), 4),
            "box_xywh": list(self.box_xywh),
        }


@dataclass
class ObjectDetectionResult:
    """Output from ObjectDetectionBackend."""
    objects: List[DetectedObject] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: Optional[str] = None
    model_name: str = "yolox_nano"
    spoken_text: str = ""
    display_text: str = ""
    confidence_status: str = "HIGH"  # "HIGH" | "LOW" | "NONE"
    annotated_frame: Optional[Any] = None  # numpy ndarray with drawn boxes (RAM only)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "count": len(self.objects),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
            "model_name": self.model_name,
            "spoken_text": self.spoken_text,
            "display_text": self.display_text,
            "confidence_status": self.confidence_status,
        }


@dataclass
class SessionJobResult:
    """Minimum data contract for a reading or object detection job (Section 4.3).
    By default, lines, detections, and spoken_text exist in RAM only.
    """
    job_id: int
    source: str  # "camera" | "file"
    status: SafetyStatus
    language: str = "vi"
    mode: str = "reading"  # "reading" | "object_detection"
    lines: List[OCRLine] = field(default_factory=list)
    detection: Optional[ObjectDetectionResult] = None
    annotated_frame: Optional[Any] = None
    spoken_text: str = ""
    display_text: str = ""
    warnings: List[str] = field(default_factory=list)
    timings_ms: Dict[str, int] = field(
        default_factory=lambda: {"capture": 0, "quality": 0, "ocr": 0, "tts_start": 0}
    )
    quality: Optional[QualityCheckResult] = None
    ocr_engine: str = ""
    tts_blocked: bool = False
    session_id: int = 0
    image_quality_status: str = "OK"
    ocr_confidence_status: str = "HIGH"
    speech_policy: str = "ALLOW_FULL_SPEECH"
    polarity: str = "dark_on_light"
    active_source: str = "CAMERA"
    selected_image_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "source": self.source,
            "active_source": self.active_source,
            "selected_image_name": self.selected_image_name,
            "mode": self.mode,
            "status": self.status.value,
            "image_quality_status": self.image_quality_status,
            "ocr_confidence_status": self.ocr_confidence_status,
            "speech_policy": self.speech_policy,
            "polarity": self.polarity,
            "language": self.language,
            "lines": [line.to_contract_dict() for line in self.lines],
            "detection": self.detection.to_dict() if self.detection is not None else None,
            "spoken_text": self.spoken_text,
            "warnings": list(self.warnings),
            "timings_ms": {
                "capture": int(self.timings_ms.get("capture", 0)),
                "quality": int(self.timings_ms.get("quality", 0)),
                "ocr": int(self.timings_ms.get("ocr", 0)),
                "detection": int(self.timings_ms.get("detection", 0)),
                "tts_start": int(self.timings_ms.get("tts_start", 0)),
            },
        }
