# -*- coding: utf-8 -*-
"""
SessionController (Section 4.1, 4.2, 4.3, FR-02, FR-07, FR-08, A09).
Implements the state machine, monotonically increasing job_id, cancellation tokens,
stale-callback suppression, and safe shutdown.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from visionmate.camera.sources import FrameSource, ImageFileSource
from visionmate.config import AppConfig
from visionmate.contracts import (
    ActiveSource,
    AppMode,
    DetectedObject,
    ObjectDetectionResult,
    PolicyDecision,
    QualityCheckResult,
    QualityIssue,
    SafetyStatus,
    SessionJobResult,
    SessionState,
)
from visionmate.detection.detector import ObjectDetector, YOLOXObjectDetector
from visionmate.ocr.base import OCRBackend
from visionmate.quality.gate import ImageQualityGate
from visionmate.reading.policy import SafetyPolicy
from visionmate.speech.base import SpeechBackend


HELP_MESSAGE_VI = (
    "Hướng dẫn phím tắt VisionMate: "
    "Nhấn phím Cách để chụp và đọc chữ. "
    "Nhấn phím R để đọc lại kết quả vừa nhận. "
    "Nhấn phím Escape để dừng đọc ngay lập tức. "
    "Nhấn tổ hợp Ctrl cộng L để xóa kết quả hiện tại. "
    "Nhấn phím O để mở ảnh thử nghiệm. "
    "Nhấn phím F1 để nghe lại hướng dẫn."
)

MSG_CAMERA_MISSING = (
    "Không mở được camera. Vui lòng kiểm tra kết nối webcam hoặc nhấn phím O để mở ảnh thử nghiệm."
)
MSG_NO_REPEAT = "Chưa có nội dung nào được đọc trước đó để phát lại."
MSG_TTS_BLOCKED = "TTS tiếng Việt chưa sẵn sàng (TTS_BLOCKED). Chỉ hiển thị văn bản trên màn hình."
MSG_RESULTS_CLEARED = "Đã xóa kết quả nhận dạng trước đó. Sẵn sàng nhận dạng mới."


class SessionController:
    """Thread-safe controller coordinating Camera, QualityGate, OCR, Policy, and Speech."""

    def __init__(
        self,
        camera: FrameSource,
        quality_gate: ImageQualityGate,
        ocr_backend: OCRBackend,
        safety_policy: SafetyPolicy,
        speech_backend: SpeechBackend,
        config: Optional[AppConfig] = None,
        object_detector: Optional[ObjectDetector] = None,
        on_state_changed: Optional[Callable[[SessionState, int, str], None]] = None,
        on_job_completed: Optional[Callable[[SessionJobResult], None]] = None,
        on_results_cleared: Optional[Callable[[int, str], None]] = None,
        debug_dir: Optional[Path] = None,
    ) -> None:
        self.camera = camera
        self.quality_gate = quality_gate
        self.ocr_backend = ocr_backend
        self.safety_policy = safety_policy
        self.speech_backend = speech_backend
        self.config = config or AppConfig()
        self.object_detector = object_detector
        if self.object_detector is None and getattr(self.config, "od_enabled", True):
            try:
                self.object_detector = YOLOXObjectDetector(
                    model_path=getattr(self.config, "od_model_path", None),
                    conf_threshold=getattr(self.config, "od_conf_threshold", 0.40),
                    nms_threshold=getattr(self.config, "od_nms_threshold", 0.45),
                )
            except Exception:
                self.object_detector = None

        self._mode: AppMode = (
            AppMode.OBJECT_DETECTION
            if getattr(self.config, "active_mode", "reading") == "object_detection"
            else AppMode.READING
        )

        self._on_state_changed = on_state_changed
        self._on_job_completed = on_job_completed
        self._on_results_cleared = on_results_cleared
        self._debug_dir = debug_dir or (
            Path(__file__).resolve().parent.parent.parent / "debug_captures"
        )

        self._lock = threading.Lock()
        self._state: SessionState = SessionState.IDLE
        self._current_job_id: int = 0
        self._session_id: int = 0
        self._current_cancel_token: Optional[threading.Event] = None
        self._shutting_down: bool = False
        self._active_threads: List[threading.Thread] = []
        self._last_confirmed_result: Optional[SessionJobResult] = None
        self._active_source: ActiveSource = ActiveSource.CAMERA
        self._selected_image_name: str = ""

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def current_job_id(self) -> int:
        with self._lock:
            return self._current_job_id

    @property
    def session_id(self) -> int:
        with self._lock:
            return self._session_id

    @property
    def last_confirmed_result(self) -> Optional[SessionJobResult]:
        with self._lock:
            return self._last_confirmed_result

    @property
    def active_source(self) -> ActiveSource:
        with self._lock:
            return self._active_source

    @property
    def selected_image_name(self) -> str:
        with self._lock:
            return self._selected_image_name

    @property
    def mode(self) -> AppMode:
        with self._lock:
            return self._mode

    def set_mode(self, new_mode: AppMode) -> None:
        """Switch operating mode (Reading OCR vs Object Detection)."""
        with self._lock:
            if self._mode == new_mode:
                return
            self._mode = new_mode
            self.config.active_mode = new_mode.value
        self.stop_current()

    def toggle_mode(self) -> AppMode:
        """Toggle between Reading OCR and Object Detection modes."""
        with self._lock:
            if self._mode == AppMode.READING:
                self._mode = AppMode.OBJECT_DETECTION
            else:
                self._mode = AppMode.READING
            self.config.active_mode = self._mode.value
            cur_mode = self._mode
        self.stop_current()
        return cur_mode

    def start_camera(self) -> bool:
        """Open the configured camera and transition to CAMERA_READY or ERROR."""
        with self._lock:
            if self._shutting_down:
                return False
        ok = self.camera.open()
        with self._lock:
            if self._shutting_down:
                return False
            job_id = self._current_job_id
            if ok:
                self._state = SessionState.CAMERA_READY
                msg = "Camera đã sẵn sàng. Nhấn phím Cách để chụp và đọc chữ."
            else:
                self._state = SessionState.ERROR
                msg = MSG_CAMERA_MISSING
        self._notify_state(self.state, job_id, msg)
        return ok

    def switch_camera(self, new_camera: FrameSource) -> bool:
        """Switch to a different camera source safely."""
        self.stop_current()
        try:
            self.camera.close()
        except Exception:
            pass
        self.camera = new_camera
        return self.start_camera()

    def switch_ocr_backend(self, new_ocr: OCRBackend) -> None:
        """Switch OCR backend safely."""
        with self._lock:
            self.ocr_backend = new_ocr

    def _begin_new_job(self) -> tuple[int, int, threading.Event]:
        """Increment session_id and job_id, cancel any running job, and stop active speech immediately."""
        with self._lock:
            if self._current_cancel_token is not None:
                self._current_cancel_token.set()
            self._session_id += 1
            self._current_job_id += 1
            session_id = self._session_id
            job_id = self._current_job_id
            token = threading.Event()
            self._current_cancel_token = token
        self.speech_backend.stop()
        return session_id, job_id, token

    def _is_job_valid(self, session_id: int, job_id: int, cancel_token: threading.Event) -> bool:
        if cancel_token.is_set():
            return False
        with self._lock:
            if (
                self._shutting_down
                or self._session_id != session_id
                or self._current_job_id != job_id
            ):
                return False
        return True

    def _set_state_if_valid(
        self,
        session_id: int,
        job_id: int,
        cancel_token: threading.Event,
        new_state: SessionState,
        message: str = "",
    ) -> bool:
        with self._lock:
            if (
                self._shutting_down
                or cancel_token.is_set()
                or self._session_id != session_id
                or self._current_job_id != job_id
            ):
                return False
            self._state = new_state
        self._notify_state(new_state, job_id, message)
        return True

    def _notify_state(self, state: SessionState, job_id: int, message: str) -> None:
        with self._lock:
            if self._shutting_down:
                return
            cb = self._on_state_changed
        if cb is not None:
            try:
                cb(state, job_id, message)
            except Exception:
                pass

    def _notify_result(
        self,
        session_id: int,
        job_id: int,
        cancel_token: threading.Event,
        result: SessionJobResult,
    ) -> None:
        with self._lock:
            if (
                self._shutting_down
                or cancel_token.is_set()
                or self._session_id != session_id
                or self._current_job_id != job_id
            ):
                return
            self._last_confirmed_result = result
            cb = self._on_job_completed
        if cb is not None:
            try:
                cb(result)
            except Exception:
                pass

    def capture_and_read(self, roi_norm: Optional[Tuple[float, float, float, float]] = None) -> int:
        """Start an asynchronous single-frame capture & process job from the active camera (Space).
        Routes to Object Detection if in OBJECT_DETECTION mode, otherwise OCR reading.
        """
        with self._lock:
            cur_mode = self._mode
        if cur_mode == AppMode.OBJECT_DETECTION:
            return self.detect_objects(roi_norm=roi_norm)

        session_id, job_id, token = self._begin_new_job()
        with self._lock:
            self._active_source = ActiveSource.CAMERA
            self._selected_image_name = ""
        self._spawn_worker(
            target=self._run_pipeline_worker,
            args=(session_id, job_id, token, self.camera, "camera", False, roi_norm),
            name=f"VisionMate-Job-{job_id}",
        )
        return job_id

    def read_image_file(
        self,
        file_path: str | Path,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> int:
        """Start an asynchronous read or object detection job from a local test image file (Key O, Scenario U2)."""
        with self._lock:
            cur_mode = self._mode
        if cur_mode == AppMode.OBJECT_DETECTION:
            return self.detect_objects_from_file(file_path, roi_norm=roi_norm)

        session_id, job_id, token = self._begin_new_job()
        file_name = Path(file_path).name
        with self._lock:
            self._active_source = ActiveSource.IMAGE
            self._selected_image_name = file_name
        source = ImageFileSource(file_path)
        self._spawn_worker(
            target=self._run_pipeline_worker,
            args=(session_id, job_id, token, source, "file", True, roi_norm),
            name=f"VisionMate-FileJob-{job_id}",
        )
        return job_id

    def detect_objects(self, roi_norm: Optional[Tuple[float, float, float, float]] = None) -> int:
        """Start an asynchronous object detection job from the active camera."""
        session_id, job_id, token = self._begin_new_job()
        with self._lock:
            self._active_source = ActiveSource.CAMERA
            self._selected_image_name = ""
        self._spawn_worker(
            target=self._run_detection_worker,
            args=(session_id, job_id, token, self.camera, "camera", False, roi_norm),
            name=f"VisionMate-DetectJob-{job_id}",
        )
        return job_id

    def detect_objects_from_file(
        self,
        file_path: str | Path,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> int:
        """Start an asynchronous object detection job from a local image file."""
        session_id, job_id, token = self._begin_new_job()
        file_name = Path(file_path).name
        with self._lock:
            self._active_source = ActiveSource.IMAGE
            self._selected_image_name = file_name
        source = ImageFileSource(file_path)
        self._spawn_worker(
            target=self._run_detection_worker,
            args=(session_id, job_id, token, source, "file", True, roi_norm),
            name=f"VisionMate-DetectFileJob-{job_id}",
        )
        return job_id

    def repeat_last(self) -> int:
        """Repeat the last confirmed spoken message (Key R)."""
        with self._lock:
            last_res = self._last_confirmed_result
        session_id, job_id, token = self._begin_new_job()
        text_to_speak = last_res.spoken_text if (last_res and last_res.spoken_text) else MSG_NO_REPEAT
        self._spawn_worker(
            target=self._run_direct_speech_worker,
            args=(session_id, job_id, token, text_to_speak, "Đang đọc lại nội dung trước đó..."),
            name=f"VisionMate-Repeat-{job_id}",
        )
        return job_id

    def speak_help(self) -> int:
        """Speak the keyboard shortcut guide (Key F1)."""
        session_id, job_id, token = self._begin_new_job()
        self._spawn_worker(
            target=self._run_direct_speech_worker,
            args=(session_id, job_id, token, HELP_MESSAGE_VI, HELP_MESSAGE_VI),
            name=f"VisionMate-Help-{job_id}",
        )
        return job_id

    def speak_announcement(self, text: str, status_msg: str = "") -> int:
        """Speak a direct system announcement safely via speech backend."""
        session_id, job_id, token = self._begin_new_job()
        msg = status_msg or text
        self._spawn_worker(
            target=self._run_direct_speech_worker,
            args=(session_id, job_id, token, text, msg),
            name=f"VisionMate-Announce-{job_id}",
        )
        return job_id

    def stop_current(self) -> None:
        """Cancel any active OCR/TTS job immediately (Key Esc, Section 4.2)."""
        with self._lock:
            if self._current_cancel_token is not None:
                self._current_cancel_token.set()
            self._session_id += 1
            self._current_job_id += 1
            job_id = self._current_job_id
            if not self._shutting_down:
                self._state = SessionState.CANCELLED
        self.speech_backend.stop()
        self._notify_state(SessionState.CANCELLED, job_id, "Đã dừng tác vụ hiện tại.")
        self._restore_ready_state(job_id)

    def clear_results(self) -> int:
        """
        Clear recognized text, spoken content, and last confirmed result; cancel active jobs
        and stop speech immediately without closing the live webcam preview (Ctrl+L, Phase 1.1).
        """
        with self._lock:
            if self._current_cancel_token is not None:
                self._current_cancel_token.set()
            self._session_id += 1
            self._current_job_id += 1
            session_id = self._session_id
            job_id = self._current_job_id
            self._last_confirmed_result = None
            self._active_source = ActiveSource.CAMERA
            self._selected_image_name = ""
            next_state = (
                SessionState.CAMERA_READY
                if (self.camera is not None and self.camera.is_open())
                else SessionState.IDLE
            )
            if not self._shutting_down:
                self._state = next_state
            cb_clear = self._on_results_cleared

        self.speech_backend.stop()
        self._notify_state(next_state, job_id, MSG_RESULTS_CLEARED)
        if cb_clear is not None:
            try:
                cb_clear(session_id, MSG_RESULTS_CLEARED)
            except Exception:
                pass
        return session_id

    def _restore_ready_state(self, job_id: int) -> None:
        with self._lock:
            if self._shutting_down or self._current_job_id != job_id:
                return
            next_state = (
                SessionState.CAMERA_READY
                if (self.camera is not None and self.camera.is_open())
                else SessionState.IDLE
            )
            self._state = next_state
        self._notify_state(next_state, job_id, "")

    def _spawn_worker(self, target: Callable, args: tuple, name: str) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._active_threads = [t for t in self._active_threads if t.is_alive()]
            worker = threading.Thread(target=target, args=args, name=name, daemon=True)
            self._active_threads.append(worker)
        worker.start()

    def _run_direct_speech_worker(
        self,
        session_id: int,
        job_id: int,
        cancel_token: threading.Event,
        text: str,
        status_msg: str,
    ) -> None:
        if not self._set_state_if_valid(
            session_id, job_id, cancel_token, SessionState.READY_TO_SPEAK, status_msg
        ):
            return
        if not self.speech_backend.is_vi_vn_available():
            self._set_state_if_valid(
                session_id, job_id, cancel_token, SessionState.ERROR, MSG_TTS_BLOCKED
            )
            return
        if not self._set_state_if_valid(
            session_id, job_id, cancel_token, SessionState.SPEAKING, status_msg
        ):
            return
        self.speech_backend.speak(text=text, job_id=job_id, cancel_token=cancel_token)
        if self._is_job_valid(session_id, job_id, cancel_token):
            self._restore_ready_state(job_id)

    @staticmethod
    def _apply_roi_crop(
        frame: np.ndarray,
        roi_norm: Optional[Tuple[float, float, float, float]],
    ) -> np.ndarray:
        if roi_norm is None or frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        rx0, ry0, rx1, ry1 = roi_norm
        x0 = max(0, min(w - 1, int(round(rx0 * w))))
        y0 = max(0, min(h - 1, int(round(ry0 * h))))
        x1 = max(x0 + 1, min(w, int(round(rx1 * w))))
        y1 = max(y0 + 1, min(h, int(round(ry1 * h))))
        return frame[y0:y1, x0:x1].copy()

    def _run_pipeline_worker(
        self,
        session_id: int,
        job_id: int,
        cancel_token: threading.Event,
        source: FrameSource,
        source_label: str,
        close_source_after: bool,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        timings = {"capture": 0, "quality": 0, "ocr": 0, "tts_start": 0}
        t_request_start = time.perf_counter()

        try:
            # Step 1: CAPTURING
            if not self._set_state_if_valid(
                session_id, job_id, cancel_token, SessionState.CAPTURING, "Đang lấy khung hình..."
            ):
                return

            t_cap0 = time.perf_counter()
            if not source.is_open():
                source.open()
            ok, frame = source.read_frame()
            if ok and frame is not None and roi_norm is not None:
                frame = self._apply_roi_crop(frame, roi_norm)
            timings["capture"] = int(round((time.perf_counter() - t_cap0) * 1000.0))

            if not self._is_job_valid(session_id, job_id, cancel_token):
                return

            active_src = "IMAGE" if source_label == "file" else "CAMERA"
            img_name = Path(source.file_path).name if isinstance(source, ImageFileSource) else ""

            if not ok or frame is None:
                err_msg = (
                    MSG_CAMERA_MISSING
                    if source_label == "camera"
                    else "Không thể đọc file ảnh thử nghiệm (file lỗi hoặc không tồn tại)."
                )
                res = SessionJobResult(
                    job_id=job_id,
                    source=source_label,
                    active_source=active_src,
                    selected_image_name=img_name,
                    status=SafetyStatus.BAD_IMAGE if source_label == "file" else SafetyStatus.ENGINE_ERROR,
                    spoken_text=err_msg,
                    display_text=err_msg,
                    warnings=[err_msg],
                    timings_ms=timings,
                    tts_blocked=not self.speech_backend.is_vi_vn_available(),
                    session_id=session_id,
                )
                self._finish_job_with_speech(session_id, job_id, cancel_token, res, t_request_start)
                return

            # Step 2: QUALITY_CHECK
            if not self._set_state_if_valid(
                session_id,
                job_id,
                cancel_token,
                SessionState.QUALITY_CHECK,
                "Đang kiểm tra chất lượng ảnh...",
            ):
                return

            quality_res: QualityCheckResult = self.quality_gate.evaluate(frame)
            timings["quality"] = int(round(quality_res.elapsed_ms))

            if not self._is_job_valid(session_id, job_id, cancel_token):
                return

            if not quality_res.passed:
                decision: PolicyDecision = self.safety_policy.evaluate(quality_res, None)
                res = SessionJobResult(
                    job_id=job_id,
                    source=source_label,
                    active_source=active_src,
                    selected_image_name=img_name,
                    status=decision.status,
                    lines=[],
                    spoken_text=decision.spoken_text,
                    display_text=decision.display_text or decision.spoken_text,
                    warnings=decision.warnings,
                    timings_ms=timings,
                    quality=quality_res,
                    tts_blocked=not self.speech_backend.is_vi_vn_available(),
                    session_id=session_id,
                    image_quality_status=decision.image_quality_status,
                    ocr_confidence_status=decision.ocr_confidence_status,
                    speech_policy=decision.speech_policy,
                )
                self._finish_job_with_speech(session_id, job_id, cancel_token, res, t_request_start)
                return

            # Step 3: RECOGNIZING
            if not self._set_state_if_valid(
                session_id,
                job_id,
                cancel_token,
                SessionState.RECOGNIZING,
                "Đang nhận dạng chữ tiếng Việt...",
            ):
                return

            with self._lock:
                ocr_engine = self.ocr_backend

            ocr_res = ocr_engine.recognize(frame, cancel_token=cancel_token)
            timings["ocr"] = int(round(ocr_res.elapsed_ms))

            if not self._is_job_valid(session_id, job_id, cancel_token):
                return

            decision = self.safety_policy.evaluate(quality_res, ocr_res)
            target_lang = getattr(decision, "language", getattr(ocr_res, "language", "vi"))
            is_tts_ready = (
                self.speech_backend.is_en_us_available()
                if target_lang == "en" and hasattr(self.speech_backend, "is_en_us_available")
                else self.speech_backend.is_vi_vn_available()
            )
            tts_blocked = not is_tts_ready
            warnings = list(decision.warnings)
            if tts_blocked:
                warnings.append(MSG_TTS_BLOCKED)

            res = SessionJobResult(
                job_id=job_id,
                source=source_label,
                active_source=active_src,
                selected_image_name=img_name,
                status=decision.status,
                language=target_lang,
                lines=decision.ordered_lines,
                spoken_text=decision.spoken_text,
                display_text=decision.display_text,
                warnings=warnings,
                timings_ms=timings,
                quality=quality_res,
                ocr_engine=ocr_res.engine_name,
                tts_blocked=tts_blocked,
                session_id=session_id,
                image_quality_status=decision.image_quality_status,
                ocr_confidence_status=decision.ocr_confidence_status,
                speech_policy=decision.speech_policy,
                polarity=ocr_res.polarity,
            )

            self._finish_job_with_speech(session_id, job_id, cancel_token, res, t_request_start)
        finally:
            if close_source_after:
                try:
                    source.close()
                except Exception:
                    pass

    def _run_detection_worker(
        self,
        session_id: int,
        job_id: int,
        cancel_token: threading.Event,
        source: FrameSource,
        source_label: str,
        close_source_after: bool,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        timings = {"capture": 0, "quality": 0, "detection": 0, "tts_start": 0}
        t_request_start = time.perf_counter()

        try:
            # Step 1: CAPTURING
            if not self._set_state_if_valid(
                session_id, job_id, cancel_token, SessionState.CAPTURING, "Đang lấy khung hình nhận diện..."
            ):
                return

            t_cap0 = time.perf_counter()
            if not source.is_open():
                source.open()
            ok, frame = source.read_frame()
            if ok and frame is not None and roi_norm is not None:
                frame = self._apply_roi_crop(frame, roi_norm)
            timings["capture"] = int(round((time.perf_counter() - t_cap0) * 1000.0))

            if not self._is_job_valid(session_id, job_id, cancel_token):
                return

            active_src = "IMAGE" if source_label == "file" else "CAMERA"
            img_name = Path(source.file_path).name if isinstance(source, ImageFileSource) else ""

            if not ok or frame is None:
                err_msg = (
                    MSG_CAMERA_MISSING
                    if source_label == "camera"
                    else "Không thể đọc file ảnh (file lỗi hoặc không tồn tại)."
                )
                res = SessionJobResult(
                    job_id=job_id,
                    source=source_label,
                    active_source=active_src,
                    selected_image_name=img_name,
                    mode="object_detection",
                    status=SafetyStatus.BAD_IMAGE if source_label == "file" else SafetyStatus.ENGINE_ERROR,
                    spoken_text="Chưa xác định được đồ vật.",
                    display_text=err_msg,
                    warnings=[err_msg],
                    timings_ms=timings,
                    tts_blocked=not self.speech_backend.is_vi_vn_available(),
                    session_id=session_id,
                )
                self._finish_job_with_speech(session_id, job_id, cancel_token, res, t_request_start)
                return

            # Step 2: QUALITY_CHECK
            if not self._set_state_if_valid(
                session_id, job_id, cancel_token, SessionState.QUALITY_CHECK, "Đang kiểm tra khung hình..."
            ):
                return
            t_q0 = time.perf_counter()
            quality_res = self.quality_gate.evaluate(frame)
            timings["quality"] = int(round((time.perf_counter() - t_q0) * 1000.0))

            if not self._is_job_valid(session_id, job_id, cancel_token):
                return

            # Extreme underexposure safety check
            if quality_res.issue == QualityIssue.TOO_DARK and quality_res.mean_brightness < 10.0:
                res = SessionJobResult(
                    job_id=job_id,
                    source=source_label,
                    active_source=active_src,
                    selected_image_name=img_name,
                    mode="object_detection",
                    status=SafetyStatus.BAD_IMAGE,
                    spoken_text="Ảnh quá tối, chưa xác định được đồ vật.",
                    display_text="Ảnh quá tối, chưa xác định được đồ vật. Hãy bật thêm đèn hoặc hướng camera ra nơi đủ sáng.",
                    warnings=["Ảnh quá tối, hãy bật thêm đèn hoặc hướng camera ra nơi đủ sáng."],
                    timings_ms=timings,
                    quality=quality_res,
                    session_id=session_id,
                )
                self._finish_job_with_speech(session_id, job_id, cancel_token, res, t_request_start)
                return

            # Step 3: RECOGNIZING (Object Detection)
            if not self._set_state_if_valid(
                session_id, job_id, cancel_token, SessionState.RECOGNIZING, "Đang nhận diện đồ vật..."
            ):
                return

            t_det0 = time.perf_counter()
            with self._lock:
                detector = self.object_detector

            if detector is None:
                det_res = ObjectDetectionResult(
                    objects=[],
                    error="Mô hình nhận diện đồ vật chưa được cài đặt hoặc chưa sẵn sàng.",
                    spoken_text="Chưa xác định được đồ vật.",
                    display_text="Mô hình nhận diện đồ vật chưa được cài đặt hoặc chưa sẵn sàng.",
                    confidence_status="NONE",
                )
            else:
                det_res = detector.detect(frame, cancel_token=cancel_token)
            timings["detection"] = int(round((time.perf_counter() - t_det0) * 1000.0))

            if not self._is_job_valid(session_id, job_id, cancel_token):
                return

            # Step 4: PREPARE RESULT
            status = SafetyStatus.READABLE if det_res.objects else SafetyStatus.NO_TEXT
            tts_blocked = not self.speech_backend.is_vi_vn_available()
            warnings = []
            if tts_blocked:
                warnings.append(MSG_TTS_BLOCKED)
            if det_res.error:
                warnings.append(det_res.error)

            res = SessionJobResult(
                job_id=job_id,
                source=source_label,
                active_source=active_src,
                selected_image_name=img_name,
                mode="object_detection",
                status=status,
                language="vi",
                detection=det_res,
                annotated_frame=det_res.annotated_frame,
                spoken_text=det_res.spoken_text,
                display_text=det_res.display_text,
                warnings=warnings,
                timings_ms=timings,
                quality=quality_res,
                tts_blocked=tts_blocked,
                session_id=session_id,
            )

            # Step 5: READY_TO_SPEAK & DISPATCH
            self._finish_job_with_speech(session_id, job_id, cancel_token, res, t_request_start)

        finally:
            if close_source_after:
                try:
                    source.close()
                except Exception:
                    pass

    def _finish_job_with_speech(
        self,
        session_id: int,
        job_id: int,
        cancel_token: threading.Event,
        result: SessionJobResult,
        t_request_start: float,
    ) -> None:
        if not self._set_state_if_valid(
            session_id, job_id, cancel_token, SessionState.READY_TO_SPEAK, result.spoken_text
        ):
            return

        if result.tts_blocked:
            self._notify_result(session_id, job_id, cancel_token, result)
            self._maybe_write_debug_log(result)
            self._set_state_if_valid(
                session_id, job_id, cancel_token, SessionState.ERROR, MSG_TTS_BLOCKED
            )
            return

        def _on_speech_start(started_job_id: int, _synth_ms: float) -> None:
            if started_job_id == job_id:
                result.timings_ms["tts_start"] = int(
                    round((time.perf_counter() - t_request_start) * 1000.0)
                )

        # Notify UI with recognized text right before speaking so text appears immediately
        result.timings_ms["tts_start"] = int(round((time.perf_counter() - t_request_start) * 1000.0))
        self._notify_result(session_id, job_id, cancel_token, result)

        if not self._set_state_if_valid(
            session_id, job_id, cancel_token, SessionState.SPEAKING, "Đang đọc kết quả..."
        ):
            return

        speak_kwargs = {
            "text": result.spoken_text,
            "job_id": job_id,
            "cancel_token": cancel_token,
            "on_speech_started": _on_speech_start,
        }
        if hasattr(self.speech_backend, "speak"):
            import inspect
            sig = inspect.signature(self.speech_backend.speak)
            if "lang" in sig.parameters:
                speak_kwargs["lang"] = getattr(result, "language", "vi")
        self.speech_backend.speak(**speak_kwargs)

        self._maybe_write_debug_log(result)

        if self._is_job_valid(session_id, job_id, cancel_token):
            if result.status == SafetyStatus.ENGINE_ERROR:
                self._set_state_if_valid(
                    session_id, job_id, cancel_token, SessionState.ERROR, result.spoken_text
                )
            else:
                self._restore_ready_state(job_id)

    def _maybe_write_debug_log(self, result: SessionJobResult) -> None:
        """Write metadata-only debug record ONLY if developer explicitly opted in for this session."""
        if not self.config.debug_opt_in_runtime:
            return
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            log_file = self._debug_dir / f"job_{result.job_id}.json"
            with log_file.open("w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def purge_debug_logs(self) -> int:
        """Delete all opt-in developer debug logs from disk (Section 6.1, FR-11)."""
        if not self._debug_dir.exists():
            return 0
        count = sum(1 for _ in self._debug_dir.glob("*"))
        try:
            shutil.rmtree(self._debug_dir, ignore_errors=True)
        except Exception:
            pass
        return count

    def wait_for_idle(self, timeout_s: float = 5.0) -> bool:
        """Helper for tests/benchmarks to wait until active worker threads finish."""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                alive = [t for t in self._active_threads if t.is_alive()]
            if not alive:
                return True
            time.sleep(0.01)
        return False

    def shutdown(self) -> None:
        """Safely cancel all jobs, stop speech, close camera, and suppress late callbacks (FR-07)."""
        with self._lock:
            self._shutting_down = True
            if self._current_cancel_token is not None:
                self._current_cancel_token.set()
            self._session_id += 1
            self._current_job_id += 1
            self._on_state_changed = None
            self._on_job_completed = None
            self._on_results_cleared = None
            threads = [t for t in self._active_threads if t.is_alive()]

        self.speech_backend.shutdown()
        try:
            self.camera.close()
        except Exception:
            pass

        for t in threads:
            try:
                t.join(timeout=0.5)
            except Exception:
                pass
