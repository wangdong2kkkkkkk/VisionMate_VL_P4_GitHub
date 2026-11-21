# -*- coding: utf-8 -*-
"""
CameraSource, ImageFileSource, and MockCameraSource implementations (Section 4.1, FR-01, FR-02).
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraDeviceInfo:
    """Information about a detected UVC/integrated camera."""
    index: int
    label: str
    width: int
    height: int


def load_image_unicode(path: str | Path) -> Optional[np.ndarray]:
    """Load an image from disk supporting full Unicode / Vietnamese file paths on Windows."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        raw_bytes = np.fromfile(str(p), dtype=np.uint8)
        if raw_bytes.size == 0:
            return None
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class FrameSource(ABC):
    """Abstract interface for opening, reading a single frame on demand, and closing a source."""

    @abstractmethod
    def open(self) -> bool:
        """Open the frame source. Returns True if ready."""

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single BGR frame on demand (in RAM only)."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources safely."""

    @abstractmethod
    def is_open(self) -> bool:
        """Return True if the source is currently open."""

    @abstractmethod
    def source_name(self) -> str:
        """Return identifier for contract ('camera' or 'file')."""


class CameraSource(FrameSource):
    """UVC / integrated webcam frame source with thread-safe capture-on-demand (FR-01, FR-02)."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0

    def open(self) -> bool:
        with self._lock:
            self._release_unlocked()
            # Try Windows Media Foundation (MSMF) first for 30 FPS 720p, fallback to DSHOW / ANY
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(self.device_index, backend)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        self._cap = cap
                        self._last_frame = frame.copy()
                        self._last_frame_ts = time.perf_counter()
                        return True
                    cap.release()
                except Exception:
                    continue
            return False

    def read_preview_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame for UI preview and update the short-lived RAM buffer."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                self._last_frame = frame
                self._last_frame_ts = time.perf_counter()
                return True, frame.copy()
            return False, None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture one frame on demand (FR-02, Phase 1.3 Section B.4).
        Samples up to 3 consecutive frames within a short burst (~60ms total) and selects
        the frame with the highest Laplacian sharpness variance to mitigate transient hand shake
        or motion blur without storing any frames to disk.
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                if self._last_frame is not None and (time.perf_counter() - self._last_frame_ts) <= 0.12:
                    return True, self._last_frame.copy()
                return False, None

            candidates: List[Tuple[float, np.ndarray]] = []
            if self._last_frame is not None and (time.perf_counter() - self._last_frame_ts) <= 0.08:
                candidates.append((self._frame_sharpness(self._last_frame), self._last_frame.copy()))

            for _ in range(3 if not candidates else 2):
                ok, frm = self._cap.read()
                if ok and frm is not None and frm.size > 0:
                    self._last_frame = frm
                    self._last_frame_ts = time.perf_counter()
                    candidates.append((self._frame_sharpness(frm), frm.copy()))

            if not candidates:
                return False, None

            candidates.sort(key=lambda item: item[0], reverse=True)
            return True, candidates[0][1]

    @staticmethod
    def _frame_sharpness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        if w > 320:
            gray = cv2.resize(gray, (320, max(24, int(h * 320 / w))), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def close(self) -> None:
        with self._lock:
            self._release_unlocked()

    def _release_unlocked(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._last_frame = None
        self._last_frame_ts = 0.0

    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def source_name(self) -> str:
        return "camera"


class ImageFileSource(FrameSource):
    """Local test image frame source for dev/test mode (Scenario U2, FR-06)."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._frame: Optional[np.ndarray] = None
        self._opened: bool = False

    def open(self) -> bool:
        img = load_image_unicode(self.file_path)
        if img is None or img.size == 0:
            self._frame = None
            self._opened = False
            return False
        self._frame = img
        self._opened = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened or self._frame is None:
            if not self.open():
                return False, None
        assert self._frame is not None
        return True, self._frame.copy()

    def close(self) -> None:
        self._frame = None
        self._opened = False

    def is_open(self) -> bool:
        return self._opened and self._frame is not None

    def source_name(self) -> str:
        return "file"


class MockCameraSource(FrameSource):
    """Configurable mock camera source for unit and integration testing."""

    def __init__(
        self,
        frame: Optional[np.ndarray] = None,
        allow_open: bool = True,
        read_delay_s: float = 0.0,
        burst_frames: Optional[List[np.ndarray]] = None,
    ) -> None:
        self._frame = frame
        self._burst_frames = list(burst_frames) if burst_frames else []
        self._allow_open = allow_open
        self._read_delay_s = read_delay_s
        self._opened = False

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        self._frame = frame

    def set_burst_frames(self, burst_frames: List[np.ndarray]) -> None:
        self._burst_frames = list(burst_frames)

    def open(self) -> bool:
        if not self._allow_open:
            self._opened = False
            return False
        self._opened = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        if self._read_delay_s > 0:
            time.sleep(self._read_delay_s)
        if self._burst_frames:
            scored = [(CameraSource._frame_sharpness(f), f.copy()) for f in self._burst_frames if f is not None and f.size > 0]
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                return True, scored[0][1]
        if self._frame is None or self._frame.size == 0:
            return False, None
        return True, self._frame.copy()

    def close(self) -> None:
        self._opened = False

    def is_open(self) -> bool:
        return self._opened

    def source_name(self) -> str:
        return "camera"


def enumerate_cameras(max_index: int = 3) -> List[CameraDeviceInfo]:
    """Probe camera indices 0..max_index-1 and return available devices (FR-01)."""
    devices: List[CameraDeviceInfo] = []
    for idx in range(max_index):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if ok and frame is not None and frame.size > 0:
                    devices.append(
                        CameraDeviceInfo(
                            index=idx,
                            label=f"Camera {idx} ({w}x{h})",
                            width=w,
                            height=h,
                        )
                    )
                    break
            else:
                cap.release()
    return devices
