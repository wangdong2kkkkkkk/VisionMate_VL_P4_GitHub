# -*- coding: utf-8 -*-
"""Fake OCR backend for deterministic unit, integration, and race-condition testing."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from visionmate.contracts import OCRLine, OCRResult
from visionmate.ocr.base import OCRBackend


class FakeOCRBackend(OCRBackend):
    """Configurable in-memory OCR backend for testing."""

    def __init__(
        self,
        lines: Optional[List[OCRLine]] = None,
        delay_s: float = 0.0,
        error: Optional[str] = None,
        available: bool = True,
    ) -> None:
        self._lines = list(lines) if lines is not None else []
        self._delay_s = delay_s
        self._error = error
        self._available = available
        self.call_count = 0

    def set_response(
        self,
        lines: Optional[List[OCRLine]] = None,
        delay_s: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self._lines = list(lines) if lines is not None else []
        self._delay_s = delay_s
        self._error = error

    def is_available(self) -> bool:
        return self._available

    def backend_info(self) -> Dict[str, Any]:
        return {"engine_name": "fake_ocr", "available": self._available}

    def recognize(
        self,
        image: np.ndarray,
        cancel_token: Optional[threading.Event] = None,
    ) -> OCRResult:
        self.call_count += 1
        t0 = time.perf_counter()
        if self._delay_s > 0:
            step = 0.01
            waited = 0.0
            while waited < self._delay_s:
                if cancel_token is not None and cancel_token.is_set():
                    return OCRResult(
                        lines=[],
                        engine_name="fake_ocr",
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        error="CANCELLED",
                    )
                time.sleep(min(step, self._delay_s - waited))
                waited += step

        if cancel_token is not None and cancel_token.is_set():
            return OCRResult(
                lines=[],
                engine_name="fake_ocr",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="CANCELLED",
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if self._error:
            return OCRResult(lines=[], engine_name="fake_ocr", elapsed_ms=elapsed_ms, error=self._error)
        return OCRResult(lines=list(self._lines), engine_name="fake_ocr", elapsed_ms=elapsed_ms, error=None)
