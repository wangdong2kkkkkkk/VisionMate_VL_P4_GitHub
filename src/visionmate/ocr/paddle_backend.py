# -*- coding: utf-8 -*-
"""
PaddleOCR (PP-OCRv3/v5 ONNX CPU) Backend for comparative evaluation (Section 5.1, FR-12).
Uses local ONNX models (`models/paddleocr_latin/latin_rec.onnx` + `latin_dict.txt`) via
`rapidocr_onnxruntime` with zero runtime network calls.
Note: PaddleOCR maps `lang='vi'` to its `latin` recognition model, which lacks stacked
Vietnamese tonal diacritics (`ắ`, `ệ`, `ộ`, `ợ`, `ữ`, `ư`, `ơ`). Included for empirical
benchmark comparison and backend pluggability.
"""
from __future__ import annotations

import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from visionmate.contracts import OCRLine, OCRResult
from visionmate.ocr.base import OCRBackend


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class PaddleONNXOCRBackend(OCRBackend):
    """Offline PaddleOCR (Latin/vi ONNX) backend running on CPU via ONNX Runtime."""

    def __init__(self) -> None:
        self.engine_name = "paddleocr_onnx_latin_vi"
        self._rec_model_path = PROJECT_ROOT / "models" / "paddleocr_latin" / "latin_rec.onnx"
        self._rec_keys_path = PROJECT_ROOT / "models" / "paddleocr_latin" / "latin_dict.txt"
        self._engine: Any = None
        self._init_error: Optional[str] = None

    def _ensure_engine(self) -> bool:
        if self._engine is not None:
            return True
        if self._init_error is not None:
            return False
        if not self._rec_model_path.is_file() or not self._rec_keys_path.is_file():
            self._init_error = "Thiếu file model ONNX cục bộ của PaddleOCR (latin_rec.onnx / latin_dict.txt)."
            return False
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR(
                rec_model_path=str(self._rec_model_path),
                rec_keys_path=str(self._rec_keys_path),
            )
            return True
        except Exception as exc:
            self._init_error = str(exc)
            return False

    def is_available(self) -> bool:
        return self._rec_model_path.is_file() and self._rec_keys_path.is_file()

    def backend_info(self) -> Dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "rec_model_path": str(self._rec_model_path),
            "rec_keys_path": str(self._rec_keys_path),
            "available": self.is_available(),
        }

    def recognize(
        self,
        image: np.ndarray,
        cancel_token: Optional[threading.Event] = None,
    ) -> OCRResult:
        t0 = time.perf_counter()
        if cancel_token is not None and cancel_token.is_set():
            return OCRResult(lines=[], engine_name=self.engine_name, elapsed_ms=0.0, error="CANCELLED")

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return OCRResult(
                lines=[],
                engine_name=self.engine_name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="Ảnh đầu vào rỗng hoặc không hợp lệ.",
            )

        if not self._ensure_engine():
            return OCRResult(
                lines=[],
                engine_name=self.engine_name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error=self._init_error or "PaddleOCR ONNX chưa sẵn sàng.",
            )

        try:
            result, _ = self._engine(image)
            if cancel_token is not None and cancel_token.is_set():
                return OCRResult(
                    lines=[],
                    engine_name=self.engine_name,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    error="CANCELLED",
                )

            lines: List[OCRLine] = []
            if result:
                for item in result:
                    pts, raw_text, score = item[0], str(item[1]).strip(), float(item[2]) * 100.0
                    text = unicodedata.normalize("NFC", raw_text)
                    alnum = [c for c in text if c.isalnum()]
                    if not alnum:
                        continue
                    if len(alnum) <= 2 and score < 70.0:
                        continue
                    if score < 35.0:
                        continue
                    xs = [int(round(p[0])) for p in pts]
                    ys = [int(round(p[1])) for p in pts]
                    bbox = (
                        min(xs),
                        min(ys),
                        max(1, max(xs) - min(xs)),
                        max(1, max(ys) - min(ys)),
                    )
                    lines.append(OCRLine(text=text, bbox=bbox, raw_score=score))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return OCRResult(lines=lines, engine_name=self.engine_name, elapsed_ms=elapsed_ms, error=None)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return OCRResult(
                lines=[],
                engine_name=self.engine_name,
                elapsed_ms=elapsed_ms,
                error=f"Lỗi PaddleOCR ONNX: {exc}",
            )
