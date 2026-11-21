# -*- coding: utf-8 -*-
"""OCR backends package."""
from .base import OCRBackend
from .fake_backend import FakeOCRBackend
from .paddle_backend import PaddleONNXOCRBackend
from .tesseract_backend import TesseractOCRBackend

__all__ = [
    "OCRBackend",
    "FakeOCRBackend",
    "PaddleONNXOCRBackend",
    "TesseractOCRBackend",
]
