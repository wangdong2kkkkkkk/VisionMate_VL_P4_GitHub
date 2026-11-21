# -*- coding: utf-8 -*-
"""Abstract OCRBackend interface (Section 4.1)."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from visionmate.contracts import OCRResult


class OCRBackend(ABC):
    """Interface for local offline Vietnamese OCR engines."""

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        cancel_token: Optional[threading.Event] = None,
    ) -> OCRResult:
        """Recognize text lines from an in-memory BGR/grayscale image array."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the OCR binary and local model files are ready offline."""

    @abstractmethod
    def backend_info(self) -> Dict[str, Any]:
        """Return metadata about the backend, version, and model variant."""
