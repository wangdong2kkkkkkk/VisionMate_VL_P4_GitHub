"""Top-level deliverable entry module exposing VL-P1b Hard-Negative Research classes."""
from src.hard_negative_research import (
    DINOv2Embedder,
    HardNegConfig,
    HardNegativeRecognizer,
    extract_roi_geometry_features,
    preprocess_roi_dinov2,
)

__all__ = [
    "DINOv2Embedder",
    "HardNegConfig",
    "HardNegativeRecognizer",
    "extract_roi_geometry_features",
    "preprocess_roi_dinov2",
]
