"""Tests for ROI handling, crop_fit compatibility, and Task 2 full-image proposers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from src.full_image_pipeline import compute_iou_xywh, foreground_contour_proposals
from src.image_utils import load_image_unicode, pad_to_square

VL_P1_ROOT = Path(__file__).resolve().parent.parent


def _load_crop_fit():
    import sys
    if str(VL_P1_ROOT) not in sys.path:
        sys.path.insert(0, str(VL_P1_ROOT))
    spec = importlib.util.spec_from_file_location("crop_fit", VL_P1_ROOT / "crop_fit.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_crop_fit_preview_and_backprojection() -> None:
    cf = _load_crop_fit()
    pw, ph = cf.preview_size(4000, 3000, 1366, 768)
    assert pw <= 1200 and ph <= 720
    assert abs((pw / ph) - (4000 / 3000)) < 0.02
    x, y, w, h = cf.roi_to_original((10, 20, 100, 150), (pw, ph), (4000, 3000))
    assert x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 4000 and y + h <= 3000


def test_pad_to_square_preserves_narrow_aspect_object() -> None:
    narrow_path = VL_P1_ROOT / "data" / "train" / "object_01" / "03.png"
    img = load_image_unicode(narrow_path)
    h, w = img.shape[:2]
    assert w == 61 and h == 1131
    sq = pad_to_square(img)
    assert sq.shape[0] == sq.shape[1] == max(h, w)


def test_foreground_proposer_finds_synthetic_object_without_gt() -> None:
    canvas = np.full((320, 480, 3), 240, dtype=np.uint8)
    canvas[90:230, 160:330] = (30, 60, 180)
    props = foreground_contour_proposals(canvas, max_boxes=2)
    assert len(props) >= 1
    best_iou = max(compute_iou_xywh((160, 90, 170, 140), p.bbox_xywh) for p in props)
    assert best_iou >= 0.60
