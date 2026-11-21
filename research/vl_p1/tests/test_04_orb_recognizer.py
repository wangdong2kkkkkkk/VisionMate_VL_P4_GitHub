"""Tests for Method A: Rebuilt OpenCV ORB Baseline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.image_utils import load_image_unicode
from src.orb_recognizer import ORBConfig, ORBRecognizer

VL_P1_ROOT = Path(__file__).resolve().parent.parent


def test_orb_adaptive_fixes_zero_keypoint_bug_on_narrow_cane() -> None:
    narrow = load_image_unicode(VL_P1_ROOT / "data" / "train" / "object_01" / "03.png")
    raw_rec = ORBRecognizer(ORBConfig(adaptive_narrow_roi=False))
    adapt_rec = ORBRecognizer(ORBConfig(adaptive_narrow_roi=True))
    kps_raw, _ = raw_rec.extract_features(narrow)
    kps_adapt, desc_adapt = adapt_rec.extract_features(narrow)
    assert len(kps_raw) == 0
    assert len(kps_adapt) > 100
    assert desc_adapt is not None


def test_orb_matches_self_as_known_and_noise_as_unknown() -> None:
    rec = ORBRecognizer(ORBConfig(adaptive_narrow_roi=True, min_inliers=8, min_inlier_ratio=0.35, min_margin_inliers=3))
    rec.fit_directory(VL_P1_ROOT / "data" / "train")

    sample = load_image_unicode(VL_P1_ROOT / "data" / "train" / "object_03" / "01.png")
    known_res = rec.predict(sample)
    assert known_res["status"] == "KNOWN"
    assert known_res["label"] == "object_03"

    blank = np.full((240, 240, 3), 127, dtype=np.uint8)
    unk_res = rec.predict(blank)
    assert unk_res["status"] == "UNKNOWN"
    assert unk_res["label"] is None
