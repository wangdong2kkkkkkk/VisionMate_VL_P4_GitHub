"""Smoke tests using synthetic images ONLY for code behavior, not accuracy metrics."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from orb_baseline import extract_orb, match_pair, predict_one


class OrbSmoke(unittest.TestCase):
    def test_blank_image_has_no_descriptors(self):
        kp, descriptors = extract_orb(np.full((128, 128, 3), 255, dtype=np.uint8))
        self.assertEqual(len(kp), 0)
        self.assertIsNone(descriptors)

    def test_blank_image_returns_unknown(self):
        blank = np.full((128, 128, 3), 255, dtype=np.uint8)
        gallery = {"object_01": [(Path("dummy.png"), extract_orb(blank))]}
        result = predict_one(blank, gallery)
        self.assertEqual(result["status"], "UNKNOWN")

    def test_same_textured_image_matches(self):
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (300, 300), dtype=np.uint8)
        a = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        scores = match_pair(extract_orb(a), extract_orb(a.copy()))
        self.assertGreater(scores["inliers"], 10)


if __name__ == "__main__":
    unittest.main()
