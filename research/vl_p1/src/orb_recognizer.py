"""Method A: Rebuilt OpenCV ORB Baseline for VL-P1 with adaptive narrow-ROI handling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from .image_utils import OBJECT_IDS, list_images, load_image_unicode


@dataclass
class ORBConfig:
    nfeatures: int = 1500
    fast_threshold: int = 10
    adaptive_narrow_roi: bool = True
    edge_threshold: int = 15
    patch_size: int = 15
    lowe_ratio: float = 0.78
    ransac_reproj_thresh: float = 4.5
    min_inliers: int = 10
    min_inlier_ratio: float = 0.45
    min_margin_inliers: int = 4


class ORBRecognizer:
    """Local keypoint (ORB + BFMatcher Hamming + Lowe ratio + RANSAC homography) matcher."""

    def __init__(self, config: ORBConfig | None = None) -> None:
        self.cfg = config or ORBConfig()
        if self.cfg.adaptive_narrow_roi:
            self._orb = cv2.ORB_create(
                nfeatures=self.cfg.nfeatures,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=self.cfg.edge_threshold,
                patchSize=self.cfg.patch_size,
                fastThreshold=self.cfg.fast_threshold,
            )
        else:
            self._orb = cv2.ORB_create(
                nfeatures=1300,
                fastThreshold=self.cfg.fast_threshold,
            )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.gallery: dict[str, list[tuple[str, list[cv2.KeyPoint], np.ndarray | None]]] = {}

    def _preprocess_gray(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided to ORBRecognizer")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.cfg.adaptive_narrow_roi:
            h, w = gray.shape[:2]
            if min(h, w) < 128:
                scale = 128.0 / float(max(1, min(h, w)))
                nw = max(1, min(2048, int(round(w * scale))))
                nh = max(1, min(2048, int(round(h * scale))))
                gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
            gray = cv2.copyMakeBorder(gray, 36, 36, 36, 36, cv2.BORDER_REFLECT_101)
        return gray

    def extract_features(self, image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        gray = self._preprocess_gray(image)
        kps, desc = self._orb.detectAndCompute(gray, None)
        return list(kps) if kps is not None else [], desc

    def fit_directory(self, train_dir: Path, object_ids: list[str] | None = None) -> dict:
        oids = object_ids or OBJECT_IDS
        t0 = time.perf_counter()
        self.gallery = {}
        kp_counts = {}
        for oid in oids:
            items = []
            for p in list_images(train_dir / oid):
                img = load_image_unicode(p)
                kps, desc = self.extract_features(img)
                items.append((p.name, kps, desc))
                kp_counts[f"{oid}/{p.name}"] = len(kps)
            self.gallery[oid] = items
        if not any(self.gallery.values()):
            raise ValueError(f"No training images found under {train_dir}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {"enroll_total_ms": round(elapsed_ms, 2), "kp_counts": kp_counts}

    def match_pair(
        self,
        feat_a: tuple[list[cv2.KeyPoint], np.ndarray | None],
        feat_b: tuple[list[cv2.KeyPoint], np.ndarray | None],
    ) -> dict:
        kp_a, desc_a = feat_a
        kp_b, desc_b = feat_b
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return {"good": 0, "inliers": 0, "ratio": 0.0}
        pairs = self._matcher.knnMatch(desc_a, desc_b, k=2)
        good = [
            m
            for pair in pairs
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self.cfg.lowe_ratio * n.distance
        ]
        if len(good) < 4:
            return {"good": len(good), "inliers": 0, "ratio": 0.0}
        src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.cfg.ransac_reproj_thresh)
            inliers = int(mask.sum()) if mask is not None else 0
        except cv2.error:
            inliers = 0
        return {
            "good": len(good),
            "inliers": inliers,
            "ratio": round(inliers / max(1, len(good)), 4),
        }

    def predict(self, image: np.ndarray) -> dict:
        if not self.gallery:
            raise RuntimeError("ORBRecognizer gallery is empty; call fit_directory() first.")
        t_feat0 = time.perf_counter()
        q_kps, q_desc = self.extract_features(image)
        t_match0 = time.perf_counter()
        feat_ms = (t_match0 - t_feat0) * 1000.0

        ranked = []
        for oid, items in self.gallery.items():
            best = {"good": 0, "inliers": 0, "ratio": 0.0}
            best_sample = ""
            for sample_name, g_kps, g_desc in items:
                m = self.match_pair((g_kps, g_desc), (q_kps, q_desc))
                if (m["inliers"], m["good"]) > (best["inliers"], best["good"]):
                    best = m
                    best_sample = sample_name
            ranked.append({"id": oid, "sample": best_sample, **best})

        ranked.sort(key=lambda r: (r["inliers"], r["good"], r["ratio"]), reverse=True)
        match_ms = (time.perf_counter() - t_match0) * 1000.0

        if not ranked:
            return {
                "status": "UNKNOWN",
                "label": None,
                "score": 0,
                "margin": 0,
                "query_keypoints": len(q_kps),
                "candidates": [],
                "feat_ms": round(feat_ms, 3),
                "match_ms": round(match_ms, 3),
            }

        top = ranked[0]
        second = ranked[1]["inliers"] if len(ranked) > 1 else 0
        margin = top["inliers"] - second

        if top["inliers"] < self.cfg.min_inliers or top["ratio"] < self.cfg.min_inlier_ratio:
            status = "UNKNOWN"
            label = None
        elif margin < self.cfg.min_margin_inliers:
            status = "AMBIGUOUS"
            label = None
        else:
            status = "KNOWN"
            label = top["id"]

        return {
            "status": status,
            "label": label,
            "score": int(top["inliers"]),
            "margin": int(margin),
            "query_keypoints": len(q_kps),
            "candidates": ranked[:3],
            "feat_ms": round(feat_ms, 3),
            "match_ms": round(match_ms, 3),
        }
