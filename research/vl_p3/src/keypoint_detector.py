"""Supervised White Cane Keypoint, Centerline & Structural Bridge Detector (Method A1).

Predicts 112x112 spatial heatmaps for:
  - grip_hm (top/hand endpoint)
  - tip_hm (bottom/ground-tip endpoint)
  - centerline_hm (full cane centerline including occluded spans)
  - double-angle orientation field (cos2, sin2)

Uses a fast pure-NumPy dense MLP (exported from sklearn MLPRegressor) + sparse top-K
HistGradientBoosting refinement, followed by collinear segment bridging and
multi-candidate structural decoding into keypoints, 11-pt centerline, and rotated OBB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .common_utils import (
    compute_sha256,
    obb_from_endpoints,
    sample_centerline_points,
)
from .spatial_features import GRID_SIZE, SpatialFeatureBundle

ABLATION_MODES = (
    "1_keypoints_only",
    "2_centerline_only",
    "3_keypoints_plus_centerline",
    "4_kp_cl_plus_hard_neg_mining",
    "5_full_aug_and_occlusion",
)

_CTX3_CH = (0, 2, 3, 6, 7, 9, 10, 12)  # 8 ch
_CTX9_CH = (2, 3, 6, 9, 10, 12, 28, 33, 34, 35)  # 10 ch
_MAX5_CH = (2, 3, 6)  # 3 ch
BASE_CTX_DIM = 36 + len(_CTX3_CH) + len(_CTX9_CH) + len(_MAX5_CH) + 3  # 60


def extract_pixel_context_112(features_112: np.ndarray) -> np.ndarray:
    """Enrich [112, 112, 36] spatial tensor with multi-scale neighborhood & end-stop cues -> [112, 112, 60]."""
    f = np.asarray(features_112, dtype=np.float32)
    ctx3 = cv2.boxFilter(f[:, :, _CTX3_CH], cv2.CV_32F, (3, 3), borderType=cv2.BORDER_REPLICATE)
    ctx9 = cv2.boxFilter(f[:, :, _CTX9_CH], cv2.CV_32F, (9, 9), borderType=cv2.BORDER_REPLICATE)
    k5 = np.ones((5, 5), dtype=np.uint8)
    max5 = cv2.dilate(f[:, :, _MAX5_CH], k5, borderType=cv2.BORDER_REPLICATE)

    ridge_s = cv2.boxFilter(f[:, :, 6], cv2.CV_32F, (3, 3), borderType=cv2.BORDER_REPLICATE)
    up6 = np.zeros_like(ridge_s)
    dn6 = np.zeros_like(ridge_s)
    lt6 = np.zeros_like(ridge_s)
    rt6 = np.zeros_like(ridge_s)
    up6[6:, :] = ridge_s[:-6, :]
    dn6[:-6, :] = ridge_s[6:, :]
    lt6[:, 6:] = ridge_s[:, :-6]
    rt6[:, :-6] = ridge_s[:, 6:]

    asym_v = (up6 - dn6)[..., None]
    sum_v = (0.5 * (up6 + dn6))[..., None]
    asym_h = np.abs(lt6 - rt6)[..., None]

    out = np.concatenate([f, ctx3, ctx9, max5, asym_v, sum_v, asym_h], axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _nms_peaks_2d(hm: np.ndarray, top_k: int = 6, min_dist: int = 6, min_val: float = 0.08) -> list[tuple[float, float, float]]:
    """Extract top-K local maxima [(gx, gy, score)] via fast morphological dilation."""
    k = max(3, min_dist | 1)
    dil = cv2.dilate(hm.astype(np.float32), np.ones((k, k), np.uint8))
    is_peak = (hm >= dil - 1e-6) & (hm >= min_val)
    ys, xs = np.nonzero(is_peak)
    if len(ys) == 0:
        idx = int(np.argmax(hm))
        py, px = divmod(idx, hm.shape[1])
        return [(float(px), float(py), float(hm[py, px]))]
    vals = hm[ys, xs]
    order = np.argsort(-vals)
    peaks: list[tuple[float, float, float]] = []
    for idx in order:
        px, py, sc = float(xs[idx]), float(ys[idx]), float(vals[idx])
        if all((px - qx) ** 2 + (py - qy) ** 2 >= min_dist * min_dist for qx, qy, _ in peaks):
            peaks.append((px, py, sc))
            if len(peaks) >= top_k:
                break
    return peaks


def _axial_deg(dx: float, dy: float) -> float:
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    return ang


def _axial_diff(a1: float, a2: float) -> float:
    d = abs(a1 - a2) % 180.0
    return min(d, 180.0 - d)


def _extract_ridge_segments_112(cl_hm: np.ndarray, valid_mask: np.ndarray, thr: float = 0.22) -> list[tuple[float, float, float, float, float]]:
    """Extract and collinearly bridge elongated centerline segments in 112x112 grid."""
    bw = ((cl_hm >= thr) & (valid_mask > 0.2)).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    raw_segs: list[dict] = []
    for cid in range(1, num):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        ys, xs = np.nonzero(labels == cid)
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T) if len(pts) >= 3 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)
        u = evecs[:, int(np.argmax(evals))]
        proj = (pts - mean) @ u
        pmin = mean + float(proj.min()) * u
        pmax = mean + float(proj.max()) * u
        if pmin[1] > pmax[1]:
            pmin, pmax = pmax, pmin
        L = float(math.hypot(pmax[0] - pmin[0], pmax[1] - pmin[1]))
        if L < 5.0:
            continue
        ang = _axial_deg(pmax[0] - pmin[0], pmax[1] - pmin[1])
        sc = float(np.mean(cl_hm[ys, xs]))
        raw_segs.append({
            "x1": float(pmin[0]), "y1": float(pmin[1]),
            "x2": float(pmax[0]), "y2": float(pmax[1]),
            "len": L, "ang": ang, "score": sc,
            "ux": float((pmax[0] - pmin[0]) / L),
            "uy": float((pmax[1] - pmin[1]) / L),
        })

    out: list[tuple[float, float, float, float, float]] = [
        (s["x1"], s["y1"], s["x2"], s["y2"], s["score"] * min(1.5, s["len"] / 20.0))
        for s in raw_segs
    ]
    # Collinear gap bridging across occlusion gaps (hand/clothing/pole occlusions)
    n = len(raw_segs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = raw_segs[i], raw_segs[j]
            if _axial_diff(a["ang"], b["ang"]) > 12.5:
                continue
            pts = [(a["x1"], a["y1"]), (a["x2"], a["y2"]), (b["x1"], b["y1"]), (b["x2"], b["y2"])]
            # Check perpendicular distance from b's midpoint to a's infinite line
            mx, my = 0.5 * (b["x1"] + b["x2"]), 0.5 * (b["y1"] + b["y2"])
            perp = abs((mx - a["x1"]) * a["uy"] - (my - a["y1"]) * a["ux"])
            if perp > 4.2:
                continue
            # Minimum gap between closest endpoints
            min_gap = min(
                math.hypot(a[k1] - b[k2], a[k3] - b[k4])
                for k1, k3 in (("x1", "y1"), ("x2", "y2"))
                for k2, k4 in (("x1", "y1"), ("x2", "y2"))
            )
            if min_gap > 36.0:
                continue
            # Take farthest two endpoints among the 4
            best_d, best_pair = -1.0, (a["x1"], a["y1"], b["x2"], b["y2"])
            for p_idx in range(4):
                for q_idx in range(p_idx + 1, 4):
                    d = math.hypot(pts[p_idx][0] - pts[q_idx][0], pts[p_idx][1] - pts[q_idx][1])
                    if d > best_d:
                        best_d = d
                        best_pair = (pts[p_idx][0], pts[p_idx][1], pts[q_idx][0], pts[q_idx][1])
            bx1, by1, bx2, by2 = best_pair
            if by1 > by2:
                bx1, by1, bx2, by2 = bx2, by2, bx1, by1
            out.append((bx1, by1, bx2, by2, 0.55 * (a["score"] + b["score"]) * min(1.6, best_d / 22.0)))

    out.sort(key=lambda t: t[4], reverse=True)
    return out[:10]


@dataclass
class LocalizationPrediction:
    detected: bool
    endpoints_xyxy: list[float]
    centerline_points_xy: list[list[float]]
    obb_cxcywh_angle: tuple[float, float, float, float, float]
    bbox_xywh: tuple[int, int, int, int]
    orientation_deg: float
    length_px: float
    length_ratio: float
    visible_ratio: float
    occlusion_gap_ratio: float
    occluded: bool
    confidence: float
    kp_confidence: float
    cl_confidence: float
    grip_score: float
    tip_score: float
    bridge_used: bool
    grip_hm: np.ndarray = field(repr=False)
    tip_hm: np.ndarray = field(repr=False)
    cl_hm: np.ndarray = field(repr=False)
    cos2_map: np.ndarray = field(repr=False)
    sin2_map: np.ndarray = field(repr=False)
    proposals: list[dict] = field(default_factory=list, repr=False)
    decode_ms: float = 0.0


class SupervisedKeypointCenterlineDetector:
    """Supervised CPU-first detector predicting grip/tip keypoints, centerline, orientation & bridged OBB."""

    def __init__(self, mode: str = "5_full_aug_and_occlusion") -> None:
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unsupported ablation mode: {mode}")
        self.mode = mode
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        self.mlp_coefs: list[np.ndarray] = []
        self.mlp_intercepts: list[np.ndarray] = []
        self.hgb_grip: HistGradientBoostingRegressor | None = None
        self.hgb_tip: HistGradientBoostingRegressor | None = None
        self.hgb_cl: HistGradientBoostingRegressor | None = None
        self.sparse_refine_k: int = 160
        self.is_fitted: bool = False
        self.train_summary: dict = {}

    def fit(
        self,
        X_rows: np.ndarray,
        Y_targets: np.ndarray,
        sample_weights: np.ndarray | None = None,
        sparse_X: np.ndarray | None = None,
        sparse_Y: np.ndarray | None = None,
        sparse_w: np.ndarray | None = None,
    ) -> "SupervisedKeypointCenterlineDetector":
        """Train Stage-1 dense MLP (5 outputs: grip, tip, cl, cos2, sin2) and Stage-2 sparse HGB refiners."""
        X = np.asarray(X_rows, dtype=np.float32)
        Y = np.asarray(Y_targets, dtype=np.float32)  # [N, 5]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X).astype(np.float32)
        self.scaler_mean = scaler.mean_.astype(np.float32)
        self.scaler_scale = np.maximum(scaler.scale_, 1e-6).astype(np.float32)

        # Oversample rare positives explicitly in dense MLP training if sample_weights given
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float64)
            w = np.maximum(w, 1e-3)
            probs = w / w.sum()
            rng = np.random.default_rng(42)
            n_draw = min(len(Xs), 160_000)
            idx = rng.choice(len(Xs), size=n_draw, replace=True, p=probs)
            X_fit, Y_fit = Xs[idx], Y[idx]
        else:
            X_fit, Y_fit = Xs, Y

        mlp = MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            solver="adam",
            alpha=1e-2,
            batch_size=512,
            learning_rate_init=1.5e-3,
            max_iter=85,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=42,
        )
        mlp.fit(X_fit, Y_fit)
        self.mlp_coefs = [c.astype(np.float32) for c in mlp.coefs_]
        self.mlp_intercepts = [b.astype(np.float32) for b in mlp.intercepts_]

        # Stage-2 Sparse HGBR refiners on hard/candidate pixels
        Xr = X if sparse_X is None else np.asarray(sparse_X, dtype=np.float32)
        Yr = Y if sparse_Y is None else np.asarray(sparse_Y, dtype=np.float32)
        wr = sparse_w if sparse_w is not None else (sample_weights if sparse_X is None else None)

        if self.mode != "2_centerline_only":
            self.hgb_grip = HistGradientBoostingRegressor(max_iter=55, max_depth=5, learning_rate=0.08, min_samples_leaf=24, l2_regularization=1.0, random_state=42)
            self.hgb_grip.fit(Xr, Yr[:, 0], sample_weight=wr)
            self.hgb_tip = HistGradientBoostingRegressor(max_iter=55, max_depth=5, learning_rate=0.08, min_samples_leaf=24, l2_regularization=1.0, random_state=43)
            self.hgb_tip.fit(Xr, Yr[:, 1], sample_weight=wr)
        else:
            self.hgb_grip = None
            self.hgb_tip = None

        if self.mode != "1_keypoints_only":
            self.hgb_cl = HistGradientBoostingRegressor(max_iter=65, max_depth=5, learning_rate=0.08, min_samples_leaf=24, l2_regularization=1.0, random_state=44)
            self.hgb_cl.fit(Xr, Yr[:, 2], sample_weight=wr)
        else:
            self.hgb_cl = None

        self.is_fitted = True
        return self

    def _mlp_forward(self, X_flat: np.ndarray) -> np.ndarray:
        assert self.scaler_mean is not None and self.scaler_scale is not None
        h = (X_flat - self.scaler_mean[None, :]) / self.scaler_scale[None, :]
        for li in range(len(self.mlp_coefs) - 1):
            h = h @ self.mlp_coefs[li] + self.mlp_intercepts[li][None, :]
            np.maximum(h, 0.0, out=h)
        out = h @ self.mlp_coefs[-1] + self.mlp_intercepts[-1][None, :]
        return out.astype(np.float32)

    def predict_heatmaps_112(self, features_112: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (grip_hm, tip_hm, cl_hm, cos2_map, sin2_map, ctx_112) on 112x112."""
        ctx112 = extract_pixel_context_112(features_112)
        if not self.is_fitted:
            z = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            return z, z, z, z, z, ctx112

        X_flat = ctx112.reshape(-1, BASE_CTX_DIM)
        raw = self._mlp_forward(X_flat)
        grip_1 = np.clip(raw[:, 0], 0.0, 1.0).reshape(GRID_SIZE, GRID_SIZE)
        tip_1 = np.clip(raw[:, 1], 0.0, 1.0).reshape(GRID_SIZE, GRID_SIZE)
        cl_1 = np.clip(raw[:, 2], 0.0, 1.0).reshape(GRID_SIZE, GRID_SIZE)
        cos2_m = np.clip(raw[:, 3], -1.0, 1.0).reshape(GRID_SIZE, GRID_SIZE)
        sin2_m = np.clip(raw[:, 4], -1.0, 1.0).reshape(GRID_SIZE, GRID_SIZE)

        # Stage-2 sparse boosted refinement only on top-K candidate cells (~2 ms on CPU)
        ridge = features_112[:, :, 6]
        white = features_112[:, :, 2]
        sal = np.maximum.reduce([grip_1, tip_1, cl_1, 0.45 * ridge + 0.25 * white]).ravel()
        k = min(self.sparse_refine_k, len(sal))
        cand_idx = np.argpartition(sal, -k)[-k:]
        X_cand = X_flat[cand_idx]

        grip_flat = grip_1.ravel().copy()
        tip_flat = tip_1.ravel().copy()
        cl_flat = cl_1.ravel().copy()
        if self.hgb_grip is not None:
            g2 = np.clip(self.hgb_grip.predict(X_cand), 0.0, 1.0).astype(np.float32)
            grip_flat[cand_idx] = 0.55 * g2 + 0.45 * grip_flat[cand_idx]
        if self.hgb_tip is not None:
            t2 = np.clip(self.hgb_tip.predict(X_cand), 0.0, 1.0).astype(np.float32)
            tip_flat[cand_idx] = 0.55 * t2 + 0.45 * tip_flat[cand_idx]
        if self.hgb_cl is not None:
            c2 = np.clip(self.hgb_cl.predict(X_cand), 0.0, 1.0).astype(np.float32)
            cl_flat[cand_idx] = 0.60 * c2 + 0.40 * cl_flat[cand_idx]

        valid = (features_112[:, :, 15] > 0.2).astype(np.float32)
        grip_hm = cv2.GaussianBlur(grip_flat.reshape(GRID_SIZE, GRID_SIZE), (3, 3), 0.7) * valid
        tip_hm = cv2.GaussianBlur(tip_flat.reshape(GRID_SIZE, GRID_SIZE), (3, 3), 0.7) * valid
        cl_hm = (cl_flat.reshape(GRID_SIZE, GRID_SIZE)) * valid

        if self.mode == "1_keypoints_only":
            cl_hm = np.zeros_like(cl_hm)
        elif self.mode == "2_centerline_only":
            grip_hm = np.zeros_like(grip_hm)
            tip_hm = np.zeros_like(tip_hm)

        return grip_hm.astype(np.float32), tip_hm.astype(np.float32), cl_hm.astype(np.float32), cos2_m, sin2_m, ctx112

    def _trace_centerline_11(
        self,
        gx1: float,
        gy1: float,
        gx2: float,
        gy2: float,
        cl_hm: np.ndarray,
        ridge: np.ndarray,
    ) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
        """Sample 11 points between endpoints snapping gently to orthogonal ridge peaks."""
        L = max(1e-4, math.hypot(gx2 - gx1, gy2 - gy1))
        ux, uy = (gx2 - gx1) / L, (gy2 - gy1) / L
        nx, ny = -uy, ux
        guide = np.maximum(cl_hm, 0.55 * ridge)
        pts_g: list[tuple[float, float]] = []
        cl_vals = []
        vis_vals = []

        ts = np.linspace(0.0, 1.0, 11)
        for ti, t in enumerate(ts):
            bx = gx1 + float(t) * (gx2 - gx1)
            by = gy1 + float(t) * (gy2 - gy1)
            best_x, best_y = bx, by
            if 0 < ti < 10:
                best_s = -1.0
                for off in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    sx = float(np.clip(bx + off * nx, 0.0, GRID_SIZE - 1.0))
                    sy = float(np.clip(by + off * ny, 0.0, GRID_SIZE - 1.0))
                    sc = float(guide[int(round(sy)), int(round(sx))]) - 0.04 * abs(off)
                    if sc > best_s:
                        best_s = sc
                        best_x, best_y = sx, sy
            ix, iy = int(round(np.clip(best_x, 0, GRID_SIZE - 1))), int(round(np.clip(best_y, 0, GRID_SIZE - 1)))
            pts_g.append((best_x, best_y))
            cl_vals.append(float(cl_hm[iy, ix]))
            vis_vals.append(float(max(cl_hm[iy, ix], 0.75 * ridge[iy, ix])))
        return pts_g, np.array(cl_vals, dtype=np.float32), np.array(vis_vals, dtype=np.float32)

    def predict(self, bundle: SpatialFeatureBundle) -> LocalizationPrediction:
        t0 = time.perf_counter()
        f112 = bundle.features_112
        tfm = bundle.transform
        grip_hm, tip_hm, cl_hm, cos2_m, sin2_m, _ = self.predict_heatmaps_112(f112)

        white = f112[:, :, 2]
        red = f112[:, :, 3]
        ridge = f112[:, :, 6]
        wood = f112[:, :, 12]
        valid = f112[:, :, 15]
        umb = f112[:, :, 35]

        cands: list[tuple[float, float, float, float, bool]] = []  # (gx1, gy1, gx2, gy2, bridge_flag)
        if self.mode != "2_centerline_only":
            g_peaks = _nms_peaks_2d(grip_hm, top_k=5, min_dist=6, min_val=0.06)
            t_peaks = _nms_peaks_2d(tip_hm, top_k=5, min_dist=6, min_val=0.06)
            for gx, gy, _ in g_peaks:
                for tx, ty, _ in t_peaks:
                    if math.hypot(tx - gx, ty - gy) >= 10.0:
                        if gy <= ty:
                            cands.append((gx, gy, tx, ty, False))
                        else:
                            cands.append((tx, ty, gx, gy, False))

        if self.mode != "1_keypoints_only":
            segs = _extract_ridge_segments_112(cl_hm, valid, thr=0.20)
            for idx_s, (sx1, sy1, sx2, sy2, _) in enumerate(segs):
                if math.hypot(sx2 - sx1, sy2 - sy1) >= 10.0:
                    cands.append((sx1, sy1, sx2, sy2, idx_s >= 4))

        # Deduplicate near-identical candidates & evaluate structural chord scores
        scored_props: list[dict] = []
        seen_cands: list[tuple[float, float, float, float]] = []
        for gx1, gy1, gx2, gy2, br in cands:
            gx1 = float(np.clip(gx1, 0.0, GRID_SIZE - 1.0))
            gy1 = float(np.clip(gy1, 0.0, GRID_SIZE - 1.0))
            gx2 = float(np.clip(gx2, 0.0, GRID_SIZE - 1.0))
            gy2 = float(np.clip(gy2, 0.0, GRID_SIZE - 1.0))
            if any(
                math.hypot(gx1 - sx1, gy1 - sy1) + math.hypot(gx2 - sx2, gy2 - sy2) < 6.5
                for sx1, sy1, sx2, sy2 in seen_cands
            ):
                continue
            seen_cands.append((gx1, gy1, gx2, gy2))

            L_g = math.hypot(gx2 - gx1, gy2 - gy1)
            if L_g < 9.0:
                continue
            th = math.atan2(gy2 - gy1, gx2 - gx1)
            c2, s2 = math.cos(2.0 * th), math.sin(2.0 * th)

            ts = np.linspace(0.0, 1.0, 21)
            xs = np.clip(np.round(gx1 + ts * (gx2 - gx1)).astype(int), 0, GRID_SIZE - 1)
            ys = np.clip(np.round(gy1 + ts * (gy2 - gy1)).astype(int), 0, GRID_SIZE - 1)

            if float(np.mean(valid[ys, xs])) < 0.70:
                continue

            cl_s = cl_hm[ys, xs]
            wh_s = white[ys, xs]
            rd_s = ridge[ys, xs]
            wd_s = wood[ys, xs]
            um_s = umb[ys, xs]
            ori_align = np.maximum(0.0, cos2_m[ys, xs] * c2 + sin2_m[ys, xs] * s2)

            ix1, iy1 = int(np.clip(round(gx1), 0, GRID_SIZE - 1)), int(np.clip(round(gy1), 0, GRID_SIZE - 1))
            ix2, iy2 = int(np.clip(round(gx2), 0, GRID_SIZE - 1)), int(np.clip(round(gy2), 0, GRID_SIZE - 1))
            g_sc = float(np.max(grip_hm[max(0, iy1 - 2):min(GRID_SIZE, iy1 + 3), max(0, ix1 - 2):min(GRID_SIZE, ix1 + 3)]))
            t_sc = float(np.max(tip_hm[max(0, iy2 - 2):min(GRID_SIZE, iy2 + 3), max(0, ix2 - 2):min(GRID_SIZE, ix2 + 3)]))
            cl_mean = float(np.mean(cl_s))
            cl_cov = float(np.mean(cl_s >= 0.20))
            wh_cov = float(np.mean((wh_s >= 0.30) | (rd_s >= 0.25)))
            red_bot = float(np.max(red[ys[-8:], xs[-8:]]))
            wood_pen = float(np.mean(wd_s))
            umb_pen = float(np.mean(um_s))
            ori_mean = float(np.mean(ori_align))

            if self.mode == "1_keypoints_only":
                score = (
                    0.46 * (0.5 * (g_sc + t_sc))
                    + 0.24 * wh_cov
                    + 0.14 * float(np.mean(wh_s))
                    + 0.12 * ori_mean
                    + 0.08 * red_bot
                    - 0.12 * wood_pen
                    - 0.14 * umb_pen
                )
            elif self.mode == "2_centerline_only":
                score = (
                    0.50 * cl_mean
                    + 0.26 * cl_cov
                    + 0.12 * ori_mean
                    + 0.10 * min(1.0, L_g / 48.0)
                    + 0.06 * wh_cov
                    - 0.10 * wood_pen
                )
            else:
                score = (
                    0.34 * cl_mean
                    + 0.20 * cl_cov
                    + 0.22 * (0.5 * (g_sc + t_sc))
                    + 0.12 * ori_mean
                    + 0.10 * wh_cov
                    + 0.07 * min(1.0, L_g / 52.0)
                    + 0.05 * red_bot
                    - 0.10 * wood_pen
                    - 0.12 * umb_pen
                )

            scored_props.append({
                "grid_ep": [gx1, gy1, gx2, gy2],
                "score": float(np.clip(score, 0.0, 1.0)),
                "g_sc": g_sc,
                "t_sc": t_sc,
                "cl_mean": cl_mean,
                "cl_cov": cl_cov,
                "bridge_used": br,
            })

        if not scored_props:
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
            ox, oy = tfm.orig_w * 0.5, tfm.orig_h * 0.5
            ep = [round(ox - 10, 2), round(oy - 20, 2), round(ox + 10, 2), round(oy + 20, 2)]
            return LocalizationPrediction(
                detected=False,
                endpoints_xyxy=ep,
                centerline_points_xy=sample_centerline_points(ep, 11),
                obb_cxcywh_angle=obb_from_endpoints(ep[0], ep[1], ep[2], ep[3], 10.0)[0],
                bbox_xywh=(int(ox - 12), int(oy - 22), 24, 44),
                orientation_deg=63.4,
                length_px=44.7,
                length_ratio=round(44.7 / max(1, min(tfm.orig_h, tfm.orig_w)), 4),
                visible_ratio=0.0,
                occlusion_gap_ratio=1.0,
                occluded=False,
                confidence=0.0,
                kp_confidence=0.0,
                cl_confidence=0.0,
                grip_score=0.0,
                tip_score=0.0,
                bridge_used=False,
                grip_hm=grip_hm,
                tip_hm=tip_hm,
                cl_hm=cl_hm,
                cos2_map=cos2_m,
                sin2_map=sin2_m,
                proposals=[],
                decode_ms=ms,
            )

        scored_props.sort(key=lambda d: d["score"], reverse=True)
        best = scored_props[0]
        gx1, gy1, gx2, gy2 = best["grid_ep"]
        pts_g, cl_vals, vis_vals = self._trace_centerline_11(gx1, gy1, gx2, gy2, cl_hm, ridge)

        # Compute largest contiguous dark gap in interior [1..9] to detect partial occlusion
        dark = vis_vals < 0.22
        max_run, cur_run = 0, 0
        for d in dark:
            cur_run = cur_run + 1 if d else 0
            max_run = max(max_run, cur_run)
        gap_ratio = float(max_run / len(dark))
        vis_ratio = float(np.mean(~dark))
        occluded = bool(gap_ratio >= 0.14 and vis_ratio >= 0.36)

        ox1, oy1 = tfm.grid_to_orig(gx1, gy1)
        ox2, oy2 = tfm.grid_to_orig(gx2, gy2)
        ox1 = float(np.clip(ox1, 0.0, tfm.orig_w - 1.0))
        oy1 = float(np.clip(oy1, 0.0, tfm.orig_h - 1.0))
        ox2 = float(np.clip(ox2, 0.0, tfm.orig_w - 1.0))
        oy2 = float(np.clip(oy2, 0.0, tfm.orig_h - 1.0))
        ep_orig = [round(ox1, 2), round(oy1, 2), round(ox2, 2), round(oy2, 2)]

        cl_orig = [
            [
                round(float(np.clip(tfm.grid_to_orig(px, py)[0], 0.0, tfm.orig_w - 1.0)), 2),
                round(float(np.clip(tfm.grid_to_orig(px, py)[1], 0.0, tfm.orig_h - 1.0)), 2),
            ]
            for px, py in pts_g
        ]
        L_px = float(math.hypot(ox2 - ox1, oy2 - oy1))
        obb, _ = obb_from_endpoints(ep_orig[0], ep_orig[1], ep_orig[2], ep_orig[3], thickness=max(8.0, 0.028 * L_px))
        bx1 = max(0, int(math.floor(min(ox1, ox2) - 10)))
        by1 = max(0, int(math.floor(min(oy1, oy2) - 10)))
        bx2 = min(tfm.orig_w, int(math.ceil(max(ox1, ox2) + 10)))
        by2 = min(tfm.orig_h, int(math.ceil(max(oy1, oy2) + 10)))

        top_props_orig = []
        for p in scored_props[:4]:
            px1, py1 = tfm.grid_to_orig(p["grid_ep"][0], p["grid_ep"][1])
            px2, py2 = tfm.grid_to_orig(p["grid_ep"][2], p["grid_ep"][3])
            top_props_orig.append({
                "endpoints_xyxy": [round(px1, 1), round(py1, 1), round(px2, 1), round(py2, 1)],
                "grid_ep": [round(float(v), 2) for v in p["grid_ep"]],
                "score": round(p["score"], 4),
            })

        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return LocalizationPrediction(
            detected=bool(best["score"] >= 0.14),
            endpoints_xyxy=ep_orig,
            centerline_points_xy=cl_orig,
            obb_cxcywh_angle=obb,
            bbox_xywh=(bx1, by1, max(1, bx2 - bx1), max(1, by2 - by1)),
            orientation_deg=round(_axial_deg(ox2 - ox1, oy2 - oy1), 2),
            length_px=round(L_px, 2),
            length_ratio=round(L_px / max(1, min(tfm.orig_h, tfm.orig_w)), 4),
            visible_ratio=round(vis_ratio, 4),
            occlusion_gap_ratio=round(gap_ratio, 4),
            occluded=occluded,
            confidence=round(best["score"], 4),
            kp_confidence=round(0.5 * (best["g_sc"] + best["t_sc"]), 4),
            cl_confidence=round(best["cl_mean"], 4),
            grip_score=round(best["g_sc"], 4),
            tip_score=round(best["t_sc"], 4),
            bridge_used=bool(best["bridge_used"]),
            grip_hm=grip_hm,
            tip_hm=tip_hm,
            cl_hm=cl_hm,
            cos2_map=cos2_m,
            sin2_map=sin2_m,
            proposals=top_props_orig,
            decode_ms=ms,
        )

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": self.mode,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "mlp_coefs": self.mlp_coefs,
            "mlp_intercepts": self.mlp_intercepts,
            "hgb_grip": self.hgb_grip,
            "hgb_tip": self.hgb_tip,
            "hgb_cl": self.hgb_cl,
            "sparse_refine_k": self.sparse_refine_k,
            "is_fitted": self.is_fitted,
            "train_summary": self.train_summary,
        }
        joblib.dump(payload, path, compress=3)
        return compute_sha256(path)

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> "SupervisedKeypointCenterlineDetector":
        if expected_sha256:
            actual = compute_sha256(path).lower()
            if actual != expected_sha256.lower():
                raise RuntimeError(f"SHA-256 mismatch for {path}")
        payload = joblib.load(path)
        obj = cls(mode=payload["mode"])
        obj.scaler_mean = payload["scaler_mean"]
        obj.scaler_scale = payload["scaler_scale"]
        obj.mlp_coefs = payload["mlp_coefs"]
        obj.mlp_intercepts = payload["mlp_intercepts"]
        obj.hgb_grip = payload["hgb_grip"]
        obj.hgb_tip = payload["hgb_tip"]
        obj.hgb_cl = payload["hgb_cl"]
        obj.sparse_refine_k = int(payload.get("sparse_refine_k", 480))
        obj.is_fitted = bool(payload["is_fitted"])
        obj.train_summary = payload.get("train_summary", {})
        return obj
