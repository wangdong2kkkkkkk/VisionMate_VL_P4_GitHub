"""Supervised White Cane Shaft Segmentation (Methods B1 Standalone & B2 Keypoint-Guided).

Predicts crisp pixel-level cane masks (`mask_rle`) and physical shaft attributes:
  - visible_shaft_ratio
  - white_purity_on_mask
  - red_tip_presence
  - width_regularity
  - edge_straightness / mean_perp_err_norm
  - bg_contamination_hint & foreground area ratio
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .common_utils import compute_sha256, encode_mask_rle, obb_from_endpoints
from .keypoint_detector import BASE_CTX_DIM, LocalizationPrediction, extract_pixel_context_112
from .spatial_features import GRID_SIZE, WORK_RIDGE_SIZE, SpatialFeatureBundle

KP_GUIDANCE_DIM = 5


def build_kp_guidance_112(
    loc_pred: LocalizationPrediction | None,
    bundle: SpatialFeatureBundle,
) -> np.ndarray:
    """Build 5 keypoint-guidance channels [112, 112, 5] for B2 segmentation."""
    if loc_pred is None or not loc_pred.detected:
        return np.zeros((GRID_SIZE, GRID_SIZE, KP_GUIDANCE_DIM), dtype=np.float32)

    tfm = bundle.transform
    gx1, gy1 = tfm.orig_to_grid(loc_pred.endpoints_xyxy[0], loc_pred.endpoints_xyxy[1])
    gx2, gy2 = tfm.orig_to_grid(loc_pred.endpoints_xyxy[2], loc_pred.endpoints_xyxy[3])

    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE].astype(np.float32)
    vx, vy = gx2 - gx1, gy2 - gy1
    seg_len2 = max(1e-4, vx * vx + vy * vy)
    raw_t = ((xx - gx1) * vx + (yy - gy1) * vy) / seg_len2
    clamped_t = np.clip(raw_t, 0.0, 1.0)
    px = gx1 + clamped_t * vx
    py = gy1 + clamped_t * vy
    dist2 = (xx - px) ** 2 + (yy - py) ** 2

    dist_gauss = np.exp(-dist2 / (2.0 * (2.4 ** 2))).astype(np.float32)
    # Smooth end gate decaying past t < 0 or t > 1
    end_over = np.maximum(0.0, np.maximum(-raw_t, raw_t - 1.0))
    inside_gate = np.exp(-(end_over ** 2) / (2.0 * (0.09 ** 2))).astype(np.float32)

    th = math.atan2(vy, vx)
    c2, s2 = math.cos(2.0 * th), math.sin(2.0 * th)
    f = bundle.features_112
    orient_agree = np.clip(0.5 * (1.0 + f[:, :, 9] * c2 + f[:, :, 10] * s2), 0.0, 1.0).astype(np.float32)
    ep_focus = np.clip(np.maximum(loc_pred.grip_hm, loc_pred.tip_hm), 0.0, 1.0).astype(np.float32)

    return np.stack(
        [loc_pred.cl_hm.astype(np.float32), dist_gauss, inside_gate, orient_agree, ep_focus],
        axis=-1,
    ).astype(np.float32)


def _fit_line_from_mask_448(mask448: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask448 > 0)
    if len(ys) < 4:
        return 220.0, 180.0, 228.0, 260.0
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, int(np.argmax(evals))]
    proj = (pts - mean) @ u
    p1 = mean + float(proj.min()) * u
    p2 = mean + float(proj.max()) * u
    if p1[1] > p2[1]:
        p1, p2 = p2, p1
    return float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])


def _compute_mask_physical_metrics(
    mask448: np.ndarray,
    prob448: np.ndarray,
    bundle: SpatialFeatureBundle,
    endpoints_orig: list[float] | None,
) -> dict:
    """Measure physical white-cane structure directly on the high-res 448x448 segmented mask."""
    tfm = bundle.transform
    hi = bundle.hi_cache
    fg = mask448 > 0
    fg_count = int(np.sum(fg))
    valid_px = max(1, int(np.sum(hi["valid_448"] > 0.5)))

    if fg_count < 8:
        return {
            "visible_shaft_ratio": 0.0,
            "white_purity_on_mask": 0.0,
            "red_tip_presence": 0.0,
            "width_regularity": 0.0,
            "edge_straightness": 0.0,
            "mean_perp_err_norm": 0.5,
            "fg_area_ratio": 0.0,
            "wood_on_mask": 0.0,
            "dark_on_mask": 0.0,
            "mask_prob_mean": 0.0,
            "median_width_px448": 0.0,
        }

    sc4 = float(WORK_RIDGE_SIZE) / float(tfm.side)
    if endpoints_orig is not None:
        gx1 = (endpoints_orig[0] + tfm.pad_left) * sc4
        gy1 = (endpoints_orig[1] + tfm.pad_top) * sc4
        gx2 = (endpoints_orig[2] + tfm.pad_left) * sc4
        gy2 = (endpoints_orig[3] + tfm.pad_top) * sc4
    else:
        gx1, gy1, gx2, gy2 = _fit_line_from_mask_448(mask448)

    if gy1 > gy2:
        gx1, gy1, gx2, gy2 = gx2, gy2, gx1, gy1

    vx, vy = gx2 - gx1, gy2 - gy1
    L4 = max(12.0, math.hypot(vx, vy))
    ux, uy = vx / L4, vy / L4
    nx, ny = -uy, ux

    # Along-shaft visible coverage & width consistency across 15 stations
    vis_hits = 0
    widths = []
    red_tip_vals = []
    for si in range(15):
        t = si / 14.0
        cx = gx1 + t * vx
        cy = gy1 + t * vy
        # Sample transverse cross-section +-18 px at 448 scale
        offs = np.arange(-18, 19, dtype=np.float32)
        sxs = np.clip(np.round(cx + offs * nx).astype(int), 0, WORK_RIDGE_SIZE - 1)
        sys = np.clip(np.round(cy + offs * ny).astype(int), 0, WORK_RIDGE_SIZE - 1)
        hit = fg[sys, sxs]
        core = np.any(hit[14:23])  # within +-4 px of predicted centerline
        if core:
            vis_hits += 1
        w_station = int(np.sum(hit))
        if w_station > 0 and 1 <= si <= 13:
            widths.append(float(w_station))
        if t >= 0.62:
            rvals = hi["red_tip_448"][sys, sxs]
            if np.any(hit):
                red_tip_vals.append(float(np.max(rvals[hit])))
            else:
                red_tip_vals.append(float(np.max(rvals)) * 0.65)

    visible_shaft_ratio = float(vis_hits / 15.0)
    red_tip_presence = float(np.max(red_tip_vals)) if red_tip_vals else 0.0

    # Photometric purity on mask (using top 75% brightest FG pixels to tolerate red tip / grip handle)
    wh_vals = hi["white_achrom_448"][fg]
    wd_vals = hi["wood_cue_448"][fg]
    dk_vals = hi["dark_cue_448"][fg]
    pr_vals = prob448[fg]
    k_top = max(1, int(math.ceil(0.75 * len(wh_vals))))
    wh_top = np.sort(wh_vals)[-k_top:]
    white_purity = float(np.mean(wh_top))
    wood_on_mask = float(np.mean(wd_vals))
    dark_on_mask = float(np.mean(dk_vals))
    prob_mean = float(np.mean(pr_vals))

    if len(widths) >= 3:
        w_arr = np.asarray(widths, dtype=np.float32)
        w_mean = float(np.mean(w_arr))
        w_med = float(np.median(w_arr))
        cv = float(np.std(w_arr) / max(1.0, w_mean))
        thin_bonus = 1.0 if w_med <= 11.0 else max(0.25, 1.0 - (w_med - 11.0) / 14.0)
        width_regularity = float(np.clip((1.0 - cv) * thin_bonus, 0.0, 1.0))
    else:
        w_med = 0.0
        width_regularity = 0.0

    # Orthogonal distance of masked FG pixels to predicted segment
    fys, fxs = np.nonzero(fg)
    px_rel = fxs.astype(np.float32) - gx1
    py_rel = fys.astype(np.float32) - gy1
    t_proj = np.clip((px_rel * ux + py_rel * uy) / L4, 0.0, 1.0) * L4
    perp = np.hypot(px_rel - t_proj * ux, py_rel - t_proj * uy)
    mean_perp = float(np.mean(perp))
    edge_straightness = float(np.clip(1.0 - mean_perp / 9.0, 0.0, 1.0))

    return {
        "visible_shaft_ratio": round(visible_shaft_ratio, 4),
        "white_purity_on_mask": round(white_purity, 4),
        "red_tip_presence": round(red_tip_presence, 4),
        "width_regularity": round(width_regularity, 4),
        "edge_straightness": round(edge_straightness, 4),
        "mean_perp_err_norm": round(mean_perp / float(WORK_RIDGE_SIZE), 4),
        "fg_area_ratio": round(float(fg_count / valid_px), 5),
        "wood_on_mask": round(wood_on_mask, 4),
        "dark_on_mask": round(dark_on_mask, 4),
        "mask_prob_mean": round(prob_mean, 4),
        "median_width_px448": round(w_med, 2),
    }


@dataclass
class SegmentationPrediction:
    mode: str
    mask_rle: dict
    mask_orig: np.ndarray = field(repr=False)
    prob_112: np.ndarray = field(repr=False)
    endpoints_xyxy: list[float]
    obb_cxcywh_angle: tuple[float, float, float, float, float]
    visible_shaft_ratio: float
    white_purity_on_mask: float
    red_tip_presence: float
    width_regularity: float
    edge_straightness: float
    mean_perp_err_norm: float
    fg_area_ratio: float
    wood_on_mask: float
    dark_on_mask: float
    mask_prob_mean: float
    median_width_px448: float
    confidence: float
    seg_ms: float = 0.0


class SupervisedShaftSegmenter:
    """Supervised CPU-fast pixel segmenter in B1 (standalone) or B2 (keypoint-guided) mode."""

    def __init__(self, mode: str = "kp_guided") -> None:
        if mode not in ("standalone", "kp_guided"):
            raise ValueError(f"Unsupported segmenter mode: {mode}")
        self.mode = mode
        self.in_dim = BASE_CTX_DIM + (KP_GUIDANCE_DIM if mode == "kp_guided" else 0)
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        self.mlp_coefs: list[np.ndarray] = []
        self.mlp_intercepts: list[np.ndarray] = []
        self.hgb_clf: HistGradientBoostingClassifier | None = None
        self.mask_thr: float = 0.42
        self.sparse_refine_k: int = 160
        self.is_fitted: bool = False
        self.train_summary: dict = {}

    def _build_input_112(self, bundle: SpatialFeatureBundle, loc_pred: LocalizationPrediction | None = None) -> np.ndarray:
        ctx = extract_pixel_context_112(bundle.features_112)
        if self.mode == "kp_guided":
            kp_g = build_kp_guidance_112(loc_pred, bundle)
            return np.concatenate([ctx, kp_g], axis=-1).astype(np.float32)
        return ctx

    def fit(
        self,
        X_rows: np.ndarray,
        y_bin: np.ndarray,
        sample_weights: np.ndarray | None = None,
        sparse_X: np.ndarray | None = None,
        sparse_y: np.ndarray | None = None,
        sparse_w: np.ndarray | None = None,
    ) -> "SupervisedShaftSegmenter":
        X = np.asarray(X_rows, dtype=np.float32)
        y = (np.asarray(y_bin) > 0).astype(np.float32)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X).astype(np.float32)
        self.scaler_mean = scaler.mean_.astype(np.float32)
        self.scaler_scale = np.maximum(scaler.scale_, 1e-6).astype(np.float32)

        if sample_weights is not None:
            w = np.maximum(np.asarray(sample_weights, dtype=np.float64), 1e-3)
            probs = w / w.sum()
            rng = np.random.default_rng(101 if self.mode == "standalone" else 202)
            idx = rng.choice(len(Xs), size=min(len(Xs), 150_000), replace=True, p=probs)
            X_fit, y_fit = Xs[idx], y[idx]
        else:
            X_fit, y_fit = Xs, y

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-2,
            batch_size=512,
            learning_rate_init=1.5e-3,
            max_iter=75,
            early_stopping=True,
            n_iter_no_change=7,
            random_state=101 if self.mode == "standalone" else 202,
        )
        mlp.fit(X_fit, y_fit)
        self.mlp_coefs = [c.astype(np.float32) for c in mlp.coefs_]
        self.mlp_intercepts = [b.astype(np.float32) for b in mlp.intercepts_]

        Xr = X if sparse_X is None else np.asarray(sparse_X, dtype=np.float32)
        yr = y.astype(int) if sparse_y is None else (np.asarray(sparse_y) > 0).astype(int)
        wr = sparse_w if sparse_w is not None else (sample_weights if sparse_X is None else None)
        self.hgb_clf = HistGradientBoostingClassifier(
            max_iter=65,
            max_depth=5,
            learning_rate=0.08,
            min_samples_leaf=24,
            l2_regularization=1.0,
            random_state=101 if self.mode == "standalone" else 202,
        )
        self.hgb_clf.fit(Xr, yr, sample_weight=wr)
        self.is_fitted = True
        return self

    def _mlp_forward(self, X_flat: np.ndarray) -> np.ndarray:
        assert self.scaler_mean is not None and self.scaler_scale is not None
        h = (X_flat - self.scaler_mean[None, :]) / self.scaler_scale[None, :]
        for li in range(len(self.mlp_coefs) - 1):
            h = h @ self.mlp_coefs[li] + self.mlp_intercepts[li][None, :]
            np.maximum(h, 0.0, out=h)
        out = h @ self.mlp_coefs[-1] + self.mlp_intercepts[-1][None, :]
        return np.clip(out.ravel(), 0.0, 1.0).astype(np.float32)

    def predict(
        self,
        bundle: SpatialFeatureBundle,
        loc_pred: LocalizationPrediction | None = None,
        mask_thr: float | None = None,
    ) -> SegmentationPrediction:
        t0 = time.perf_counter()
        tfm = bundle.transform
        hi = bundle.hi_cache
        thr = self.mask_thr if mask_thr is None else float(mask_thr)

        if not self.is_fitted:
            prob_112 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            mask_orig = np.zeros((tfm.orig_h, tfm.orig_w), dtype=np.uint8)
            ep = loc_pred.endpoints_xyxy if loc_pred is not None else [0.0, 0.0, 10.0, 10.0]
            return SegmentationPrediction(
                mode=self.mode,
                mask_rle=encode_mask_rle(mask_orig),
                mask_orig=mask_orig,
                prob_112=prob_112,
                endpoints_xyxy=ep,
                obb_cxcywh_angle=obb_from_endpoints(ep[0], ep[1], ep[2], ep[3], 10.0)[0],
                visible_shaft_ratio=0.0,
                white_purity_on_mask=0.0,
                red_tip_presence=0.0,
                width_regularity=0.0,
                edge_straightness=0.0,
                mean_perp_err_norm=0.5,
                fg_area_ratio=0.0,
                wood_on_mask=0.0,
                dark_on_mask=0.0,
                mask_prob_mean=0.0,
                median_width_px448=0.0,
                confidence=0.0,
                seg_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        inp112 = self._build_input_112(bundle, loc_pred)
        X_flat = inp112.reshape(-1, self.in_dim)
        p1 = self._mlp_forward(X_flat)

        # Sparse top-K HGBC refiner on salient candidates
        sal = np.maximum(p1, 0.50 * bundle.features_112[:, :, 6].ravel())
        if self.mode == "kp_guided" and loc_pred is not None and loc_pred.detected:
            sal = np.maximum(sal, 0.65 * loc_pred.cl_hm.ravel())
        k = min(self.sparse_refine_k, len(sal))
        cand_idx = np.argpartition(sal, -k)[-k:]
        if self.hgb_clf is not None:
            p2 = self.hgb_clf.predict_proba(X_flat[cand_idx])[:, 1].astype(np.float32)
            p1_ref = p1.copy()
            p1_ref[cand_idx] = 0.65 * p2 + 0.35 * p1_ref[cand_idx]
        else:
            p1_ref = p1

        valid112 = (bundle.features_112[:, :, 15] > 0.2).astype(np.float32)
        prob_112 = np.clip(p1_ref.reshape(GRID_SIZE, GRID_SIZE) * valid112, 0.0, 1.0).astype(np.float32)

        # Upsample to 448x448 and sharpen with high-res photometric ridge boundary modulation
        prob448 = cv2.resize(prob_112, (WORK_RIDGE_SIZE, WORK_RIDGE_SIZE), interpolation=cv2.INTER_LINEAR)
        phys448 = np.clip(
            0.55 * hi["steered_ridge_448"] + 0.45 * hi["white_achrom_448"] + 0.45 * hi["red_tip_448"] - 0.30 * hi["wood_cue_448"],
            0.0,
            1.2,
        )
        refined448 = prob448 * (0.58 + 0.50 * phys448)

        ep_use = None
        if self.mode == "kp_guided" and loc_pred is not None and loc_pred.detected:
            ep_use = loc_pred.endpoints_xyxy
            # Build crisp high-res distance corridor along predicted centerline (~10px sigma at 448 scale)
            sc4 = float(WORK_RIDGE_SIZE) / float(tfm.side)
            gx1 = (ep_use[0] + tfm.pad_left) * sc4
            gy1 = (ep_use[1] + tfm.pad_top) * sc4
            gx2 = (ep_use[2] + tfm.pad_left) * sc4
            gy2 = (ep_use[3] + tfm.pad_top) * sc4
            yy4, xx4 = np.mgrid[0:WORK_RIDGE_SIZE, 0:WORK_RIDGE_SIZE].astype(np.float32)
            vx, vy = gx2 - gx1, gy2 - gy1
            seg2 = max(1e-4, vx * vx + vy * vy)
            t4 = np.clip(((xx4 - gx1) * vx + (yy4 - gy1) * vy) / seg2, 0.0, 1.0)
            dist2_4 = (xx4 - (gx1 + t4 * vx)) ** 2 + (yy4 - (gy1 + t4 * vy)) ** 2
            tube_gate = np.exp(-dist2_4 / (2.0 * (9.5 ** 2))).astype(np.float32)
            refined448 = refined448 * (0.18 + 0.92 * tube_gate)

        refined448 *= hi["valid_448"]
        bw448 = (refined448 >= thr).astype(np.uint8)
        bw448 = cv2.morphologyEx(bw448, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Prune tiny noise specks while preserving multi-part occluded cane fragments
        n_cc, lbls, stats, _ = cv2.connectedComponentsWithStats(bw448, connectivity=8)
        clean448 = np.zeros_like(bw448)
        for cid in range(1, n_cc):
            if int(stats[cid, cv2.CC_STAT_AREA]) >= 14:
                clean448[lbls == cid] = 255

        if ep_use is None:
            fx1, fy1, fx2, fy2 = _fit_line_from_mask_448(clean448)
            sc4 = float(WORK_RIDGE_SIZE) / float(tfm.side)
            ox1 = float(np.clip(fx1 / sc4 - tfm.pad_left, 0.0, tfm.orig_w - 1.0))
            oy1 = float(np.clip(fy1 / sc4 - tfm.pad_top, 0.0, tfm.orig_h - 1.0))
            ox2 = float(np.clip(fx2 / sc4 - tfm.pad_left, 0.0, tfm.orig_w - 1.0))
            oy2 = float(np.clip(fy2 / sc4 - tfm.pad_top, 0.0, tfm.orig_h - 1.0))
            ep_out = [round(ox1, 2), round(oy1, 2), round(ox2, 2), round(oy2, 2)]
        else:
            ep_out = list(ep_use)

        phys = _compute_mask_physical_metrics(clean448, refined448, bundle, ep_out)
        mask_orig = tfm.unwarp_mask_to_orig(clean448)
        L_px = math.hypot(ep_out[2] - ep_out[0], ep_out[3] - ep_out[1])
        obb, _ = obb_from_endpoints(ep_out[0], ep_out[1], ep_out[2], ep_out[3], thickness=max(8.0, 0.028 * L_px))

        conf = float(np.clip(
            0.35 * phys["mask_prob_mean"]
            + 0.25 * phys["white_purity_on_mask"]
            + 0.20 * phys["visible_shaft_ratio"]
            + 0.20 * phys["edge_straightness"]
            + 0.12 * phys["red_tip_presence"]
            - 0.18 * phys["wood_on_mask"],
            0.0,
            1.0,
        ))

        return SegmentationPrediction(
            mode=self.mode,
            mask_rle=encode_mask_rle(mask_orig),
            mask_orig=mask_orig,
            prob_112=prob_112,
            endpoints_xyxy=ep_out,
            obb_cxcywh_angle=obb,
            visible_shaft_ratio=phys["visible_shaft_ratio"],
            white_purity_on_mask=phys["white_purity_on_mask"],
            red_tip_presence=phys["red_tip_presence"],
            width_regularity=phys["width_regularity"],
            edge_straightness=phys["edge_straightness"],
            mean_perp_err_norm=phys["mean_perp_err_norm"],
            fg_area_ratio=phys["fg_area_ratio"],
            wood_on_mask=phys["wood_on_mask"],
            dark_on_mask=phys["dark_on_mask"],
            mask_prob_mean=phys["mask_prob_mean"],
            median_width_px448=phys["median_width_px448"],
            confidence=round(conf, 4),
            seg_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": self.mode,
            "in_dim": self.in_dim,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "mlp_coefs": self.mlp_coefs,
            "mlp_intercepts": self.mlp_intercepts,
            "hgb_clf": self.hgb_clf,
            "mask_thr": self.mask_thr,
            "sparse_refine_k": self.sparse_refine_k,
            "is_fitted": self.is_fitted,
            "train_summary": self.train_summary,
        }
        joblib.dump(payload, path, compress=3)
        return compute_sha256(path)

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> "SupervisedShaftSegmenter":
        if expected_sha256:
            actual = compute_sha256(path).lower()
            if actual != expected_sha256.lower():
                raise RuntimeError(f"SHA-256 mismatch for {path}")
        payload = joblib.load(path)
        obj = cls(mode=payload["mode"])
        obj.in_dim = int(payload["in_dim"])
        obj.scaler_mean = payload["scaler_mean"]
        obj.scaler_scale = payload["scaler_scale"]
        obj.mlp_coefs = payload["mlp_coefs"]
        obj.mlp_intercepts = payload["mlp_intercepts"]
        obj.hgb_clf = payload["hgb_clf"]
        obj.mask_thr = float(payload.get("mask_thr", 0.42))
        obj.sparse_refine_k = int(payload.get("sparse_refine_k", 560))
        obj.is_fitted = bool(payload["is_fitted"])
        obj.train_summary = payload.get("train_summary", {})
        return obj
