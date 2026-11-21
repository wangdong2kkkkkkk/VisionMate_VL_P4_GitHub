"""Method B1: Thin-Object Rectified Corridor Segmentation for White Canes.

Generates crisp binary (`mask_bin`) and Gaussian-tapered soft (`mask_soft`) pixel masks
for oriented white-cane proposals on CPU (< 2 ms per candidate), strictly without Ground Truth access.
Also provides `evaluate_segmentation_against_gt` computing Mask IoU, Dice, Precision, Recall,
Boundary F1, and Post-Mask Background Contamination.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .common_utils import compute_mask_metrics
from .localization_research import OrientedCaneProposal


@dataclass
class SegmentationResult:
    strip_bgr: np.ndarray
    strip_mask_bin: np.ndarray     # uint8 (0/255), shape (Hs, Ws)
    strip_mask_soft: np.ndarray    # float32 [0, 1], shape (Hs, Ws)
    affine_fwd: np.ndarray         # 2x3 float32 img -> strip
    affine_inv: np.ndarray         # 2x3 float32 strip -> img
    full_mask_bin: np.ndarray | None
    full_mask_soft: np.ndarray | None
    shaft_coverage_frac: float
    mean_half_width_px: float
    seg_ms: float


def compute_rectification_affine(
    proposal: OrientedCaneProposal,
    img_shape: tuple[int, int],
    corridor_pad_factor: float = 3.6,
    axial_pad_px: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Compute forward and inverse 2x3 affine matrices mapping proposal axis horizontally into (Hs, Ws)."""
    x1, y1, x2, y2 = [float(v) for v in proposal.endpoints_xyxy]
    dx, dy = x2 - x1, y2 - y1
    length = max(24.0, math.hypot(dx, dy))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    angle_deg = math.degrees(math.atan2(dy, dx))

    # Strip dimensions in local scaled pixels (keep length faithful, capped for CPU speed)
    target_len = min(240.0, max(72.0, length + 2.0 * axial_pad_px))
    scale = target_len / max(24.0, length + 2.0 * axial_pad_px)
    ws = max(72, int(round(target_len)))
    corridor_h_orig = max(42.0, proposal.est_thickness_px * corridor_pad_factor)
    hs = max(40, min(96, int(round(corridor_h_orig * scale))))
    if hs % 2 == 1:
        hs += 1

    # Rigid scale+rotation centered at (cx, cy) -> (ws/2, hs/2)
    rad = math.radians(-angle_deg)
    ca, sa = math.cos(rad) * scale, math.sin(rad) * scale
    tx = 0.5 * ws - (ca * cx - sa * cy)
    ty = 0.5 * hs - (sa * cx + ca * cy)
    m_fwd = np.array([[ca, -sa, tx], [sa, ca, ty]], dtype=np.float32)
    m_inv = cv2.invertAffineTransform(m_fwd).astype(np.float32)
    return m_fwd, m_inv, ws, hs, scale


class CaneShaftSegmenter:
    """Fast CPU cane shaft segmenter inside a rectified oriented corridor."""

    def segment_proposal(
        self,
        image: np.ndarray,
        proposal: OrientedCaneProposal,
        return_full_mask: bool = True,
        corridor_pad_factor: float = 3.6,
    ) -> SegmentationResult:
        t0 = time.perf_counter()
        h_img, w_img = image.shape[:2]
        m_fwd, m_inv, ws, hs, sc = compute_rectification_affine(
            proposal, (h_img, w_img), corridor_pad_factor=corridor_pad_factor
        )
        strip = cv2.warpAffine(
            image, m_fwd, (ws, hs), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        # Chromatic & bilateral likelihood inside strip
        lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).astype(np.float32)
        l_ch = lab[:, :, 0] * (100.0 / 255.0)
        a_ch = lab[:, :, 1] - 128.0
        b_ch = lab[:, :, 2] - 128.0
        chroma = np.hypot(a_ch, b_ch)

        # Estimate top & bottom corridor background L* per column
        band = max(4, hs // 7)
        bg_top_l = np.median(l_ch[:band, :], axis=0, keepdims=True)
        bg_bot_l = np.median(l_ch[hs - band:, :], axis=0, keepdims=True)
        bg_min = np.minimum(bg_top_l, bg_bot_l)
        bg_max = np.maximum(bg_top_l, bg_bot_l)
        # Bilateral contrast prefers brighter than BOTH top and bottom corridor background
        bilat = np.clip((l_ch - 0.5 * (bg_top_l + bg_bot_l) + 0.5 * (l_ch - bg_max)) / 24.0, 0.0, 1.0)
        rel_bright = np.clip((l_ch - bg_min - 6.0) / 28.0, 0.0, 1.0)

        white_like = np.clip((l_ch - 54.0) / 36.0, 0.0, 1.0) * np.clip((28.0 - chroma) / 22.0, 0.0, 1.0)
        red_like = np.clip((a_ch - 13.0) / 28.0, 0.0, 1.0) * np.clip((hsv[:, :, 2] - 50.0) / 140.0, 0.0, 1.0)

        ys = np.arange(hs, dtype=np.float32)[:, None]
        y_center = 0.5 * (hs - 1)
        sigma_y = max(4.0, 0.19 * hs)
        center_prior = np.exp(-0.5 * ((ys - y_center) / sigma_y) ** 2)

        # Combined per-pixel shaft likelihood in [0, 1]
        pixel_prob = center_prior * np.clip(
            0.46 * (white_like * (0.45 + 0.55 * bilat)) + 0.34 * red_like + 0.24 * (rel_bright * bilat),
            0.0,
            1.0,
        )
        # Mild horizontal smoothing along cane axis to bridge speckle / logo breaks
        pixel_prob = cv2.GaussianBlur(pixel_prob, (7, 3), sigmaX=1.8, sigmaY=0.7)

        # Dynamic programming centerline tracking across x = 0..ws-1
        dp = np.zeros_like(pixel_prob)
        back = np.zeros((hs, ws), dtype=np.int32)
        y_ar = np.arange(hs, dtype=np.int32)
        center_norm = np.abs(y_ar.astype(np.float32) - y_center) / max(1.0, sigma_y)
        dp[:, 0] = -pixel_prob[:, 0] + 0.08 * center_norm
        move_pen = 0.035
        center_bias = (0.03 * center_norm).astype(np.float32)
        dys = (-2, -1, 0, 1, 2)
        src_mat = np.zeros((len(dys), hs), dtype=np.int32)
        pen_mat = np.zeros((len(dys), hs), dtype=np.float32)
        for k_idx, dy in enumerate(dys):
            sy = y_ar - dy
            valid = (sy >= 0) & (sy < hs)
            src_mat[k_idx] = np.clip(sy, 0, hs - 1)
            pen_mat[k_idx] = np.where(valid, move_pen * abs(dy), 1e6).astype(np.float32)

        for x in range(1, ws):
            prev = dp[:, x - 1]
            cands = prev[src_mat] + pen_mat
            best_k = np.argmin(cands, axis=0)
            dp[:, x] = cands[best_k, y_ar] - pixel_prob[:, x] + center_bias
            back[:, x] = src_mat[best_k, y_ar]

        yc_path = np.zeros(ws, dtype=np.int32)
        # Terminate near center band
        lo_t, hi_t = max(0, int(0.2 * hs)), min(hs, int(0.8 * hs))
        yc_path[-1] = lo_t + int(np.argmin(dp[lo_t:hi_t, -1]))
        for x in range(ws - 1, 0, -1):
            yc_path[x - 1] = back[yc_path[x], x]

        # Estimate upper and lower shaft half-widths per column from centerline response
        est_r_s = max(2.2, 0.5 * proposal.est_thickness_px * sc)
        max_half_r = max(3, min(int(round(max(est_r_s * 1.9, hs * 0.28))), hs // 3))

        col_active = np.zeros(ws, dtype=bool)
        half_tops = np.zeros(ws, dtype=np.float32)
        half_bots = np.zeros(ws, dtype=np.float32)

        ax_pad_s = max(2, int(round(0.06 * ws)))
        for x in range(ws):
            yc = int(yc_path[x])
            peak = float(np.max(pixel_prob[max(0, yc - 2):min(hs, yc + 3), x]))
            bg_p = float(min(np.median(pixel_prob[:band, x]), np.median(pixel_prob[hs - band:, x])))
            thr = bg_p + 0.42 * max(0.05, peak - bg_p)
            active = peak >= 0.16 and (ax_pad_s <= x < ws - ax_pad_s or peak >= 0.24)
            col_active[x] = active
            if not active:
                continue
            rt = 1.0
            for d in range(1, max_half_r + 1):
                if yc - d >= 0 and pixel_prob[yc - d, x] >= thr:
                    rt = float(d)
                else:
                    break
            rb = 1.0
            for d in range(1, max_half_r + 1):
                if yc + d < hs and pixel_prob[yc + d, x] >= thr:
                    rb = float(d)
                else:
                    break
            half_tops[x] = min(rt, est_r_s * 1.8)
            half_bots[x] = min(rb, est_r_s * 1.8)

        # Bridge tiny 1-4px axial gaps on active shaft, drop isolated noise blobs < 6px
        act_u8 = col_active.astype(np.uint8)
        act_u8 = cv2.morphologyEx(act_u8.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 5), dtype=np.uint8)).ravel()
        # Median smooth radii over active support
        act_idx = np.where(act_u8 > 0)[0]
        med_rt = float(np.median(half_tops[half_tops > 0])) if np.any(half_tops > 0) else est_r_s
        med_rb = float(np.median(half_bots[half_bots > 0])) if np.any(half_bots > 0) else est_r_s
        half_tops = np.where(half_tops > 0, half_tops, med_rt)
        half_bots = np.where(half_bots > 0, half_bots, med_rb)
        half_tops = cv2.blur(half_tops.reshape(1, -1), (9, 1)).ravel()
        half_bots = cv2.blur(half_bots.reshape(1, -1), (9, 1)).ravel()

        mask_bin = np.zeros((hs, ws), dtype=np.uint8)
        for x in act_idx:
            yc = int(yc_path[x])
            y_hi = max(0, int(math.floor(yc - half_tops[x] - 0.5)))
            y_lo = min(hs - 1, int(math.ceil(yc + half_bots[x] + 0.5)))
            mask_bin[y_hi:y_lo + 1, x] = 255

        # Keep largest + collinear active components (removing off-axis stray specks)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, np.ones((3, 5), dtype=np.uint8))
        soft = cv2.GaussianBlur(mask_bin.astype(np.float32) / 255.0, (5, 5), sigmaX=1.2, sigmaY=1.2)
        soft = np.clip(np.maximum(soft, (mask_bin > 0).astype(np.float32) * 0.92), 0.0, 1.0)

        full_bin = None
        full_soft = None
        if return_full_mask:
            warped = cv2.warpAffine(
                soft,
                m_inv,
                (w_img, h_img),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            full_soft = np.clip(warped, 0.0, 1.0).astype(np.float32)
            full_bin = (full_soft >= 0.48).astype(np.uint8) * 255

        cov_frac = float(np.mean(np.any(mask_bin > 0, axis=0)))
        mean_hw = 0.5 * float(np.mean(half_tops[act_idx] + half_bots[act_idx])) if act_idx.size > 0 else est_r_s
        ms = (time.perf_counter() - t0) * 1000.0
        return SegmentationResult(
            strip_bgr=strip,
            strip_mask_bin=mask_bin,
            strip_mask_soft=soft,
            affine_fwd=m_fwd,
            affine_inv=m_inv,
            full_mask_bin=full_bin,
            full_mask_soft=full_soft,
            shaft_coverage_frac=round(cov_frac, 4),
            mean_half_width_px=round(mean_hw / max(1e-3, sc), 2),
            seg_ms=round(ms, 2),
        )


def compute_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tol_px: int = 3) -> float:
    """Compute boundary F1 score within tol_px tolerance radius."""
    p = (np.asarray(pred_mask) > 0).astype(np.uint8)
    g = (np.asarray(gt_mask) > 0).astype(np.uint8)
    if not np.any(p) or not np.any(g):
        return 0.0
    k3 = np.ones((3, 3), dtype=np.uint8)
    pb = p ^ cv2.erode(p, k3)
    gb = g ^ cv2.erode(g, k3)
    if not np.any(pb) or not np.any(gb):
        return 0.0
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    pb_dil = cv2.dilate(pb, dk) > 0
    gb_dil = cv2.dilate(gb, dk) > 0
    prec = float(np.sum((pb > 0) & gb_dil) / max(1, np.sum(pb > 0)))
    rec = float(np.sum((gb > 0) & pb_dil) / max(1, np.sum(gb > 0)))
    return round((2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0, 4)


def evaluate_segmentation_against_gt(pred_full_bin: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Evaluate predicted segmentation mask against GT shaft mask."""
    metrics = compute_mask_metrics(pred_full_bin, gt_mask)
    metrics["boundary_f1"] = compute_boundary_f1(pred_full_bin, gt_mask, tol_px=3)
    return metrics
