"""Method A: Oriented White Cane Localization, Collinear Gap Bridging & Metric Evaluation.

Strictly operates on raw input images without Ground Truth access during inference.
Detects tilted, slender, handheld, and partially occluded white mobility canes directly
in oriented geometry (OBB + centerline endpoints), solving the 71-76% axis-aligned box
background contamination bottleneck identified in VL-P1c.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from .common_utils import (
    axial_angle_diff_deg,
    compute_containment_and_bg_contamination,
    compute_iou_xywh,
    compute_mask_metrics,
    compute_rotated_obb_iou,
    obb_from_endpoints,
)

_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_yolox_grid(img_size: tuple[int, int] = (416, 416)) -> tuple[np.ndarray, np.ndarray]:
    if img_size not in _GRID_CACHE:
        grids, expanded_strides = [], []
        for stride in (8, 16, 32):
            hs, ws = img_size[0] // stride, img_size[1] // stride
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            g = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(g)
            expanded_strides.append(np.full((*g.shape[:2], 1), stride))
        _GRID_CACHE[img_size] = (np.concatenate(grids, 1), np.concatenate(expanded_strides, 1))
    return _GRID_CACHE[img_size]


class YOLOXPersonAndObjectDetector:
    """Lightweight CPU YOLOX-Nano detector used strictly for person-anchor & general context."""

    def __init__(self, yolox_model_path: Path, input_size: tuple[int, int] = (416, 416)) -> None:
        self.model_path = Path(yolox_model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLOX model not found: {self.model_path}")
        self.input_size = input_size
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect_boxes(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.16,
        max_boxes: int = 12,
    ) -> list[dict]:
        """Return raw COCO detections [{bbox_xywh, confidence, class_id}]."""
        h, w = image.shape[:2]
        r = min(self.input_size[0] / h, self.input_size[1] / w)
        rh, rw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)
        padded[:rh, :rw] = resized
        tensor = np.ascontiguousarray(padded.transpose((2, 0, 1)).astype(np.float32)[None, ...])

        raw = self.session.run(None, {self.input_name: tensor})[0]
        grids, strides = _get_yolox_grid(self.input_size)
        dec = raw.copy()
        dec[..., :2] = (dec[..., :2] + grids) * strides
        dec[..., 2:4] = np.exp(dec[..., 2:4]) * strides
        dec = dec[0]

        scores_mat = dec[:, 4:5] * dec[:, 5:]
        cids = np.argmax(scores_mat, axis=1)
        scores = scores_mat[np.arange(len(scores_mat)), cids]
        mask = scores >= conf_threshold
        if not np.any(mask):
            return []

        fboxes = dec[mask, :4]
        fscores = scores[mask]
        fcids = cids[mask]
        cx, cy, bw, bh = fboxes[:, 0], fboxes[:, 1], fboxes[:, 2], fboxes[:, 3]
        x1 = np.clip(np.round((cx - bw / 2.0) / r).astype(int), 0, w - 1)
        y1 = np.clip(np.round((cy - bh / 2.0) / r).astype(int), 0, h - 1)
        bw_o = np.clip(np.round(bw / r).astype(int), 1, w - x1)
        bh_o = np.clip(np.round(bh / r).astype(int), 1, h - y1)

        boxes_list = [[int(x1[i]), int(y1[i]), int(bw_o[i]), int(bh_o[i])] for i in range(len(fscores))]
        scores_list = [float(s) for s in fscores]
        idxs = cv2.dnn.NMSBoxes(boxes_list, scores_list, conf_threshold, 0.45)
        if len(idxs) == 0:
            return []
        flat = idxs.flatten().tolist()
        flat.sort(key=lambda idx: scores_list[idx], reverse=True)
        out = []
        for idx in flat[:max_boxes]:
            bx, by, bwi, bhi = boxes_list[idx]
            if bwi >= 16 and bhi >= 16:
                out.append({
                    "bbox_xywh": (int(bx), int(by), int(bwi), int(bhi)),
                    "confidence": float(scores_list[idx]),
                    "class_id": int(fcids[idx]),
                })
        return out


@dataclass
class OrientedCaneProposal:
    bbox_xywh: tuple[int, int, int, int]
    obb_cxcywh_angle: list[float]  # [cx, cy, length, thickness, angle_deg]
    obb_corners_xy: list[list[float]]
    endpoints_xyxy: list[float]  # [x1, y1, x2, y2] ordered top->bottom
    centerline_points_xy: list[list[float]]
    confidence: float
    bilateral_profile_score: float
    white_purity_score: float
    red_tip_score: float
    hand_anchor_score: float
    border_penalty: float
    collinear_bridge_count: int
    est_thickness_px: float
    source: str = "oriented_cane_v2"

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_cane_feature_maps(work_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Compute lightness, tophat ridge, white achromatic map, red-tip map, and bilateral gradient."""
    lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    l_ch = lab[:, :, 0] * (100.0 / 255.0)  # [0, 100]
    a_ch = lab[:, :, 1] - 128.0
    b_ch = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_ch, b_ch)
    s_hsv = hsv[:, :, 1]
    v_hsv = hsv[:, :, 2]

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k17 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    gray = lab[:, :, 0].astype(np.uint8)
    th9 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k9).astype(np.float32)
    th17 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k17).astype(np.float32)
    ridge = np.maximum(th9, 0.9 * th17)

    # White shaft response: high L*, low chroma/saturation, strong local ridge over background
    white_achrom = np.clip((l_ch - 56.0) / 34.0, 0.0, 1.0) * np.clip((28.0 - chroma) / 22.0, 0.0, 1.0)
    white_ridge = np.clip(ridge / 28.0, 0.0, 1.0) * (0.4 + 0.6 * white_achrom)

    # Red reflective band/tip response (handles red bottom tape / joint bands on canes)
    red_a = np.clip((a_ch - 14.0) / 28.0, 0.0, 1.0)
    red_val = np.clip((v_hsv - 55.0) / 140.0, 0.0, 1.0) * np.clip((s_hsv - 65.0) / 110.0, 0.0, 1.0)
    red_map = red_a * red_val

    cane_energy = np.clip(np.maximum(white_ridge, 0.85 * red_map) + 0.25 * white_achrom, 0.0, 1.0)
    return {
        "gray": gray,
        "l_ch": l_ch,
        "chroma": chroma,
        "s_hsv": s_hsv,
        "v_hsv": v_hsv,
        "ridge": ridge,
        "white_achrom": white_achrom,
        "red_map": red_map,
        "cane_energy": cane_energy,
    }


def _extract_raw_segments(
    work_bgr: np.ndarray,
    fmaps: dict[str, np.ndarray],
    person_boxes_work: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    """Extract oriented line segments from LSD + HoughLinesP on intensity, ridge, and person-vicinity ROIs."""
    h, w = work_bgr.shape[:2]
    gray = fmaps["gray"]
    ridge_u8 = np.clip(fmaps["ridge"] * 2.5, 0, 255).astype(np.uint8)
    energy_u8 = np.clip(fmaps["cane_energy"] * 255.0, 0, 255).astype(np.uint8)

    segments: list[np.ndarray] = []
    min_seg_len = max(14.0, 0.032 * min(h, w))

    # 1. LSD on contrast-normalized grayscale & ridge energy
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    for src_u8 in (gray, ridge_u8):
        lines, _, _, _ = lsd.detect(src_u8)
        if lines is not None:
            for ln in lines[:, 0, :]:
                x1, y1, x2, y2 = [float(v) for v in ln]
                if math.hypot(x2 - x1, y2 - y1) >= min_seg_len:
                    segments.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    # 2. HoughLinesP on multi-cue edge map of cane_energy
    edges = cv2.Canny(energy_u8, 45, 130)
    hlines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(18, int(round(0.055 * min(h, w)))),
        minLineLength=int(round(min_seg_len)),
        maxLineGap=max(10, int(round(0.04 * max(h, w)))),
    )
    if hlines is not None:
        for ln in hlines[:, 0, :]:
            segments.append(ln.astype(np.float32))

    # 3. Thin bright + red connected-component skeletons -> PCA segments
    bin_mask = ((fmaps["white_achrom"] > 0.36) & (fmaps["ridge"] > 8.0)) | (fmaps["red_map"] > 0.42)
    bin_u8 = (bin_mask.astype(np.uint8) * 255)
    bin_u8 = cv2.morphologyEx(bin_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    num_lbl, _, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    for cid in range(1, num_lbl):
        x, y, bw, bh, area = stats[cid]
        if area < 24 or max(bw, bh) < min_seg_len:
            continue
        diag = math.hypot(bw, bh)
        if diag / max(1.0, area / max(1.0, diag)) < 2.5:
            continue
        pts_y, pts_x = np.where(bin_u8[y:y + bh, x:x + bw] > 0)
        if len(pts_x) < 12:
            continue
        pts = np.stack([pts_x + x, pts_y + y], axis=1).astype(np.float32)
        mean_pt = pts.mean(axis=0)
        cov = np.cov((pts - mean_pt).T)
        vals, vecs = np.linalg.eigh(cov)
        if vals[-1] / max(1e-3, vals[0]) < 5.0:
            continue
        u = vecs[:, -1]
        proj = (pts - mean_pt) @ u
        p1 = mean_pt + u * float(np.percentile(proj, 2))
        p2 = mean_pt + u * float(np.percentile(proj, 98))
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= min_seg_len:
            segments.append(np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32))

    # 4. Person-vicinity sensitive recovery for occluded handheld canes
    for px, py, pw, ph in person_boxes_work:
        rx1 = max(0, int(round(px - 0.35 * pw)))
        ry1 = max(0, int(round(py + 0.20 * ph)))
        rx2 = min(w, int(round(px + 1.35 * pw)))
        ry2 = min(h, int(round(py + 1.25 * ph)))
        if rx2 - rx1 < 24 or ry2 - ry1 < 24:
            continue
        sub_e = cv2.Canny(energy_u8[ry1:ry2, rx1:rx2], 25, 85)
        plines = cv2.HoughLinesP(
            sub_e,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=14,
            minLineLength=max(12, int(round(0.03 * min(h, w)))),
            maxLineGap=max(14, int(round(0.06 * max(h, w)))),
        )
        if plines is not None:
            for ln in plines[:, 0, :]:
                segments.append(
                    np.array([ln[0] + rx1, ln[1] + ry1, ln[2] + rx1, ln[3] + ry1], dtype=np.float32)
                )

    if len(segments) > 48:
        energy = fmaps["cane_energy"]
        arr = np.asarray(segments, dtype=np.float32)
        ts = np.linspace(0.0, 1.0, 12, dtype=np.float32)[None, :]
        xs = np.clip(np.round(arr[:, 0:1] + ts * (arr[:, 2:3] - arr[:, 0:1])).astype(int), 0, w - 1)
        ys = np.clip(np.round(arr[:, 1:2] + ts * (arr[:, 3:4] - arr[:, 1:2])).astype(int), 0, h - 1)
        e_mean = np.mean(energy[ys, xs], axis=1)
        lens = np.hypot(arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1])
        scores = e_mean * np.sqrt(np.clip(lens, 1.0, 400.0))
        top_idx = np.argsort(-scores)[:48]
        segments = [segments[int(i)] for i in top_idx]

    return segments


def _bridge_collinear_segments(
    segments: list[np.ndarray],
    img_shape: tuple[int, int],
) -> list[tuple[np.ndarray, int]]:
    """Cluster & bridge nearly collinear line segments separated by hand/clothing occlusion."""
    if not segments:
        return []
    h, w = img_shape
    max_dim = float(max(h, w))
    perp_tol = max(7.5, 0.017 * max_dim)

    seg_meta = []
    for s in segments:
        x1, y1, x2, y2 = [float(v) for v in s]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 10.0:
            continue
        ux, uy = dx / length, dy / length
        if uy < 0 or (abs(uy) < 1e-6 and ux < 0):
            ux, uy = -ux, -uy
            x1, y1, x2, y2 = x2, y2, x1, y1
        ang = math.degrees(math.atan2(uy, ux)) % 180.0
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        seg_meta.append({
            "pts": (x1, y1, x2, y2),
            "u": (ux, uy),
            "mid": (mx, my),
            "len": length,
            "ang": ang,
        })

    seg_meta.sort(key=lambda d: d["len"], reverse=True)
    seg_meta = seg_meta[:48]
    used = [False] * len(seg_meta)
    merged: list[tuple[np.ndarray, int]] = []
    merged_raw_singles: list[tuple[np.ndarray, float]] = []
    min_kept_len = max(24.0, 0.055 * min(h, w))

    for i, base in enumerate(seg_meta):
        if used[i]:
            continue
        used[i] = True
        group = [base]
        ux, uy = base["u"]
        bx, by = base["mid"]

        changed = True
        while changed:
            changed = False
            all_t = []
            for g in group:
                for px, py in ((g["pts"][0], g["pts"][1]), (g["pts"][2], g["pts"][3])):
                    all_t.append((px - bx) * ux + (py - by) * uy)
            tmin, tmax = min(all_t), max(all_t)
            span = max(12.0, tmax - tmin)
            max_gap = max(26.0, min(0.28 * max_dim, 0.55 * span))

            for j, cand in enumerate(seg_meta):
                if used[j]:
                    continue
                if axial_angle_diff_deg(base["ang"], cand["ang"]) > 9.5:
                    continue
                mx, my = cand["mid"]
                d_perp = abs((mx - bx) * (-uy) + (my - by) * ux)
                if d_perp > perp_tol:
                    continue
                t1 = (cand["pts"][0] - bx) * ux + (cand["pts"][1] - by) * uy
                t2 = (cand["pts"][2] - bx) * ux + (cand["pts"][3] - by) * uy
                cmin, cmax = min(t1, t2), max(t1, t2)
                if cmax < tmin - max_gap or cmin > tmax + max_gap:
                    continue
                used[j] = True
                group.append(cand)
                changed = True

        # Fit weighted principal line across all segment endpoints in group
        pts_list = []
        weights = []
        for g in group:
            x1, y1, x2, y2 = g["pts"]
            w_g = g["len"]
            pts_list.append([x1, y1])
            pts_list.append([x2, y2])
            weights.extend([w_g, w_g])
            if len(group) > 1 and g["len"] >= min_kept_len:
                sx1, sy1, sx2, sy2 = g["pts"]
                if sy1 > sy2:
                    sx1, sy1, sx2, sy2 = sx2, sy2, sx1, sy1
                merged_raw_singles.append((np.array([sx1, sy1, sx2, sy2], dtype=np.float32), g["len"]))
        pts_arr = np.asarray(pts_list, dtype=np.float32)
        wts = np.asarray(weights, dtype=np.float32)[:, None]
        center = np.sum(pts_arr * wts, axis=0) / np.sum(wts)
        centered = (pts_arr - center) * np.sqrt(wts)
        cov = centered.T @ centered
        _, vecs = np.linalg.eigh(cov)
        u_vec = vecs[:, -1]
        proj = (pts_arr - center) @ u_vec
        p1 = center + u_vec * float(np.min(proj))
        p2 = center + u_vec * float(np.max(proj))
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
        total_len = float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if total_len >= min_kept_len:
            merged.append((np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32), len(group) - 1))

    if merged_raw_singles:
        merged_raw_singles.sort(key=lambda item: item[1], reverse=True)
        for seg_arr, _ in merged_raw_singles[:6]:
            merged.append((seg_arr, 0))

    return merged


def _evaluate_and_refine_candidate(
    seg: np.ndarray,
    bridge_cnt: int,
    fmaps: dict[str, np.ndarray],
    person_boxes_work: list[tuple[int, int, int, int]],
    img_shape: tuple[int, int],
) -> dict | None:
    """Measure bilateral cross-sectional shaft contrast, whiteness, red tip, refine endpoints, and score."""
    h, w = img_shape
    x1, y1, x2, y2 = [float(v) for v in seg]
    dx, dy = x2 - x1, y2 - y1
    raw_len = math.hypot(dx, dy)
    if raw_len < 20.0:
        return None
    ux, uy = dx / raw_len, dy / raw_len
    nx, ny = -uy, ux

    l_ch = fmaps["l_ch"]
    wa = fmaps["white_achrom"]
    rmap = fmaps["red_map"]
    ridge = fmaps["ridge"]

    # Sample M stations along initial segment and snap to ridge center before measuring bilateral profile
    m_stations = 28
    ts = np.linspace(0.0, raw_len, m_stations)
    normal_offsets = np.arange(-14, 15, 1, dtype=np.float32)  # 29 offsets across normal, center index = 14

    def _sample_profiles(sx1: float, sy1: float):
        xs = sx1 + ux * ts[:, None] + nx * normal_offsets[None, :]
        ys = sy1 + uy * ts[:, None] + ny * normal_offsets[None, :]
        ix = np.clip(np.round(xs).astype(int), 0, w - 1)
        iy = np.clip(np.round(ys).astype(int), 0, h - 1)
        in_b = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        return (
            l_ch[iy, ix],
            wa[iy, ix],
            rmap[iy, ix],
            ridge[iy, ix],
            in_b,
        )

    prof_l, prof_wa, prof_red, prof_ridge, in_bounds = _sample_profiles(x1, y1)
    score_n = prof_l + 30.0 * prof_wa + 40.0 * prof_red + 0.6 * prof_ridge
    peak_idx = 8 + np.argmax(score_n[:, 8:21], axis=1)
    valid_st = np.all(in_bounds[:, 8:21], axis=1)
    shift_n = float(np.median(peak_idx[valid_st] - 14)) if np.any(valid_st) else float(np.median(peak_idx - 14))
    if abs(shift_n) > 0.25:
        x1 += shift_n * nx
        y1 += shift_n * ny
        x2 += shift_n * nx
        y2 += shift_n * ny
        prof_l, prof_wa, prof_red, prof_ridge, in_bounds = _sample_profiles(x1, y1)

    # Mask center [-3..+3] vs left [-14..-7] vs right [+7..+14]
    c_idx = slice(11, 18)   # offsets -3..+3
    l_idx = slice(0, 8)     # offsets -14..-7
    r_idx = slice(21, 29)   # offsets +7..+14

    center_l = np.max(prof_l[:, c_idx], axis=1)
    left_l = np.median(prof_l[:, l_idx], axis=1)
    right_l = np.median(prof_l[:, r_idx], axis=1)
    bilat_delta = np.minimum(center_l - left_l, center_l - right_l)

    core_wa = np.max(prof_wa[:, c_idx], axis=1)
    core_red = np.max(prof_red[:, c_idx], axis=1)
    core_ridge = np.max(prof_ridge[:, c_idx], axis=1)
    side_wa = 0.5 * (np.median(prof_wa[:, l_idx], axis=1) + np.median(prof_wa[:, r_idx], axis=1))

    # Genuine bright shaft stations: positive bilateral step OR red tip on cane axis, and brighter/whiter than sides
    station_ok = (
        ((bilat_delta > 2.2) & ((core_ridge > 3.2) | (core_wa > 0.22)) & ((core_wa - side_wa) > -0.02))
        | (core_red > 0.28)
    ) & np.all(in_bounds[:, c_idx], axis=1)

    raw_ok = np.where(station_ok)[0]
    if len(raw_ok) < 5:
        return None

    # Longest contiguous run allowing interior occlusion gaps <= 4 stations
    runs: list[list[int]] = [[int(raw_ok[0])]]
    for idx_k in raw_ok[1:]:
        ik = int(idx_k)
        if ik - runs[-1][-1] <= 5:
            runs[-1].append(ik)
        else:
            runs.append([ik])
    best_run = max(runs, key=lambda r: (len(r), float(np.sum(bilat_delta[r]))))
    ok_indices = np.array(best_run, dtype=int)
    ok_frac = float(len(ok_indices) / float(m_stations))
    if len(ok_indices) < 5 or ok_frac < 0.18:
        return None

    t_start = float(ts[int(ok_indices[0])])
    t_end = float(ts[int(ok_indices[-1])])

    # Refine endpoints outward along tangent while bilateral cane cue persists (bridge <= 3 gap steps)
    def _step_outward(t0: float, step: float, limit: int = 16) -> float:
        cur_t = t0
        last_good_t = t0
        gap_cnt = 0
        for _ in range(limit):
            nxt_t = cur_t + step
            cur_t = nxt_t
            px, py = x1 + ux * nxt_t, y1 + uy * nxt_t
            if not (2.0 <= px < w - 2.0 and 2.0 <= py < h - 2.0):
                break
            sx = np.clip(np.round(px + nx * np.arange(-10, 11, 2)).astype(int), 0, w - 1)
            sy = np.clip(np.round(py + ny * np.arange(-10, 11, 2)).astype(int), 0, h - 1)
            pl = l_ch[sy, sx]
            pw = wa[sy, sx]
            pr = rmap[sy, sx]
            cl = float(np.max(pl[4:7]))
            ll, rl = float(np.median(pl[:3])), float(np.median(pl[8:]))
            bd = min(cl - ll, cl - rl)
            cw = float(np.max(pw[4:7]))
            cr = float(np.max(pr[4:7]))
            if (bd > 2.5 and (cw > 0.18 or cl > 64.0)) or cr > 0.28:
                last_good_t = nxt_t
                gap_cnt = 0
            else:
                gap_cnt += 1
                if gap_cnt > 3:
                    break
        return last_good_t

    t_start = _step_outward(t_start, -3.0)
    t_end = _step_outward(t_end, +3.0)

    rx1, ry1 = x1 + ux * t_start, y1 + uy * t_start
    rx2, ry2 = x1 + ux * t_end, y1 + uy * t_end
    if ry1 > ry2:
        rx1, ry1, rx2, ry2 = rx2, ry2, rx1, ry1
    ref_len = math.hypot(rx2 - rx1, ry2 - ry1)
    if ref_len < max(26.0, 0.06 * min(h, w)):
        return None

    # Estimate cross-sectional FWHM shaft thickness in working image pixels
    widths = []
    for k in ok_indices:
        row_l = prof_l[k]
        bg = 0.5 * (left_l[k] + right_l[k])
        peak = float(np.max(row_l[c_idx]))
        half_lvl = bg + 0.5 * max(4.0, peak - bg)
        above = np.where(row_l >= half_lvl)[0]
        if above.size > 0:
            widths.append(float(above[-1] - above[0] + 1))
    est_thick = float(np.clip(np.median(widths) if widths else 5.0, 3.5, 22.0))

    # Reject overly wide slabs (walls/curbs/stripes) where cross-section does not taper back down
    side_bright_frac = float(np.mean((left_l[ok_indices] > 76.0) & (right_l[ok_indices] > 76.0) & (bilat_delta[ok_indices] < 5.0)))
    if side_bright_frac > 0.45:
        return None

    bilat_score = float(np.clip(np.median(bilat_delta[ok_indices]) / 24.0, 0.0, 1.0))
    white_score = float(np.clip(np.percentile(core_wa[ok_indices], 75), 0.0, 1.0))
    # Weight red tip higher when located in bottom 45% of stations (cane lower half)
    n_st = len(core_red)
    bottom_red = float(np.percentile(core_red[int(0.55 * n_st):], 85)) if n_st >= 6 else 0.0
    red_score = float(np.clip(max(bottom_red, 0.75 * float(np.percentile(core_red, 88))), 0.0, 1.0))

    # Hand/person anchor score: check if upper half is inside or close to a person torso/hand zone
    hand_anchor = 0.0
    up_x, up_y = rx1, ry1
    mid_x, mid_y = 0.5 * (rx1 + rx2), 0.5 * (ry1 + ry2)
    for px, py, pw, ph in person_boxes_work:
        hand_y1 = py + 0.25 * ph
        hand_y2 = py + 0.92 * ph
        if (px - 0.30 * pw) <= up_x <= (px + 1.30 * pw) and (hand_y1 - 0.15 * ph) <= up_y <= hand_y2:
            # Diagonal downwards away from body/waist scores highest
            ang_h = abs(math.degrees(math.atan2(ry2 - ry1, rx2 - rx1)))
            if 15.0 <= ang_h <= 86.0:
                hand_anchor = max(hand_anchor, 1.0)
            else:
                hand_anchor = max(hand_anchor, 0.55)
        elif (px - 0.25 * pw) <= mid_x <= (px + 1.25 * pw) and py <= mid_y <= (py + 1.15 * ph):
            hand_anchor = max(hand_anchor, 0.5)

    # Border / architectural edge penalty
    ang_deg = float(math.degrees(math.atan2(ry2 - ry1, rx2 - rx1)) % 180.0)
    near_hv = min(ang_deg, 180.0 - ang_deg, abs(ang_deg - 90.0)) < 3.5
    touches_edge = (min(rx1, rx2) <= 3 or max(rx1, rx2) >= w - 4) and (abs(rx2 - rx1) > 0.78 * w)
    border_pen = 0.0
    if near_hv and hand_anchor < 0.3 and red_score < 0.25:
        border_pen += 0.25
    if touches_edge and ok_frac < 0.65:
        border_pen += 0.35

    len_norm = float(np.clip(ref_len / (0.65 * math.hypot(h, w)), 0.15, 1.0))
    slim_ratio = ref_len / max(3.0, est_thick)
    slim_bonus = float(np.clip((slim_ratio - 4.0) / 14.0, 0.0, 1.0))

    raw_conf = (
        0.30 * bilat_score
        + 0.24 * white_score
        + 0.16 * red_score
        + 0.15 * hand_anchor
        + 0.10 * len_norm
        + 0.05 * slim_bonus
        + 0.04 * ok_frac
        + 0.03 * min(2, bridge_cnt)
        - border_pen
    )
    conf = float(np.clip(raw_conf, 0.0, 1.0))
    return {
        "endpoints_work": (rx1, ry1, rx2, ry2),
        "thickness_work": est_thick,
        "confidence": conf,
        "bilat_score": bilat_score,
        "white_score": white_score,
        "red_score": red_score,
        "hand_anchor": hand_anchor,
        "border_pen": border_pen,
        "bridge_cnt": bridge_cnt,
    }


class OrientedCaneLocalizer:
    """Method A: Zero-GT oriented white-cane localizer on CPU."""

    def __init__(self, yolox_detector: YOLOXPersonAndObjectDetector | None = None, top_k: int = 5) -> None:
        self.yolox = yolox_detector
        self.top_k = top_k

    def detect_with_context(
        self,
        image: np.ndarray,
        top_k: int | None = None,
    ) -> tuple[list[OrientedCaneProposal], dict]:
        """Detect oriented cane candidates and return auxiliary person/YOLOX context and timing."""
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        k = self.top_k if top_k is None else int(top_k)

        t_yx = time.perf_counter()
        yolox_boxes = self.yolox.detect_boxes(image) if self.yolox is not None else []
        yolox_ms = (time.perf_counter() - t_yx) * 1000.0
        person_boxes_orig = [d["bbox_xywh"] for d in yolox_boxes if d["class_id"] == 0]

        scale = min(1.0, 360.0 / float(max(h, w, 1)))
        ww = max(32, int(round(w * scale)))
        wh = max(32, int(round(h * scale)))
        work = cv2.resize(image, (ww, wh), interpolation=cv2.INTER_AREA) if scale < 1.0 else image

        person_boxes_work = [
            (
                int(round(px * scale)),
                int(round(py * scale)),
                max(8, int(round(pw * scale))),
                max(8, int(round(ph * scale))),
            )
            for px, py, pw, ph in person_boxes_orig
        ]

        fmaps = _compute_cane_feature_maps(work)
        raw_segs = _extract_raw_segments(work, fmaps, person_boxes_work)
        bridged = _bridge_collinear_segments(raw_segs, (wh, ww))

        cands: list[dict] = []
        for seg, bcnt in bridged[:24]:
            ev = _evaluate_and_refine_candidate(seg, bcnt, fmaps, person_boxes_work, (wh, ww))
            if ev is not None:
                cands.append(ev)

        # Fallback: also include diagonal/slender ridge boxes inside person lower zone if few candidates
        cands.sort(key=lambda c: c["confidence"], reverse=True)

        inv_s = 1.0 / scale
        proposals: list[OrientedCaneProposal] = []
        for c in cands:
            rx1, ry1, rx2, ry2 = [v * inv_s for v in c["endpoints_work"]]
            thick = float(np.clip(max(8.0, c["thickness_work"] * inv_s * 1.15), 8.0, min(22.0, 0.028 * min(h, w))))
            obb, corners = obb_from_endpoints(rx1, ry1, rx2, ry2, thick)

            # Duplicate check via rotated OBB IoU and near-collinear centerlines
            dup = False
            for kept in proposals:
                riou = compute_rotated_obb_iou(obb, kept.obb_cxcywh_angle)
                if riou >= 0.35:
                    dup = True
                    break
                ang_d = axial_angle_diff_deg(obb[4], kept.obb_cxcywh_angle[4])
                cdist = math.hypot(obb[0] - kept.obb_cxcywh_angle[0], obb[1] - kept.obb_cxcywh_angle[1])
                if ang_d <= 8.0 and cdist <= 0.10 * max(h, w):
                    dup = True
                    break
            if dup:
                continue

            xs_c = [pt[0] for pt in corners]
            ys_c = [pt[1] for pt in corners]
            ax1 = max(0, int(math.floor(min(xs_c))))
            ay1 = max(0, int(math.floor(min(ys_c))))
            ax2 = min(w, int(math.ceil(max(xs_c))))
            ay2 = min(h, int(math.ceil(max(ys_c))))
            bbox_xywh = (ax1, ay1, max(8, ax2 - ax1), max(8, ay2 - ay1))

            # Sample 5 points along centerline
            cl_pts = []
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                cl_pts.append([round(rx1 + frac * (rx2 - rx1), 2), round(ry1 + frac * (ry2 - ry1), 2)])

            proposals.append(
                OrientedCaneProposal(
                    bbox_xywh=bbox_xywh,
                    obb_cxcywh_angle=obb,
                    obb_corners_xy=corners,
                    endpoints_xyxy=[round(rx1, 2), round(ry1, 2), round(rx2, 2), round(ry2, 2)],
                    centerline_points_xy=cl_pts,
                    confidence=round(c["confidence"], 4),
                    bilateral_profile_score=round(c["bilat_score"], 4),
                    white_purity_score=round(c["white_score"], 4),
                    red_tip_score=round(c["red_score"], 4),
                    hand_anchor_score=round(c["hand_anchor"], 4),
                    border_penalty=round(c["border_pen"], 4),
                    collinear_bridge_count=int(c["bridge_cnt"]),
                    est_thickness_px=round(thick, 2),
                )
            )
            if len(proposals) >= k:
                break

        tot_ms = (time.perf_counter() - t0) * 1000.0
        ctx = {
            "yolox_boxes": yolox_boxes,
            "person_boxes": person_boxes_orig,
            "yolox_ms": round(yolox_ms, 2),
            "localization_ms": round(tot_ms, 2),
            "num_raw_segments": len(raw_segs),
            "num_bridged": len(bridged),
        }
        return proposals, ctx

    def detect(self, image: np.ndarray, top_k: int | None = None) -> tuple[list[OrientedCaneProposal], float]:
        props, ctx = self.detect_with_context(image, top_k=top_k)
        return props, float(ctx["localization_ms"])


def _point_to_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom < 1e-6:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / denom))
    proj_x, proj_y = x1 + t * vx, y1 + t * vy
    return math.hypot(px - proj_x, py - proj_y)


def evaluate_proposal_against_gt(
    proposal: OrientedCaneProposal,
    gt_annot: dict,
    gt_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate a single OrientedCaneProposal against ground truth annotation (OBB, axis box, endpoints, bg)."""
    gt_bbox = tuple(gt_annot["bbox_xywh"])
    gt_obb = list(gt_annot["obb_cxcywh_angle"])
    gt_eps = [float(v) for v in gt_annot["endpoints_xyxy"]]
    gt_ang = float(gt_obb[4])

    axis_iou = compute_iou_xywh(proposal.bbox_xywh, gt_bbox)
    _, axis_bg_box = compute_containment_and_bg_contamination(proposal.bbox_xywh, gt_bbox)
    obb_iou = compute_rotated_obb_iou(proposal.obb_cxcywh_angle, gt_obb)
    ang_err = axial_angle_diff_deg(proposal.obb_cxcywh_angle[4], gt_ang)

    # Endpoint error: min average L2 distance between predicted and GT endpoints normalized by GT length
    px1, py1, px2, py2 = proposal.endpoints_xyxy
    gx1, gy1, gx2, gy2 = gt_eps
    gt_len = max(10.0, math.hypot(gx2 - gx1, gy2 - gy1))
    d_direct = 0.5 * (math.hypot(px1 - gx1, py1 - gy1) + math.hypot(px2 - gx2, py2 - gy2))
    d_swap = 0.5 * (math.hypot(px1 - gx2, py1 - gy2) + math.hypot(px2 - gx1, py2 - gy1))
    ep_err_norm = min(d_direct, d_swap) / gt_len

    # Endpoint inclusion: both GT endpoints inside expanded OBB corridor or within 12% of GT length of pred segment
    tol = max(14.0, 0.14 * gt_len)
    d1 = _point_to_segment_dist(gx1, gy1, px1, py1, px2, py2)
    d2 = _point_to_segment_dist(gx2, gy2, px1, py1, px2, py2)
    endpoints_covered = (d1 <= tol and d2 <= tol)

    # True background contamination relative to GT shaft mask (inside axis box vs inside OBB)
    axis_bg_mask = axis_bg_box
    obb_bg_mask = 1.0
    if gt_mask is not None and np.any(gt_mask > 0):
        h, w = gt_mask.shape[:2]
        ax, ay, aw, ah = proposal.bbox_xywh
        sub_gt = gt_mask[max(0, ay):min(h, ay + ah), max(0, ax):min(w, ax + aw)] > 0
        axis_bg_mask = float(1.0 - (np.sum(sub_gt) / max(1, aw * ah)))

        obb_m = np.zeros((h, w), dtype=np.uint8)
        pts = np.round(np.asarray(proposal.obb_corners_xy, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(obb_m, [pts], 255)
        obb_area = max(1, int(np.sum(obb_m > 0)))
        inter_obb = int(np.sum((obb_m > 0) & (gt_mask > 0)))
        obb_bg_mask = float(1.0 - (inter_obb / obb_area))

    return {
        "axis_iou": round(axis_iou, 4),
        "obb_iou": round(obb_iou, 4),
        "angle_err_deg": round(ang_err, 2),
        "endpoint_err_norm": round(ep_err_norm, 4),
        "endpoints_covered": bool(endpoints_covered),
        "axis_bg_contamination": round(axis_bg_mask, 4),
        "obb_bg_contamination": round(obb_bg_mask, 4),
    }
