"""Method B: Improved Multi-Scale & Geometry-Aware Class-Agnostic Object Proposal Research.

Never inspects or receives Ground Truth bounding boxes during proposal generation or ranking.
Supports ablation modes:
- yolox_strict (conf >= 0.40)
- yolox_low (conf >= 0.15)
- B0_vl_p1_proposer (exact VL-P1 hybrid proposer)
- B1_multiscale (YOLOX + person-anchored sub-proposals + multiscale saliency windows)
- B2_geometry_thin (YOLOX + white/contrasting thin-line top-hat Hough + small-detail gradient islands)
- B_full_v1b (all cues unified + ProposalRanker + diversity NMS top-K)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

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


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return float(inter / max(1, aw * ah + bw * bh - inter))


@dataclass
class ProposalBox:
    bbox_xywh: tuple[int, int, int, int]
    source: str
    raw_confidence: float
    rank_score: float = 0.0
    coco_class_id: int | None = None


class YOLOXNanoExtractor:
    """Offline YOLOX-Nano ONNX proposal generator (inherited unchanged from OD-1 baseline)."""

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

    def extract(self, image: np.ndarray, conf_threshold: float = 0.15, max_boxes: int = 8) -> list[ProposalBox]:
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
                cid = int(fcids[idx])
                out.append(ProposalBox(
                    bbox_xywh=(bx, by, bwi, bhi),
                    source=f"yolox_coco_{cid}",
                    raw_confidence=round(scores_list[idx], 4),
                    coco_class_id=cid,
                ))
        return out


def _compute_saliency_map(small_bgr: np.ndarray) -> np.ndarray:
    """Two-cue class-agnostic saliency: LAB border contrast + multi-scale center-surround."""
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]], axis=0)
    bg_med = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg_med, axis=2)
    d_norm = dist / (float(np.percentile(dist, 98)) + 1e-6)

    # Center-surround DoG-like luminance & chroma contrast (robust when borders are cluttered)
    blur_fine = cv2.GaussianBlur(lab, (5, 5), 0)
    blur_coarse = cv2.GaussianBlur(lab, (31, 31), 0)
    cs = np.linalg.norm(blur_fine - blur_coarse, axis=2)
    cs_norm = cs / (float(np.percentile(cs, 98)) + 1e-6)

    sal = np.clip(0.55 * d_norm + 0.45 * cs_norm, 0.0, 1.0)
    return sal.astype(np.float32)


def _vl_p1_fg_contour_boxes(image: np.ndarray, max_boxes: int = 2) -> list[ProposalBox]:
    h, w = image.shape[:2]
    scale = min(1.0, 256.0 / float(max(h, w)))
    sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]], axis=0)
    bg_med = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg_med, axis=2)
    dist_u8 = np.clip((dist / (np.percentile(dist, 98) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    _, sal_mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    combined = cv2.bitwise_or(sal_mask, edges)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.015 * sw * sh
    cands = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > 0.96 * sw * sh:
            continue
        cx, cy = (x + bw / 2.0) / sw, (y + bh / 2.0) / sh
        center_prior = 1.0 - 0.5 * np.hypot(cx - 0.5, cy - 0.5)
        score = float((area / (sw * sh)) * center_prior)
        ox = max(0, int(round((x - 4) / scale)))
        oy = max(0, int(round((y - 4) / scale)))
        ow = min(w - ox, int(round((bw + 8) / scale)))
        oh = min(h - oy, int(round((bh + 8) / scale)))
        if ow >= 24 and oh >= 24:
            cands.append((score, (ox, oy, ow, oh)))
    cands.sort(key=lambda t: t[0], reverse=True)
    return [ProposalBox(box, "fg_contour", round(sc, 4)) for sc, box in cands[:max_boxes]]


def propose_person_interaction_zones(
    image: np.ndarray,
    yolox_boxes: list[ProposalBox],
    sal_map: np.ndarray,
) -> list[ProposalBox]:
    """B1: Generate sub-proposals around detected people so handheld/lower canes & keys are isolated."""
    h, w = image.shape[:2]
    sh, sw = sal_map.shape[:2]
    sx, sy = sw / float(max(1, w)), sh / float(max(1, h))
    out: list[ProposalBox] = []

    for pb in yolox_boxes:
        if pb.coco_class_id != 0:
            continue
        px, py, pw, ph = pb.bbox_xywh
        if pw < 32 or ph < 48:
            continue

        # Candidate sub-zones tailored to handheld/lower objects near a person:
        # 1. Lower diagonal/cane corridor extending from waist down & slightly outward
        pad_x = int(round(0.28 * pw))
        c1_x = max(0, px - pad_x)
        c1_y = max(0, py + int(round(0.28 * ph)))
        c1_w = min(w - c1_x, pw + 2 * pad_x)
        c1_h = min(h - c1_y, h - c1_y if (py + ph) > 0.85 * h else int(round(0.80 * ph)))

        # 2. Left & right lower foreground wings (person + extended diagonal cane)
        lw_x, lw_y = max(0, px - int(round(0.35 * pw))), max(0, py + int(round(0.20 * ph)))
        lw_w, lw_h = min(w - lw_x, int(round(0.85 * pw))), min(h - lw_y, int(round(0.85 * ph)))

        rw_x = max(0, px + int(round(0.35 * pw)))
        rw_y = lw_y
        rw_w, rw_h = min(w - rw_x, int(round(0.85 * pw))), min(h - rw_y, int(round(0.85 * ph)))

        # 3. Tight mid-torso hand holding zone (wallets, keys, handles)
        hz_x = max(0, px + int(round(0.10 * pw)))
        hz_y = max(0, py + int(round(0.22 * ph)))
        hz_w = min(w - hz_x, int(round(0.80 * pw)))
        hz_h = min(h - hz_y, int(round(0.45 * ph)))

        for name, (bx, by, bw, bh) in [
            ("person_lower_cane", (c1_x, c1_y, c1_w, c1_h)),
            ("person_wing_left", (lw_x, lw_y, lw_w, lw_h)),
            ("person_wing_right", (rw_x, rw_y, rw_w, rw_h)),
            ("person_hand_zone", (hz_x, hz_y, hz_w, hz_h)),
        ]:
            if bw >= 32 and bh >= 32:
                # Measure saliency intensity inside sub-zone to check if active
                x1s, y1s = int(bx * sx), int(by * sy)
                x2s, y2s = max(x1s + 1, int((bx + bw) * sx)), max(y1s + 1, int((by + bh) * sy))
                patch = sal_map[y1s:y2s, x1s:x2s]
                sal_energy = float(np.mean(patch)) if patch.size > 0 else 0.3
                out.append(ProposalBox((bx, by, bw, bh), name, round(0.45 + 0.35 * sal_energy, 4)))

    return out


def propose_multiscale_saliency_windows(
    image: np.ndarray,
    sal_map: np.ndarray,
) -> list[ProposalBox]:
    """B1/B3: Class-agnostic multi-threshold saliency & localized peak windows (catches edge/off-center objects)."""
    h, w = image.shape[:2]
    sh, sw = sal_map.shape[:2]
    scale = sw / float(max(1, w))
    out: list[ProposalBox] = []

    # Tight contour masks at two complementary thresholds (0.35 catches medium foreground, 0.55 splits merged bg)
    sal_u8 = np.clip(sal_map * 255.0, 0, 255).astype(np.uint8)
    for thr_frac, tag in ((0.42, "ms_sal_mid"), (0.60, "ms_sal_tight")):
        thr_val = max(45, int(round(float(np.percentile(sal_u8, 65)) * thr_frac * 2.0)))
        mask = (sal_u8 >= thr_val).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area_frac = (bw * bh) / float(max(1, sw * sh))
            if area_frac < 0.008 or area_frac > 0.92:
                continue
            ox = max(0, int(round((x - 4) / scale)))
            oy = max(0, int(round((y - 4) / scale)))
            ow = min(w - ox, int(round((bw + 8) / scale)))
            oh = min(h - oy, int(round((bh + 8) / scale)))
            if ow >= 24 and oh >= 24:
                energy = float(np.mean(sal_map[y:y + bh, x:x + bw]))
                out.append(ProposalBox((ox, oy, ow, oh), tag, round(energy, 4)))

    # Saliency center-of-mass peak sub-windows at 55% scale (catches objects shifted off center)
    cy_idx, cx_idx = np.unravel_index(np.argmax(cv2.GaussianBlur(sal_map, (15, 15), 0)), sal_map.shape)
    pk_x, pk_y = int(round(cx_idx / scale)), int(round(cy_idx / scale))
    for frac in (0.55, 0.72):
        ww, hh = max(48, int(round(w * frac))), max(48, int(round(h * frac)))
        x0 = max(0, min(w - ww, pk_x - ww // 2))
        y0 = max(0, min(h - hh, pk_y - hh // 2))
        out.append(ProposalBox((x0, y0, ww, hh), f"sal_peak_{int(frac * 100)}", 0.42))

    return out


def propose_thin_and_small_geometry(
    image: np.ndarray,
    max_thin: int = 3,
    max_small: int = 3,
) -> list[ProposalBox]:
    """B2: Class-agnostic geometry proposer for long thin/diagonal sticks and small compact metal clusters."""
    h, w = image.shape[:2]
    scale = min(1.0, 320.0 / float(max(h, w, 1)))
    sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 1. Top-hat + Black-hat thin shaft isolation + Canny + HoughLinesP (detects thin elongated sticks/canes)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kern)
    shaft_map = np.maximum(tophat, (0.65 * blackhat).astype(np.uint8))
    edges_shaft = cv2.Canny(cv2.GaussianBlur(shaft_map, (3, 3), 0), 35, 110)
    edges_gray = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
    edge_union = cv2.bitwise_or(edges_shaft, edges_gray)

    lines = cv2.HoughLinesP(
        edge_union,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(24, min(sw, sh) // 8),
        minLineLength=max(28, int(round(0.18 * max(sw, sh)))),
        maxLineGap=max(10, int(round(0.05 * max(sw, sh)))),
    )

    out: list[ProposalBox] = []
    if lines is not None and len(lines) > 0:
        segs = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
            segs.append((length, ang, (x1, y1, x2, y2)))
        segs.sort(key=lambda t: t[0], reverse=True)
        # Keep top 24 longest segments so dense tiled textures don't explode grouping
        segs = segs[:24]
        used = [False] * len(segs)
        thin_cands = []
        for i in range(len(segs)):
            if used[i]:
                continue
            used[i] = True
            _, ang_i, (ax1, ay1, ax2, ay2) = segs[i]
            pts = [(ax1, ay1), (ax2, ay2)]
            vx, vy = ax2 - ax1, ay2 - ay1
            norm_v = max(1e-6, float(np.hypot(vx, vy)))
            ux, uy = vx / norm_v, vy / norm_v
            for j in range(i + 1, len(segs)):
                if used[j]:
                    continue
                _, ang_j, (bx1, by1, bx2, by2) = segs[j]
                adiff = abs(ang_i - ang_j)
                adiff = min(adiff, 180.0 - adiff)
                if adiff > 18.0:
                    continue
                mx, my = (bx1 + bx2) / 2.0 - ax1, (by1 + by2) / 2.0 - ay1
                perp = abs(mx * (-uy) + my * ux)
                if perp <= max(18.0, 0.08 * min(sw, sh)):
                    used[j] = True
                    pts.extend([(bx1, by1), (bx2, by2)])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span = float(np.hypot(max_x - min_x, max_y - min_y))
            if span < 0.22 * max(sw, sh):
                continue
            pad_w = max(10, int(round(0.08 * span)))
            pad_h = max(10, int(round(0.08 * span)))
            bx = max(0, int(round((min_x - pad_w) / scale)))
            by = max(0, int(round((min_y - pad_h) / scale)))
            bw = min(w - bx, int(round((max_x - min_x + 2 * pad_w) / scale)))
            bh = min(h - by, int(round((max_y - min_y + 2 * pad_h) / scale)))
            if bw >= 24 and bh >= 24 and (bw * bh) < 0.93 * w * h:
                thin_cands.append((span / max(1, max(sw, sh)), (bx, by, bw, bh)))
        thin_cands.sort(key=lambda t: t[0], reverse=True)
        for sc, box in thin_cands[:max_thin]:
            out.append(ProposalBox(box, "geom_thin_line", round(float(sc), 4)))

    # 2. Compact high-gradient island detector (small objects like key rings, coins, pouches)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    g_u8 = cv2.GaussianBlur(grad, (7, 7), 0)
    _, g_mask = cv2.threshold(g_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    g_mask = cv2.morphologyEx(g_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    num_lbls, _, stats, _ = cv2.connectedComponentsWithStats(g_mask)
    small_cands = []
    for lid in range(1, num_lbls):
        x, y, bw, bh, area = [int(v) for v in stats[lid]]
        area_frac = (bw * bh) / float(max(1, sw * sh))
        if area_frac < 0.006 or area_frac > 0.45:
            continue
        density = float(np.mean(grad[y:y + bh, x:x + bw])) / 255.0
        ox = max(0, int(round((x - 6) / scale)))
        oy = max(0, int(round((y - 6) / scale)))
        ow = min(w - ox, int(round((bw + 12) / scale)))
        oh = min(h - oy, int(round((bh + 12) / scale)))
        if ow >= 24 and oh >= 24:
            small_cands.append((density * min(1.0, area_frac * 8.0), (ox, oy, ow, oh)))
    small_cands.sort(key=lambda t: t[0], reverse=True)
    for sc, box in small_cands[:max_small]:
        out.append(ProposalBox(box, "geom_small_detail", round(float(sc), 4)))

    return out


class ProposalRanker:
    """B4: Rank proposals by objectness, interior-vs-border saliency contrast, and low background clutter."""

    @staticmethod
    def score_and_select(
        image: np.ndarray,
        proposals: list[ProposalBox],
        sal_map: np.ndarray,
        top_k: int = 4,
        nms_iou: float = 0.72,
    ) -> list[ProposalBox]:
        if not proposals:
            return []
        h, w = image.shape[:2]
        sh, sw = sal_map.shape[:2]
        sx, sy = sw / float(max(1, w)), sh / float(max(1, h))

        gray = cv2.cvtColor(cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        edges = (cv2.Canny(gray, 55, 150) > 0).astype(np.float32)

        ranked: list[ProposalBox] = []
        for p in proposals:
            bx, by, bw, bh = p.bbox_xywh
            if bw < 16 or bh < 16:
                continue
            area_frac = (bw * bh) / float(max(1, w * h))
            x1 = max(0, min(sw - 1, int(bx * sx)))
            y1 = max(0, min(sh - 1, int(by * sy)))
            x2 = max(x1 + 1, min(sw, int((bx + bw) * sx)))
            y2 = max(y1 + 1, min(sh, int((by + bh) * sy)))

            patch_sal = sal_map[y1:y2, x1:x2]
            patch_edg = edges[y1:y2, x1:x2]
            sal_mean = float(np.mean(patch_sal)) if patch_sal.size > 0 else 0.0
            edg_mean = float(np.mean(patch_edg)) if patch_edg.size > 0 else 0.0

            # Center vs outer-margin contrast within the proposal box
            pw_s, ph_s = x2 - x1, y2 - y1
            ix1, iy1 = x1 + pw_s // 6, y1 + ph_s // 6
            ix2, iy2 = max(ix1 + 1, x2 - pw_s // 6), max(iy1 + 1, y2 - ph_s // 6)
            inner_sal = float(np.mean(sal_map[iy1:iy2, ix1:ix2]))

            source_bonus = 0.0
            if p.source.startswith("yolox_coco_"):
                if p.coco_class_id == 0:  # person box: penalize because handheld object sits inside person bg
                    source_bonus = -0.10 + 0.15 * p.raw_confidence
                else:
                    source_bonus = 0.35 + 0.35 * p.raw_confidence
            elif p.source in ("fg_contour", "ms_sal_mid", "ms_sal_tight"):
                source_bonus = 0.22 + 0.25 * p.raw_confidence
            elif p.source == "geom_thin_line":
                source_bonus = 0.26 + 0.22 * p.raw_confidence
            elif p.source == "geom_small_detail":
                source_bonus = 0.24 + 0.20 * p.raw_confidence
            elif p.source.startswith("person_"):
                source_bonus = 0.14 + 0.22 * p.raw_confidence
            elif p.source == "full_frame":
                source_bonus = -0.22
            else:
                source_bonus = 0.05

            # Mild penalty for overly huge > 80% boxes when tighter foreground boxes exist
            size_penalty = 0.22 * max(0.0, area_frac - 0.72)
            # Mild penalty for sliver noise boxes < 0.8% area unless sharp thin line
            tiny_penalty = 0.20 if (area_frac < 0.008 and p.source != "geom_thin_line") else 0.0

            rscore = source_bonus + 0.32 * inner_sal + 0.18 * sal_mean + 0.45 * edg_mean - size_penalty - tiny_penalty
            p.rank_score = round(float(rscore), 5)
            ranked.append(p)

        ranked.sort(key=lambda item: item.rank_score, reverse=True)
        kept: list[ProposalBox] = []
        for p in ranked:
            if not any(_iou(p.bbox_xywh, k.bbox_xywh) >= nms_iou for k in kept):
                kept.append(p)
            if len(kept) >= top_k:
                break
        return kept


class ObjectProposerVL1b:
    """Unified proposer interface for VL-P1b ablation evaluation (never uses GT boxes)."""

    def __init__(
        self,
        yolox: YOLOXNanoExtractor | None,
        mode: str = "B_full_v1b",  # yolox_strict | yolox_low | B0_vl_p1_proposer | B1_multiscale | B2_geometry_thin | B_full_v1b
        top_k: int = 4,
    ) -> None:
        self.yolox = yolox
        self.mode = mode
        self.top_k = top_k

    def propose(self, image: np.ndarray) -> tuple[list[ProposalBox], float]:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided to ObjectProposerVL1b")
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        if self.mode == "yolox_strict":
            out = self.yolox.extract(image, conf_threshold=0.40, max_boxes=self.top_k) if self.yolox else []
            return out, round((time.perf_counter() - t0) * 1000.0, 3)

        if self.mode == "yolox_low":
            out = self.yolox.extract(image, conf_threshold=0.15, max_boxes=self.top_k) if self.yolox else []
            return out, round((time.perf_counter() - t0) * 1000.0, 3)

        if self.mode == "B0_vl_p1_proposer":
            props: list[ProposalBox] = []
            if self.yolox:
                props.extend(self.yolox.extract(image, conf_threshold=0.15, max_boxes=2))
            props.extend(_vl_p1_fg_contour_boxes(image, max_boxes=2))
            cw, ch = int(round(w * 0.72)), int(round(h * 0.72))
            props.append(ProposalBox(((w - cw) // 2, (h - ch) // 2, cw, ch), "center_72", 0.5))
            props.append(ProposalBox((0, 0, w, h), "full_frame", 0.4))
            kept: list[ProposalBox] = []
            for p in props:
                if not any(_iou(p.bbox_xywh, k.bbox_xywh) >= 0.78 for k in kept):
                    kept.append(p)
                if len(kept) >= 4:
                    break
            return kept, round((time.perf_counter() - t0) * 1000.0, 3)

        # Shared lightweight 256px saliency map
        scale = min(1.0, 256.0 / float(max(h, w, 1)))
        small = cv2.resize(
            image,
            (max(32, int(round(w * scale))), max(32, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        sal_map = _compute_saliency_map(small)
        y_boxes = self.yolox.extract(image, conf_threshold=0.15, max_boxes=4) if self.yolox else []
        candidates: list[ProposalBox] = list(y_boxes)

        if self.mode in ("B1_multiscale", "B_full_v1b"):
            candidates.extend(propose_person_interaction_zones(image, y_boxes, sal_map))
            candidates.extend(propose_multiscale_saliency_windows(image, sal_map))

        if self.mode in ("B2_geometry_thin", "B_full_v1b"):
            candidates.extend(_vl_p1_fg_contour_boxes(image, max_boxes=2))
            candidates.extend(propose_thin_and_small_geometry(image, max_thin=3, max_small=3))

        # Always keep fallback center_72 & full_frame at low prior rank so zero-box never occurs unless empty image
        cw, ch = int(round(w * 0.72)), int(round(h * 0.72))
        candidates.append(ProposalBox(((w - cw) // 2, (h - ch) // 2, cw, ch), "center_72", 0.3))
        candidates.append(ProposalBox((0, 0, w, h), "full_frame", 0.2))

        selected = ProposalRanker.score_and_select(
            image, candidates, sal_map, top_k=self.top_k, nms_iou=0.72
        )
        return selected, round((time.perf_counter() - t0) * 1000.0, 3)
