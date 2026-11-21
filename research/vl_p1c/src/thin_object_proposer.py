"""Section VII / Method D: Oriented Thin-Object & Handheld White-Cane Localization (Class-Agnostic, Offline).

Never receives Ground Truth bounding boxes during proposal generation or cropping.
Addresses two root failures of VL-P1b on object_01:
1. YOLOX detects the person (coco_class_id=0) while ignoring the thin cane held diagonally.
2. Standard axis-aligned crops of diagonal canes contain ~68% background pavement clutter, masking cane cues in DINOv2.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .common_utils import compute_containment_and_bg_contamination, compute_iou_xywh
from .proposal_types import ProposalBox


def _directional_tophat_response(gray: np.ndarray) -> np.ndarray:
    """Enhance narrow light/contrasting linear shafts across 0, 45, 90, 135 degrees."""
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))  # suppresses horizontal band clutter, keeps thin horiz/diag? wait:
    # Vertical 1x15 structuring element extracts horizontal-narrow shafts cleanly via morphological opening subtraction (tophat)
    # Cross-directional 15-px line kernels suppress uniform patches larger than ~15px wide.
    k0 = np.ones((15, 1), dtype=np.uint8)
    k90 = np.ones((1, 15), dtype=np.uint8)
    k45 = np.eye(13, dtype=np.uint8)
    k135 = np.fliplr(k45).copy()
    k45 = cv2.dilate(k45, np.ones((2, 2), dtype=np.uint8))
    k135 = cv2.dilate(k135, np.ones((2, 2), dtype=np.uint8))

    resps = []
    for kern in (k0, k90, k45, k135):
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kern)
        resps.append(np.maximum(th.astype(np.float32), 0.45 * bh.astype(np.float32)))
    combined = np.max(np.stack(resps, axis=0), axis=0)
    mx = float(np.percentile(combined, 99.5)) + 1e-5
    return np.clip(combined / mx, 0.0, 1.0).astype(np.float32)


def _white_and_red_cane_masks(small_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract soft chromatic response maps for white shaft + red warning tip/band."""
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sat, val, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # White / off-white / silver reflective shaft mask
    white_hsv = (sat < 85.0) & (val > 140.0)
    white_lab = (L > 155.0) & (np.abs(a - 128.0) < 20.0) & (np.abs(b - 128.0) < 24.0)
    white_mask = (white_hsv | white_lab).astype(np.float32)

    # Red reflective warning tape / tip accent
    red_mask = (((hue <= 14.0) | (hue >= 164.0)) & (sat > 80.0) & (val > 65.0)).astype(np.float32)
    return white_mask, red_mask


def detect_oriented_thin_cane_candidates(
    image: np.ndarray,
    yolox_boxes: list[ProposalBox] | None = None,
    max_candidates: int = 4,
) -> list[ProposalBox]:
    """Detect collinear thin white/red-accent shafts and optionally anchor them to person hand zones."""
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    scale = min(1.0, 384.0 / float(max(h, w, 1)))
    sw, sh = max(48, int(round(w * scale))), max(48, int(round(h * scale)))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    shaft_resp = _directional_tophat_response(gray)
    white_mask, red_mask = _white_and_red_cane_masks(small)
    chrom_map = np.clip(0.85 * white_mask + 1.0 * red_mask, 0.0, 1.0)
    cane_sal = shaft_resp * (0.35 + 0.65 * chrom_map)

    edges_shaft = cv2.Canny(np.clip(cane_sal * 255.0, 0, 255).astype(np.uint8), 35, 110)
    edges_gray = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 145)
    edge_union = cv2.bitwise_or(edges_shaft, edges_gray)

    diag = float(math.hypot(sw, sh))
    lines = cv2.HoughLinesP(
        edge_union,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(22, min(sw, sh) // 9),
        minLineLength=max(26, int(round(0.14 * max(sw, sh)))),
        maxLineGap=max(12, int(round(0.06 * max(sw, sh)))),
    )

    segs: list[tuple[float, float, tuple[int, int, int, int]]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            length = float(math.hypot(x2 - x1, y2 - y1))
            ang = float(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
            segs.append((length, ang, (int(x1), int(y1), int(x2), int(y2))))

    # Also add connected-component elongated bright ridge axes (handles curved/dashed/red-tipped shafts)
    sal_bin = (cane_sal >= max(0.28, float(np.percentile(cane_sal, 92)))).astype(np.uint8) * 255
    sal_bin = cv2.morphologyEx(sal_bin, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    num_lbls, _, stats, _ = cv2.connectedComponentsWithStats(sal_bin)
    for lid in range(1, num_lbls):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < 28:
            continue
        bx_c, by_c, bw_c, bh_c = [int(stats[lid, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
        )]
        ys_c, xs_c = np.where(sal_bin[by_c:by_c + bh_c, bx_c:bx_c + bw_c] > 0)
        if xs_c.size < 16:
            continue
        pts_cc = np.column_stack([xs_c + bx_c, ys_c + by_c]).astype(np.float32)
        rect = cv2.minAreaRect(pts_cc)
        rw, rh = rect[1]
        long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
        if long_side >= 0.16 * max(sw, sh) and (long_side / short_side) >= 3.2:
            box_pts = cv2.boxPoints(rect)
            # Midpoints of the two short edges define principal shaft endpoints
            d01 = np.linalg.norm(box_pts[0] - box_pts[1])
            if d01 <= np.linalg.norm(box_pts[1] - box_pts[2]):
                p_a = 0.5 * (box_pts[0] + box_pts[1])
                p_b = 0.5 * (box_pts[2] + box_pts[3])
            else:
                p_a = 0.5 * (box_pts[1] + box_pts[2])
                p_b = 0.5 * (box_pts[3] + box_pts[0])
            x1c, y1c = int(round(p_a[0])), int(round(p_a[1]))
            x2c, y2c = int(round(p_b[0])), int(round(p_b[1]))
            ang = float(math.degrees(math.atan2(y2c - y1c, x2c - x1c))) % 180.0
            segs.append((float(long_side), ang, (x1c, y1c, x2c, y2c)))

    if not segs:
        return []

    # Prepare scaled person boxes + hand zones (for person-cane interaction scoring)
    person_boxes_s: list[tuple[int, int, int, int]] = []
    hand_zones_s: list[tuple[int, int, int, int]] = []
    if yolox_boxes:
        for pb in yolox_boxes:
            if pb.coco_class_id == 0:
                px, py, pw, ph = pb.bbox_xywh
                psx, psy = int(round(px * scale)), int(round(py * scale))
                psw, psh = max(8, int(round(pw * scale))), max(12, int(round(ph * scale)))
                person_boxes_s.append((psx, psy, psw, psh))
                hx = max(0, psx - int(round(0.25 * psw)))
                hy = max(0, psy + int(round(0.18 * psh)))
                hw = min(sw - hx, int(round(1.50 * psw)))
                hh = min(sh - hy, int(round(0.58 * psh)))
                hand_zones_s.append((hx, hy, hw, hh))

    segs.sort(key=lambda t: t[0], reverse=True)
    segs = segs[:32]
    used = [False] * len(segs)
    out_scored: list[tuple[float, ProposalBox]] = []

    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        _, ang_i, (ax1, ay1, ax2, ay2) = segs[i]
        vx, vy = float(ax2 - ax1), float(ay2 - ay1)
        nv = max(1e-6, math.hypot(vx, vy))
        ux, uy = vx / nv, vy / nv
        pts = [(ax1, ay1), (ax2, ay2)]

        # Merge collinear segments (reconnects white shaft + red tip + partially occluded shaft)
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            _, ang_j, (bx1, by1, bx2, by2) = segs[j]
            adiff = abs(ang_i - ang_j)
            adiff = min(adiff, 180.0 - adiff)
            if adiff > 14.0:
                continue
            mx, my = (bx1 + bx2) / 2.0 - ax1, (by1 + by2) / 2.0 - ay1
            perp = abs(mx * (-uy) + my * ux)
            if perp > max(12.0, 0.055 * min(sw, sh)):
                continue
            # Check longitudinal proximity along unit axis u so distant parallel lines are NOT merged
            proj_i = [px * ux + py * uy for px, py in pts]
            proj_j = [bx1 * ux + by1 * uy, bx2 * ux + by2 * uy]
            gap = max(min(proj_j) - max(proj_i), min(proj_i) - max(proj_j), 0.0)
            if gap <= 0.28 * max(sw, sh):
                used[j] = True
                pts.extend([(bx1, by1), (bx2, by2)])

        # Fit exact endpoints along principal axis
        projs = [px * ux + py * uy for px, py in pts]
        p_min_idx = int(np.argmin(projs))
        p_max_idx = int(np.argmax(projs))
        e1x, e1y = pts[p_min_idx]
        e2x, e2y = pts[p_max_idx]
        # Order upper point first (y1 <= y2)
        if e1y > e2y:
            e1x, e1y, e2x, e2y = e2x, e2y, e1x, e1y
        span = float(math.hypot(e2x - e1x, e2y - e1y))
        if span < 0.20 * max(sw, sh):
            continue

        # Sample points strictly along the 1D shaft corridor to guard against bright table/floor edges
        n_samp = max(12, int(span // 4))
        ts = np.linspace(0.0, 1.0, n_samp)
        xs_s = np.clip(np.round(e1x + ts * (e2x - e1x)).astype(int), 0, sw - 1)
        ys_s = np.clip(np.round(e1y + ts * (e2y - e1y)).astype(int), 0, sh - 1)

        white_line_frac = float(np.mean(white_mask[ys_s, xs_s]))
        red_line_frac = float(np.mean(red_mask[ys_s, xs_s]))
        ridge_line_mean = float(np.mean(shaft_resp[ys_s, xs_s]))

        # Check perpendicular thinness vs background (penalize uniform white walls / bright sky patches)
        perp_x, perp_y = -uy * 7.0, ux * 7.0
        xl = np.clip(np.round(xs_s - perp_x).astype(int), 0, sw - 1)
        yl = np.clip(np.round(ys_s - perp_y).astype(int), 0, sh - 1)
        xr = np.clip(np.round(xs_s + perp_x).astype(int), 0, sw - 1)
        yr = np.clip(np.round(ys_s + perp_y).astype(int), 0, sh - 1)
        center_luma = gray[ys_s, xs_s].astype(np.float32)
        side_luma = 0.5 * (gray[yl, xl].astype(np.float32) + gray[yr, xr].astype(np.float32))
        bilateral_contrast = float(np.mean(np.abs(center_luma - side_luma)) / 255.0)

        # Inclination: horizontal architectural lines (<14 deg) without strong red/person evidence are suppressed
        horiz_deg = abs(float(math.degrees(math.atan2(e2y - e1y, e2x - e1x))))
        if horiz_deg > 90.0:
            horiz_deg = 180.0 - horiz_deg
        border_margin = min(e1x, sw - 1 - e1x, e1y, sh - 1 - e1y)
        is_image_edge_line = (horiz_deg < 6.0 or horiz_deg > 87.0) and border_margin < 6

        # Person-hand anchor check: upper endpoint near hand zone AND lower endpoint descending below it
        person_anchored = False
        for hx, hy, hw_z, hh_z in hand_zones_s:
            if hx <= e1x <= (hx + hw_z) and hy <= e1y <= (hy + hh_z) and e2y > (hy + 0.55 * hh_z):
                person_anchored = True
                break

        diag_score = 1.0 if 18.0 <= horiz_deg <= 84.0 else (0.55 if horiz_deg > 84.0 else 0.25)
        cane_evidence = (
            0.30 * min(1.0, span / (0.65 * diag))
            + 0.28 * white_line_frac
            + 0.22 * min(1.0, red_line_frac * 4.0)
            + 0.18 * min(1.0, bilateral_contrast * 3.0)
            + 0.12 * ridge_line_mean
            + (0.28 if person_anchored else 0.0)
            + 0.12 * diag_score
            - (0.30 if is_image_edge_line else 0.0)
        )
        cane_evidence = float(np.clip(cane_evidence, 0.0, 1.0))

        # Tight axis-aligned box containing endpoints + proportional shaft thickness margin
        pad_x = max(10, int(round(0.06 * span)))
        pad_y = max(10, int(round(0.06 * span)))
        bx1_s = max(0, min(e1x, e2x) - pad_x)
        by1_s = max(0, min(e1y, e2y) - pad_y)
        bx2_s = min(sw, max(e1x, e2x) + pad_x)
        by2_s = min(sh, max(e1y, e2y) + pad_y)

        ox = max(0, int(round(bx1_s / scale)))
        oy = max(0, int(round(by1_s / scale)))
        ow = min(w - ox, int(round((bx2_s - bx1_s) / scale)))
        oh = min(h - oy, int(round((by2_s - by1_s) / scale)))
        if ow < 24 or oh < 24 or (ow * oh) >= 0.90 * w * h:
            continue

        ox1 = max(0, min(w - 1, int(round(e1x / scale))))
        oy1 = max(0, min(h - 1, int(round(e1y / scale))))
        ox2 = max(0, min(w - 1, int(round(e2x / scale))))
        oy2 = max(0, min(h - 1, int(round(e2y / scale))))
        src_tag = "person_cane_ray" if person_anchored else "oriented_thin_cane"
        pb = ProposalBox(
            bbox_xywh=(ox, oy, ow, oh),
            source=src_tag,
            raw_confidence=round(cane_evidence, 4),
            source_family="F_thin",
            endpoints_xyxy=(ox1, oy1, ox2, oy2),
            oriented_angle_deg=round(float(horiz_deg), 2),
            thin_evidence_score=round(cane_evidence, 4),
            meta={
                "white_line_frac": round(white_line_frac, 4),
                "red_line_frac": round(red_line_frac, 4),
                "bilateral_contrast": round(bilateral_contrast, 4),
                "person_anchored": person_anchored,
                "span_px": round(span / scale, 1),
            },
        )
        out_scored.append((cane_evidence, pb))

    out_scored.sort(key=lambda t: t[0], reverse=True)
    kept: list[ProposalBox] = []
    for _, pb in out_scored:
        if not any(compute_iou_xywh(pb.bbox_xywh, k.bbox_xywh) >= 0.65 for k in kept):
            kept.append(pb)
        if len(kept) >= max_candidates:
            break
    return kept


def extract_thin_aware_roi(
    image: np.ndarray,
    prop: ProposalBox,
    corridor_blend: float = 0.65,
) -> np.ndarray:
    """Crop proposal ROI and softly suppress distant background corners along diagonal thin shafts."""
    bx, by, bw, bh = prop.bbox_xywh
    roi = image[by:by + bh, bx:bx + bw]
    if roi.size == 0 or prop.endpoints_xyxy is None or prop.source_family != "F_thin":
        return roi

    # Only apply oriented corridor suppression on genuinely diagonal, sufficiently large crops
    ex1, ey1, ex2, ey2 = prop.endpoints_xyxy
    p1 = (ex1 - bx, ey1 - by)
    p2 = (ex2 - bx, ey2 - by)
    span = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if span < 32.0 or min(bw, bh) < 48:
        return roi
    # Check diagonalness: axis-aligned thin crops don't have huge dead opposite corners
    diag_box = math.hypot(bw, bh)
    if min(bw, bh) / max(1, max(bw, bh)) < 0.35 or span < 0.55 * diag_box:
        return roi

    rh, rw = roi.shape[:2]
    mask = np.zeros((rh, rw), dtype=np.uint8)
    corridor_thick = max(24, int(round(0.22 * span)))
    cv2.line(mask, p1, p2, 255, thickness=corridor_thick, lineType=cv2.LINE_AA)
    soft_alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (15, 15), 0)[..., None]

    # Blend outside pixels gently toward border median color so road/clutter corners do not dominate DINOv2 patches
    border_px = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]], axis=0)
    bg_med = np.median(border_px, axis=0).astype(np.float32)[None, None, :]
    eff_alpha = corridor_blend * soft_alpha + (1.0 - corridor_blend) * 1.0
    out = roi.astype(np.float32) * eff_alpha + bg_med * (1.0 - eff_alpha)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def evaluate_thin_object_metrics(
    proposals: list[ProposalBox],
    gt_box_xywh: tuple[int, int, int, int],
) -> dict:
    """Compute specialized thin-object localization metrics (object_01 audit vs GT)."""
    if not proposals:
        return {
            "best_iou": 0.0,
            "best_containment": 0.0,
            "best_bg_contamination": 1.0,
            "endpoint_included": False,
            "axis_angle_error_deg": 90.0,
        }

    gx, gy, gw, gh = gt_box_xywh
    # Approximate GT endpoints by principal bounding-box diagonal/axis
    tol_x = max(8.0, 0.10 * gw)
    tol_y = max(8.0, 0.10 * gh)
    gt_angle = float(math.degrees(math.atan2(gh, gw)))  # magnitude [0..90]

    best_p = max(proposals, key=lambda p: compute_iou_xywh(tuple(p.bbox_xywh), gt_box_xywh))
    bx, by, bw, bh = best_p.bbox_xywh
    best_iou = compute_iou_xywh((bx, by, bw, bh), gt_box_xywh)
    cont, bg_cont = compute_containment_and_bg_contamination((bx, by, bw, bh), gt_box_xywh)

    # Endpoint inclusion: does box cover from near top of GT to near bottom AND near left to right?
    ep_inc = (
        bx <= (gx + tol_x)
        and by <= (gy + tol_y)
        and (bx + bw) >= (gx + gw - tol_x)
        and (by + bh) >= (gy + gh - tol_y)
    )

    if best_p.oriented_angle_deg is not None:
        pred_angle = float(best_p.oriented_angle_deg)
    else:
        pred_angle = float(math.degrees(math.atan2(bh, max(1, bw))))
    ang_err = abs(pred_angle - gt_angle)
    ang_err = min(ang_err, abs(180.0 - ang_err))

    return {
        "best_iou": round(float(best_iou), 4),
        "best_containment": round(float(cont), 4),
        "best_bg_contamination": round(float(bg_cont), 4),
        "endpoint_included": bool(ep_inc),
        "axis_angle_error_deg": round(float(ang_err), 2),
        "best_source": best_p.source,
    }
