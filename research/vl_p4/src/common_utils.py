"""Shared geometry, hashing, RLE mask, image I/O, and statistical utilities for VL-P4."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np
from scipy.stats import binomtest

VL_P4_DIR = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = VL_P4_DIR.parents[1]
WORKSPACE_ROOT = WORKTREE_ROOT.parent


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_title(title: str) -> str:
    s = str(title or "").strip().lower()
    s = re.sub(r"^file:", "", s)
    s = re.sub(r"\.[a-z0-9]{2,5}$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def compute_ahash(img: np.ndarray) -> str:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    m = float(np.mean(small))
    bits = (small.flatten() >= m).astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def compute_dhash(img: np.ndarray) -> str:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    diff = (small[:, 1:] > small[:, :-1]).flatten()
    val = 0
    for b in diff:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def hex_hamming_distance(h1: str, h2: str) -> int:
    if not h1 or not h2:
        return 64
    return int((int(h1, 16) ^ int(h2, 16)).bit_count())


def load_image_unicode(path: Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to decode image: {p}")
    return img


def save_image_unicode(path: Path, img: np.ndarray, quality: int = 92) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {p}")
    buf.tofile(str(p))


def order_endpoints_top_bottom(ep: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in ep]
    if (y1 > y2) or (abs(y1 - y2) <= 1e-3 and x1 > x2):
        return [round(x2, 2), round(y2, 2), round(x1, 2), round(y1, 2)]
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def sample_centerline_points(ep: Iterable[float], n_pts: int = 11) -> list[list[float]]:
    x1, y1, x2, y2 = [float(v) for v in ep]
    pts = []
    for i in range(n_pts):
        t = i / max(1, n_pts - 1)
        pts.append([round(x1 + t * (x2 - x1), 2), round(y1 + t * (y2 - y1), 2)])
    return pts


def obb_from_endpoints(x1: float, y1: float, x2: float, y2: float, thickness: float = 10.0) -> tuple[list[float], list[list[float]]]:
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    dx = x2 - x1
    dy = y2 - y1
    length = max(4.0, math.hypot(dx, dy))
    angle = math.degrees(math.atan2(dy, dx))
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    hl = 0.5 * length
    ht = 0.5 * max(2.0, thickness)
    corners = [
        [round(cx - hl * ux - ht * nx, 2), round(cy - hl * uy - ht * ny, 2)],
        [round(cx + hl * ux - ht * nx, 2), round(cy + hl * uy - ht * ny, 2)],
        [round(cx + hl * ux + ht * nx, 2), round(cy + hl * uy + ht * ny, 2)],
        [round(cx - hl * ux + ht * nx, 2), round(cy - hl * uy + ht * ny, 2)],
    ]
    return [round(cx, 2), round(cy, 2), round(length, 2), round(thickness, 2), round(angle, 2)], corners


def axial_angle_diff_deg(a1: float, a2: float) -> float:
    d = abs(a1 - a2) % 180.0
    return min(d, 180.0 - d)


def encode_mask_rle(mask: np.ndarray) -> dict:
    bin_m = (mask > 0).astype(np.uint8)
    h, w = bin_m.shape[:2]
    flat = bin_m.flatten()
    if flat.size == 0:
        return {"size": [h, w], "counts": []}
    changes = np.where(flat[1:] != flat[:-1])[0] + 1
    idx = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(idx).tolist()
    counts = [0] + runs if flat[0] == 1 else runs
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def decode_mask_rle(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in rle["counts"]:
        if c > 0:
            if val == 1:
                flat[pos : pos + c] = 255
            pos += c
        val = 1 - val
    return flat.reshape((h, w))


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)
    inter = int(np.sum(p & g))
    union = int(np.sum(p | g))
    p_sum = int(np.sum(p))
    g_sum = int(np.sum(g))
    iou = inter / union if union > 0 else (1.0 if p_sum == 0 and g_sum == 0 else 0.0)
    dice = (2.0 * inter) / (p_sum + g_sum) if (p_sum + g_sum) > 0 else (1.0 if p_sum == 0 and g_sum == 0 else 0.0)
    prec = inter / p_sum if p_sum > 0 else 0.0
    rec = inter / g_sum if g_sum > 0 else 0.0
    return {
        "mask_iou": round(float(iou), 4),
        "dice": round(float(dice), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "bg_contamination": round(float(1.0 - prec), 4) if p_sum > 0 else 0.0,
    }


def compute_endpoint_and_centerline_metrics(
    pred_ep: Iterable[float],
    gt_ep: Iterable[float],
    pred_cl: list[list[float]],
    gt_cl: list[list[float]],
    hw: tuple[int, int],
) -> dict[str, float]:
    h, w = hw
    diag = max(1.0, math.hypot(w, h))
    px1, py1, px2, py2 = [float(v) for v in pred_ep]
    gx1, gy1, gx2, gy2 = [float(v) for v in gt_ep]
    gt_len = max(1.0, math.hypot(gx2 - gx1, gy2 - gy1))

    d_direct = 0.5 * (math.hypot(px1 - gx1, py1 - gy1) + math.hypot(px2 - gx2, py2 - gy2))
    d_swap = 0.5 * (math.hypot(px1 - gx2, py1 - gy2) + math.hypot(px2 - gx1, py2 - gy1))
    if d_direct <= d_swap:
        e1 = math.hypot(px1 - gx1, py1 - gy1)
        e2 = math.hypot(px2 - gx2, py2 - gy2)
    else:
        e1 = math.hypot(px1 - gx2, py1 - gy2)
        e2 = math.hypot(px2 - gx1, py2 - gy1)

    ep_err = 0.5 * (e1 + e2)
    pdj_05 = 0.5 * (float(e1 <= 0.05 * diag) + float(e2 <= 0.05 * diag))
    pdj_10 = 0.5 * (float(e1 <= 0.10 * diag) + float(e2 <= 0.10 * diag))
    both_pdj_10 = float(e1 <= 0.10 * diag and e2 <= 0.10 * diag)

    pa = np.asarray(pred_cl, dtype=np.float32)
    ga = np.asarray(gt_cl, dtype=np.float32)
    if pa.size and ga.size:
        dmat = np.linalg.norm(pa[:, None, :] - ga[None, :, :], axis=-1)
        chamfer = 0.5 * (float(np.mean(np.min(dmat, axis=1))) + float(np.mean(np.min(dmat, axis=0))))
    else:
        chamfer = diag

    ang_p = math.degrees(math.atan2(py2 - py1, px2 - px1))
    ang_g = math.degrees(math.atan2(gy2 - gy1, gx2 - gx1))
    ang_err = axial_angle_diff_deg(ang_p, ang_g)

    return {
        "endpoint_err_px": round(ep_err, 4),
        "endpoint_err_norm_diag": round(ep_err / diag, 4),
        "endpoint_err_norm_len": round(ep_err / gt_len, 4),
        "keypoint_pdj_05": round(pdj_05, 4),
        "keypoint_pdj_10": round(pdj_10, 4),
        "keypoint_both_pdj_10": round(both_pdj_10, 4),
        "centerline_chamfer_px": round(chamfer, 4),
        "centerline_chamfer_norm": round(chamfer / diag, 4),
        "angle_error_deg": round(ang_err, 4),
    }


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_binary_comparison(new_ok: list[int], ref_ok: list[int]) -> dict:
    assert len(new_ok) == len(ref_ok)
    n = len(new_ok)
    b11 = sum(1 for a, b in zip(new_ok, ref_ok) if a == 1 and b == 1)
    b10 = sum(1 for a, b in zip(new_ok, ref_ok) if a == 1 and b == 0)
    b01 = sum(1 for a, b in zip(new_ok, ref_ok) if a == 0 and b == 1)
    b00 = sum(1 for a, b in zip(new_ok, ref_ok) if a == 0 and b == 0)
    disc = b10 + b01
    pval = float(binomtest(b10, disc, 0.5, alternative="two-sided").pvalue) if disc > 0 else 1.0
    diff = (b10 - b01) / max(1, n)
    se = math.sqrt(max(0.0, (b10 + b01) - ((b10 - b01) ** 2) / max(1, n))) / max(1, n)
    return {
        "n": n,
        "new_wins": b10,
        "ref_wins": b01,
        "both_1": b11,
        "both_0": b00,
        "diff_pct": round(diff * 100.0, 2),
        "diff_ci95_pct": [round((diff - 1.96 * se) * 100.0, 2), round((diff + 1.96 * se) * 100.0, 2)],
        "mcnemar_exact_p": round(pval, 6),
    }


def make_percent_grid_sheet(items: list[tuple[str, np.ndarray]], out_path: Path, cols: int = 3, cell_w: int = 380, cell_h: int = 320) -> None:
    """Render contact sheet with 0..100 normalized percent coordinate grids for manual GT annotation."""
    if not items:
        return
    rows = math.ceil(len(items) / cols)
    pad_t = 24
    canvas = np.full((rows * (cell_h + pad_t), cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, (label, img) in enumerate(items):
        r, c = divmod(idx, cols)
        x0 = c * cell_w
        y0 = r * (cell_h + pad_t) + pad_t
        h, w = img.shape[:2]
        sc = min((cell_w - 8) / max(1, w), (cell_h - 8) / max(1, h))
        rw, rh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
        res = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
        ox = x0 + (cell_w - rw) // 2
        oy = y0 + (cell_h - rh) // 2
        canvas[oy : oy + rh, ox : ox + rw] = res
        # Draw 0..100% grid lines every 20%
        for pct in (0, 20, 40, 60, 80, 100):
            gx = ox + int(round((pct / 100.0) * (rw - 1)))
            gy = oy + int(round((pct / 100.0) * (rh - 1)))
            color = (0, 210, 210) if pct in (20, 40, 60, 80) else (0, 140, 255)
            cv2.line(canvas, (gx, oy), (gx, oy + rh - 1), color, 1)
            cv2.line(canvas, (ox, gy), (ox + rw - 1, gy), color, 1)
            if pct in (20, 40, 60, 80):
                cv2.putText(canvas, str(pct), (gx + 2, oy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, str(pct), (ox + 2, gy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, label[:46], (x0 + 6, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
    save_image_unicode(out_path, canvas, quality=92)
