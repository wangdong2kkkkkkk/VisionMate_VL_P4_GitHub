"""Shared I/O, hashing, OBB/keypoint/centerline/mask geometry & statistical utilities for VL-P3."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OBJECT_IDS = [f"object_{i:02d}" for i in range(1, 6)]
LABELS_VI = {
    "object_01": "Gậy trắng hỗ trợ người khiếm thị",
    "object_02": "Chùm chìa khóa",
    "object_03": "Ví tiền",
    "object_04": "Đôi dép",
    "object_05": "Đôi giày",
}
DINOV2_SHA256 = "3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917"
YOLOX_SHA256 = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VL_P3_DIR = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = VL_P3_DIR.parents[1]


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_file_sha256(path: Path, expected_sha256: str) -> str:
    """Verify exact SHA-256 of an asset or model file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required asset: {path}")
    actual = compute_sha256(path).lower()
    if actual != expected_sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return actual


def load_image_unicode(path: Path) -> np.ndarray:
    """Read a BGR image safely from Unicode paths on Windows."""
    if not path.is_file():
        raise ValueError(f"Missing image file: {path}")
    try:
        blob = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(blob, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError(f"Cannot decode image {path}: {exc}") from exc
    if img is None or img.size == 0:
        raise ValueError(f"Invalid or empty image: {path}")
    return img


def save_image_unicode(path: Path, img: np.ndarray) -> None:
    """Write a BGR image safely to Unicode paths on Windows."""
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"Unsupported output extension: {ext}")
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Failed to encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def compute_ahash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute 64-bit average perceptual hash (hex)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    bits = (resized >= resized.mean()).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:0{(hash_size * hash_size) // 4}x}"


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute 64-bit difference perceptual hash (hex)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:0{(hash_size * hash_size) // 4}x}"


def hex_hamming_distance(h1: str, h2: str) -> int:
    """Compute bit-level Hamming distance between two equal-length hex hashes."""
    if len(h1) != len(h2):
        raise ValueError("Hash lengths must match")
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def normalize_title(title: str) -> str:
    """Normalize a Commons title for duplicate/leakage detection."""
    t = str(title or "").strip().lower()
    if t.startswith("file:"):
        t = t[5:]
    if "." in t:
        t = t.rsplit(".", 1)[0]
    return " ".join(t.replace("_", " ").replace("-", " ").split())


def pad_to_square(img: np.ndarray, pad_color: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Pad an image to square preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)


def compute_iou_xywh(box_a: tuple[int, int, int, int] | list[int], box_b: tuple[int, int, int, int] | list[int]) -> float:
    """Compute IoU of two axis-aligned (x, y, w, h) boxes."""
    ax, ay, aw, ah = [int(v) for v in box_a]
    bx, by, bw, bh = [int(v) for v in box_b]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(1, union))


def compute_containment_and_bg_contamination(
    prop_box: tuple[int, int, int, int] | list[int],
    gt_box: tuple[int, int, int, int] | list[int],
) -> tuple[float, float]:
    """Return (containment=|P∩GT|/|GT|, bg_contamination=1 - |P∩GT|/|P|)."""
    px, py, pw, ph = [int(v) for v in prop_box]
    gx, gy, gw, gh = [int(v) for v in gt_box]
    x1, y1 = max(px, gx), max(py, gy)
    x2, y2 = min(px + pw, gx + gw), min(py + ph, gy + gh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = max(1, gw * gh)
    prop_area = max(1, pw * ph)
    containment = float(inter / gt_area)
    bg_contam = float(1.0 - (inter / prop_area))
    return containment, max(0.0, min(1.0, bg_contam))


def obb_from_endpoints(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    thickness: float,
) -> tuple[list[float], list[list[float]]]:
    """Construct [cx, cy, length, thickness, angle_deg] and 4 corner points from endpoints."""
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    dx = x2 - x1
    dy = y2 - y1
    length = max(4.0, float(math.hypot(dx, dy)))
    thick = max(4.0, float(thickness))
    angle_deg = float(math.degrees(math.atan2(dy, dx)) % 180.0)
    rect = ((cx, cy), (length, thick), angle_deg)
    corners = cv2.boxPoints(rect)
    return (
        [round(cx, 2), round(cy, 2), round(length, 2), round(thick, 2), round(angle_deg, 2)],
        [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in corners],
    )


def sample_centerline_points(endpoints_xyxy: list[float], n_pts: int = 11) -> list[list[float]]:
    """Sample equidistant points along handle -> tip endpoints."""
    x1, y1, x2, y2 = [float(v) for v in endpoints_xyxy]
    return [
        [round(x1 + (k / max(1, n_pts - 1)) * (x2 - x1), 2), round(y1 + (k / max(1, n_pts - 1)) * (y2 - y1), 2)]
        for k in range(n_pts)
    ]


def compute_rotated_obb_iou(
    obb_a: tuple[float, float, float, float, float] | list[float],
    obb_b: tuple[float, float, float, float, float] | list[float],
) -> float:
    """Compute exact Intersection-over-Union between two oriented bounding boxes (cx, cy, w, h, angle_deg)."""
    ca_x, ca_y, wa, ha, aa = [float(v) for v in obb_a]
    cb_x, cb_y, wb, hb, ab = [float(v) for v in obb_b]
    if wa <= 0.0 or ha <= 0.0 or wb <= 0.0 or hb <= 0.0:
        return 0.0
    ra = ((ca_x, ca_y), (wa, ha), aa % 180.0)
    rb = ((cb_x, cb_y), (wb, hb), ab % 180.0)
    inter_type, pts = cv2.rotatedRectangleIntersection(ra, rb)
    if inter_type == cv2.INTERSECT_NONE or pts is None or len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32))
    inter_area = float(cv2.contourArea(hull))
    if inter_area <= 0.0:
        return 0.0
    union_area = wa * ha + wb * hb - inter_area
    return float(max(0.0, min(1.0, inter_area / max(1e-6, union_area))))


def axial_angle_diff_deg(angle_a: float, angle_b: float) -> float:
    """Compute undirected angular difference in [0, 90] degrees."""
    d = abs((float(angle_a) - float(angle_b)) % 180.0)
    return float(min(d, 180.0 - d))


def compute_endpoint_and_centerline_metrics(
    pred_ep: list[float],
    gt_ep: list[float],
    pred_centerline: list[list[float]],
    gt_centerline: list[list[float]],
    img_shape: tuple[int, int],
) -> dict[str, float]:
    """Compute endpoint error, PDJ@0.05/0.10, Chamfer centerline distance, inclusion, and full-shaft rate."""
    h, w = int(img_shape[0]), int(img_shape[1])
    diag = max(1.0, math.hypot(w, h))
    px1, py1, px2, py2 = [float(v) for v in pred_ep]
    gx1, gy1, gx2, gy2 = [float(v) for v in gt_ep]
    gt_len = max(8.0, math.hypot(gx2 - gx1, gy2 - gy1))

    d_direct = 0.5 * (math.hypot(px1 - gx1, py1 - gy1) + math.hypot(px2 - gx2, py2 - gy2))
    d_swap = 0.5 * (math.hypot(px1 - gx2, py1 - gy2) + math.hypot(px2 - gx1, py2 - gy1))
    if d_direct <= d_swap:
        ep1_err = math.hypot(px1 - gx1, py1 - gy1)
        ep2_err = math.hypot(px2 - gx2, py2 - gy2)
    else:
        ep1_err = math.hypot(px1 - gx2, py1 - gy2)
        ep2_err = math.hypot(px2 - gx1, py2 - gy1)

    mean_ep_px = 0.5 * (ep1_err + ep2_err)
    max_ep_px = max(ep1_err, ep2_err)
    norm_diag = mean_ep_px / diag
    norm_len = mean_ep_px / gt_len

    pdj_05 = float(0.5 * ((ep1_err <= 0.05 * diag) + (ep2_err <= 0.05 * diag)))
    pdj_10 = float(0.5 * ((ep1_err <= 0.10 * diag) + (ep2_err <= 0.10 * diag)))
    both_pdj_10 = 1.0 if max_ep_px <= 0.10 * diag else 0.0

    # Symmetric Chamfer distance between dense centerlines (resample 21 pts along pred & gt)
    p_pts = np.asarray(pred_centerline if len(pred_centerline) >= 2 else [[px1, py1], [px2, py2]], dtype=np.float32)
    g_pts = np.asarray(gt_centerline if len(gt_centerline) >= 2 else [[gx1, gy1], [gx2, gy2]], dtype=np.float32)

    def _dense(pts: np.ndarray, n: int = 25) -> np.ndarray:
        if len(pts) == 2:
            t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            return pts[0:1] * (1.0 - t) + pts[1:2] * t
        seg_d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg_d)])
        tot = max(1e-4, float(cum[-1]))
        u = np.linspace(0.0, tot, n, dtype=np.float32)
        return np.stack([np.interp(u, cum, pts[:, 0]), np.interp(u, cum, pts[:, 1])], axis=1).astype(np.float32)

    pd = _dense(p_pts, 25)
    gd = _dense(g_pts, 25)
    dmat = np.sqrt(np.sum((pd[:, None, :] - gd[None, :, :]) ** 2, axis=2))
    g2p = np.min(dmat, axis=0)
    p2g = np.min(dmat, axis=1)
    chamfer_px = float(0.5 * (np.mean(g2p) + np.mean(p2g)))
    chamfer_norm = float(chamfer_px / diag)

    cov_tol = max(16.0, 0.04 * diag)
    shaft_cov_frac = float(np.mean(g2p <= cov_tol))
    full_shaft_ok = 1.0 if shaft_cov_frac >= 0.70 else 0.0
    incl_tol = max(18.0, 0.08 * diag)
    endpoint_inclusion = 1.0 if max_ep_px <= incl_tol else 0.0

    return {
        "endpoint_err_px": round(mean_ep_px, 2),
        "endpoint_max_err_px": round(max_ep_px, 2),
        "endpoint_err_norm_diag": round(norm_diag, 4),
        "endpoint_err_norm_len": round(norm_len, 4),
        "keypoint_pdj_05": round(pdj_05, 4),
        "keypoint_pdj_10": round(pdj_10, 4),
        "keypoint_both_pdj_10": both_pdj_10,
        "centerline_chamfer_px": round(chamfer_px, 2),
        "centerline_chamfer_norm": round(chamfer_norm, 4),
        "shaft_coverage_frac": round(shaft_cov_frac, 4),
        "full_shaft_detected": full_shaft_ok,
        "endpoint_inclusion": endpoint_inclusion,
    }


def encode_mask_rle(mask: np.ndarray) -> dict:
    """Encode a binary 2D uint8 mask into compact COCO-style row-major counts RLE."""
    bin_m = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = bin_m.shape[:2]
    flat = bin_m.ravel()
    if flat.size == 0:
        return {"size": [int(h), int(w)], "counts": []}
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [flat.size]))
    lengths = (ends - starts).tolist()
    if flat[0] == 1:
        lengths = [0] + lengths
    return {"size": [int(h), int(w)], "counts": [int(v) for v in lengths]}


def decode_mask_rle(rle: dict) -> np.ndarray:
    """Decode an RLE dict back into a uint8 (0/255) 2D binary mask."""
    h, w = [int(v) for v in rle["size"]]
    counts = [int(v) for v in rle["counts"]]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        if c > 0 and pos < flat.size:
            end = min(flat.size, pos + c)
            if val == 1:
                flat[pos:end] = 255
        pos += c
        val ^= 1
    return flat.reshape((h, w))


def compute_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tol_px: int = 2) -> float:
    """Compute Boundary F1-score with spatial tolerance tol_px."""
    p = (np.asarray(pred_mask) > 0).astype(np.uint8)
    g = (np.asarray(gt_mask) > 0).astype(np.uint8)
    k = np.ones((3, 3), dtype=np.uint8)
    p_b = (p ^ cv2.erode(p, k)) > 0
    g_b = (g ^ cv2.erode(g, k)) > 0
    if not np.any(p_b) or not np.any(g_b):
        return 0.0
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    g_dil = cv2.dilate(g_b.astype(np.uint8), dk) > 0
    p_dil = cv2.dilate(p_b.astype(np.uint8), dk) > 0
    prec = float(np.mean(g_dil[p_b]))
    rec = float(np.mean(p_dil[g_b]))
    if prec + rec <= 1e-6:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Compute Mask IoU, Dice, Precision, Recall, Boundary F1, and Background Contamination."""
    p = np.asarray(pred_mask) > 0
    g = np.asarray(gt_mask) > 0
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    union = tp + fp + fn
    p_sum = tp + fp
    g_sum = tp + fn
    iou = float(tp / union) if union > 0 else 0.0
    dice = float((2.0 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    prec = float(tp / p_sum) if p_sum > 0 else 0.0
    rec = float(tp / g_sum) if g_sum > 0 else 0.0
    bg_contam = float(1.0 - prec) if p_sum > 0 else 1.0
    bf1 = compute_boundary_f1(p.astype(np.uint8), g.astype(np.uint8))
    area_rel_err = float(abs(p_sum - g_sum) / max(1, g_sum))
    return {
        "mask_iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "boundary_f1": round(bf1, 4),
        "bg_contamination": round(bg_contam, 4),
        "area_rel_error": round(area_rel_err, 4),
        "tp_pixels": tp,
        "pred_pixels": p_sum,
        "gt_pixels": g_sum,
    }


def wilson_ci_95(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute 95% binomial Wilson score confidence interval (in [0.0, 1.0])."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def format_count_pct_ci(k: int, n: int) -> dict:
    """Format count/total, percentage, and Wilson 95% CI in one dict."""
    lo, hi = wilson_ci_95(k, n)
    rate = round(k / max(1, n), 4)
    return {
        "k": int(k),
        "n": int(n),
        "ratio": f"{k}/{n}",
        "pct": round(rate * 100.0, 2),
        "ci95_low_pct": round(lo * 100.0, 2),
        "ci95_high_pct": round(hi * 100.0, 2),
        "formatted": f"{k}/{n} ({round(rate * 100.0, 2):.1f}%, 95% CI [{round(lo * 100.0, 2):.2f}, {round(hi * 100.0, 2):.2f}])",
    }


def paired_binary_comparison(
    outcomes_new: list[int],
    outcomes_ref: list[int],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Exact binomial McNemar two-sided test + paired bootstrap 95% CI on rate diff (new - ref)."""
    if len(outcomes_new) != len(outcomes_ref):
        raise ValueError("Paired lists must have equal length")
    n = len(outcomes_new)
    if n == 0:
        return {
            "n": 0,
            "new_wins": 0,
            "ref_wins": 0,
            "both_1": 0,
            "both_0": 0,
            "diff_pct": 0.0,
            "diff_ci95_pct": [0.0, 0.0],
            "mcnemar_exact_p": 1.0,
        }
    a = np.asarray(outcomes_new, dtype=np.int32)
    b = np.asarray(outcomes_ref, dtype=np.int32)
    b11 = int(np.sum((a == 1) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    b01 = int(np.sum((a == 0) & (b == 1)))
    b00 = int(np.sum((a == 0) & (b == 0)))

    disc = b10 + b01
    if disc == 0:
        p_val = 1.0
    else:
        k_min = min(b10, b01)
        tail = sum(math.comb(disc, i) * (0.5 ** disc) for i in range(k_min + 1))
        p_val = min(1.0, 2.0 * tail)

    diff = (a - b).astype(np.float64) * 100.0
    mean_diff = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diff[idx], axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n": n,
        "new_wins": b10,
        "ref_wins": b01,
        "both_1": b11,
        "both_0": b00,
        "diff_pct": round(mean_diff, 2),
        "diff_ci95_pct": [round(float(lo), 2), round(float(hi), 2)],
        "mcnemar_exact_p": round(float(p_val), 6),
    }


def make_contact_sheet(
    items: list[tuple[str, np.ndarray]],
    out_path: Path,
    cols: int = 5,
    cell_w: int = 260,
    cell_h: int = 210,
) -> Path:
    """Create a labeled contact sheet grid of visualization images."""
    if not items:
        raise ValueError("Cannot create contact sheet from empty item list")
    rows = math.ceil(len(items) / cols)
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, (label_text, vis) in enumerate(items):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell_h, c * cell_w
        ih, iw = vis.shape[:2]
        avail_h, avail_w = cell_h - 26, cell_w - 8
        sc = min(avail_w / max(1, iw), avail_h / max(1, ih))
        rw, rh = max(1, int(round(iw * sc))), max(1, int(round(ih * sc)))
        thumb = cv2.resize(vis, (rw, rh), interpolation=cv2.INTER_AREA)
        px = x0 + 4 + (avail_w - rw) // 2
        py = y0 + 22 + (avail_h - rh) // 2
        canvas[py:py + rh, px:px + rw] = thumb
        cv2.putText(
            canvas,
            label_text[:36],
            (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    save_image_unicode(out_path, canvas)
    return out_path
