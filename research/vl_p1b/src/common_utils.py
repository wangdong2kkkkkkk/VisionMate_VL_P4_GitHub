"""Shared image I/O, hashing, geometry, contact-sheet, and Wilson 95% CI utilities for VL-P1b."""
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


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


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


def list_images(folder: Path) -> list[Path]:
    """List sorted valid image files in a folder."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def compute_ahash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute 64-bit average perceptual hash (hex)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    bits = (resized >= mean_val).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:0{(hash_size * hash_size) // 4}x}"


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute 64-bit difference perceptual hash (hex)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:0{(hash_size * hash_size) // 4}x}"


def hex_hamming_distance(h1: str, h2: str) -> int:
    """Compute bit-level Hamming distance between two equal-length hex hashes."""
    if len(h1) != len(h2):
        raise ValueError("Hash lengths must match")
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


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


def compute_iou_xywh(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection over Union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(1, union))


def compute_containment_and_bg_contamination(
    prop_box: tuple[int, int, int, int],
    gt_box: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Return (containment=|P∩GT|/|GT|, bg_contamination=1 - |P∩GT|/|P|)."""
    px, py, pw, ph = prop_box
    gx, gy, gw, gh = gt_box
    x1, y1 = max(px, gx), max(py, gy)
    x2, y2 = min(px + pw, gx + gw), min(py + ph, gy + gh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = max(1, gw * gh)
    prop_area = max(1, pw * ph)
    containment = float(inter / gt_area)
    bg_contam = float(1.0 - (inter / prop_area))
    return containment, max(0.0, min(1.0, bg_contam))


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
    """Format count/total, percentage, and Wilson 95% CI in one JSON-friendly dict."""
    lo, hi = wilson_ci_95(k, n)
    rate = round(k / max(1, n), 4)
    return {
        "k": k,
        "n": n,
        "ratio": f"{k}/{n}",
        "pct": round(rate * 100.0, 2),
        "ci95_low_pct": round(lo * 100.0, 2),
        "ci95_high_pct": round(hi * 100.0, 2),
    }


def make_contact_sheet(
    items: list[tuple[str, np.ndarray, tuple[int, int, int, int] | None]],
    out_path: Path,
    cols: int = 5,
    cell_w: int = 240,
    cell_h: int = 190,
) -> Path:
    """Create a labeled thumbnail grid (optionally drawing GT bbox in green) for visual audit."""
    if not items:
        raise ValueError("Cannot create contact sheet from empty item list")
    rows = math.ceil(len(items) / cols)
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, (label_text, img, bbox) in enumerate(items):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell_h, c * cell_w
        vis = img.copy()
        if bbox is not None:
            bx, by, bw, bh = bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 200, 0), max(2, min(vis.shape[:2]) // 180))
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
            label_text[:34],
            (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    save_image_unicode(out_path, canvas)
    return out_path
