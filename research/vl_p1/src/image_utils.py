"""Image I/O, hashing, perceptual similarity, and preprocessing utilities for VL-P1."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OBJECT_IDS = [f"object_{i:02d}" for i in range(1, 6)]


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_image_unicode(path: Path) -> np.ndarray:
    """Read an image safely from Unicode paths on Windows."""
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
    """Write an image safely to Unicode paths on Windows."""
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"Unsupported output extension: {ext}")
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Failed to encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def compute_ahash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute average perceptual hash (hex string of hash_size*hash_size bits)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    bits = (resized >= mean_val).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:0{(hash_size * hash_size) // 4}x}"


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute difference perceptual hash (hex string of hash_size*hash_size bits)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def pad_to_square(
    img: np.ndarray,
    pad_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Pad an image to square preserving aspect ratio (useful for thin objects like canes)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)


def list_images(folder: Path) -> list[Path]:
    """List sorted valid image files in a folder."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
