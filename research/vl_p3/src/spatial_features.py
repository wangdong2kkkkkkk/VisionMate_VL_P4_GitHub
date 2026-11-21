"""Multi-scale Spatial Feature Extractor (DINOv2 16x16 Patch Tokens + Steered Ridges + YOLOX Anchors)
and Geometry-Preserving Data Augmentations for VL-P3.

Uses frozen DINOv2-Small INT8 ONNX (`last_hidden_state` [1, 257, 384] where [:, 1:, :]
gives the 16x16x384 spatial patch tokens!) and frozen YOLOX-Nano ONNX on CPU.
No PyTorch dependency required.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from .common_utils import (
    DINOV2_SHA256,
    IMAGENET_MEAN,
    IMAGENET_STD,
    YOLOX_SHA256,
    compute_sha256,
    decode_mask_rle,
    obb_from_endpoints,
    sample_centerline_points,
)

GRID_SIZE = 112
WORK_RIDGE_SIZE = 448
NUM_BASE_CHANNELS = 16
NUM_DINO_PCA = 12
NUM_DINO_SEM = 5
NUM_YOLOX_CTX = 3
TOTAL_SPATIAL_CH = NUM_BASE_CHANNELS + NUM_DINO_PCA + NUM_DINO_SEM + NUM_YOLOX_CTX  # 36

# Build 8 oriented thin-bar Gabor/line kernels at 448x448 once at import time
_ORIENT_ANGLES_RAD = [k * math.pi / 8.0 for k in range(8)]


def _build_steered_bar_kernels() -> list[tuple[np.ndarray, np.ndarray]]:
    """Build matched center-bar vs flanking-surround 11x11 kernels across 8 angles."""
    kernels: list[tuple[np.ndarray, np.ndarray]] = []
    yy, xx = np.mgrid[-6:7, -6:7].astype(np.float32)
    for th in _ORIENT_ANGLES_RAD:
        ux, uy = math.cos(th), math.sin(th)
        nx, ny = -uy, ux
        along = np.abs(xx * ux + yy * uy)
        perp = np.abs(xx * nx + yy * ny)
        # Central narrow ridge (|perp| <= 1.3, |along| <= 5.5) vs adjacent surround (2.5 <= |perp| <= 5.2)
        center = np.exp(-0.5 * (perp / 1.05) ** 2) * (along <= 5.5).astype(np.float32)
        center /= max(1e-6, float(np.sum(center)))
        side = np.exp(-0.5 * ((perp - 3.6) / 1.0) ** 2) * (along <= 5.5).astype(np.float32)
        side /= max(1e-6, float(np.sum(side)))
        kernels.append((center.astype(np.float32), side.astype(np.float32)))
    return kernels


_STEERED_KERNELS = _build_steered_bar_kernels()

_YX_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_yolox_grid(img_size: tuple[int, int] = (416, 416)) -> tuple[np.ndarray, np.ndarray]:
    if img_size not in _YX_GRID_CACHE:
        grids, strides = [], []
        for st in (8, 16, 32):
            hs, ws = img_size[0] // st, img_size[1] // st
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            g = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(g)
            strides.append(np.full((*g.shape[:2], 1), st))
        _YX_GRID_CACHE[img_size] = (np.concatenate(grids, 1), np.concatenate(strides, 1))
    return _YX_GRID_CACHE[img_size]


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-8)


@dataclass
class SquarePadTransform:
    """Bijective aspect-preserving mapping between original (H, W) and square (grid_size, grid_size)."""

    orig_h: int
    orig_w: int
    side: int
    pad_left: int
    pad_top: int
    scale_to_grid: float
    grid_size: int = GRID_SIZE

    def orig_to_grid(self, x: float, y: float) -> tuple[float, float]:
        return (x + self.pad_left) * self.scale_to_grid, (y + self.pad_top) * self.scale_to_grid

    def grid_to_orig(self, gx: float, gy: float) -> tuple[float, float]:
        x = gx / max(1e-8, self.scale_to_grid) - self.pad_left
        y = gy / max(1e-8, self.scale_to_grid) - self.pad_top
        return float(x), float(y)

    def warp_mask_to_grid(self, mask_orig: np.ndarray, out_size: int | None = None) -> np.ndarray:
        sz = out_size or self.grid_size
        sq = np.zeros((self.side, self.side), dtype=np.uint8)
        sq[self.pad_top : self.pad_top + self.orig_h, self.pad_left : self.pad_left + self.orig_w] = (mask_orig > 0).astype(np.uint8)
        return cv2.resize(sq, (sz, sz), interpolation=cv2.INTER_NEAREST)

    def unwarp_mask_to_orig(self, mask_sq: np.ndarray) -> np.ndarray:
        full_sq = cv2.resize(mask_sq.astype(np.float32), (self.side, self.side), interpolation=cv2.INTER_LINEAR)
        crop = full_sq[self.pad_top : self.pad_top + self.orig_h, self.pad_left : self.pad_left + self.orig_w]
        return crop


def pad_image_to_square_with_transform(
    img: np.ndarray,
    grid_size: int = GRID_SIZE,
    pad_val: tuple[int, int, int] = (128, 128, 128),
) -> tuple[np.ndarray, SquarePadTransform]:
    h, w = img.shape[:2]
    side = max(h, w, 1)
    top = (side - h) // 2
    left = (side - w) // 2
    sq = cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=pad_val)
    tfm = SquarePadTransform(
        orig_h=h,
        orig_w=w,
        side=side,
        pad_left=left,
        pad_top=top,
        scale_to_grid=float(grid_size) / float(side),
        grid_size=grid_size,
    )
    return sq, tfm


class DINOv2SpatialBackbone:
    """Runs frozen DINOv2-Small INT8 ONNX once per square canvas, yielding CLS (384) & 16x16x384 patch tokens."""

    def __init__(self, model_path: Path, num_threads: int = 4) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing DINOv2 ONNX: {self.model_path}")
        if compute_sha256(self.model_path) != DINOV2_SHA256:
            raise ValueError(f"DINOv2 SHA-256 mismatch: {self.model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def forward_square_and_patches(self, sq_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (cls_384_normed, patches_16x16x384_normed, ms)."""
        t0 = time.perf_counter()
        im224 = sq_bgr if sq_bgr.shape[:2] == (224, 224) else cv2.resize(sq_bgr, (224, 224), interpolation=cv2.INTER_AREA if max(sq_bgr.shape[:2]) > 224 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(im224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normed = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.ascontiguousarray(np.transpose(normed, (2, 0, 1))[None, ...], dtype=np.float32)
        raw = self.session.run(None, {self.input_name: tensor})[0]  # [1, 257, 384]
        cls_emb = _l2_normalize_rows(raw[0, 0, :])
        patches = _l2_normalize_rows(raw[0, 1:257, :]).reshape(16, 16, 384).astype(np.float32)
        ms = (time.perf_counter() - t0) * 1000.0
        return cls_emb.astype(np.float32), patches, round(ms, 2)

    def embed_canvas_cls(self, canvas_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        cls_emb, _, ms = self.forward_square_and_patches(canvas_bgr)
        return cls_emb, ms


class YOLOXContextExtractor:
    """Runs frozen YOLOX-Nano ONNX for person (class 0) & umbrella (class 25) spatial context."""

    def __init__(self, model_path: Path, num_threads: int = 4) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing YOLOX ONNX: {self.model_path}")
        if compute_sha256(self.model_path) != YOLOX_SHA256:
            raise ValueError(f"YOLOX SHA-256 mismatch: {self.model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (416, 416)

    def detect_context(self, image: np.ndarray, conf_thr: float = 0.20) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        r = min(self.input_size[0] / h, self.input_size[1] / w)
        rh, rw = max(1, int(round(h * r))), max(1, int(round(w * r)))
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
        mask = scores >= conf_thr
        if not np.any(mask):
            return [], round((time.perf_counter() - t0) * 1000.0, 2)
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
        idxs = cv2.dnn.NMSBoxes(boxes_list, scores_list, conf_thr, 0.45)
        out = []
        if len(idxs) > 0:
            for idx in idxs.flatten().tolist()[:14]:
                out.append({
                    "bbox_xywh": tuple(boxes_list[idx]),
                    "confidence": float(scores_list[idx]),
                    "class_id": int(fcids[idx]),
                })
        return out, round((time.perf_counter() - t0) * 1000.0, 2)


def _max_pool_4x(arr448: np.ndarray) -> np.ndarray:
    """Fast exact 4x4 max-pool from 448x448 -> 112x112 preserving thin 1-3px ridges."""
    a = arr448.reshape(GRID_SIZE, 4, GRID_SIZE, 4)
    return a.max(axis=(1, 3)).astype(np.float32)


def _mean_pool_4x(arr448: np.ndarray) -> np.ndarray:
    a = arr448.reshape(GRID_SIZE, 4, GRID_SIZE, 4)
    return a.mean(axis=(1, 3)).astype(np.float32)


def compute_base_and_ridge_maps(sq_bgr: np.ndarray, tfm: SquarePadTransform) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute 16 photometric + multi-orientation thin-bar ridge maps on (112, 112) & high-res cache."""
    sq448 = cv2.resize(sq_bgr, (WORK_RIDGE_SIZE, WORK_RIDGE_SIZE), interpolation=cv2.INTER_AREA if tfm.side > WORK_RIDGE_SIZE else cv2.INTER_LINEAR)
    lab = cv2.cvtColor(sq448, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(sq448, cv2.COLOR_BGR2HSV).astype(np.float32)
    l_ch = lab[:, :, 0] * (100.0 / 255.0)
    a_ch = lab[:, :, 1] - 128.0
    b_ch = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_ch, b_ch)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    gray_f = l_ch
    gray_u8 = lab[:, :, 0].astype(np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    th5 = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, k5).astype(np.float32) / 45.0
    th11 = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, k11).astype(np.float32) / 45.0
    bh7 = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k11).astype(np.float32) / 45.0

    # Steered 8-orientation thin-bar contrast filter on 448x448 lightness
    resps = []
    bilat_diffs = []
    for k_cen, k_sid in _STEERED_KERNELS:
        c_mean = cv2.filter2D(gray_f, cv2.CV_32F, k_cen)
        s_mean = cv2.filter2D(gray_f, cv2.CV_32F, k_sid)
        diff = np.maximum(0.0, c_mean - s_mean)
        bilat_diffs.append(diff)
        # Positive ridge combined with low-chroma whiteness gate
        resps.append(diff)
    stack_r = np.stack(resps, axis=0)  # [8, 448, 448]
    best_k = np.argmax(stack_r, axis=0)
    max_r = np.max(stack_r, axis=0)
    part_r = np.partition(stack_r, (3, 4), axis=0)
    med_r = 0.5 * (part_r[3] + part_r[4])
    aniso = np.clip((max_r - med_r) / np.maximum(max_r + 3.0, 4.0), 0.0, 1.0)
    steered_ridge = np.clip(max_r / 24.0, 0.0, 1.0) * (0.35 + 0.65 * aniso)
    bilat_contrast = np.clip(max_r / 22.0, 0.0, 1.0)

    # Double-angle orientation vector weighted by anisotropy
    ang_lut = np.array(_ORIENT_ANGLES_RAD, dtype=np.float32)
    theta_map = ang_lut[best_k]
    cos2_448 = np.cos(2.0 * theta_map) * steered_ridge
    sin2_448 = np.sin(2.0 * theta_map) * steered_ridge

    white_achrom = np.clip((l_ch - 52.0) / 36.0, 0.0, 1.0) * np.clip((30.0 - chroma) / 24.0, 0.0, 1.0)
    red_tip = (
        np.clip((a_ch - 12.0) / 26.0, 0.0, 1.0)
        * np.clip((sat - 60.0) / 110.0, 0.0, 1.0)
        * np.clip((val - 45.0) / 140.0, 0.0, 1.0)
        * ((hue <= 14.0) | (hue >= 164.0)).astype(np.float32)
    )
    wood_cue = (
        np.clip((a_ch - 3.0) / 16.0, 0.0, 1.0)
        * np.clip((b_ch - 8.0) / 22.0, 0.0, 1.0)
        * np.clip((72.0 - l_ch) / 35.0, 0.0, 1.0)
    )
    loc_bg = cv2.boxFilter(l_ch, cv2.CV_32F, (25, 25))
    loc_diff = np.clip((l_ch - loc_bg) / 25.0, -1.0, 1.0)
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.clip(np.hypot(gx, gy) / 180.0, 0.0, 1.0)

    # Valid unpadded mask
    valid448 = np.zeros((WORK_RIDGE_SIZE, WORK_RIDGE_SIZE), dtype=np.float32)
    sc4 = float(WORK_RIDGE_SIZE) / float(tfm.side)
    vx1 = max(0, int(round(tfm.pad_left * sc4)))
    vy1 = max(0, int(round(tfm.pad_top * sc4)))
    vx2 = min(WORK_RIDGE_SIZE, int(round((tfm.pad_left + tfm.orig_w) * sc4)))
    vy2 = min(WORK_RIDGE_SIZE, int(round((tfm.pad_top + tfm.orig_h) * sc4)))
    valid448[vy1:vy2, vx1:vx2] = 1.0

    # Pool onto 112x112 preserving thin ridges via max pooling for sparse cues and mean for dense cues
    ch_112 = np.stack([
        _mean_pool_4x(l_ch / 100.0),
        _mean_pool_4x(np.clip((28.0 - chroma) / 28.0, 0.0, 1.0)),
        _max_pool_4x(white_achrom),
        _max_pool_4x(red_tip),
        _max_pool_4x(np.clip(th5, 0.0, 1.0)),
        _max_pool_4x(np.clip(th11, 0.0, 1.0)),
        _max_pool_4x(steered_ridge),
        _max_pool_4x(aniso),
        _max_pool_4x(bilat_contrast),
        _mean_pool_4x(cos2_448),
        _mean_pool_4x(sin2_448),
        _max_pool_4x(np.clip(bh7, 0.0, 1.0)),
        _max_pool_4x(wood_cue),
        _max_pool_4x(loc_diff),
        _max_pool_4x(gmag),
        _mean_pool_4x(valid448),
    ], axis=-1).astype(np.float32)  # [112, 112, 16]

    hi_cache = {
        "l_ch_448": l_ch,
        "white_achrom_448": white_achrom,
        "red_tip_448": red_tip,
        "steered_ridge_448": steered_ridge,
        "wood_cue_448": wood_cue,
        "dark_cue_448": np.clip((42.0 - l_ch) / 30.0, 0.0, 1.0),
        "valid_448": valid448,
    }
    return ch_112, hi_cache


def compute_yolox_context_maps(detections: list[dict], tfm: SquarePadTransform) -> np.ndarray:
    """Build 3 context channels on 112x112: [person_mask, hand_sweep_prior, umbrella_mask]."""
    ctx = np.zeros((GRID_SIZE, GRID_SIZE, NUM_YOLOX_CTX), dtype=np.float32)
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE].astype(np.float32)
    for det in detections:
        cid = det["class_id"]
        conf = float(det["confidence"])
        bx, by, bw, bh = det["bbox_xywh"]
        gx1, gy1 = tfm.orig_to_grid(bx, by)
        gx2, gy2 = tfm.orig_to_grid(bx + bw, by + bh)
        ix1, iy1 = max(0, int(math.floor(gx1))), max(0, int(math.floor(gy1)))
        ix2, iy2 = min(GRID_SIZE, int(math.ceil(gx2))), min(GRID_SIZE, int(math.ceil(gy2)))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if cid == 0:
            ctx[iy1:iy2, ix1:ix2, 0] = np.maximum(ctx[iy1:iy2, ix1:ix2, 0], conf)
            # Pedestrian hand/sweep lower-side zone prior
            cx_g = 0.5 * (gx1 + gx2)
            cy_g = gy1 + 0.65 * max(1.0, gy2 - gy1)
            sx = max(4.0, (gx2 - gx1) * 0.75)
            sy = max(4.0, (gy2 - gy1) * 0.45)
            gauss = np.exp(-0.5 * (((xx - cx_g) / sx) ** 2 + ((yy - cy_g) / sy) ** 2)).astype(np.float32)
            ctx[:, :, 1] = np.maximum(ctx[:, :, 1], conf * gauss)
        elif cid == 25:  # COCO umbrella
            ctx[iy1:iy2, ix1:ix2, 2] = np.maximum(ctx[iy1:iy2, ix1:ix2, 2], conf)
    return ctx


class PatchSemanticHead:
    """Lightweight supervised PCA(12) + 5-class patch head over DINOv2 16x16x384 tokens."""

    def __init__(
        self,
        pca_mean: np.ndarray | None = None,
        pca_components: np.ndarray | None = None,
        sem_W: np.ndarray | None = None,
        sem_b: np.ndarray | None = None,
    ) -> None:
        self.pca_mean = pca_mean  # [384]
        self.pca_components = pca_components  # [384, 12]
        self.sem_W = sem_W  # [384, 5]
        self.sem_b = sem_b  # [5]

    def project_grid_to_112(self, patches_16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project 16x16x384 DINOv2 patch tokens -> (pca_112x112x12, sem_112x112x5)."""
        flat = patches_16.reshape(256, 384)
        if self.pca_mean is not None and self.pca_components is not None:
            centered = flat - self.pca_mean[None, :]
            pca16 = (centered @ self.pca_components).reshape(16, 16, NUM_DINO_PCA)
            # Soft scale PCA coefficients into ~[-1, 1]
            pca16 = np.tanh(pca16 * 1.6).astype(np.float32)
        else:
            # Deterministic fallback orthogonal slice until trained
            pca16 = flat[:, :NUM_DINO_PCA].reshape(16, 16, NUM_DINO_PCA) * 6.0

        if self.sem_W is not None and self.sem_b is not None:
            logits = flat @ self.sem_W + self.sem_b[None, :]
            probs = (1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))).reshape(16, 16, NUM_DINO_SEM).astype(np.float32)
        else:
            probs = np.full((16, 16, NUM_DINO_SEM), 0.5, dtype=np.float32)

        pca112 = cv2.resize(pca16, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_LINEAR)
        sem112 = cv2.resize(probs, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_LINEAR)
        return pca112.astype(np.float32), sem112.astype(np.float32)

    def to_state_dict(self) -> dict:
        return {
            "pca_mean": self.pca_mean.tolist() if self.pca_mean is not None else None,
            "pca_components": self.pca_components.tolist() if self.pca_components is not None else None,
            "sem_W": self.sem_W.tolist() if self.sem_W is not None else None,
            "sem_b": self.sem_b.tolist() if self.sem_b is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state: dict | None) -> "PatchSemanticHead":
        if not state:
            return cls()
        return cls(
            pca_mean=np.array(state["pca_mean"], dtype=np.float32) if state.get("pca_mean") is not None else None,
            pca_components=np.array(state["pca_components"], dtype=np.float32) if state.get("pca_components") is not None else None,
            sem_W=np.array(state["sem_W"], dtype=np.float32) if state.get("sem_W") is not None else None,
            sem_b=np.array(state["sem_b"], dtype=np.float32) if state.get("sem_b") is not None else None,
        )


@dataclass
class SpatialFeatureBundle:
    """Single-pass spatial feature packet for one image (~50-75 ms CPU)."""

    features_112: np.ndarray  # [112, 112, 36]
    patches_16: np.ndarray  # [16, 16, 384]
    cls_emb: np.ndarray  # [384]
    hi_cache: dict[str, np.ndarray]
    yolox_detections: list[dict]
    transform: SquarePadTransform
    sq_bgr: np.ndarray
    feature_ms: float
    yolox_ms: float
    dino_ms: float


def extract_spatial_features(
    image: np.ndarray,
    dino_backbone: DINOv2SpatialBackbone,
    yolox_ctx: YOLOXContextExtractor | None = None,
    patch_head: PatchSemanticHead | None = None,
) -> SpatialFeatureBundle:
    t0 = time.perf_counter()
    sq_bgr, tfm = pad_image_to_square_with_transform(image, grid_size=GRID_SIZE)
    cls_emb, patches_16, dino_ms = dino_backbone.forward_square_and_patches(sq_bgr)
    if yolox_ctx is not None:
        dets, yx_ms = yolox_ctx.detect_context(image)
    else:
        dets, yx_ms = [], 0.0
    base112, hi_cache = compute_base_and_ridge_maps(sq_bgr, tfm)
    head = patch_head or PatchSemanticHead()
    pca112, sem112 = head.project_grid_to_112(patches_16)
    ctx112 = compute_yolox_context_maps(dets, tfm)
    feat112 = np.concatenate([base112, pca112, sem112, ctx112], axis=-1).astype(np.float32)
    tot_ms = (time.perf_counter() - t0) * 1000.0
    return SpatialFeatureBundle(
        features_112=feat112,
        patches_16=patches_16,
        cls_emb=cls_emb,
        hi_cache=hi_cache,
        yolox_detections=dets,
        transform=tfm,
        sq_bgr=sq_bgr,
        feature_ms=round(tot_ms, 2),
        yolox_ms=yx_ms,
        dino_ms=dino_ms,
    )


# ---------------------------------------------------------------------------
# Geometry-preserving training augmentations (flips, affine rot/scale, occlusion, photometric)
# ---------------------------------------------------------------------------

def augment_horizontal_flip(
    img: np.ndarray,
    endpoints_xyxy: list[float] | None,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, list[float] | None, np.ndarray | None]:
    _, w = img.shape[:2]
    out_img = cv2.flip(img, 1)
    out_ep = None
    if endpoints_xyxy is not None:
        x1, y1, x2, y2 = [float(v) for v in endpoints_xyxy]
        out_ep = [round(w - 1.0 - x1, 2), round(y1, 2), round(w - 1.0 - x2, 2), round(y2, 2)]
    out_mask = cv2.flip(mask, 1) if mask is not None else None
    return out_img, out_ep, out_mask


def augment_rotation_scale(
    img: np.ndarray,
    endpoints_xyxy: list[float] | None,
    mask: np.ndarray | None,
    angle_deg: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, list[float] | None, np.ndarray | None]:
    h, w = img.shape[:2]
    cx, cy = 0.5 * w, 0.5 * h
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    out_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out_ep = None
    if endpoints_xyxy is not None:
        pts = np.array([[endpoints_xyxy[0], endpoints_xyxy[1], 1.0], [endpoints_xyxy[2], endpoints_xyxy[3], 1.0]], dtype=np.float32)
        tpts = (M @ pts.T).T
        x1 = float(np.clip(tpts[0, 0], 0.0, w - 1.0))
        y1 = float(np.clip(tpts[0, 1], 0.0, h - 1.0))
        x2 = float(np.clip(tpts[1, 0], 0.0, w - 1.0))
        y2 = float(np.clip(tpts[1, 1], 0.0, h - 1.0))
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        out_ep = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
    out_mask = None
    if mask is not None:
        out_mask = (cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0).astype(np.uint8)
    return out_img, out_ep, out_mask


def augment_brightness_contrast_and_occlusion(
    img: np.ndarray,
    endpoints_xyxy: list[float] | None,
    mask: np.ndarray | None,
    rng: np.random.Generator,
    apply_occlusion: bool = True,
) -> tuple[np.ndarray, list[float] | None, np.ndarray | None]:
    alpha = float(rng.uniform(0.82, 1.18))
    beta = float(rng.uniform(-16.0, 16.0))
    out_img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    out_mask = mask.copy() if mask is not None else None
    if apply_occlusion and endpoints_xyxy is not None:
        h, w = out_img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in endpoints_xyxy]
        # Place 1 realistic dark clothing/bag occluder over mid-shaft (t in [0.30, 0.70]) leaving endpoints intact
        t_c = float(rng.uniform(0.32, 0.68))
        ocx = x1 + t_c * (x2 - x1)
        ocy = y1 + t_c * (y2 - y1)
        L = math.hypot(x2 - x1, y2 - y1)
        rw = max(12, int(round(0.08 * min(h, w))))
        rh = max(14, int(round(0.18 * L)))
        ox1 = max(0, int(round(ocx - 0.5 * rw)))
        oy1 = max(0, int(round(ocy - 0.5 * rh)))
        ox2 = min(w, ox1 + rw)
        oy2 = min(h, oy1 + rh)
        col = tuple(int(c) for c in rng.integers(35, 95, size=3))
        out_img[oy1:oy2, ox1:ox2] = col
        # Visible shaft mask loses occluded pixels, while full endpoints & centerline remain intact!
        if out_mask is not None:
            out_mask[oy1:oy2, ox1:ox2] = 0
    return out_img, endpoints_xyxy, out_mask


def build_supervision_targets_112(
    endpoints_xyxy: list[float],
    mask_orig: np.ndarray | None,
    tfm: SquarePadTransform,
) -> dict[str, np.ndarray]:
    """Generate pixel supervision maps on 112x112 grid: endpoint/centerline heatmaps, orientation, mask."""
    gx1, gy1 = tfm.orig_to_grid(endpoints_xyxy[0], endpoints_xyxy[1])
    gx2, gy2 = tfm.orig_to_grid(endpoints_xyxy[2], endpoints_xyxy[3])
    if gy1 > gy2:
        gx1, gy1, gx2, gy2 = gx2, gy2, gx1, gy1

    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE].astype(np.float32)
    # Grip (top) and Tip (bottom) Gaussian heatmaps (sigma = 2.6 grid px)
    sigma2 = 2.0 * (2.6 ** 2)
    grip_hm = np.exp(-(((xx - gx1) ** 2 + (yy - gy1) ** 2) / sigma2)).astype(np.float32)
    tip_hm = np.exp(-(((xx - gx2) ** 2 + (yy - gy2) ** 2) / sigma2)).astype(np.float32)

    # Distance to line segment [g1, g2]
    vx, vy = gx2 - gx1, gy2 - gy1
    seg_len2 = max(1e-4, vx * vx + vy * vy)
    t_proj = np.clip(((xx - gx1) * vx + (yy - gy1) * vy) / seg_len2, 0.0, 1.0)
    proj_x = gx1 + t_proj * vx
    proj_y = gy1 + t_proj * vy
    dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    cl_hm = np.exp(-dist2 / (2.0 * (1.8 ** 2))).astype(np.float32)

    theta = math.atan2(vy, vx)
    cos2_target = np.full((GRID_SIZE, GRID_SIZE), math.cos(2.0 * theta), dtype=np.float32)
    sin2_target = np.full((GRID_SIZE, GRID_SIZE), math.sin(2.0 * theta), dtype=np.float32)

    if mask_orig is not None:
        mask_112 = tfm.warp_mask_to_grid(mask_orig, GRID_SIZE).astype(np.float32)
    else:
        mask_112 = (dist2 <= 2.2 ** 2).astype(np.float32)

    return {
        "grip_hm": grip_hm,
        "tip_hm": tip_hm,
        "centerline_hm": cl_hm,
        "cos2": cos2_target,
        "sin2": sin2_target,
        "shaft_mask": mask_112,
        "endpoints_grid": np.array([gx1, gy1, gx2, gy2], dtype=np.float32),
    }
