"""Lightweight MobileNetV3-Small Multi-Task CNN & ONNX CPU Runner for VL-P4."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .common_utils import encode_mask_rle, obb_from_endpoints, order_endpoints_top_bottom, sample_centerline_points

INPUT_SIZE = 256
GRID_SIZE = 64  # stride 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class SquarePadTransform256:
    orig_h: int
    orig_w: int
    side: int
    pad_left: int
    pad_top: int
    scale_to_input: float
    scale_to_grid: float

    def orig_to_grid(self, x: float, y: float) -> tuple[float, float]:
        return (x + self.pad_left) * self.scale_to_grid, (y + self.pad_top) * self.scale_to_grid

    def grid_to_orig(self, gx: float, gy: float) -> tuple[float, float]:
        x = gx / max(1e-8, self.scale_to_grid) - self.pad_left
        y = gy / max(1e-8, self.scale_to_grid) - self.pad_top
        return float(np.clip(x, 0.0, self.orig_w)), float(np.clip(y, 0.0, self.orig_h))


def preprocess_image_256(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, SquarePadTransform256]:
    """Aspect-preserving square letterbox -> [1, 3, 256, 256] float32 RGB normalized."""
    h, w = img_bgr.shape[:2]
    side = max(h, w, 1)
    pad_left = (side - w) // 2
    pad_top = (side - h) // 2
    sq = np.full((side, side, 3), 127, dtype=np.uint8)
    sq[pad_top : pad_top + h, pad_left : pad_left + w] = img_bgr
    sq256 = cv2.resize(sq, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA if side > INPUT_SIZE else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(sq256, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    tensor = np.ascontiguousarray(norm.transpose(2, 0, 1)[None, ...], dtype=np.float32)
    tfm = SquarePadTransform256(
        orig_h=h,
        orig_w=w,
        side=side,
        pad_left=pad_left,
        pad_top=pad_top,
        scale_to_input=float(INPUT_SIZE) / float(side),
        scale_to_grid=float(GRID_SIZE) / float(side),
    )
    return tensor, sq256, tfm


class DepthwiseSepBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.dw(x)))
        y = self.act(self.bn2(self.pw(y)))
        return x + y


class MobileNetV3SmallMultiHead(nn.Module):
    """Lightweight FPN multi-head CNN (stride-4 64x64 spatial + global objectness).

    Outputs:
      spatial_logits: [B, 4, 64, 64] -> [top_kp, bot_kp, centerline, shaft_mask]
      obj_logit:      [B, 1]
    """

    def __init__(self, pretrained: bool = True, fpn_ch: int = 64) -> None:
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # indices: 1->16(64), 3->24(32), 8->48(16), 12->576(8)

        self.lat4 = nn.Conv2d(576, fpn_ch, kernel_size=1)
        self.lat3 = nn.Conv2d(48, fpn_ch, kernel_size=1)
        self.lat2 = nn.Conv2d(24, fpn_ch, kernel_size=1)
        self.lat1 = nn.Conv2d(16, fpn_ch, kernel_size=1)

        self.refine3 = DepthwiseSepBlock(fpn_ch)
        self.refine2 = DepthwiseSepBlock(fpn_ch)
        self.refine1 = DepthwiseSepBlock(fpn_ch)
        self.head_spatial = nn.Sequential(
            DepthwiseSepBlock(fpn_ch),
            nn.Conv2d(fpn_ch, 4, kernel_size=1),
        )

        # Global classification head combining semantic GAP(c4) + spatial heatmap statistics
        self.cls_head = nn.Sequential(
            nn.Linear(576 + 8, 96),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c1 = c2 = c3 = c4 = x
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 1:
                c1 = x
            elif idx == 3:
                c2 = x
            elif idx == 8:
                c3 = x
            elif idx == 12:
                c4 = x

        p4 = self.lat4(c4)  # [B, 64, 8, 8]
        p3 = self.refine3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.refine2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.refine1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False))
        spatial_logits = self.head_spatial(p1)  # [B, 4, 64, 64]

        gap_sem = F.adaptive_avg_pool2d(c4, 1).flatten(1)  # [B, 576]
        sp_prob = torch.sigmoid(spatial_logits)
        sp_max = F.adaptive_max_pool2d(sp_prob, 1).flatten(1)  # [B, 4]
        sp_avg = F.adaptive_avg_pool2d(sp_prob, 1).flatten(1)  # [B, 4]
        obj_logit = self.cls_head(torch.cat([gap_sem, sp_max, sp_avg], dim=1))  # [B, 1]
        return spatial_logits, obj_logit


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _peak_subpixel(hm: np.ndarray) -> tuple[float, float, float]:
    """Find peak + local 3x3 weighted subpixel centroid on (64, 64) heatmap."""
    iy, ix = np.unravel_index(int(np.argmax(hm)), hm.shape)
    val = float(hm[iy, ix])
    y0, y1 = max(0, iy - 1), min(hm.shape[0], iy + 2)
    x0, x1 = max(0, ix - 1), min(hm.shape[1], ix + 2)
    patch = np.maximum(0.0, hm[y0:y1, x0:x1])
    s = float(np.sum(patch))
    if s > 1e-6:
        yy, xx = np.mgrid[y0:y1, x0:x1]
        sy = float(np.sum(yy * patch) / s)
        sx = float(np.sum(xx * patch) / s)
        return sx, sy, val
    return float(ix), float(iy), val


def decode_cnn_outputs(
    spatial_logits: np.ndarray,
    obj_logit: np.ndarray,
    tfm: SquarePadTransform256,
    mode: str = "B2_CNN_MultiTask",
    mask_thr: float = 0.42,
) -> dict:
    """Decode raw [1,4,64,64] + [1,1] CNN tensors into original-image cane predictions."""
    sp = _sigmoid(spatial_logits[0])  # [4, 64, 64]
    top_hm, bot_hm, cl_hm, mask_hm = sp[0], sp[1], sp[2], sp[3]
    obj_prob = float(_sigmoid(obj_logit.reshape(-1))[0])

    if mode == "B1_CNN_Segmentation":
        # Derive centerline/endpoints from predicted shaft mask PCA when kp heads are inactive
        fg = mask_hm >= mask_thr
        ys, xs = np.nonzero(fg)
        if len(xs) >= 4:
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            mean_pt = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - mean_pt, full_matrices=False)
            d = vt[0]
            proj = (pts - mean_pt) @ d
            p1g = mean_pt + proj.min() * d
            p2g = mean_pt + proj.max() * d
            gx1, gy1, gx2, gy2 = float(p1g[0]), float(p1g[1]), float(p2g[0]), float(p2g[1])
        else:
            gx1, gy1, gx2, gy2 = 30.0, 20.0, 34.0, 44.0
        top_v = bot_v = cl_v = float(np.max(mask_hm))
    else:
        gx1, gy1, top_v = _peak_subpixel(top_hm)
        gx2, gy2, bot_v = _peak_subpixel(bot_hm)
        cl_v = float(np.max(cl_hm))
        # Fallback if endpoints collapsed too close (< 4 grid px) while centerline is strong
        if math.hypot(gx2 - gx1, gy2 - gy1) < 4.0 and cl_v >= 0.25:
            ys, xs = np.nonzero(cl_hm >= max(0.20, 0.5 * cl_v))
            if len(xs) >= 3:
                pts = np.stack([xs, ys], axis=1).astype(np.float32)
                mean_pt = pts.mean(axis=0)
                _, _, vt = np.linalg.svd(pts - mean_pt, full_matrices=False)
                d = vt[0]
                proj = (pts - mean_pt) @ d
                p1g = mean_pt + proj.min() * d
                p2g = mean_pt + proj.max() * d
                gx1, gy1, gx2, gy2 = float(p1g[0]), float(p1g[1]), float(p2g[0]), float(p2g[1])

    # Line support along predicted segment on centerline heatmap
    ts = np.linspace(0.0, 1.0, 11)
    xs_line = np.clip(np.round(gx1 + ts * (gx2 - gx1)).astype(int), 0, GRID_SIZE - 1)
    ys_line = np.clip(np.round(gy1 + ts * (gy2 - gy1)).astype(int), 0, GRID_SIZE - 1)
    cl_support = float(np.mean(cl_hm[ys_line, xs_line])) if mode != "B1_CNN_Segmentation" else float(np.mean(mask_hm[ys_line, xs_line]))

    ox1, oy1 = tfm.grid_to_orig(gx1, gy1)
    ox2, oy2 = tfm.grid_to_orig(gx2, gy2)
    ep = order_endpoints_top_bottom([ox1, oy1, ox2, oy2])
    cl_pts = sample_centerline_points(ep, 11)
    length_px = max(4.0, math.hypot(ep[2] - ep[0], ep[3] - ep[1]))
    obb, _ = obb_from_endpoints(ep[0], ep[1], ep[2], ep[3], thickness=max(6.0, 0.024 * length_px))

    # Map mask back to original image dimensions
    if mode == "A1_CNN_Keypoint":
        mask_orig = np.zeros((tfm.orig_h, tfm.orig_w), dtype=np.uint8)
        cv2.line(mask_orig, (int(round(ep[0])), int(round(ep[1]))), (int(round(ep[2])), int(round(ep[3]))), 255, max(3, int(round(0.015 * min(tfm.orig_h, tfm.orig_w)))))
    else:
        mask_sq = cv2.resize(mask_hm.astype(np.float32), (tfm.side, tfm.side), interpolation=cv2.INTER_LINEAR)
        crop_prob = mask_sq[tfm.pad_top : tfm.pad_top + tfm.orig_h, tfm.pad_left : tfm.pad_left + tfm.orig_w]
        mask_orig = ((crop_prob >= mask_thr) * 255).astype(np.uint8)

    return {
        "obj_prob": round(obj_prob, 5),
        "top_kp_peak": round(top_v, 5),
        "bot_kp_peak": round(bot_v, 5),
        "cl_peak": round(cl_v, 5),
        "cl_line_support": round(cl_support, 5),
        "mask_max": round(float(np.max(mask_hm)), 5),
        "mask_fg_ratio": round(float(np.mean(mask_hm >= mask_thr)), 5),
        "endpoints_xyxy": ep,
        "centerline_points_xy": cl_pts,
        "obb_cxcywh_angle": obb,
        "length_px": round(length_px, 2),
        "length_ratio": round(length_px / max(1.0, math.hypot(tfm.orig_w, tfm.orig_h)), 5),
        "mask_orig": mask_orig,
        "mask_rle": encode_mask_rle(mask_orig),
    }


class ONNXCaneRunner:
    """Offline CPU ONNX Runtime inference engine (zero PyTorch dependency)."""

    def __init__(self, onnx_path: Path, mode: str = "B2_CNN_MultiTask", num_threads: int = 4) -> None:
        self.model_path = Path(onnx_path)
        self.mode = mode
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(num_threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        tensor, sq256, tfm = preprocess_image_256(img_bgr)
        t_pre = time.perf_counter()
        sp_out, obj_out = self.session.run(None, {self.input_name: tensor})
        t_cnn = time.perf_counter()
        dec = decode_cnn_outputs(sp_out, obj_out, tfm, mode=self.mode)
        t_dec = time.perf_counter()
        dec["sq256"] = sq256
        dec["tfm"] = tfm
        dec["pre_ms"] = round((t_pre - t0) * 1000.0, 3)
        dec["cnn_ms"] = round((t_cnn - t_pre) * 1000.0, 3)
        dec["decode_ms"] = round((t_dec - t_cnn) * 1000.0, 3)
        dec["e2e_ms"] = round((t_dec - t0) * 1000.0, 3)
        return dec
