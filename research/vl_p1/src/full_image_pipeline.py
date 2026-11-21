"""Task 2 Full-Image End-to-End Pipeline: Region Proposal (YOLOX / Hybrid) + Embedding Recognizer.

Never uses Ground Truth bounding boxes at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from .embedding_recognizer import EmbeddingRecognizer

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


def compute_iou_xywh(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(1, union))


@dataclass
class Proposal:
    bbox_xywh: tuple[int, int, int, int]
    source: str
    objectness: float


class YOLOXProposer:
    """Offline YOLOX-Nano ONNX proposal extractor (COCO-80)."""

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

    def propose(self, image: np.ndarray, conf_threshold: float = 0.15, max_boxes: int = 4) -> list[Proposal]:
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
                out.append(Proposal((bx, by, bwi, bhi), f"yolox_coco_{int(cids[mask][idx])}", round(scores_list[idx], 4)))
        return out


def foreground_contour_proposals(image: np.ndarray, max_boxes: int = 2) -> list[Proposal]:
    """Class-agnostic OpenCV foreground/border-contrast + edge envelope ROI proposal."""
    h, w = image.shape[:2]
    scale = min(1.0, 256.0 / float(max(h, w)))
    sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)

    # Color distance from border median + Canny edges
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
    cands: list[tuple[float, tuple[int, int, int, int]]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > 0.96 * sw * sh:
            continue
        # Mild preference for centered, non-trivial foreground objects
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
    return [Proposal(box, "fg_contour", round(sc, 4)) for sc, box in cands[:max_boxes]]


class FullImagePipeline:
    """End-to-end Task 2 pipeline: automatic ROI proposal + Method B embedding classification."""

    def __init__(
        self,
        recognizer: EmbeddingRecognizer,
        yolox_proposer: YOLOXProposer | None = None,
        proposer_mode: str = "hybrid",  # yolox_strict | yolox_low | hybrid
    ) -> None:
        self.recognizer = recognizer
        self.yolox = yolox_proposer
        self.proposer_mode = proposer_mode

    def _dedup_proposals(self, props: list[Proposal], iou_thr: float = 0.75) -> list[Proposal]:
        kept: list[Proposal] = []
        for p in props:
            if not any(compute_iou_xywh(p.bbox_xywh, k.bbox_xywh) >= iou_thr for k in kept):
                kept.append(p)
        return kept

    def generate_proposals(self, image: np.ndarray) -> tuple[list[Proposal], float]:
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        props: list[Proposal] = []

        if self.proposer_mode == "yolox_strict":
            if self.yolox is not None:
                props.extend(self.yolox.propose(image, conf_threshold=0.40, max_boxes=4))
        elif self.proposer_mode == "yolox_low":
            if self.yolox is not None:
                props.extend(self.yolox.propose(image, conf_threshold=0.15, max_boxes=4))
        elif self.proposer_mode == "hybrid":
            if self.yolox is not None:
                props.extend(self.yolox.propose(image, conf_threshold=0.15, max_boxes=2))
            props.extend(foreground_contour_proposals(image, max_boxes=2))
            # Center 72% framing prior + full frame prior as fallback candidates
            cw, ch = int(round(w * 0.72)), int(round(h * 0.72))
            props.append(Proposal(((w - cw) // 2, (h - ch) // 2, cw, ch), "center_72", 0.5))
            props.append(Proposal((0, 0, w, h), "full_frame", 0.4))
            props = self._dedup_proposals(props, iou_thr=0.78)[:4]
        else:
            raise ValueError(f"Unknown proposer_mode: {self.proposer_mode}")

        prop_ms = (time.perf_counter() - t0) * 1000.0
        return props, round(prop_ms, 3)

    def predict(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided to FullImagePipeline")
        t_start = time.perf_counter()
        proposals, prop_ms = self.generate_proposals(image)
        if not proposals:
            return {
                "status": "PROPOSAL_MISSED",
                "label": None,
                "score": 0.0,
                "margin": 0.0,
                "proposals": [],
                "selected_box": None,
                "selected_source": None,
                "proposal_ms": prop_ms,
                "total_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
            }

        best_known = None
        best_any = None
        eval_props = []
        for prop in proposals:
            x, y, bw, bh = prop.bbox_xywh
            roi = image[y:y + bh, x:x + bw]
            if roi.size == 0:
                continue
            res = self.recognizer.predict(roi)
            entry = {
                "bbox_xywh": prop.bbox_xywh,
                "source": prop.source,
                "objectness": prop.objectness,
                "decision": res,
            }
            eval_props.append(entry)
            if res["status"] == "KNOWN":
                if best_known is None or (res["score"], res["margin"]) > (
                    best_known["decision"]["score"],
                    best_known["decision"]["margin"],
                ):
                    best_known = entry
            if best_any is None or res["score"] > best_any["decision"]["score"]:
                best_any = entry

        chosen = best_known or best_any
        if chosen is None:
            return {
                "status": "PROPOSAL_MISSED",
                "label": None,
                "score": 0.0,
                "margin": 0.0,
                "proposals": [],
                "selected_box": None,
                "selected_source": None,
                "proposal_ms": prop_ms,
                "total_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
            }

        dec = chosen["decision"]
        return {
            "status": dec["status"],
            "label": dec["label"],
            "score": dec["score"],
            "margin": dec["margin"],
            "selected_box": chosen["bbox_xywh"],
            "selected_source": chosen["source"],
            "proposals": [p.bbox_xywh for p in proposals],
            "proposal_sources": [p.source for p in proposals],
            "proposal_ms": prop_ms,
            "total_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }
