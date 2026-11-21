"""Method A: Hard-Negative Recognition Research with DINOv2-Small INT8.

Implements four ablation stages on top of frozen DINOv2-Small INT8:
- A0_baseline_hybrid: VL-P1 global hybrid threshold baseline
- A1_neg_prototypes: hard-negative prototype & exemplar similarity penalty
- A2_adaptive_thresholds: per-category thresholds + pos-vs-neg margin + 3-state logic
- A3_full_verifier: A2 + lightweight L2-regularized candidate verifier (DINOv2 + ROI geometry)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from .common_utils import OBJECT_IDS, list_images, load_image_unicode, pad_to_square
from .labeling_protocol import HARD_NEGATIVE_GROUPS, TARGET_CONFUSER_MAP

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class HardNegConfig:
    mode: str = "A3_full_verifier"  # A0_baseline_hybrid | A1_neg_prototypes | A2_adaptive_thresholds | A3_full_verifier
    lambda_neg: float = 0.35
    neg_floor: float = 0.30
    tau_global: float = 0.4758
    tau_per_class: dict[str, float] = field(default_factory=lambda: {oid: 0.45 for oid in OBJECT_IDS})
    delta_pos: float = 0.02
    delta_neg_min: dict[str, float] = field(default_factory=lambda: {oid: 0.04 for oid in OBJECT_IDS})
    verifier_min_score: dict[str, float] = field(default_factory=lambda: {oid: -0.05 for oid in OBJECT_IDS})
    ridge_alpha: float = 12.0
    num_threads: int = 4


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < eps:
        out = np.zeros_like(v, dtype=np.float32)
        if out.size > 0:
            out[0] = 1.0
        return out
    return (v / norm).astype(np.float32)


def preprocess_roi_dinov2(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Aspect-preserving RGB ImageNet-normalized NCHW float32 preprocessing."""
    if image is None or image.size == 0:
        raise ValueError("Empty image provided to preprocess_roi_dinov2")
    work = image
    if work.ndim == 2:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    elif work.shape[2] == 4:
        work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)

    border_px = np.concatenate([work[0, :], work[-1, :], work[:, 0], work[:, -1]], axis=0)
    med = tuple(int(c) for c in np.median(border_px, axis=0))
    sq = pad_to_square(work, pad_color=med)

    h, w = sq.shape[:2]
    interp = cv2.INTER_AREA if min(h, w) > size else cv2.INTER_LINEAR
    resized = cv2.resize(sq, (size, size), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(normed, (2, 0, 1))[None, ...], dtype=np.float32)


def extract_roi_geometry_features(image: np.ndarray) -> np.ndarray:
    """Extract 16 content-only geometric, chromatic, and texture descriptors from an ROI."""
    if image is None or image.size == 0:
        return np.zeros(16, dtype=np.float32)
    h, w = image.shape[:2]
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3]
    scale = min(1.0, 160.0 / float(max(h, w, 1)))
    sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 1-2: Box aspect & narrowness
    log_ar = float(np.clip(np.log(max(w, 1) / float(max(h, 1))), -3.5, 3.5)) / 3.5
    narrowness = 1.0 - (float(min(w, h)) / float(max(w, h, 1)))

    # 3-5: Foreground contour shape cues (elongation, fill, solidity)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]], axis=0)
    bg_med = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg_med, axis=2)
    fg_mask = (dist > max(8.0, float(np.percentile(dist, 55)))).astype(np.uint8)
    fg_frac = float(np.mean(fg_mask))
    elongation = 0.0
    solidity = 0.0
    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 1.0:
            solidity = float(np.clip(area / hull_area, 0.0, 1.0))
        rect = cv2.minAreaRect(largest)
        rw, rh = rect[1]
        if max(rw, rh) > 1.0:
            elongation = 1.0 - float(min(rw, rh) / max(rw, rh))

    # 6-8: White cane chromatic/reflective cues (high lum low sat, red tape accent, brown wood)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)
    white_ratio = float(np.mean((sat < 55.0) & (val > 175.0)))
    red_ratio = float(np.mean(((hue <= 12.0) | (hue >= 165.0)) & (sat > 85.0) & (val > 70.0)))
    wood_brown_ratio = float(np.mean((hue >= 8.0) & (hue <= 28.0) & (sat > 55.0) & (val < 180.0)))

    # 9-11: Metallic & edge high-frequency cues (keys/locks vs smooth leather)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_norm = float(np.tanh(np.std(lap) / 45.0))
    center_crop = edges[sh // 4: (3 * sh) // 4, sw // 4: (3 * sw) // 4]
    center_edge_ratio = float(np.mean(center_crop > 0)) if center_crop.size > 0 else edge_density

    # 12-13: Straight line proportion via HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(15, min(sw, sh) // 5),
                            minLineLength=max(12, min(sw, sh) // 4), maxLineGap=6)
    line_count_norm = float(np.tanh((0 if lines is None else len(lines)) / 12.0))
    max_line_len_norm = 0.0
    if lines is not None and len(lines) > 0:
        lens = [float(np.hypot(x2 - x1, y2 - y1)) for x1, y1, x2, y2 in lines[:, 0, :]]
        max_line_len_norm = float(np.clip(max(lens) / max(1.0, float(np.hypot(sw, sh))), 0.0, 1.0))

    # 14-16: Contrast & specular peak cues
    specular_ratio = float(np.mean(val > 235.0))
    dark_ratio = float(np.mean(val < 55.0))
    gray_std_norm = float(np.clip(np.std(gray) / 90.0, 0.0, 1.0))

    feat = np.array([
        log_ar, narrowness, fg_frac, elongation, solidity,
        white_ratio, red_ratio, wood_brown_ratio,
        edge_density, lap_norm, center_edge_ratio,
        line_count_norm, max_line_len_norm,
        specular_ratio, dark_ratio, gray_std_norm,
    ], dtype=np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)


class DINOv2Embedder:
    """Offline ONNX Runtime wrapper for quantized DINOv2 ViT-S/14."""

    def __init__(self, models_dir: Path, num_threads: int = 4) -> None:
        self.model_path = Path(models_dir) / "dinov2_small_quantized.onnx"
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Missing DINOv2 ONNX model at {self.model_path}. Offline inference requires local weights."
            )
        t0 = time.perf_counter()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0

    def extract(self, image: np.ndarray) -> np.ndarray:
        tensor = preprocess_roi_dinov2(image, size=224)
        hidden = self.session.run(None, {self.input_name: tensor})[0]  # (1, 257, 384)
        cls_vec = _l2_normalize(hidden[0, 0, :])
        patch_mean = _l2_normalize(hidden[0, 1:, :].mean(axis=0))
        return _l2_normalize(np.concatenate([cls_vec, patch_mean], axis=0))


class HardNegativeRecognizer:
    """Few-shot DINOv2 recognizer with negative prototypes, adaptive thresholds, and candidate verification."""

    def __init__(self, models_dir: Path, config: HardNegConfig | None = None) -> None:
        self.cfg = config or HardNegConfig()
        self.embedder = DINOv2Embedder(models_dir, num_threads=self.cfg.num_threads)

        # Positive gallery (from 5-shot train_pos)
        self.pos_embeddings: dict[str, np.ndarray] = {}
        self.pos_prototypes: dict[str, np.ndarray] = {}
        self.pos_names: dict[str, list[str]] = {}

        # Hard-negative gallery (from hard_neg_gallery across 7 groups)
        self.neg_embeddings: dict[str, np.ndarray] = {}
        self.neg_prototypes: dict[str, np.ndarray] = {}
        self.neg_names: dict[str, list[str]] = {}

        # Lightweight linear ridge verifiers (one per object_id: +1 for class oid, -1 for hard-neg confusers)
        self.verifiers: dict[str, tuple[np.ndarray, float]] = {}  # oid -> (w_784, bias)
        self.geom_mean: np.ndarray = np.zeros(16, dtype=np.float32)
        self.geom_std: np.ndarray = np.ones(16, dtype=np.float32)

        # Cached feature vectors during gallery fit so verifier training avoids re-running inference
        self._pos_geoms: dict[str, np.ndarray] = {}
        self._neg_geoms: dict[str, np.ndarray] = {}

    def fit_positives(self, train_pos_dir: Path) -> dict:
        t0 = time.perf_counter()
        self.pos_embeddings.clear()
        self.pos_prototypes.clear()
        self.pos_names.clear()
        self._pos_geoms.clear()
        for oid in OBJECT_IDS:
            embs, geoms, names = [], [], []
            for p in list_images(train_pos_dir / oid):
                img = load_image_unicode(p)
                embs.append(self.embedder.extract(img))
                geoms.append(extract_roi_geometry_features(img))
                names.append(p.name)
            if not embs:
                raise ValueError(f"No positive training images found in {train_pos_dir / oid}")
            mat = np.vstack(embs).astype(np.float32)
            self.pos_embeddings[oid] = mat
            self.pos_prototypes[oid] = _l2_normalize(mat.mean(axis=0))
            self.pos_names[oid] = names
            self._pos_geoms[oid] = np.vstack(geoms).astype(np.float32)
        return {"enroll_pos_ms": round((time.perf_counter() - t0) * 1000.0, 2)}

    def fit_hard_negatives(self, hard_neg_dir: Path) -> dict:
        t0 = time.perf_counter()
        self.neg_embeddings.clear()
        self.neg_prototypes.clear()
        self.neg_names.clear()
        self._neg_geoms.clear()
        count = 0
        for grp in HARD_NEGATIVE_GROUPS:
            folder = hard_neg_dir / grp
            embs, geoms, names = [], [], []
            for p in list_images(folder):
                img = load_image_unicode(p)
                embs.append(self.embedder.extract(img))
                geoms.append(extract_roi_geometry_features(img))
                names.append(p.name)
            if embs:
                mat = np.vstack(embs).astype(np.float32)
                self.neg_embeddings[grp] = mat
                self.neg_prototypes[grp] = _l2_normalize(mat.mean(axis=0))
                self.neg_names[grp] = names
                self._neg_geoms[grp] = np.vstack(geoms).astype(np.float32)
                count += len(embs)
        if count == 0:
            raise ValueError(f"No hard-negative gallery images found under {hard_neg_dir}")
        return {
            "neg_gallery_count": count,
            "enroll_neg_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def serialized_db_sizes(self) -> dict[str, int]:
        def _sz(d1: dict, d2: dict, prefix: str) -> int:
            buf = io.BytesIO()
            payload = {f"{prefix}_e_{k}": v for k, v in d1.items()}
            payload.update({f"{prefix}_p_{k}": v for k, v in d2.items()})
            if payload:
                np.savez(buf, **payload)
            return len(buf.getvalue())

        pos_b = _sz(self.pos_embeddings, self.pos_prototypes, "pos")
        neg_b = _sz(self.neg_embeddings, self.neg_prototypes, "neg")
        ver_b = len(json.dumps({
            k: [w.tolist(), float(b)] for k, (w, b) in self.verifiers.items()
        }).encode("utf-8")) if self.verifiers else 0
        return {"pos_db_bytes": pos_b, "neg_db_bytes": neg_b, "verifier_bytes": ver_b, "total_db_bytes": pos_b + neg_b + ver_b}

    def fit_candidate_verifiers(self, cal_items: list[tuple[str, str, np.ndarray]]) -> None:
        """Fit per-class L2-regularized Ridge verifiers using ONLY train_pos, hard_neg_gallery, and cal_items."""
        # Collect feature matrix across train_pos + hard_neg_gallery + calibration samples
        pos_feats_by_cls: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {oid: [] for oid in OBJECT_IDS}
        neg_feats_by_grp: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {g: [] for g in HARD_NEGATIVE_GROUPS}
        cal_neg_all: list[tuple[np.ndarray, np.ndarray]] = []

        for oid in OBJECT_IDS:
            for e, g in zip(self.pos_embeddings[oid], self._pos_geoms[oid]):
                pos_feats_by_cls[oid].append((e, g))
        for grp, mat in self.neg_embeddings.items():
            for e, g in zip(mat, self._neg_geoms[grp]):
                neg_feats_by_grp.setdefault(grp, []).append((e, g))

        for lbl, cgroup, img in cal_items:
            e = self.embedder.extract(img)
            g = extract_roi_geometry_features(img)
            if lbl in OBJECT_IDS:
                pos_feats_by_cls[lbl].append((e, g))
            else:
                cal_neg_all.append((e, g))
                if cgroup in neg_feats_by_grp:
                    neg_feats_by_grp[cgroup].append((e, g))

        all_geoms = (
            [g for lst in pos_feats_by_cls.values() for _, g in lst]
            + [g for lst in neg_feats_by_grp.values() for _, g in lst]
            + [g for _, g in cal_neg_all]
        )
        gmat = np.vstack(all_geoms).astype(np.float32)
        self.geom_mean = gmat.mean(axis=0)
        self.geom_std = np.where(gmat.std(axis=0) > 1e-4, gmat.std(axis=0), 1.0).astype(np.float32)

        def _combine(e: np.ndarray, g: np.ndarray) -> np.ndarray:
            gn = (g - self.geom_mean) / self.geom_std
            return np.concatenate([e, 0.15 * gn], axis=0).astype(np.float64)

        self.verifiers.clear()
        alpha = float(self.cfg.ridge_alpha)
        for oid in OBJECT_IDS:
            x_rows, y_vals, weights = [], [], []
            for e, g in pos_feats_by_cls[oid]:
                x_rows.append(_combine(e, g))
                y_vals.append(1.0)
                weights.append(1.0)
            # Targeted hard-negatives carry higher negative weight; other classes + general neg carry standard weight
            target_groups = set(TARGET_CONFUSER_MAP.get(oid, []))
            for grp, lst in neg_feats_by_grp.items():
                w_neg = 1.6 if grp in target_groups else 0.7
                for e, g in lst:
                    x_rows.append(_combine(e, g))
                    y_vals.append(-1.0)
                    weights.append(w_neg)
            for e, g in cal_neg_all:
                x_rows.append(_combine(e, g))
                y_vals.append(-1.0)
                weights.append(0.8)
            for other_oid, lst in pos_feats_by_cls.items():
                if other_oid != oid:
                    for e, g in lst:
                        x_rows.append(_combine(e, g))
                        y_vals.append(-0.6)
                        weights.append(0.6)

            X = np.vstack(x_rows)
            y = np.array(y_vals, dtype=np.float64)
            sw = np.sqrt(np.array(weights, dtype=np.float64))[:, None]
            # Balance positive vs negative total weight so class imbalance never biases boundary toward -1
            pos_mask = y > 0
            neg_mask = y <= 0
            if np.any(pos_mask) and np.any(neg_mask):
                ratio = float(np.sum(sw[neg_mask] ** 2) / max(1e-6, np.sum(sw[pos_mask] ** 2)))
                sw[pos_mask] *= np.sqrt(ratio)

            Xw = X * sw
            yw = y * sw[:, 0]
            # Append bias column (unregularized)
            X_aug = np.hstack([Xw, sw])
            dim = X_aug.shape[1]
            reg = np.eye(dim, dtype=np.float64) * alpha
            reg[-1, -1] = 1e-4
            theta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ yw)
            self.verifiers[oid] = (theta[:-1].astype(np.float32), float(theta[-1]))

    def _verifier_score(self, oid: str, emb: np.ndarray, geom: np.ndarray) -> float:
        if oid not in self.verifiers:
            return 0.0
        w, b = self.verifiers[oid]
        gn = (geom - self.geom_mean) / self.geom_std
        feat = np.concatenate([emb, 0.15 * gn], axis=0)
        return float(np.dot(w, feat) + b)

    def _score_embedding_and_geom(self, emb: np.ndarray, geom: np.ndarray) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        ranked = []
        mode = self.cfg.mode

        # Compute hard-negative similarities per group once
        grp_sims: dict[str, tuple[float, float]] = {}
        for grp, nmat in self.neg_embeddings.items():
            sims_n = (nmat @ emb).astype(np.float64)
            proto_n = float(self.neg_prototypes[grp] @ emb)
            max_n = float(np.max(sims_n))
            top2_n = float(np.mean(np.sort(sims_n)[-min(2, len(sims_n)):]))
            grp_sims[grp] = (0.5 * proto_n + 0.5 * top2_n, max_n)

        global_neg_max = max((v[0] for v in grp_sims.values()), default=0.0)
        global_neg_1nn = max((v[1] for v in grp_sims.values()), default=0.0)

        for oid, pmat in self.pos_embeddings.items():
            sims = (pmat @ emb).astype(np.float64)
            proto_sim = float(self.pos_prototypes[oid] @ emb)
            top2_mean = float(np.mean(np.sort(sims)[-min(2, len(sims)):]))
            pos_score = 0.5 * proto_sim + 0.5 * top2_mean

            target_grps = TARGET_CONFUSER_MAP.get(oid, [])
            targ_hybrid = max((grp_sims[g][0] for g in target_grps if g in grp_sims), default=global_neg_max)
            targ_1nn = max((grp_sims[g][1] for g in target_grps if g in grp_sims), default=global_neg_1nn)
            neg_sim = 0.5 * max(targ_hybrid, 0.85 * global_neg_max) + 0.5 * max(targ_1nn, 0.85 * global_neg_1nn)

            if mode == "A0_baseline_hybrid":
                adj_score = pos_score
            else:
                penalty = self.cfg.lambda_neg * max(0.0, neg_sim - self.cfg.neg_floor)
                adj_score = pos_score - penalty

            v_score = self._verifier_score(oid, emb, geom) if mode == "A3_full_verifier" else 0.0
            ranked.append({
                "id": oid,
                "pos_score": round(pos_score, 6),
                "neg_sim": round(neg_sim, 6),
                "pos_neg_margin": round(pos_score - neg_sim, 6),
                "adj_score": round(adj_score, 6),
                "verifier_score": round(v_score, 6),
                "best_sample": self.pos_names[oid][int(np.argmax(sims))],
            })

        ranked.sort(key=lambda r: r["adj_score"], reverse=True)
        match_ms = (time.perf_counter() - t0) * 1000.0
        return ranked, match_ms

    def decide(self, ranked: list[dict]) -> dict:
        if not ranked:
            return {"status": "UNKNOWN", "label": None, "score": 0.0, "margin": 0.0, "candidates": []}
        top = ranked[0]
        oid = top["id"]
        second_adj = float(ranked[1]["adj_score"]) if len(ranked) > 1 else -1.0
        pos_margin = float(top["adj_score"]) - second_adj
        mode = self.cfg.mode

        if mode == "A0_baseline_hybrid":
            if top["pos_score"] < self.cfg.tau_global:
                status, label = "UNKNOWN", None
            elif pos_margin < 0.0:
                status, label = "AMBIGUOUS", None
            else:
                status, label = "KNOWN", oid
        elif mode == "A1_neg_prototypes":
            if top["adj_score"] < self.cfg.tau_global:
                status, label = "UNKNOWN", None
            elif pos_margin < self.cfg.delta_pos:
                status, label = "AMBIGUOUS", None
            else:
                status, label = "KNOWN", oid
        else:
            tau_c = float(self.cfg.tau_per_class.get(oid, self.cfg.tau_global))
            dneg_c = float(self.cfg.delta_neg_min.get(oid, 0.0))
            ver_min = float(self.cfg.verifier_min_score.get(oid, -999.0))
            if top["adj_score"] < tau_c or top["pos_neg_margin"] < dneg_c:
                status, label = "UNKNOWN", None
            elif mode == "A3_full_verifier" and top["verifier_score"] < ver_min:
                status, label = "UNKNOWN", None
            elif pos_margin < self.cfg.delta_pos:
                status, label = "AMBIGUOUS", None
            else:
                status, label = "KNOWN", oid

        return {
            "status": status,
            "label": label,
            "score": float(top["adj_score"]),
            "pos_score": float(top["pos_score"]),
            "neg_sim": float(top["neg_sim"]),
            "pos_neg_margin": float(top["pos_neg_margin"]),
            "verifier_score": float(top["verifier_score"]),
            "margin": round(pos_margin, 6),
            "candidates": ranked[:3],
        }

    def predict(self, image: np.ndarray) -> dict:
        t0 = time.perf_counter()
        emb = self.embedder.extract(image)
        geom = extract_roi_geometry_features(image) if self.cfg.mode == "A3_full_verifier" else np.zeros(16, dtype=np.float32)
        feat_ms = (time.perf_counter() - t0) * 1000.0
        ranked, match_ms = self._score_embedding_and_geom(emb, geom)
        dec = self.decide(ranked)
        dec["feat_ms"] = round(feat_ms, 3)
        dec["match_ms"] = round(match_ms, 4)
        return dec

    def calibrate_on_split(
        self,
        cal_items: list[tuple[str, str, np.ndarray]],
        max_far_target: float = 0.06,
    ) -> dict:
        """Calibrate parameters strictly on calibration split (never on test or holdout)."""
        if self.cfg.mode == "A0_baseline_hybrid":
            self.cfg.tau_global = 0.4758
            self.cfg.delta_pos = 0.0
            return {"mode": self.cfg.mode, "tau_global": self.cfg.tau_global}

        if self.cfg.mode == "A3_full_verifier":
            self.fit_candidate_verifiers(cal_items)

        # Precompute embeddings + geometry on calibration split once
        cached = []
        for lbl, cgroup, img in cal_items:
            emb = self.embedder.extract(img)
            geom = extract_roi_geometry_features(img)
            cached.append((lbl, cgroup, emb, geom))

        # 1. Grid-search shared lambda_neg & tau_global on calibration split
        best_global = (-1e9, 0.35, 0.40, 0.015)
        for lam in (0.20, 0.35, 0.50, 0.65):
            self.cfg.lambda_neg = lam
            scored = [(y, self._score_embedding_and_geom(e, g)[0]) for y, _, e, g in cached]
            pos_scores = [r[0]["adj_score"] for y, r in scored if y != "unknown"]
            neg_scores = [r[0]["adj_score"] for y, r in scored if y == "unknown"]
            cand_taus = sorted(set([round(s + 1e-4, 4) for s in neg_scores + pos_scores] + [0.35, 0.40, 0.44, 0.48]))
            for tau in cand_taus:
                for dpos in (0.01, 0.02, 0.04):
                    tp = fa = 0
                    for y, r in scored:
                        s0 = r[0]["adj_score"]
                        s1 = r[1]["adj_score"] if len(r) > 1 else -1.0
                        if s0 >= tau and (s0 - s1) >= dpos:
                            if y == "unknown":
                                fa += 1
                            elif r[0]["id"] == y:
                                tp += 1
                    n_pos = max(1, sum(1 for y, _ in scored if y != "unknown"))
                    n_neg = max(1, sum(1 for y, _ in scored if y == "unknown"))
                    acc = tp / n_pos
                    far = fa / n_neg
                    obj = acc - max(0.0, far - max_far_target) * 4.5 - 0.8 * far
                    if obj > best_global[0]:
                        best_global = (obj, lam, tau, dpos)

        self.cfg.lambda_neg = best_global[1]
        self.cfg.tau_global = best_global[2]
        self.cfg.delta_pos = best_global[3]
        if self.cfg.mode == "A1_neg_prototypes":
            return {
                "mode": self.cfg.mode,
                "lambda_neg": self.cfg.lambda_neg,
                "tau_global": self.cfg.tau_global,
                "delta_pos": self.cfg.delta_pos,
            }

        # 2. Per-category adaptive thresholds & negative margin (+ verifier cutoffs for A3)
        scored = [(y, self._score_embedding_and_geom(e, g)[0]) for y, _, e, g in cached]
        for oid in OBJECT_IDS:
            # Evaluate on calibration samples where top predicted class == oid
            pos_c = [r[0] for y, r in scored if y == oid]
            neg_c = [r[0] for y, r in scored if y == "unknown" and r[0]["id"] == oid]
            best_c = (-1e9, self.cfg.tau_global, 0.02, -0.15)
            tau_grid = sorted(set(
                [round(x["adj_score"] - 0.005, 4) for x in pos_c]
                + [round(x["adj_score"] + 0.002, 4) for x in neg_c]
                + [round(self.cfg.tau_global - 0.04, 4), self.cfg.tau_global, round(self.cfg.tau_global + 0.04, 4)]
            ))
            dneg_grid = (0.0, 0.03, 0.06, 0.10)
            ver_grid = (-0.25, -0.10, 0.0, 0.10) if self.cfg.mode == "A3_full_verifier" else (-999.0,)

            # Also keep recall stable across ALL positives predicted as oid
            for tc in tau_grid:
                for dnc in dneg_grid:
                    for vc in ver_grid:
                        tp_c = fa_c = 0
                        for y, r in scored:
                            if r[0]["id"] != oid:
                                continue
                            s0 = r[0]["adj_score"]
                            s1 = r[1]["adj_score"] if len(r) > 1 else -1.0
                            ok = (
                                s0 >= tc
                                and r[0]["pos_neg_margin"] >= dnc
                                and (self.cfg.mode != "A3_full_verifier" or r[0]["verifier_score"] >= vc)
                                and (s0 - s1) >= self.cfg.delta_pos
                            )
                            if ok:
                                if y == oid:
                                    tp_c += 1
                                elif y == "unknown":
                                    fa_c += 1
                        rec_c = tp_c / max(1, len(pos_c))
                        far_c = fa_c / max(1, sum(1 for y, _ in scored if y == "unknown"))
                        # Encourage keeping recall high while keeping per-class FA near zero on cal
                        obj_c = rec_c - 3.5 * far_c - 0.05 * tc
                        if obj_c > best_c[0]:
                            best_c = (obj_c, tc, dnc, vc)

            self.cfg.tau_per_class[oid] = float(best_c[1])
            self.cfg.delta_neg_min[oid] = float(best_c[2])
            if self.cfg.mode == "A3_full_verifier":
                self.cfg.verifier_min_score[oid] = float(best_c[3])

        return {
            "mode": self.cfg.mode,
            "lambda_neg": self.cfg.lambda_neg,
            "tau_global": self.cfg.tau_global,
            "delta_pos": self.cfg.delta_pos,
            "tau_per_class": dict(self.cfg.tau_per_class),
            "delta_neg_min": dict(self.cfg.delta_neg_min),
            "verifier_min_score": dict(self.cfg.verifier_min_score),
        }

    def export_frozen_state(self, out_path: Path) -> None:
        """Freeze thresholds and linear verifier weights prior to holdout evaluation."""
        payload = {
            "config": asdict(self.cfg),
            "geom_mean": self.geom_mean.tolist(),
            "geom_std": self.geom_std.tolist(),
            "verifiers": {k: {"weights": w.tolist(), "bias": float(b)} for k, (w, b) in self.verifiers.items()},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_frozen_state(self, in_path: Path) -> None:
        """Load frozen calibration thresholds & verifiers without touching holdout data."""
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        cfg_d = payload["config"]
        self.cfg = HardNegConfig(**cfg_d)
        self.geom_mean = np.array(payload["geom_mean"], dtype=np.float32)
        self.geom_std = np.array(payload["geom_std"], dtype=np.float32)
        self.verifiers = {
            k: (np.array(v["weights"], dtype=np.float32), float(v["bias"]))
            for k, v in payload.get("verifiers", {}).items()
        }
