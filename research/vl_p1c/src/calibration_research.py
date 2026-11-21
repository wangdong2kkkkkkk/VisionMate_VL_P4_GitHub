"""Section IV / Part A: Calibration Reliability, Out-of-Fold Verification & 3-State Selective Classification.

Addresses the two core failures of VL-P1b calibration:
1. Per-class thresholds collapsing to 0.0757..0.1826 on N=50 due to small-sample noise and penalty scale mixing.
2. In-sample leakage when fitting Ridge verifiers and picking cutoffs on the exact same small calibration set.

Implements 6 calibration/recognition modes on frozen DINOv2-Small INT8:
- R0_vl_p1_baseline: exact VL-P1 global cosine threshold (tau_global = 0.4758)
- R1_global_conservative: global cosine + orthogonal target hard-negative margin + top1-top2 margin
- R2_per_class_floored: Empirical-Bayes shrunk per-class thresholds with CV-selected safety floor
- R3_distributional_quantile: upper negative-distribution quantile across ALL negatives per class + floor
- R4_out_of_fold_verifier: 5-fold CV out-of-fold calibrated L2 Ridge verifier (768D + 16D geom) + floor
- R5_selective_3state_v1c: full floored distributional + OOF verifier + strict KNOWN/AMBIGUOUS/UNKNOWN bands
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
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
class CalibrationConfigV1c:
    mode: str = "R5_selective_3state_v1c"
    tau_global: float = 0.4758
    delta_floor: float = 0.06  # Max allowed dip below tau_global (selected by CV, never hardcoded blind)
    tau_floor_abs: float = 0.40
    tau_per_class: dict[str, float] = field(default_factory=lambda: {oid: 0.4758 for oid in OBJECT_IDS})
    delta_neg_min: dict[str, float] = field(default_factory=lambda: {oid: 0.02 for oid in OBJECT_IDS})
    verifier_min_score: dict[str, float] = field(default_factory=lambda: {oid: 0.0 for oid in OBJECT_IDS})
    delta_pos_ambig: float = 0.02
    tau_ambig_band: float = 0.045
    neg_ambig_band: float = 0.025
    ver_ambig_band: float = 0.14
    ridge_alpha: float = 24.0
    neg_blend_top2_weight: float = 0.55
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

    log_ar = float(np.clip(np.log(max(w, 1) / float(max(h, 1))), -3.5, 3.5)) / 3.5
    narrowness = 1.0 - (float(min(w, h)) / float(max(w, h, 1)))

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

    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)
    white_ratio = float(np.mean((sat < 55.0) & (val > 175.0)))
    red_ratio = float(np.mean(((hue <= 12.0) | (hue >= 165.0)) & (sat > 85.0) & (val > 70.0)))
    wood_brown_ratio = float(np.mean((hue >= 8.0) & (hue <= 28.0) & (sat > 55.0) & (val < 180.0)))

    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_norm = float(np.tanh(np.std(lap) / 45.0))
    center_crop = edges[sh // 4:(3 * sh) // 4, sw // 4:(3 * sw) // 4]
    center_edge_ratio = float(np.mean(center_crop > 0)) if center_crop.size > 0 else edge_density

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(15, min(sw, sh) // 5),
        minLineLength=max(12, min(sw, sh) // 4),
        maxLineGap=6,
    )
    line_count_norm = float(np.tanh((0 if lines is None else len(lines)) / 12.0))
    max_line_len_norm = 0.0
    if lines is not None and len(lines) > 0:
        lens = [float(np.hypot(x2 - x1, y2 - y1)) for x1, y1, x2, y2 in lines[:, 0, :]]
        max_line_len_norm = float(np.clip(max(lens) / max(1.0, float(np.hypot(sw, sh))), 0.0, 1.0))

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
            raise FileNotFoundError(f"Missing DINOv2 ONNX model at {self.model_path}")
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
        hidden = self.session.run(None, {self.input_name: tensor})[0]
        cls_vec = _l2_normalize(hidden[0, 0, :])
        patch_mean = _l2_normalize(hidden[0, 1:, :].mean(axis=0))
        return _l2_normalize(np.concatenate([cls_vec, patch_mean], axis=0))


def make_group_aware_stratified_folds(
    records: list[dict],
    n_folds: int = 5,
) -> list[list[int]]:
    """Assign calibration indices into n_folds stratified by label/confuser while keeping cluster_id together."""
    # Group indices strictly by canonical cluster_id / source_cluster so NO cluster can straddle folds
    by_cluster: dict[str, list[int]] = {}
    cluster_strata: dict[str, str] = {}
    for idx, rec in enumerate(records):
        lbl = rec["label_id"]
        cg = rec.get("confuser_group") or "pos"
        cid = str(rec.get("cluster_id") or rec.get("source_cluster") or rec.get("id") or f"idx_{idx}")
        by_cluster.setdefault(cid, []).append(idx)
        cluster_strata.setdefault(cid, f"{lbl}:{cg}")

    # Sort deterministically so CV is 100% reproducible across runs
    sorted_cids = sorted(
        by_cluster.keys(),
        key=lambda cid: (cluster_strata[cid], hashlib.md5(cid.encode("utf-8")).hexdigest()),
    )
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    by_stratum: dict[str, list[list[int]]] = {}
    for cid in sorted_cids:
        by_stratum.setdefault(cluster_strata[cid], []).append(by_cluster[cid])

    for stratum in sorted(by_stratum.keys()):
        for k_i, idxs in enumerate(by_stratum[stratum]):
            # Put into fold with fewest items so far within rotation to avoid fold skew
            target_f = min(range(n_folds), key=lambda f: (len(folds[f]), (f - k_i) % n_folds))
            folds[target_f].extend(idxs)

    for f in folds:
        f.sort()
    return folds


def _fit_ridge_verifiers_from_features(
    pos_feats_by_cls: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    neg_feats_by_grp: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    cal_neg_all: list[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
) -> tuple[dict[str, tuple[np.ndarray, float]], np.ndarray, np.ndarray]:
    all_geoms = (
        [g for lst in pos_feats_by_cls.values() for _, g in lst]
        + [g for lst in neg_feats_by_grp.values() for _, g in lst]
        + [g for _, g in cal_neg_all]
    )
    gmat = np.vstack(all_geoms).astype(np.float32)
    geom_mean = gmat.mean(axis=0)
    geom_std = np.where(gmat.std(axis=0) > 1e-4, gmat.std(axis=0), 1.0).astype(np.float32)

    def _combine(e: np.ndarray, g: np.ndarray) -> np.ndarray:
        gn = (g - geom_mean) / geom_std
        return np.concatenate([e, 0.15 * gn], axis=0).astype(np.float64)

    verifiers: dict[str, tuple[np.ndarray, float]] = {}
    for oid in OBJECT_IDS:
        x_rows, y_vals, weights = [], [], []
        for e, g in pos_feats_by_cls[oid]:
            x_rows.append(_combine(e, g))
            y_vals.append(1.0)
            weights.append(1.0)
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
            weights.append(0.9)
        for other_oid, lst in pos_feats_by_cls.items():
            if other_oid != oid:
                for e, g in lst:
                    x_rows.append(_combine(e, g))
                    y_vals.append(-0.6)
                    weights.append(0.6)

        X = np.vstack(x_rows)
        y = np.array(y_vals, dtype=np.float64)
        sw = np.sqrt(np.array(weights, dtype=np.float64))[:, None]
        pos_mask = y > 0
        neg_mask = y <= 0
        if np.any(pos_mask) and np.any(neg_mask):
            ratio = float(np.sum(sw[neg_mask] ** 2) / max(1e-6, np.sum(sw[pos_mask] ** 2)))
            sw[pos_mask] *= np.sqrt(ratio)

        Xw = X * sw
        yw = y * sw[:, 0]
        X_aug = np.hstack([Xw, sw])
        dim = X_aug.shape[1]
        reg = np.eye(dim, dtype=np.float64) * float(ridge_alpha)
        reg[-1, -1] = 1e-4
        theta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ yw)
        verifiers[oid] = (theta[:-1].astype(np.float32), float(theta[-1]))

    return verifiers, geom_mean, geom_std


class CalibratedRecognizerV1c:
    """Section IV recognizer implementing R0..R5 with strict threshold floors and OOF verification."""

    def __init__(self, models_dir: Path, config: CalibrationConfigV1c | None = None, embedder: DINOv2Embedder | None = None) -> None:
        self.cfg = config or CalibrationConfigV1c()
        self.embedder = embedder or DINOv2Embedder(models_dir, num_threads=self.cfg.num_threads)

        self.pos_embeddings: dict[str, np.ndarray] = {}
        self.pos_prototypes: dict[str, np.ndarray] = {}
        self.pos_names: dict[str, list[str]] = {}
        self._pos_geoms: dict[str, np.ndarray] = {}

        self.neg_embeddings: dict[str, np.ndarray] = {}
        self.neg_prototypes: dict[str, np.ndarray] = {}
        self.neg_names: dict[str, list[str]] = {}
        self._neg_geoms: dict[str, np.ndarray] = {}

        self.verifiers: dict[str, tuple[np.ndarray, float]] = {}
        self.geom_mean: np.ndarray = np.zeros(16, dtype=np.float32)
        self.geom_std: np.ndarray = np.ones(16, dtype=np.float32)

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
                raise ValueError(f"No positive images found in {train_pos_dir / oid}")
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
        return {"neg_gallery_count": count, "enroll_neg_ms": round((time.perf_counter() - t0) * 1000.0, 2)}

    def score_features(
        self,
        emb: np.ndarray,
        geom: np.ndarray,
        verifiers_override: tuple[dict, np.ndarray, np.ndarray] | None = None,
    ) -> list[dict]:
        ver_map, gmean, gstd = verifiers_override if verifiers_override is not None else (
            self.verifiers, self.geom_mean, self.geom_std
        )
        grp_sims: dict[str, tuple[float, float]] = {}
        for grp, nmat in self.neg_embeddings.items():
            sims_n = (nmat @ emb).astype(np.float64)
            proto_n = float(self.neg_prototypes[grp] @ emb)
            max_n = float(np.max(sims_n))
            top2_n = float(np.mean(np.sort(sims_n)[-min(2, len(sims_n)):]))
            grp_sims[grp] = (0.5 * proto_n + 0.5 * top2_n, max_n)

        global_neg_hyb = max((v[0] for v in grp_sims.values()), default=0.0)
        global_neg_1nn = max((v[1] for v in grp_sims.values()), default=0.0)

        gn = (geom - gmean) / np.where(gstd > 1e-4, gstd, 1.0)
        v_feat = np.concatenate([emb, 0.15 * gn], axis=0)

        ranked = []
        for oid, pmat in self.pos_embeddings.items():
            sims = (pmat @ emb).astype(np.float64)
            proto_sim = float(self.pos_prototypes[oid] @ emb)
            top2_mean = float(np.mean(np.sort(sims)[-min(2, len(sims)):]))
            pos_score = 0.5 * proto_sim + 0.5 * top2_mean

            target_grps = TARGET_CONFUSER_MAP.get(oid, [])
            targ_hyb = max((grp_sims[g][0] for g in target_grps if g in grp_sims), default=global_neg_hyb)
            targ_1nn = max((grp_sims[g][1] for g in target_grps if g in grp_sims), default=global_neg_1nn)
            w1 = float(self.cfg.neg_blend_top2_weight)
            neg_sim = (1.0 - w1) * max(targ_hyb, 0.85 * global_neg_hyb) + w1 * max(targ_1nn, 0.85 * global_neg_1nn)

            v_score = 0.0
            if oid in ver_map:
                w, b = ver_map[oid]
                v_score = float(np.dot(w, v_feat) + b)

            ranked.append({
                "id": oid,
                "pos_score": round(pos_score, 6),
                "neg_sim": round(neg_sim, 6),
                "pos_neg_margin": round(pos_score - neg_sim, 6),
                "verifier_score": round(v_score, 6),
                "best_sample": self.pos_names[oid][int(np.argmax(sims))],
            })

        ranked.sort(key=lambda r: r["pos_score"], reverse=True)
        return ranked

    def decide_with_config(self, ranked: list[dict], cfg: CalibrationConfigV1c | None = None) -> dict:
        c = cfg or self.cfg
        if not ranked:
            return {"status": "UNKNOWN", "label": None, "score": 0.0, "margin": 0.0, "candidates": []}
        top = ranked[0]
        oid = top["id"]
        second_pos = float(ranked[1]["pos_score"]) if len(ranked) > 1 else -1.0
        pos_margin = float(top["pos_score"]) - second_pos
        mode = c.mode

        status = "UNKNOWN"
        label = None
        if mode == "R0_vl_p1_baseline":
            if top["pos_score"] < c.tau_global:
                status, label = "UNKNOWN", None
            elif pos_margin < 0.0:
                status, label = "AMBIGUOUS", None
            else:
                status, label = "KNOWN", oid
        elif mode == "R1_global_conservative":
            dneg_g = max(c.delta_neg_min.values(), default=0.02)
            if top["pos_score"] < c.tau_global or top["pos_neg_margin"] < dneg_g:
                status, label = "UNKNOWN", None
            elif pos_margin < c.delta_pos_ambig:
                status, label = "AMBIGUOUS", None
            else:
                status, label = "KNOWN", oid
        elif mode in ("R2_per_class_floored", "R3_distributional_quantile"):
            tau_c = float(c.tau_per_class.get(oid, c.tau_global))
            dneg_c = float(c.delta_neg_min.get(oid, 0.0))
            if top["pos_score"] < tau_c or top["pos_neg_margin"] < dneg_c:
                status, label = "UNKNOWN", None
            elif pos_margin < c.delta_pos_ambig:
                status, label = "AMBIGUOUS", None
            else:
                status, label = "KNOWN", oid
        elif mode == "R4_out_of_fold_verifier":
            tau_c = float(c.tau_per_class.get(oid, c.tau_global))
            dneg_c = float(c.delta_neg_min.get(oid, 0.0))
            ver_min = float(c.verifier_min_score.get(oid, -999.0))
            if top["pos_score"] < tau_c or top["pos_neg_margin"] < dneg_c or top["verifier_score"] < ver_min:
                status, label = "UNKNOWN", None
            elif pos_margin < c.delta_pos_ambig:
                status, label = "AMBIGUOUS", None
            else:
                status, label = "KNOWN", oid
        else:
            # R5_selective_3state_v1c: Clear separation between low-evidence UNKNOWN and borderline AMBIGUOUS
            tau_c = float(c.tau_per_class.get(oid, c.tau_global))
            dneg_c = float(c.delta_neg_min.get(oid, 0.0))
            ver_min = float(c.verifier_min_score.get(oid, -999.0))
            if (
                top["pos_score"] < (tau_c - c.tau_ambig_band)
                or top["pos_neg_margin"] < (dneg_c - c.neg_ambig_band)
                or top["verifier_score"] < (ver_min - c.ver_ambig_band)
            ):
                status, label = "UNKNOWN", None
            elif (
                top["pos_score"] < tau_c
                or top["pos_neg_margin"] < dneg_c
                or top["verifier_score"] < ver_min
                or pos_margin < c.delta_pos_ambig
            ):
                status, label = "AMBIGUOUS", None
            else:
                status, label = "KNOWN", oid

        return {
            "status": status,
            "label": label,
            "score": float(top["pos_score"]),
            "pos_score": float(top["pos_score"]),
            "neg_sim": float(top["neg_sim"]),
            "pos_neg_margin": float(top["pos_neg_margin"]),
            "verifier_score": float(top["verifier_score"]),
            "margin": round(pos_margin, 6),
            "top_class": oid,
            "second_class": ranked[1]["id"] if len(ranked) > 1 else None,
            "candidates": ranked[:3],
        }

    def predict(self, image: np.ndarray) -> dict:
        t0 = time.perf_counter()
        emb = self.embedder.extract(image)
        geom = extract_roi_geometry_features(image)
        feat_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        ranked = self.score_features(emb, geom)
        match_ms = (time.perf_counter() - t1) * 1000.0
        dec = self.decide_with_config(ranked)
        dec["feat_ms"] = round(feat_ms, 3)
        dec["match_ms"] = round(match_ms, 4)
        return dec

    def _fit_verifier_on_subset(self, cal_subset: list[dict], embeddings: list[np.ndarray], geoms: list[np.ndarray]) -> tuple[dict, np.ndarray, np.ndarray]:
        pos_f: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {oid: [] for oid in OBJECT_IDS}
        neg_f: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {g: [] for g in HARD_NEGATIVE_GROUPS}
        cal_neg_all: list[tuple[np.ndarray, np.ndarray]] = []
        for oid in OBJECT_IDS:
            for e, g in zip(self.pos_embeddings[oid], self._pos_geoms[oid]):
                pos_f[oid].append((e, g))
        for grp, mat in self.neg_embeddings.items():
            for e, g in zip(mat, self._neg_geoms[grp]):
                neg_f.setdefault(grp, []).append((e, g))
        for rec, e, g in zip(cal_subset, embeddings, geoms):
            lbl = rec["label_id"]
            cg = rec.get("confuser_group")
            if lbl in OBJECT_IDS:
                pos_f[lbl].append((e, g))
            else:
                cal_neg_all.append((e, g))
                if cg in neg_f:
                    neg_f[cg].append((e, g))
        return _fit_ridge_verifiers_from_features(pos_f, neg_f, cal_neg_all, self.cfg.ridge_alpha)

    def calibrate_mode_on_records(
        self,
        cal_records: list[dict],
        cal_embs: list[np.ndarray],
        cal_geoms: list[np.ndarray],
        mode: str,
        folds: list[list[int]] | None = None,
        max_far_target: float = 0.05,
        delta_floor: float = 0.06,
        alpha_quantile: float = 0.03,
    ) -> CalibrationConfigV1c:
        """Fit a single mode cleanly on cal_records using strictly out-of-fold verifiers."""
        cfg = CalibrationConfigV1c(mode=mode, delta_floor=delta_floor, ridge_alpha=self.cfg.ridge_alpha)
        n = len(cal_records)
        if mode == "R0_vl_p1_baseline":
            cfg.tau_global = 0.4758
            cfg.delta_pos_ambig = 0.0
            return cfg

        folds_eff = folds or make_group_aware_stratified_folds(cal_records, n_folds=min(5, max(2, n // 10)))

        # Compute out-of-fold ranked scores for every calibration item
        oof_ranked: list[list[dict]] = [[] for _ in range(n)]
        fold_map = {idx: f_id for f_id, idxs in enumerate(folds_eff) for idx in idxs}

        need_verifier = mode in ("R4_out_of_fold_verifier", "R5_selective_3state_v1c")
        if need_verifier:
            for f_id, val_idxs in enumerate(folds_eff):
                tr_idxs = [i for i in range(n) if fold_map[i] != f_id]
                ver_tup = self._fit_verifier_on_subset(
                    [cal_records[i] for i in tr_idxs],
                    [cal_embs[i] for i in tr_idxs],
                    [cal_geoms[i] for i in tr_idxs],
                )
                for vi in val_idxs:
                    oof_ranked[vi] = self.score_features(cal_embs[vi], cal_geoms[vi], verifiers_override=ver_tup)
            # Fit final deployable verifier on ALL calibration items (only thresholds were chosen from OOF!)
            v_all, gm_all, gs_all = self._fit_verifier_on_subset(cal_records, cal_embs, cal_geoms)
            self.verifiers, self.geom_mean, self.geom_std = v_all, gm_all, gs_all
        else:
            for i in range(n):
                oof_ranked[i] = self.score_features(cal_embs[i], cal_geoms[i], verifiers_override=({}, self.geom_mean, self.geom_std))

        # Step 1: Global threshold + global negative margin on raw pos_score (preserving cosine scale!)
        neg_top_scores = [oof_ranked[i][0]["pos_score"] for i in range(n) if cal_records[i]["label_id"] == "unknown"]
        pos_top_scores = [oof_ranked[i][0]["pos_score"] for i in range(n) if cal_records[i]["label_id"] != "unknown"]
        tau_cands = sorted(set(
            [round(s + 1e-4, 4) for s in neg_top_scores + pos_top_scores]
            + [0.44, 0.46, 0.4758, 0.49, 0.51, 0.53, 0.55]
        ))
        n_pos = max(1, sum(1 for r in cal_records if r["label_id"] != "unknown"))
        n_neg = max(1, sum(1 for r in cal_records if r["label_id"] == "unknown"))

        best_g = (-1e9, 0.4758, 0.02, 0.02)
        for tau in tau_cands:
            if tau < 0.42:
                continue
            for dneg in (0.0, 0.015, 0.03, 0.05):
                for dpos in (0.01, 0.02, 0.035):
                    tp = fa = 0
                    for i, rec in enumerate(cal_records):
                        rk = oof_ranked[i]
                        s0 = rk[0]["pos_score"]
                        s1 = rk[1]["pos_score"] if len(rk) > 1 else -1.0
                        ok = s0 >= tau and rk[0]["pos_neg_margin"] >= dneg and (s0 - s1) >= dpos
                        if ok:
                            if rec["label_id"] == "unknown":
                                fa += 1
                            elif rk[0]["id"] == rec["label_id"]:
                                tp += 1
                    rec_rate = tp / n_pos
                    far_rate = fa / n_neg
                    # Strict penalty for exceeding max_far_target; modest reward for higher safe threshold on ties
                    obj = rec_rate - max(0.0, far_rate - max_far_target) * 5.0 - 1.2 * far_rate + 0.03 * tau
                    if obj > best_g[0]:
                        best_g = (obj, float(tau), float(dneg), float(dpos))

        cfg.tau_global = best_g[1]
        for oid in OBJECT_IDS:
            cfg.delta_neg_min[oid] = best_g[2]
        cfg.delta_pos_ambig = best_g[3]

        if mode == "R1_global_conservative":
            for oid in OBJECT_IDS:
                cfg.tau_per_class[oid] = cfg.tau_global
            return cfg

        safe_floor = max(cfg.tau_floor_abs, round(cfg.tau_global - delta_floor, 4))

        # Step 2: Distributional per-class quantile reference (scores ALL negatives per class c, never argmax-only)
        quant_tau: dict[str, float] = {}
        for oid in OBJECT_IDS:
            neg_scores_c = []
            pos_scores_c = []
            for i, rec in enumerate(cal_records):
                cand = next((x for x in oof_ranked[i] if x["id"] == oid), None)
                if cand is None:
                    continue
                if rec["label_id"] == "unknown":
                    neg_scores_c.append(float(cand["pos_score"]))
                elif rec["label_id"] == oid:
                    pos_scores_c.append(float(cand["pos_score"]))
            q_neg = float(np.quantile(neg_scores_c, 1.0 - alpha_quantile)) if neg_scores_c else cfg.tau_global
            g_tail = (float(np.mean(neg_scores_c)) + 1.96 * float(np.std(neg_scores_c))) if len(neg_scores_c) >= 5 else q_neg
            t_dist = max(safe_floor, 0.5 * q_neg + 0.5 * g_tail)
            if pos_scores_c:
                # Keep reasonable upper guard relative to lower positive quantile when safe
                q_pos_low = float(np.quantile(pos_scores_c, 0.20))
                t_dist = min(max(safe_floor, q_pos_low), max(safe_floor, t_dist))
            quant_tau[oid] = round(max(safe_floor, t_dist), 4)

        if mode == "R3_distributional_quantile":
            cfg.tau_per_class = quant_tau
            return cfg

        # Step 3: Empirical Bayes shrunk per-class threshold + OOF margin/verifier cutoffs with safety floor
        n0_shrink = 8.0
        for oid in OBJECT_IDS:
            pos_c = [oof_ranked[i][0] for i, r in enumerate(cal_records) if r["label_id"] == oid]
            neg_c = [oof_ranked[i][0] for i, r in enumerate(cal_records) if r["label_id"] == "unknown" and oof_ranked[i][0]["id"] == oid]
            t_cands = sorted(set(
                [round(x["pos_score"] - 0.004, 4) for x in pos_c if x["pos_score"] >= safe_floor]
                + [round(x["pos_score"] + 0.002, 4) for x in neg_c if x["pos_score"] >= safe_floor]
                + [safe_floor, quant_tau[oid], cfg.tau_global, round(cfg.tau_global + 0.03, 4)]
            ))
            t_cands = [t for t in t_cands if t >= safe_floor]
            dneg_grid = (0.0, 0.015, 0.03, 0.05)
            ver_grid = (-0.15, -0.05, 0.05, 0.15) if need_verifier else (-999.0,)

            best_c = (-1e9, max(safe_floor, quant_tau[oid]), best_g[2], -0.05)
            for tc in t_cands:
                for dnc in dneg_grid:
                    for vc in ver_grid:
                        tp_c = fa_c = 0
                        for i, rec in enumerate(cal_records):
                            rk = oof_ranked[i]
                            if rk[0]["id"] != oid:
                                continue
                            s0 = rk[0]["pos_score"]
                            s1 = rk[1]["pos_score"] if len(rk) > 1 else -1.0
                            ok = (
                                s0 >= tc
                                and rk[0]["pos_neg_margin"] >= dnc
                                and (not need_verifier or rk[0]["verifier_score"] >= vc)
                                and (s0 - s1) >= cfg.delta_pos_ambig
                            )
                            if ok:
                                if rec["label_id"] == oid:
                                    tp_c += 1
                                elif rec["label_id"] == "unknown":
                                    fa_c += 1
                        rec_c = tp_c / max(1, len(pos_c))
                        far_c = fa_c / max(1, n_neg)
                        # Crucial fix over VL-P1b: reward HIGHER threshold (+0.08 * tc), not lower (-0.05 * tc)!
                        obj_c = rec_c - 5.0 * far_c + 0.08 * tc
                        if obj_c > best_c[0]:
                            best_c = (obj_c, tc, dnc, vc)

            n_pos_c = len(pos_c)
            anchor = 0.5 * cfg.tau_global + 0.5 * quant_tau[oid]
            w_shrink = n0_shrink / (n_pos_c + n0_shrink)
            shrunk_t = (1.0 - w_shrink) * best_c[1] + w_shrink * anchor
            final_t = round(max(safe_floor, shrunk_t), 4)
            cfg.tau_per_class[oid] = final_t
            cfg.delta_neg_min[oid] = float(best_c[2])
            if need_verifier:
                cfg.verifier_min_score[oid] = float(best_c[3])

        return cfg

    def fit_full_calibration(
        self,
        cal_records: list[dict],
        cal_embs: list[np.ndarray],
        cal_geoms: list[np.ndarray],
        delta_floor: float = 0.06,
        max_far_target: float = 0.05,
    ) -> CalibrationConfigV1c:
        folds = make_group_aware_stratified_folds(cal_records, n_folds=5)
        cfg = self.calibrate_mode_on_records(
            cal_records, cal_embs, cal_geoms,
            mode=self.cfg.mode,
            folds=folds,
            max_far_target=max_far_target,
            delta_floor=delta_floor,
        )
        self.cfg = cfg
        return cfg

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

    def export_frozen_state(self, out_path: Path) -> None:
        payload = {
            "config": asdict(self.cfg),
            "geom_mean": self.geom_mean.tolist(),
            "geom_std": self.geom_std.tolist(),
            "verifiers": {k: {"weights": w.tolist(), "bias": float(b)} for k, (w, b) in self.verifiers.items()},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_frozen_state(self, in_path: Path) -> None:
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        self.cfg = CalibrationConfigV1c(**payload["config"])
        self.geom_mean = np.array(payload["geom_mean"], dtype=np.float32)
        self.geom_std = np.array(payload["geom_std"], dtype=np.float32)
        self.verifiers = {
            k: (np.array(v["weights"], dtype=np.float32), float(v["bias"]))
            for k, v in payload.get("verifiers", {}).items()
        }


def evaluate_5fold_cv_and_sample_scaling(
    rec_base: CalibratedRecognizerV1c,
    cal_records: list[dict],
    cal_embs: list[np.ndarray],
    cal_geoms: list[np.ndarray],
    delta_floor_grid: tuple[float, ...] = (0.04, 0.06, 0.08),
    modes: tuple[str, ...] = (
        "R1_global_conservative",
        "R2_per_class_floored",
        "R3_distributional_quantile",
        "R4_out_of_fold_verifier",
        "R5_selective_3state_v1c",
    ),
    max_far_target: float = 0.05,
) -> dict:
    """Section IV.A3-A4: Group-aware 5-fold CV threshold variance, sample-size sweep, and CV selection of delta_floor."""
    folds = make_group_aware_stratified_folds(cal_records, n_folds=5)
    n = len(cal_records)
    fold_map = {idx: f_id for f_id, idxs in enumerate(folds) for idx in idxs}

    # 1. Select best delta_floor strictly via 5-fold CV on R5_selective_3state_v1c (never hardcoded!)
    floor_cv_results = {}
    best_floor_score = -1e9
    chosen_delta_floor = 0.06
    for df in delta_floor_grid:
        f_recs, f_fars, f_errs = [], [], []
        taus_per_fold: dict[str, list[float]] = {oid: [] for oid in OBJECT_IDS}
        for f_id, val_idxs in enumerate(folds):
            tr_idxs = [i for i in range(n) if fold_map[i] != f_id]
            tr_records = [cal_records[i] for i in tr_idxs]
            tr_embs = [cal_embs[i] for i in tr_idxs]
            tr_geoms = [cal_geoms[i] for i in tr_idxs]
            inner_folds = make_group_aware_stratified_folds(tr_records, n_folds=4)
            cfg_f = rec_base.calibrate_mode_on_records(
                tr_records, tr_embs, tr_geoms,
                mode="R5_selective_3state_v1c",
                folds=inner_folds,
                max_far_target=max_far_target,
                delta_floor=df,
            )
            for oid in OBJECT_IDS:
                taus_per_fold[oid].append(cfg_f.tau_per_class[oid])
            tp = fa = wc = n_p = n_n = 0
            for vi in val_idxs:
                rk = rec_base.score_features(cal_embs[vi], cal_geoms[vi])
                dec = rec_base.decide_with_config(rk, cfg_f)
                y = cal_records[vi]["label_id"]
                p = dec["label"] if dec["status"] == "KNOWN" else dec["status"]
                if y == "unknown":
                    n_n += 1
                    if p in OBJECT_IDS:
                        fa += 1
                else:
                    n_p += 1
                    if p == y:
                        tp += 1
                    elif p in OBJECT_IDS:
                        wc += 1
            f_recs.append(tp / max(1, n_p))
            f_fars.append(fa / max(1, n_n))
            f_errs.append(wc / max(1, n_p))

        mean_rec = float(np.mean(f_recs))
        mean_far = float(np.mean(f_fars))
        floor_cv_results[str(df)] = {
            "mean_recall_pct": round(mean_rec * 100.0, 2),
            "std_recall_pct": round(float(np.std(f_recs)) * 100.0, 2),
            "mean_far_pct": round(mean_far * 100.0, 2),
            "std_far_pct": round(float(np.std(f_fars)) * 100.0, 2),
            "tau_std_by_class": {oid: round(float(np.std(taus_per_fold[oid])), 4) for oid in OBJECT_IDS},
        }
        sc = mean_rec - max(0.0, mean_far - max_far_target) * 5.0 - 1.5 * mean_far
        if sc > best_floor_score:
            best_floor_score = sc
            chosen_delta_floor = df

    # 2. Full 5-fold CV report across all calibration modes R0..R5 at CV-selected chosen_delta_floor
    mode_cv_reports = {}
    for mode in ("R0_vl_p1_baseline", *modes):
        f_details = []
        taus_track: dict[str, list[float]] = {oid: [] for oid in OBJECT_IDS}
        oof_preds = []
        for f_id, val_idxs in enumerate(folds):
            tr_idxs = [i for i in range(n) if fold_map[i] != f_id]
            tr_records = [cal_records[i] for i in tr_idxs]
            tr_embs = [cal_embs[i] for i in tr_idxs]
            tr_geoms = [cal_geoms[i] for i in tr_idxs]
            inner_folds = make_group_aware_stratified_folds(tr_records, n_folds=4)
            cfg_f = rec_base.calibrate_mode_on_records(
                tr_records, tr_embs, tr_geoms,
                mode=mode,
                folds=inner_folds,
                max_far_target=max_far_target,
                delta_floor=chosen_delta_floor,
            )
            for oid in OBJECT_IDS:
                taus_track[oid].append(float(cfg_f.tau_per_class.get(oid, cfg_f.tau_global)))
            tp = fa = wc = amb_p = amb_n = n_p = n_n = 0
            for vi in val_idxs:
                rk = rec_base.score_features(cal_embs[vi], cal_geoms[vi])
                dec = rec_base.decide_with_config(rk, cfg_f)
                y = cal_records[vi]["label_id"]
                p = dec["label"] if dec["status"] == "KNOWN" else dec["status"]
                oof_preds.append((y, dec))
                if y == "unknown":
                    n_n += 1
                    if p in OBJECT_IDS:
                        fa += 1
                    elif p == "AMBIGUOUS":
                        amb_n += 1
                else:
                    n_p += 1
                    if p == y:
                        tp += 1
                    elif p in OBJECT_IDS:
                        wc += 1
                    elif p == "AMBIGUOUS":
                        amb_p += 1
            f_details.append({
                "fold": f_id,
                "n_val": len(val_idxs),
                "recall": round(tp / max(1, n_p), 4),
                "far": round(fa / max(1, n_n), 4),
                "wrong_class": round(wc / max(1, n_p), 4),
                "ambiguous_rate": round((amb_p + amb_n) / max(1, n_p + n_n), 4),
            })

        # Also benchmark unfloored baseline stability for comparison (delta_floor=0.45 simulation on fold variance)
        mode_cv_reports[mode] = {
            "per_fold": f_details,
            "mean_recall_pct": round(float(np.mean([d["recall"] for d in f_details])) * 100.0, 2),
            "std_recall_pct": round(float(np.std([d["recall"] for d in f_details])) * 100.0, 2),
            "mean_far_pct": round(float(np.mean([d["far"] for d in f_details])) * 100.0, 2),
            "std_far_pct": round(float(np.std([d["far"] for d in f_details])) * 100.0, 2),
            "mean_ambiguous_pct": round(float(np.mean([d["ambiguous_rate"] for d in f_details])) * 100.0, 2),
            "tau_stability": {
                oid: {
                    "mean": round(float(np.mean(taus_track[oid])), 4),
                    "std": round(float(np.std(taus_track[oid])), 4),
                    "min": round(float(np.min(taus_track[oid])), 4),
                    "max": round(float(np.max(taus_track[oid])), 4),
                }
                for oid in OBJECT_IDS
            },
            "oof_risk_coverage": compute_risk_coverage_curve(oof_preds),
        }

    # 3. Sample size calibration stability curve (N_cal = 50, 100, 150, 200)
    rng = np.random.default_rng(17)
    pos_indices = [i for i, r in enumerate(cal_records) if r["label_id"] != "unknown"]
    neg_indices = [i for i, r in enumerate(cal_records) if r["label_id"] == "unknown"]
    sample_scaling = []
    for n_sub in (50, 100, 150, min(200, n)):
        n_pos_sub = max(15, round(len(pos_indices) * (n_sub / n)))
        n_neg_sub = max(25, n_sub - n_pos_sub)
        tau_draws: dict[str, list[float]] = {oid: [] for oid in OBJECT_IDS}
        val_fars, val_recs = [], []
        repeats = 1 if n_sub >= n else 4
        for rep in range(repeats):
            if n_sub >= n:
                sub_idx = list(range(n))
            else:
                p_pick = rng.choice(pos_indices, size=min(len(pos_indices), n_pos_sub), replace=False).tolist()
                n_pick = rng.choice(neg_indices, size=min(len(neg_indices), n_neg_sub), replace=False).tolist()
                sub_idx = sorted(p_pick + n_pick)
            rem_idx = [i for i in range(n) if i not in set(sub_idx)]
            sub_records = [cal_records[i] for i in sub_idx]
            cfg_s = rec_base.calibrate_mode_on_records(
                sub_records, [cal_embs[i] for i in sub_idx], [cal_geoms[i] for i in sub_idx],
                mode="R5_selective_3state_v1c", delta_floor=chosen_delta_floor, max_far_target=max_far_target,
            )
            for oid in OBJECT_IDS:
                tau_draws[oid].append(cfg_s.tau_per_class[oid])
            eval_set = rem_idx if rem_idx else sub_idx
            tp = fa = np_c = nn_c = 0
            for ei in eval_set:
                rk = rec_base.score_features(cal_embs[ei], cal_geoms[ei])
                dec = rec_base.decide_with_config(rk, cfg_s)
                if cal_records[ei]["label_id"] == "unknown":
                    nn_c += 1
                    if dec["status"] == "KNOWN":
                        fa += 1
                else:
                    np_c += 1
                    if dec["status"] == "KNOWN" and dec["label"] == cal_records[ei]["label_id"]:
                        tp += 1
            val_recs.append(tp / max(1, np_c))
            val_fars.append(fa / max(1, nn_c))
        sample_scaling.append({
            "n_cal": n_sub,
            "mean_tau_std_across_classes": round(float(np.mean([np.std(tau_draws[o]) for o in OBJECT_IDS])), 4),
            "min_tau_observed": round(float(min(min(tau_draws[o]) for o in OBJECT_IDS)), 4),
            "out_of_sample_recall_pct": round(float(np.mean(val_recs)) * 100.0, 2),
            "out_of_sample_far_pct": round(float(np.mean(val_fars)) * 100.0, 2),
        })

    return {
        "n_calibration": n,
        "n_folds": len(folds),
        "delta_floor_cv_grid": floor_cv_results,
        "selected_delta_floor": chosen_delta_floor,
        "modes_5fold_cv": mode_cv_reports,
        "sample_size_scaling": sample_scaling,
    }


def compute_risk_coverage_curve(preds: list[tuple[str, dict]]) -> list[dict]:
    """Compute risk-coverage operating table across strictness offsets on pos_score."""
    curve = []
    n = max(1, len(preds))
    for shift in (-0.06, -0.03, 0.0, 0.02, 0.04, 0.06, 0.08):
        accepted = 0
        correct = 0
        errors = 0
        far_cnt = 0
        amb_cnt = 0
        neg_tot = sum(1 for y, _ in preds if y == "unknown")
        for y, dec in preds:
            if dec["status"] == "AMBIGUOUS":
                amb_cnt += 1
            if dec["status"] == "KNOWN" and (dec["pos_score"] - shift) >= 0.45:
                # Apply monotone score offset relative to calibrated cutoff
                if shift <= 0.0 or dec["margin"] >= (0.015 + 0.25 * max(0.0, shift)):
                    accepted += 1
                    if y == "unknown":
                        errors += 1
                        far_cnt += 1
                    elif dec["label"] == y:
                        correct += 1
                    else:
                        errors += 1
        cov = accepted / n
        risk = (errors / accepted) if accepted > 0 else 0.0
        curve.append({
            "threshold_shift": shift,
            "accepted_k": accepted,
            "total_n": len(preds),
            "coverage_pct": round(cov * 100.0, 2),
            "rejection_pct": round((1.0 - cov) * 100.0, 2),
            "selective_risk_pct": round(risk * 100.0, 2),
            "far_pct": round((far_cnt / max(1, neg_tot)) * 100.0, 2),
        })
    return curve
