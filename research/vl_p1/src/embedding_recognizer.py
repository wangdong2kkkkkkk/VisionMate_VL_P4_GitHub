"""Method B: Lightweight Pretrained Image Embedding Recognizer for VL-P1 (100% offline inference)."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from .image_utils import OBJECT_IDS, list_images, load_image_unicode, pad_to_square

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BACKBONE_FILES = {
    "mobilenetv3_small": ("mobilenetv3_small_100_Opset17.onnx", "onnx_node!/flatten/Flatten", 1024),
    "efficientnet_b0": ("efficientnet_b0_Opset17.onnx", "onnx_node!/global_pool/flatten/Flatten", 1280),
    "dinov2_small_int8": ("dinov2_small_quantized.onnx", "last_hidden_state", 768),
}


@dataclass
class EmbeddingConfig:
    backbone: str = "mobilenetv3_small"  # mobilenetv3_small | efficientnet_b0 | dinov2_small_int8 | ensemble_mv3_dino
    scoring_mode: str = "hybrid"         # 1nn_max | prototype | hybrid
    tau_known: float = 0.50              # calibrated on calibration split
    delta_margin: float = 0.04           # top1 - top2 score margin for AMBIGUOUS
    pad_aspect: bool = True
    dino_pool_mode: str = "cls_plus_patch"  # cls | cls_plus_patch
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


def preprocess_roi(image: np.ndarray, size: int = 224, pad_aspect: bool = True) -> np.ndarray:
    """Convert BGR uint8 image into NCHW float32 tensor with ImageNet normalization."""
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to preprocess_roi")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    work = image
    if pad_aspect:
        # Use border median so studio-white or natural backgrounds pad cleanly without harsh seams.
        top_row = work[0, :, :]
        bot_row = work[-1, :, :]
        left_col = work[:, 0, :]
        right_col = work[:, -1, :]
        border_px = np.concatenate([top_row, bot_row, left_col, right_col], axis=0)
        med = tuple(int(c) for c in np.median(border_px, axis=0))
        work = pad_to_square(work, pad_color=med)

    h, w = work.shape[:2]
    interp = cv2.INTER_AREA if min(h, w) > size else cv2.INTER_LINEAR
    resized = cv2.resize(work, (size, size), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(normed, (2, 0, 1))
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)


class EmbeddingRecognizer:
    """Few-shot category recognizer powered by offline pretrained ONNX embeddings."""

    def __init__(self, models_dir: Path, config: EmbeddingConfig | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.cfg = config or EmbeddingConfig()
        self.load_time_ms: float = 0.0
        self._cv_nets: dict[str, tuple[cv2.dnn.Net, str]] = {}
        self._ort_sessions: dict[str, tuple[ort.InferenceSession, str]] = {}
        self._load_backbone(self.cfg.backbone)

        self.gallery_embeddings: dict[str, np.ndarray] = {}  # oid -> (N, D)
        self.gallery_names: dict[str, list[str]] = {}
        self.prototypes: dict[str, np.ndarray] = {}          # oid -> (D,)

    def _require_model(self, filename: str) -> Path:
        path = self.models_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Required offline ONNX model not found: {path}. "
                f"Ensure models are placed in {self.models_dir} before offline inference."
            )
        return path

    def _load_single_engine(self, backbone_key: str) -> None:
        if backbone_key in ("mobilenetv3_small", "efficientnet_b0"):
            fn, layer_name, _ = BACKBONE_FILES[backbone_key]
            path = self._require_model(fn)
            # Load via byte buffer so Windows non-ASCII paths (e.g., "mắt kính AI") never fail in C++ fopen.
            onnx_buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            net = cv2.dnn.readNetFromONNX(onnx_buf)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._cv_nets[backbone_key] = (net, layer_name)
        elif backbone_key == "dinov2_small_int8":
            fn, _, _ = BACKBONE_FILES[backbone_key]
            path = self._require_model(fn)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.cfg.num_threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            self._ort_sessions[backbone_key] = (sess, inp_name)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_key}")

    def _load_backbone(self, backbone: str) -> None:
        t0 = time.perf_counter()
        if backbone == "ensemble_mv3_dino":
            self._load_single_engine("mobilenetv3_small")
            self._load_single_engine("dinov2_small_int8")
        else:
            self._load_single_engine(backbone)
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0

    def model_disk_size_bytes(self) -> int:
        keys = ["mobilenetv3_small", "dinov2_small_int8"] if self.cfg.backbone == "ensemble_mv3_dino" else [self.cfg.backbone]
        total = 0
        for k in keys:
            fn = BACKBONE_FILES[k][0]
            p = self.models_dir / fn
            if p.is_file():
                total += p.stat().st_size
        return total

    def _extract_single(self, backbone_key: str, tensor: np.ndarray) -> np.ndarray:
        if backbone_key in self._cv_nets:
            net, layer = self._cv_nets[backbone_key]
            net.setInput(tensor)
            feat = net.forward(layer)
            return _l2_normalize(feat)
        if backbone_key in self._ort_sessions:
            sess, inp_name = self._ort_sessions[backbone_key]
            hidden = sess.run(None, {inp_name: tensor})[0]  # (1, 257, 384)
            cls_vec = _l2_normalize(hidden[0, 0, :])
            if self.cfg.dino_pool_mode == "cls":
                return cls_vec
            patch_mean = _l2_normalize(hidden[0, 1:, :].mean(axis=0))
            return _l2_normalize(np.concatenate([cls_vec, patch_mean], axis=0))
        raise RuntimeError(f"Engine not loaded for {backbone_key}")

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        tensor = preprocess_roi(image, size=224, pad_aspect=self.cfg.pad_aspect)
        if self.cfg.backbone == "ensemble_mv3_dino":
            e1 = self._extract_single("mobilenetv3_small", tensor)
            e2 = self._extract_single("dinov2_small_int8", tensor)
            return _l2_normalize(np.concatenate([0.4 * e1, 0.6 * e2], axis=0))
        return self._extract_single(self.cfg.backbone, tensor)

    def enroll_sample(self, oid: str, sample_name: str, image: np.ndarray) -> float:
        """Add one new labeled ROI image into the live gallery and update its class prototype."""
        t0 = time.perf_counter()
        emb = self.extract_embedding(image)[None, :]
        if oid in self.gallery_embeddings:
            self.gallery_embeddings[oid] = np.vstack([self.gallery_embeddings[oid], emb])
            self.gallery_names[oid].append(sample_name)
        else:
            self.gallery_embeddings[oid] = emb
            self.gallery_names[oid] = [sample_name]
        self.prototypes[oid] = _l2_normalize(self.gallery_embeddings[oid].mean(axis=0))
        return (time.perf_counter() - t0) * 1000.0

    def fit_directory(self, train_dir: Path, object_ids: list[str] | None = None) -> dict:
        oids = object_ids or OBJECT_IDS
        t0 = time.perf_counter()
        self.gallery_embeddings = {}
        self.gallery_names = {}
        self.prototypes = {}
        per_sample_ms = []
        for oid in oids:
            for p in list_images(train_dir / oid):
                img = load_image_unicode(p)
                ms = self.enroll_sample(oid, p.name, img)
                per_sample_ms.append(ms)
        if not self.gallery_embeddings:
            raise ValueError(f"No training images found under {train_dir}")
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "enroll_total_ms": round(total_ms, 2),
            "enroll_per_sample_ms": round(float(np.mean(per_sample_ms)), 3),
            "gallery_db_bytes": self.serialized_db_bytes(),
        }

    def serialized_db_bytes(self) -> int:
        buf = io.BytesIO()
        payload = {f"emb_{k}": v for k, v in self.gallery_embeddings.items()}
        payload.update({f"proto_{k}": v for k, v in self.prototypes.items()})
        np.savez(buf, **payload)
        return len(buf.getvalue())

    def score_embedding(self, query_emb: np.ndarray) -> tuple[list[dict], dict]:
        if not self.gallery_embeddings:
            raise RuntimeError("Gallery is empty; call fit_directory() first.")
        t0 = time.perf_counter()
        ranked = []
        mode = self.cfg.scoring_mode
        for oid, embs in self.gallery_embeddings.items():
            sims = (embs @ query_emb).astype(np.float64)
            best_idx = int(np.argmax(sims))
            max_sim = float(sims[best_idx])
            proto_sim = float(self.prototypes[oid] @ query_emb)
            top2_mean = float(np.mean(np.sort(sims)[-min(2, len(sims)):]))
            if mode == "1nn_max":
                score = max_sim
            elif mode == "prototype":
                score = proto_sim
            elif mode == "hybrid":
                score = 0.5 * proto_sim + 0.5 * top2_mean
            else:
                raise ValueError(f"Unknown scoring_mode: {mode}")
            ranked.append({
                "id": oid,
                "score": round(score, 6),
                "max_sim": round(max_sim, 6),
                "proto_sim": round(proto_sim, 6),
                "top2_mean_sim": round(top2_mean, 6),
                "best_sample": self.gallery_names[oid][best_idx],
            })
        ranked.sort(key=lambda r: r["score"], reverse=True)
        match_ms = (time.perf_counter() - t0) * 1000.0
        return ranked, {"match_ms": round(match_ms, 4)}

    def decide(self, ranked: list[dict]) -> dict:
        if not ranked:
            return {"status": "UNKNOWN", "label": None, "score": 0.0, "margin": 0.0, "candidates": []}
        top_score = float(ranked[0]["score"])
        second_score = float(ranked[1]["score"]) if len(ranked) > 1 else -1.0
        margin = top_score - second_score
        if top_score < self.cfg.tau_known:
            status = "UNKNOWN"
            label = None
        elif margin < self.cfg.delta_margin:
            status = "AMBIGUOUS"
            label = None
        else:
            status = "KNOWN"
            label = ranked[0]["id"]
        return {
            "status": status,
            "label": label,
            "score": round(top_score, 6),
            "margin": round(margin, 6),
            "candidates": ranked[:3],
        }

    def predict(self, image: np.ndarray) -> dict:
        t0 = time.perf_counter()
        q_emb = self.extract_embedding(image)
        feat_ms = (time.perf_counter() - t0) * 1000.0
        ranked, m_info = self.score_embedding(q_emb)
        decision = self.decide(ranked)
        decision["feat_ms"] = round(feat_ms, 3)
        decision["match_ms"] = m_info["match_ms"]
        return decision

    def calibrate_thresholds(
        self,
        cal_samples: list[tuple[str, np.ndarray]],
        max_far: float = 0.12,
    ) -> dict:
        """Grid-search (tau_known, delta_margin) strictly on an independent calibration split."""
        if not cal_samples:
            raise ValueError("Calibration sample list is empty")
        precomputed = []
        for true_label, img in cal_samples:
            q_emb = self.extract_embedding(img)
            ranked, _ = self.score_embedding(q_emb)
            precomputed.append((true_label, ranked))

        all_top1 = [r[0]["score"] for _, r in precomputed if r]
        lo = max(0.05, float(min(all_top1)) - 0.02)
        hi = min(0.95, float(max(all_top1)) + 0.02)
        taus = np.linspace(lo, hi, 41)
        deltas = [0.00, 0.015, 0.03, 0.05, 0.075, 0.10]

        best_obj = -1e9
        best_cfg = (float(np.median(taus)), 0.03)
        best_stats = {}

        n_pos = sum(1 for y, _ in precomputed if y != "unknown")
        n_neg = sum(1 for y, _ in precomputed if y == "unknown")

        orig_tau, orig_delta = self.cfg.tau_known, self.cfg.delta_margin
        for tau in taus:
            for delta in deltas:
                self.cfg.tau_known = float(tau)
                self.cfg.delta_margin = float(delta)
                tp = fp_pos = fn_pos = false_accept_neg = rej_neg = amb_pos = 0
                for y, ranked in precomputed:
                    dec = self.decide(ranked)
                    pred = dec["label"] or dec["status"]
                    if y == "unknown":
                        if dec["status"] == "KNOWN":
                            false_accept_neg += 1
                        else:
                            rej_neg += 1
                    else:
                        if pred == y:
                            tp += 1
                        elif dec["status"] == "AMBIGUOUS":
                            amb_pos += 1
                        elif dec["status"] == "UNKNOWN":
                            fn_pos += 1
                        else:
                            fp_pos += 1
                pos_acc = tp / max(1, n_pos)
                far = false_accept_neg / max(1, n_neg)
                neg_rej = 1.0 - far
                # Strongly penalize FAR above cap, otherwise maximize harmonic/balanced trade-off
                penalty = max(0.0, far - max_far) * 3.0
                obj = pos_acc + 0.85 * neg_rej - 0.5 * (fp_pos / max(1, n_pos)) - penalty
                if obj > best_obj:
                    best_obj = obj
                    best_cfg = (float(round(tau, 4)), float(round(delta, 4)))
                    best_stats = {
                        "cal_pos_acc": round(pos_acc, 4),
                        "cal_far": round(far, 4),
                        "cal_neg_rej_rate": round(neg_rej, 4),
                        "cal_pos_wrong_class": fp_pos,
                        "cal_pos_unknown": fn_pos,
                        "cal_pos_ambiguous": amb_pos,
                    }

        self.cfg.tau_known, self.cfg.delta_margin = best_cfg
        return {"tau_known": best_cfg[0], "delta_margin": best_cfg[1], "orig": (orig_tau, orig_delta), **best_stats}
