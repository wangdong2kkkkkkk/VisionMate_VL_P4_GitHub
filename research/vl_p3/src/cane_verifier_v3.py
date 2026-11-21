"""De-Biased Cane vs. Confuser Structural & DINOv2 Verifier (VL-P3 Methods C1 & C2).

Combines:
  1. Masked-crop & foreground-patch DINOv2-Small cosine similarity margins
     against `train_v3` positive canes and `train_v3` hard confusers.
  2. Supervised B2 shaft physical purity & shape attributes
     (`white_purity_on_mask`, `red_tip_presence`, `width_regularity`, `edge_straightness`).
  3. Structural confuser vetoes (wooden walking sticks, dark black poles/umbrellas,
     YOLOX umbrella overlaps, dual parallel trekking/Nordic walking poles).
  4. De-biased `hand_person_anchor`: gated to 0 when shaft lacks white/red evidence
     and strictly capped at <= 0.10 maximum positive score contribution.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .common_utils import compute_sha256
from .keypoint_detector import LocalizationPrediction
from .shaft_segmenter import SegmentationPrediction
from .spatial_features import DINOv2SpatialBackbone, GRID_SIZE, SpatialFeatureBundle

VERIFIER_FEATURE_NAMES = [
    "margin_crop",
    "pos_crop_top3",
    "margin_patch",
    "pos_patch_top3",
    "loc_confidence",
    "loc_kp_confidence",
    "loc_cl_confidence",
    "seg_visible_shaft_ratio",
    "seg_white_purity",
    "seg_red_tip",
    "seg_width_regularity",
    "seg_edge_straightness",
    "seg_confidence",
    "loc_length_ratio",
    "wood_dark_penalty",
    "umbrella_ctx_overlap",
    "parallel_dual_pole_score",
    "gated_hand_person_anchor",
]


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / max(n, 1e-8)).astype(np.float32)


def compute_foreground_patch_embedding(
    bundle: SpatialFeatureBundle,
    seg_pred: SegmentationPrediction | None,
    loc_pred: LocalizationPrediction | None,
) -> np.ndarray:
    """Weighted average of DINOv2 16x16x384 spatial tokens over segmented cane shaft (0 extra ONNX calls)."""
    patches = bundle.patches_16  # [16, 16, 384]
    w16 = None
    if seg_pred is not None and seg_pred.fg_area_ratio > 1e-4:
        w112 = np.maximum(seg_pred.prob_112, 0.25 * (bundle.features_112[:, :, 2] * bundle.features_112[:, :, 6]))
        w16 = cv2.resize(w112, (16, 16), interpolation=cv2.INTER_AREA)
    elif loc_pred is not None and loc_pred.detected:
        w16 = cv2.resize(loc_pred.cl_hm, (16, 16), interpolation=cv2.INTER_AREA)
    else:
        w16 = cv2.resize(bundle.features_112[:, :, 6], (16, 16), interpolation=cv2.INTER_AREA)

    w_flat = np.maximum(0.0, w16.reshape(256))
    # Sharpen toward top foreground tokens so street background tokens don't drown thin cane
    if float(w_flat.max()) > 0.05:
        thr = float(np.percentile(w_flat, 80))
        w_flat = np.where(w_flat >= thr, w_flat ** 1.5, w_flat * 0.1)
    if float(w_flat.sum()) < 1e-5:
        return bundle.cls_emb.astype(np.float32)
    emb = (w_flat[:, None] * patches.reshape(256, 384)).sum(axis=0)
    return _l2_norm(emb)


def compute_masked_shaft_crop_embedding(
    image: np.ndarray,
    bundle: SpatialFeatureBundle,
    seg_pred: SegmentationPrediction | None,
    loc_pred: LocalizationPrediction,
    dino_backbone: DINOv2SpatialBackbone,
) -> tuple[np.ndarray, float]:
    """Build tight background-neutralized shaft crop and embed via DINOv2-Small."""
    h, w = image.shape[:2]
    x1, y1, bw, bh = loc_pred.bbox_xywh
    pad_x = max(18, int(round(0.22 * max(bw, bh))))
    pad_y = max(18, int(round(0.22 * max(bw, bh))))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x1 + bw + pad_x)
    cy2 = min(h, y1 + bh + pad_y)
    if cx2 <= cx1 + 8 or cy2 <= cy1 + 8:
        return bundle.cls_emb.astype(np.float32), 0.0

    crop = image[cy1:cy2, cx1:cx2].copy()
    if seg_pred is not None and int(np.sum(seg_pred.mask_orig[cy1:cy2, cx1:cx2] > 0)) >= 12:
        m_crop = (seg_pred.mask_orig[cy1:cy2, cx1:cx2] > 0).astype(np.uint8)
        m_dil = cv2.dilate(m_crop, np.ones((15, 15), np.uint8)) > 0
        neutral = np.full_like(crop, 127)
        neutral[m_dil] = crop[m_dil]
        crop_use = neutral
    else:
        crop_use = crop

    ch, cw = crop_use.shape[:2]
    side = max(ch, cw)
    sq = np.full((side, side, 3), 127, dtype=np.uint8)
    ox = (side - cw) // 2
    oy = (side - ch) // 2
    sq[oy:oy + ch, ox:ox + cw] = crop_use
    return dino_backbone.embed_canvas_cls(sq)


def _topk_cosine_mean(emb: np.ndarray, bank: np.ndarray | None, k: int = 3) -> float:
    if bank is None or len(bank) == 0:
        return 0.0
    sims = bank @ emb.astype(np.float32)
    kk = min(k, len(sims))
    if kk == len(sims):
        return float(np.mean(sims))
    top = np.partition(sims, -kk)[-kk:]
    return float(np.mean(top))


def _estimate_parallel_dual_pole(
    bundle: SpatialFeatureBundle,
    loc_pred: LocalizationPrediction,
) -> float:
    """Detect twin off-axis parallel thin poles characteristic of trekking poles / Nordic walking."""
    if not loc_pred.detected:
        return 0.0
    tfm = bundle.transform
    gx1, gy1 = tfm.orig_to_grid(loc_pred.endpoints_xyxy[0], loc_pred.endpoints_xyxy[1])
    gx2, gy2 = tfm.orig_to_grid(loc_pred.endpoints_xyxy[2], loc_pred.endpoints_xyxy[3])
    L = max(1.0, math.hypot(gx2 - gx1, gy2 - gy1))
    ux, uy = (gx2 - gx1) / L, (gy2 - gy1) / L
    nx, ny = -uy, ux

    f = bundle.features_112
    # Require person context nearby so random slatted fences don't trigger falsely
    if float(np.max(f[:, :, 33])) < 0.25:
        return 0.0
    ridge_map = f[:, :, 6]
    c2, s2 = math.cos(2.0 * math.atan2(uy, ux)), math.sin(2.0 * math.atan2(uy, ux))
    ori_agree = np.maximum(0.0, f[:, :, 9] * c2 + f[:, :, 10] * s2)
    dual_field = ridge_map * ori_agree

    # Sweep lateral shifts 10..28 grid px away on left/right of primary shaft
    best_off_score = 0.0
    ts = np.linspace(0.1, 0.9, 13)
    for sign in (-1.0, 1.0):
        for dist_lat in (11.0, 15.0, 20.0, 25.0):
            xs = np.clip(np.round(gx1 + ts * (gx2 - gx1) + sign * dist_lat * nx).astype(int), 0, GRID_SIZE - 1)
            ys = np.clip(np.round(gy1 + ts * (gy2 - gy1) + sign * dist_lat * ny).astype(int), 0, GRID_SIZE - 1)
            sc = float(np.mean(dual_field[ys, xs] >= 0.22))
            if sc > best_off_score:
                best_off_score = sc
    return round(float(np.clip(best_off_score, 0.0, 1.0)), 4)


def extract_verifier_features_v3(
    bundle: SpatialFeatureBundle,
    loc_pred: LocalizationPrediction,
    seg_pred: SegmentationPrediction,
    crop_emb: np.ndarray,
    patch_emb: np.ndarray,
    pos_crop_bank: np.ndarray | None,
    neg_crop_bank: np.ndarray | None,
    pos_patch_bank: np.ndarray | None,
    neg_patch_bank: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build 18-dim de-biased verification vector and dictionary."""
    pos_c3 = _topk_cosine_mean(crop_emb, pos_crop_bank, k=3)
    neg_c3 = _topk_cosine_mean(crop_emb, neg_crop_bank, k=3)
    margin_c = pos_c3 - neg_c3

    pos_p3 = _topk_cosine_mean(patch_emb, pos_patch_bank, k=3)
    neg_p3 = _topk_cosine_mean(patch_emb, neg_patch_bank, k=3)
    margin_p = pos_p3 - neg_p3

    tfm = bundle.transform
    gx1, gy1 = tfm.orig_to_grid(loc_pred.endpoints_xyxy[0], loc_pred.endpoints_xyxy[1])
    gx2, gy2 = tfm.orig_to_grid(loc_pred.endpoints_xyxy[2], loc_pred.endpoints_xyxy[3])
    ts = np.linspace(0.0, 1.0, 15)
    xs = np.clip(np.round(gx1 + ts * (gx2 - gx1)).astype(int), 0, GRID_SIZE - 1)
    ys = np.clip(np.round(gy1 + ts * (gy2 - gy1)).astype(int), 0, GRID_SIZE - 1)

    f = bundle.features_112
    umb_overlap = float(np.mean(f[ys, xs, 35]))
    raw_hand_anchor = float(np.max(f[ys[:6], xs[:6], 34])) if len(ys) >= 6 else 0.0

    # Strict de-biasing gate: person/hand anchor gets ZERO credit without genuine white/red cane evidence
    if seg_pred.white_purity_on_mask < 0.42 and seg_pred.red_tip_presence < 0.25:
        gated_anchor = 0.0
    else:
        gated_anchor = float(np.clip(raw_hand_anchor, 0.0, 1.0))

    wood_dark_pen = float(max(seg_pred.wood_on_mask, 0.72 * seg_pred.dark_on_mask))
    dual_pole = _estimate_parallel_dual_pole(bundle, loc_pred)

    vals = [
        margin_c,
        pos_c3,
        margin_p,
        pos_p3,
        loc_pred.confidence,
        loc_pred.kp_confidence,
        loc_pred.cl_confidence,
        seg_pred.visible_shaft_ratio,
        seg_pred.white_purity_on_mask,
        seg_pred.red_tip_presence,
        seg_pred.width_regularity,
        seg_pred.edge_straightness,
        seg_pred.confidence,
        min(1.5, loc_pred.length_ratio),
        wood_dark_pen,
        umb_overlap,
        dual_pole,
        gated_anchor,
    ]
    vec = np.asarray(vals, dtype=np.float32)
    fdict = {name: round(float(v), 5) for name, v in zip(VERIFIER_FEATURE_NAMES, vals)}
    fdict["raw_hand_person_anchor"] = round(raw_hand_anchor, 5)
    return vec, fdict


class DeBiasedCaneVerifierV3:
    """Calibrated ensemble + structural veto classifier for object_01 vs. hard confusers."""

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.log_reg: LogisticRegression | None = None
        self.hgb_clf: HistGradientBoostingClassifier | None = None
        self.pos_crop_bank: np.ndarray | None = None
        self.neg_crop_bank: np.ndarray | None = None
        self.pos_patch_bank: np.ndarray | None = None
        self.neg_patch_bank: np.ndarray | None = None
        self.max_anchor_prob_delta: float = 0.10
        self.t_known: float = 0.62
        self.t_uncertain: float = 0.40
        self.is_fitted: bool = False
        self.oof_metrics: dict = {}

    def set_banks(
        self,
        pos_crop_bank: np.ndarray,
        neg_crop_bank: np.ndarray,
        pos_patch_bank: np.ndarray,
        neg_patch_bank: np.ndarray,
    ) -> None:
        self.pos_crop_bank = np.asarray(pos_crop_bank, dtype=np.float32)
        self.neg_crop_bank = np.asarray(neg_crop_bank, dtype=np.float32)
        self.pos_patch_bank = np.asarray(pos_patch_bank, dtype=np.float32)
        self.neg_patch_bank = np.asarray(neg_patch_bank, dtype=np.float32)

    def fit_oof(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fit OOF-calibrated LogisticRegression + de-biased HGBC (excluding raw anchor dominance)."""
        X = np.asarray(X, dtype=np.float32)
        y = (np.asarray(y) > 0).astype(int)
        anchor_idx = VERIFIER_FEATURE_NAMES.index("gated_hand_person_anchor")

        # Train non-linear HGBC WITHOUT anchor index so trees never over-rely on person boxes!
        non_anchor_cols = [i for i in range(X.shape[1]) if i != anchor_idx]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_scores = np.zeros(len(y), dtype=np.float32)
        for tr, va in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=0.8, class_weight="balanced", max_iter=400, random_state=42).fit(sc.transform(X[tr]), y[tr])
            # Cap positive anchor coefficient cleanly
            lr.coef_[0, anchor_idx] = min(float(lr.coef_[0, anchor_idx]), 0.35)
            hgb = HistGradientBoostingClassifier(
                max_iter=45, max_depth=3, min_samples_leaf=8, learning_rate=0.06, l2_regularization=2.0, random_state=42
            ).fit(X[tr][:, non_anchor_cols], y[tr])
            p_lr = lr.predict_proba(sc.transform(X[va]))[:, 1]
            p_hgb = hgb.predict_proba(X[va][:, non_anchor_cols])[:, 1]
            oof_scores[va] = 0.55 * p_hgb + 0.45 * p_lr

        self.scaler = StandardScaler().fit(X)
        self.log_reg = LogisticRegression(C=0.8, class_weight="balanced", max_iter=400, random_state=42).fit(self.scaler.transform(X), y)
        self.log_reg.coef_[0, anchor_idx] = min(float(self.log_reg.coef_[0, anchor_idx]), 0.35)
        self.hgb_clf = HistGradientBoostingClassifier(
            max_iter=45, max_depth=3, min_samples_leaf=8, learning_rate=0.06, l2_regularization=2.0, random_state=42
        ).fit(X[:, non_anchor_cols], y)
        self.is_fitted = True

        self.oof_metrics = {
            "n_samples": int(len(y)),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
            "oof_pos_mean": round(float(np.mean(oof_scores[y == 1])), 4),
            "oof_neg_mean": round(float(np.mean(oof_scores[y == 0])), 4),
        }
        return self.oof_metrics

    def _raw_model_prob(self, vec: np.ndarray) -> float:
        anchor_idx = VERIFIER_FEATURE_NAMES.index("gated_hand_person_anchor")
        non_anchor_cols = [i for i in range(len(vec)) if i != anchor_idx]
        if not self.is_fitted or self.scaler is None or self.log_reg is None or self.hgb_clf is None:
            # Sensible linear physical fallback prior to fit
            margin = float(vec[0] + 0.8 * vec[2])
            return float(np.clip(0.5 + 1.8 * margin + 0.25 * vec[8] - 0.30 * vec[14], 0.0, 1.0))

        Xs = self.scaler.transform(vec[None, :])
        p_lr = float(self.log_reg.predict_proba(Xs)[:, 1][0])
        p_hgb = float(self.hgb_clf.predict_proba(vec[None, non_anchor_cols])[:, 1][0])
        base_prob = 0.55 * p_hgb + 0.45 * p_lr

        # Enforce hard <= 0.10 positive boost from hand_person_anchor
        vec_no_anchor = vec.copy()
        vec_no_anchor[anchor_idx] = 0.0
        p_lr0 = float(self.log_reg.predict_proba(self.scaler.transform(vec_no_anchor[None, :]))[:, 1][0])
        prob0 = 0.55 * p_hgb + 0.45 * p_lr0
        capped_prob = min(base_prob, prob0 + self.max_anchor_prob_delta)
        return float(np.clip(capped_prob, 0.0, 1.0))

    def check_structural_veto(self, fdict: dict[str, float]) -> str | None:
        """Apply white-cane physical laws against dark poles, wood sticks, umbrellas, twin trekking poles."""
        if fdict["seg_visible_shaft_ratio"] < 0.20 and fdict["loc_confidence"] < 0.25:
            return "veto_no_continuous_shaft"
        if fdict["seg_white_purity"] < 0.26 and fdict["seg_red_tip"] < 0.20:
            return "veto_not_white_shaft"
        if fdict["wood_dark_penalty"] > 0.28 and fdict["seg_white_purity"] < 0.50:
            return "veto_wood_or_dark_confuser"
        if fdict["umbrella_ctx_overlap"] > 0.36 and fdict["seg_white_purity"] < 0.58:
            return "veto_umbrella_overlap"
        if fdict["parallel_dual_pole_score"] > 0.65 and fdict["seg_red_tip"] < 0.22:
            return "veto_parallel_trekking_poles"
        if fdict["loc_length_ratio"] < 0.12:
            return "veto_too_short"
        return None

    def score_and_decide(
        self,
        vec: np.ndarray,
        fdict: dict[str, float],
        apply_vetoes: bool = True,
    ) -> dict:
        t0 = time.perf_counter()
        raw_prob = self._raw_model_prob(vec)
        veto = self.check_structural_veto(fdict) if apply_vetoes else None

        adj_prob = raw_prob
        if veto is not None:
            # Strong but informative down-weighting so severe structural failure rejects cleanly
            adj_prob = min(raw_prob, 0.24 if veto in ("veto_not_white_shaft", "veto_no_continuous_shaft", "veto_wood_or_dark_confuser") else 0.35)

        if adj_prob >= self.t_known and veto is None:
            decision = "KNOWN"
            pred_label = "object_01"
        elif adj_prob >= self.t_uncertain:
            decision = "UNCERTAIN"
            pred_label = "UNCERTAIN"
        else:
            decision = "UNKNOWN"
            pred_label = "UNKNOWN"

        anchor_idx = VERIFIER_FEATURE_NAMES.index("gated_hand_person_anchor")
        v0 = vec.copy()
        v0[anchor_idx] = 0.0
        prob_without_anchor = self._raw_model_prob(v0)
        return {
            "score": round(adj_prob, 4),
            "raw_prob": round(raw_prob, 4),
            "prob_without_anchor": round(prob_without_anchor, 4),
            "anchor_prob_delta": round(raw_prob - prob_without_anchor, 4),
            "decision": decision,
            "pred_label": pred_label,
            "veto_reason": veto,
            "features": fdict,
            "verify_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self.scaler,
            "log_reg": self.log_reg,
            "hgb_clf": self.hgb_clf,
            "pos_crop_bank": self.pos_crop_bank,
            "neg_crop_bank": self.neg_crop_bank,
            "pos_patch_bank": self.pos_patch_bank,
            "neg_patch_bank": self.neg_patch_bank,
            "max_anchor_prob_delta": self.max_anchor_prob_delta,
            "t_known": self.t_known,
            "t_uncertain": self.t_uncertain,
            "is_fitted": self.is_fitted,
            "oof_metrics": self.oof_metrics,
        }
        joblib.dump(payload, path, compress=3)
        return compute_sha256(path)

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> "DeBiasedCaneVerifierV3":
        if expected_sha256:
            actual = compute_sha256(path).lower()
            if actual != expected_sha256.lower():
                raise RuntimeError(f"SHA-256 mismatch for {path}")
        payload = joblib.load(path)
        obj = cls()
        obj.scaler = payload["scaler"]
        obj.log_reg = payload["log_reg"]
        obj.hgb_clf = payload["hgb_clf"]
        obj.pos_crop_bank = payload["pos_crop_bank"]
        obj.neg_crop_bank = payload["neg_crop_bank"]
        obj.pos_patch_bank = payload["pos_patch_bank"]
        obj.neg_patch_bank = payload["neg_patch_bank"]
        obj.max_anchor_prob_delta = float(payload.get("max_anchor_prob_delta", 0.10))
        obj.t_known = float(payload.get("t_known", 0.62))
        obj.t_uncertain = float(payload.get("t_uncertain", 0.40))
        obj.is_fitted = bool(payload["is_fitted"])
        obj.oof_metrics = payload.get("oof_metrics", {})
        return obj
