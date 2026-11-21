"""Method C: Dedicated White-Cane vs Slender-Confuser Verifier & 3-State Selective Classifier.

Separates genuine white mobility canes (straight, folded, red-tipped, handheld, partially occluded)
from visually similar slender confusers (wooden walking sticks, trekking poles, umbrellas,
metal rods/pipes, white road markings/bollards) using:
- Representation-matched DINOv2 similarity & slender-confuser gallery margin,
- Interpretable physical/chromatic shaft features along the rectified predicted mask,
- Out-of-fold 5-fold group-aware CV Ridge-Logistic calibration on `dev_cal_v2` ONLY.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .common_utils import OBJECT_IDS
from .localization_research import OrientedCaneProposal
from .segmentation_research import SegmentationResult

FEATURE_NAMES = (
    "bilateral_profile_score",
    "white_purity_core",
    "wood_brown_ratio",
    "red_tip_evidence",
    "metallic_dark_ratio",
    "width_uniformity",
    "canopy_flare_penalty",
    "hand_person_anchor",
    "diagonal_plausibility",
    "shaft_coverage_frac",
    "dinov2_pos_sim",
    "dinov2_confuser_margin",
)


@dataclass
class CaneVerifierConfig:
    tau_cane_sim: float = 0.44
    tau_ambig_sim: float = 0.37
    delta_neg_min: float = 0.015
    prob_known: float = 0.56
    prob_ambig: float = 0.36
    min_white_purity: float = 0.28
    max_wood_veto: float = 0.52
    c_reg: float = 0.6
    use_structural_verifier: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def extract_cane_verification_features(
    image: np.ndarray,
    proposal: OrientedCaneProposal,
    seg_res: SegmentationResult,
    dinov2_pos_sim: float,
    dinov2_neg_sim: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract 12 physical, chromatic, geometric, and DINOv2 margin features from predicted proposal+mask."""
    strip = seg_res.strip_bgr
    m_bin = seg_res.strip_mask_bin > 0
    hs, ws = strip.shape[:2]

    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).astype(np.float32)
    l_ch = lab[:, :, 0] * (100.0 / 255.0)
    a_ch = lab[:, :, 1] - 128.0
    b_ch = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_ch, b_ch)
    hue_cv = hsv[:, :, 0]

    n_fg = int(np.sum(m_bin))
    if n_fg < 12:
        core_idx = slice(hs // 3, (2 * hs) // 3)
        l_fg = l_ch[core_idx, :].ravel()
        c_fg = chroma[core_idx, :].ravel()
        a_fg = a_ch[core_idx, :].ravel()
        b_fg = b_ch[core_idx, :].ravel()
        h_fg = hue_cv[core_idx, :].ravel()
    else:
        l_fg = l_ch[m_bin]
        c_fg = chroma[m_bin]
        a_fg = a_ch[m_bin]
        b_fg = b_ch[m_bin]
        h_fg = hue_cv[m_bin]

    # 1. Pure white shaft ratio (high L*, low C* - white matte/glossy cane shaft)
    white_px = (l_fg >= 58.0) & (c_fg <= 24.0)
    white_purity_core = float(np.mean(white_px))

    # 2. Wooden / tan / yellow-brown walking stick cue
    wood_px = (b_fg >= 10.0) & (c_fg >= 14.0) & (l_fg < 78.0) & (h_fg >= 7.0) & (h_fg <= 32.0)
    wood_brown_ratio = float(np.mean(wood_px))

    # 3. Red reflective tip or striped bands (either bottom end OR top end depending on hold angle)
    red_2d = (a_ch >= 16.0) & (hsv[:, :, 1] >= 75.0) & (hsv[:, :, 2] >= 55.0) & ((hue_cv <= 10.0) | (hue_cv >= 168.0))
    if n_fg >= 12:
        red_on_fg = red_2d & m_bin
        red_frac_all = float(np.mean(red_on_fg[m_bin]))
        # Check extremity red band (left 30% or right 30% of strip along length)
        w30 = max(6, int(0.30 * ws))
        left_fg = max(1, int(np.sum(m_bin[:, :w30])))
        right_fg = max(1, int(np.sum(m_bin[:, ws - w30:])))
        r_left = float(np.sum(red_on_fg[:, :w30]) / left_fg)
        r_right = float(np.sum(red_on_fg[:, ws - w30:]) / right_fg)
        red_tip_evidence = float(np.clip(max(red_frac_all * 2.2, max(r_left, r_right)), 0.0, 1.0))
    else:
        red_tip_evidence = float(np.clip(proposal.red_tip_score, 0.0, 1.0))

    # 4. Dark/metallic/black cane or umbrella/trekking pole shaft
    dark_px = (l_fg < 54.0) & (c_fg < 16.0)
    metallic_dark_ratio = float(np.mean(dark_px))

    # 5. Shaft thickness uniformity along active columns
    col_heights = np.sum(m_bin, axis=0)
    act_h = col_heights[col_heights > 0]
    if act_h.size >= 8 and float(np.mean(act_h)) > 0.5:
        cv_w = float(np.std(act_h) / max(1.0, np.mean(act_h)))
        width_uniformity = float(np.clip(1.0 - cv_w, 0.0, 1.0))
    else:
        width_uniformity = 0.4

    # 6. Umbrella canopy / attached dark fabric flare around the shaft outside core mask
    dil = cv2.dilate(m_bin.astype(np.uint8), np.ones((15, 7), dtype=np.uint8)) > 0
    halo = dil & ~m_bin
    if np.any(halo):
        dark_halo = float(np.mean((l_ch[halo] < 40.0) & (chroma[halo] < 22.0)))
    else:
        dark_halo = 0.0
    # Subtract flare if whole background is dark pavement; check chromatic cloth / asymmetric bulk
    top_dark = float(np.mean(l_ch[:max(4, hs // 5), :] < 38.0))
    bot_dark = float(np.mean(l_ch[hs - max(4, hs // 5):, :] < 38.0))
    asym_bulk = abs(top_dark - bot_dark)
    canopy_flare_penalty = float(np.clip(0.55 * asym_bulk + 0.45 * max(0.0, dark_halo - 0.35), 0.0, 1.0))

    # 7. Diagonal human-cane sweep plausibility
    ang = float(proposal.obb_cxcywh_angle[4])
    horiz_dist = min(ang, 180.0 - ang)
    vert_dist = abs(ang - 90.0)
    # Pedestrian white canes typically tilt 18..78 deg or lie horizontal/folded
    diag_score = float(np.clip(min(horiz_dist, vert_dist + 18.0) / 30.0, 0.2, 1.0))
    if min(horiz_dist, vert_dist) < 4.0 and proposal.hand_anchor_score < 0.25 and red_tip_evidence < 0.22:
        diag_score = 0.15

    margin = float(dinov2_pos_sim - dinov2_neg_sim)
    feat_dict = {
        "bilateral_profile_score": round(float(proposal.bilateral_profile_score), 4),
        "white_purity_core": round(white_purity_core, 4),
        "wood_brown_ratio": round(wood_brown_ratio, 4),
        "red_tip_evidence": round(red_tip_evidence, 4),
        "metallic_dark_ratio": round(metallic_dark_ratio, 4),
        "width_uniformity": round(width_uniformity, 4),
        "canopy_flare_penalty": round(canopy_flare_penalty, 4),
        "hand_person_anchor": round(float(proposal.hand_anchor_score), 4),
        "diagonal_plausibility": round(diag_score, 4),
        "shaft_coverage_frac": round(float(seg_res.shaft_coverage_frac), 4),
        "dinov2_pos_sim": round(float(dinov2_pos_sim), 4),
        "dinov2_confuser_margin": round(margin, 4),
    }
    vec = np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
    return vec, feat_dict


class CaneVsConfuserVerifier:
    """Method C classifier calibrated strictly on `dev_cal_v2` via 5-fold group-aware CV."""

    def __init__(self, cfg: CaneVerifierConfig | None = None) -> None:
        self.cfg = cfg or CaneVerifierConfig()
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept: float = 0.0
        self.neg_gallery_embs: np.ndarray | None = None

    def fit_negative_gallery(self, neg_embs: np.ndarray) -> None:
        self.neg_gallery_embs = neg_embs.astype(np.float32)

    def max_neg_similarity(self, emb: np.ndarray) -> float:
        if self.neg_gallery_embs is None or self.neg_gallery_embs.size == 0:
            return 0.0
        sims = self.neg_gallery_embs @ emb
        # Top-3 blended negative pressure reduces single-outlier noise
        top3 = np.sort(sims)[-min(3, len(sims)):]
        return float(0.45 * top3[-1] + 0.55 * np.mean(top3))

    def fit_oof_and_calibrate(
        self,
        features: np.ndarray,
        labels_is_cane: np.ndarray,
        cluster_ids: list[str],
        n_splits: int = 5,
    ) -> dict:
        """Fit group-aware 5-fold CV OOF predictions on `dev_cal_v2`, calibrate prob/sim thresholds, and refit full dev."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels_is_cane, dtype=np.int32)
        n = len(y)
        if n < 10:
            raise ValueError("Insufficient samples for CV verifier fitting")

        # Deterministic group-aware fold assignment stratified by label
        unique_clusters = sorted(set(cluster_ids))
        pos_clusters = [c for c in unique_clusters if any(y[i] == 1 and cluster_ids[i] == c for i in range(n))]
        neg_clusters = [c for c in unique_clusters if c not in pos_clusters]
        fold_map: dict[str, int] = {}
        for idx, c in enumerate(pos_clusters):
            fold_map[c] = idx % n_splits
        for idx, c in enumerate(neg_clusters):
            fold_map[c] = idx % n_splits
        folds = np.array([fold_map[c] for c in cluster_ids], dtype=np.int32)

        oof_probs = np.zeros(n, dtype=np.float64)
        for f in range(n_splits):
            tr = folds != f
            va = folds == f
            if not np.any(va):
                continue
            sc = StandardScaler()
            x_tr = sc.fit_transform(x[tr])
            x_va = sc.transform(x[va])
            clf = LogisticRegression(C=self.cfg.c_reg, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(x_tr, y[tr])
            oof_probs[va] = clf.predict_proba(x_va)[:, 1]

        # Sweep probability/sim/margin/purity thresholds on OOF dev predictions prioritizing low FAR while maximizing recall
        best_p_known = 0.56
        best_score = -1e9
        sims_all = x[:, FEATURE_NAMES.index("dinov2_pos_sim")]
        margins_all = x[:, FEATURE_NAMES.index("dinov2_confuser_margin")]
        whites_all = x[:, FEATURE_NAMES.index("white_purity_core")]
        woods_all = x[:, FEATURE_NAMES.index("wood_brown_ratio")]

        best_tau = 0.42
        best_delta_m = 0.0
        best_min_wp = 0.14
        for tau in (0.34, 0.38, 0.42, 0.46, 0.50):
            for delta_m in (-0.06, -0.04, -0.02, 0.0, 0.02):
                for min_wp in (0.08, 0.14, 0.22):
                    for p_thr in np.linspace(0.44, 0.70, 14):
                        pred_pos = (
                            (oof_probs >= p_thr)
                            & (sims_all >= tau)
                            & (margins_all >= delta_m)
                            & (whites_all >= min_wp)
                            & (woods_all <= self.cfg.max_wood_veto)
                        )
                        tp = int(np.sum(pred_pos & (y == 1)))
                        fp = int(np.sum(pred_pos & (y == 0)))
                        fn = int(np.sum(~pred_pos & (y == 1)))
                        rec = tp / max(1, tp + fn)
                        far = fp / max(1, int(np.sum(y == 0)))
                        # Strict penalty when dev FAR > 5%
                        obj = rec - 2.2 * far - max(0.0, far - 0.05) * 4.0
                        if obj > best_score:
                            best_score = obj
                            best_tau = float(tau)
                            best_delta_m = float(delta_m)
                            best_min_wp = float(min_wp)
                            best_p_known = float(round(float(p_thr), 3))

        self.cfg.tau_cane_sim = round(best_tau, 4)
        self.cfg.tau_ambig_sim = round(max(0.28, best_tau - 0.065), 4)
        self.cfg.delta_neg_min = round(best_delta_m, 4)
        self.cfg.min_white_purity = round(best_min_wp, 4)
        self.cfg.prob_known = round(best_p_known, 4)
        self.cfg.prob_ambig = round(max(0.26, best_p_known - 0.20), 4)

        # Fit final model on full dev_cal_v2
        sc_full = StandardScaler()
        x_full = sc_full.fit_transform(x)
        clf_full = LogisticRegression(C=self.cfg.c_reg, class_weight="balanced", max_iter=1000, random_state=42)
        clf_full.fit(x_full, y)
        self.scaler_mean = sc_full.mean_.astype(np.float32)
        self.scaler_scale = np.where(sc_full.scale_ < 1e-6, 1.0, sc_full.scale_).astype(np.float32)
        self.coef = clf_full.coef_[0].astype(np.float32)
        self.intercept = float(clf_full.intercept_[0])

        oof_pred = (
            (oof_probs >= self.cfg.prob_known)
            & (sims_all >= self.cfg.tau_cane_sim)
            & (margins_all >= self.cfg.delta_neg_min)
            & (whites_all >= self.cfg.min_white_purity)
            & (woods_all <= self.cfg.max_wood_veto)
        )
        return {
            "oof_n": int(n),
            "oof_pos_recall": round(float(np.mean(oof_pred[y == 1])), 4),
            "oof_neg_far": round(float(np.mean(oof_pred[y == 0])), 4),
            "selected_tau_cane_sim": self.cfg.tau_cane_sim,
            "selected_delta_neg_min": self.cfg.delta_neg_min,
            "selected_min_white_purity": self.cfg.min_white_purity,
            "selected_prob_known": self.cfg.prob_known,
            "coef_weights": {k: round(float(w), 4) for k, w in zip(FEATURE_NAMES, self.coef)},
        }

    def predict_probability(self, feat_vec: np.ndarray) -> float:
        if self.coef is None or self.scaler_mean is None or self.scaler_scale is None:
            # Rule-based fallback prior to dev fitting (unit tests)
            d = dict(zip(FEATURE_NAMES, feat_vec.tolist()))
            logit = (
                2.5 * (d["white_purity_core"] - 0.38)
                + 2.2 * d["red_tip_evidence"]
                + 2.0 * (d["bilateral_profile_score"] - 0.3)
                + 3.5 * (d["dinov2_pos_sim"] - 0.42)
                + 4.0 * d["dinov2_confuser_margin"]
                + 1.0 * d["hand_person_anchor"]
                - 2.6 * d["wood_brown_ratio"]
                - 1.8 * d["metallic_dark_ratio"]
                - 1.5 * d["canopy_flare_penalty"]
            )
            return float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit)))))
        z = (feat_vec.astype(np.float32) - self.scaler_mean) / self.scaler_scale
        logit = float(np.dot(z, self.coef) + self.intercept)
        return float(1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, logit)))))

    def decide(
        self,
        feat_vec: np.ndarray,
        feat_dict: dict[str, float],
        dinov2_only_mode: bool = False,
    ) -> dict:
        """Return 3-state selective decision: KNOWN / AMBIGUOUS / UNKNOWN with transparent evidence."""
        sim = float(feat_dict["dinov2_pos_sim"])
        margin = float(feat_dict["dinov2_confuser_margin"])
        white_p = float(feat_dict["white_purity_core"])
        wood_r = float(feat_dict["wood_brown_ratio"])
        red_e = float(feat_dict["red_tip_evidence"])
        dark_r = float(feat_dict["metallic_dark_ratio"])
        flare = float(feat_dict["canopy_flare_penalty"])
        prob = self.predict_probability(feat_vec)

        if dinov2_only_mode or not self.cfg.use_structural_verifier:
            if sim >= self.cfg.tau_cane_sim and margin >= self.cfg.delta_neg_min:
                status, label = "KNOWN", "object_01"
                reason = "dinov2_calibrated_match"
            elif sim >= self.cfg.tau_ambig_sim:
                status, label = "AMBIGUOUS", "AMBIGUOUS"
                reason = "dinov2_borderline_band"
            else:
                status, label = "UNKNOWN", "UNKNOWN"
                reason = "dinov2_below_threshold"
            return {
                "status": status,
                "label": label,
                "score": round(sim, 4),
                "margin": round(margin, 4),
                "verifier_prob": round(prob, 4),
                "reason": reason,
            }

        # Physical confuser vetoes (wooden sticks, predominantly black umbrellas/rods without white core)
        is_wood_veto = wood_r > self.cfg.max_wood_veto and red_e < 0.18
        is_dark_veto = dark_r > 0.62 and white_p < 0.22 and red_e < 0.18
        is_umbrella_veto = flare > 0.58 and white_p < 0.35

        if is_wood_veto:
            return {
                "status": "UNKNOWN",
                "label": "UNKNOWN",
                "score": round(prob, 4),
                "margin": round(margin, 4),
                "verifier_prob": round(prob, 4),
                "reason": "veto_wooden_stick_chroma",
            }
        if is_dark_veto or is_umbrella_veto:
            return {
                "status": "UNKNOWN",
                "label": "UNKNOWN",
                "score": round(prob, 4),
                "margin": round(margin, 4),
                "verifier_prob": round(prob, 4),
                "reason": "veto_dark_umbrella_or_pole",
            }

        passes_known = (
            prob >= self.cfg.prob_known
            and sim >= self.cfg.tau_cane_sim
            and margin >= self.cfg.delta_neg_min
            and white_p >= self.cfg.min_white_purity
        )
        # Allow strong multimodal mobility-cane evidence (red reflective tip + clear white shaft + person anchor)
        strong_cane_override = (
            prob >= (self.cfg.prob_known - 0.05)
            and sim >= (self.cfg.tau_cane_sim - 0.03)
            and white_p >= 0.36
            and red_e >= 0.35
            and margin >= 0.0
        )

        if passes_known or strong_cane_override:
            return {
                "status": "KNOWN",
                "label": "object_01",
                "score": round(prob, 4),
                "margin": round(margin, 4),
                "verifier_prob": round(prob, 4),
                "reason": "cane_verified_known",
            }

        # Plain white pole/stick or partial evidence -> AMBIGUOUS rather than unsafe hallucination
        if (prob >= self.cfg.prob_ambig and sim >= self.cfg.tau_ambig_sim) or (
            white_p >= 0.42 and sim >= self.cfg.tau_ambig_sim and feat_dict["hand_person_anchor"] < 0.3 and red_e < 0.20
        ):
            return {
                "status": "AMBIGUOUS",
                "label": "AMBIGUOUS",
                "score": round(prob, 4),
                "margin": round(margin, 4),
                "verifier_prob": round(prob, 4),
                "reason": "cane_ambiguous_evidence",
            }

        return {
            "status": "UNKNOWN",
            "label": "UNKNOWN",
            "score": round(prob, 4),
            "margin": round(margin, 4),
            "verifier_prob": round(prob, 4),
            "reason": "cane_verifier_rejected",
        }

    def to_state_dict(self) -> dict:
        return {
            "cfg": self.cfg.to_dict(),
            "scaler_mean": self.scaler_mean.tolist() if self.scaler_mean is not None else None,
            "scaler_scale": self.scaler_scale.tolist() if self.scaler_scale is not None else None,
            "coef": self.coef.tolist() if self.coef is not None else None,
            "intercept": self.intercept,
        }

    def load_state_dict(self, state: dict) -> None:
        self.cfg = CaneVerifierConfig(**state["cfg"])
        self.scaler_mean = np.array(state["scaler_mean"], dtype=np.float32) if state.get("scaler_mean") is not None else None
        self.scaler_scale = np.array(state["scaler_scale"], dtype=np.float32) if state.get("scaler_scale") is not None else None
        self.coef = np.array(state["coef"], dtype=np.float32) if state.get("coef") is not None else None
        self.intercept = float(state.get("intercept", 0.0))
