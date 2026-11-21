"""Soft Calibrated Verifier (C1_CNN_WithVerifier) without brittle binary vetoes or person-anchor bias."""
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

SOFT_VERIFIER_FEATURES = (
    "obj_prob",
    "top_kp_peak",
    "bot_kp_peak",
    "cl_peak",
    "cl_line_support",
    "mask_fg_ratio",
    "length_ratio",
    "white_purity_line",
    "red_tip_line",
    "dark_wood_line",
    "bilateral_contrast_line",
    "line_straight_consistency",
)


def extract_soft_line_features(img_bgr: np.ndarray, cnn_pred: dict) -> tuple[np.ndarray, dict[str, float]]:
    """Extract continuous photometric + geometric evidence strictly along predicted centerline.

    Zero person/hand anchor features: anchor_prob_delta is 0.0 by construction.
    """
    ep = cnn_pred["endpoints_xyxy"]
    h, w = img_bgr.shape[:2]
    # Sample at 256x256 square-padded space if available for speed (~0.5ms), else resize
    sq = cnn_pred.get("sq256")
    if sq is None:
        sq = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    side = max(h, w, 1)
    pad_l = (side - w) // 2
    pad_t = (side - h) // 2
    sc = 256.0 / float(side)
    x1, y1 = (ep[0] + pad_l) * sc, (ep[1] + pad_t) * sc
    x2, y2 = (ep[2] + pad_l) * sc, (ep[3] + pad_t) * sc

    lab = cv2.cvtColor(sq, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch = lab[:, :, 0] * (100.0 / 255.0)
    a_ch = lab[:, :, 1] - 128.0
    b_ch = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_ch, b_ch)

    ts = np.linspace(0.0, 1.0, 21)
    xs = np.clip(np.round(x1 + ts * (x2 - x1)).astype(int), 0, 255)
    ys = np.clip(np.round(y1 + ts * (y2 - y1)).astype(int), 0, 255)

    l_vals = l_ch[ys, xs]
    c_vals = chroma[ys, xs]
    a_vals = a_ch[ys, xs]
    b_vals = b_ch[ys, xs]

    white_pts = np.clip((l_vals - 50.0) / 35.0, 0.0, 1.0) * np.clip((32.0 - c_vals) / 24.0, 0.0, 1.0)
    red_pts = np.clip((a_vals - 12.0) / 26.0, 0.0, 1.0) * np.clip((c_vals - 15.0) / 30.0, 0.0, 1.0)
    wood_pts = np.clip((a_vals - 2.0) / 16.0, 0.0, 1.0) * np.clip((b_vals - 6.0) / 22.0, 0.0, 1.0) * np.clip((72.0 - l_vals) / 35.0, 0.0, 1.0)

    # Bilateral contrast perpendicular to line at +/- 5 px in 256x256
    dx, dy = x2 - x1, y2 - y1
    ln = max(1.0, math.hypot(dx, dy))
    nx, ny = -dy / ln, dx / ln
    xl1 = np.clip(np.round(xs + 5.0 * nx).astype(int), 0, 255)
    yl1 = np.clip(np.round(ys + 5.0 * ny).astype(int), 0, 255)
    xl2 = np.clip(np.round(xs - 5.0 * nx).astype(int), 0, 255)
    yl2 = np.clip(np.round(ys - 5.0 * ny).astype(int), 0, 255)
    bg_l = 0.5 * (l_ch[yl1, xl1] + l_ch[yl2, xl2])
    bilat = float(np.mean(np.clip((l_vals - bg_l) / 25.0, 0.0, 1.0)))

    # Line straightness / aspect consistency
    line_cons = float(min(1.0, cnn_pred["cl_line_support"] * 1.4)) * float(1.0 if cnn_pred["length_ratio"] >= 0.12 else cnn_pred["length_ratio"] / 0.12)

    fdict = {
        "obj_prob": float(cnn_pred["obj_prob"]),
        "top_kp_peak": float(cnn_pred["top_kp_peak"]),
        "bot_kp_peak": float(cnn_pred["bot_kp_peak"]),
        "cl_peak": float(cnn_pred["cl_peak"]),
        "cl_line_support": float(cnn_pred["cl_line_support"]),
        "mask_fg_ratio": float(cnn_pred["mask_fg_ratio"]),
        "length_ratio": float(min(1.5, cnn_pred["length_ratio"])),
        "white_purity_line": float(np.mean(white_pts)),
        "red_tip_line": float(np.max(red_pts)),
        "dark_wood_line": float(np.mean(wood_pts)),
        "bilateral_contrast_line": bilat,
        "line_straight_consistency": line_cons,
    }
    vec = np.asarray([fdict[k] for k in SOFT_VERIFIER_FEATURES], dtype=np.float32)
    return vec, fdict


class SoftCaneVerifierV4:
    """Calibrated continuous verifier (LogisticRegression + HGB blend, zero hard vetoes)."""

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.log_reg: LogisticRegression | None = None
        self.hgb: HistGradientBoostingClassifier | None = None
        self.t_known: float = 0.55
        self.t_uncertain: float = 0.35
        self.oof_metrics: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
        n_splits = min(5, max(2, int(np.sum(y == 1))))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=np.float32)
        for tr, va in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=0.6, max_iter=1000, class_weight="balanced", random_state=seed)
            lr.fit(sc.transform(X[tr]), y[tr])
            hgb = HistGradientBoostingClassifier(max_iter=80, max_depth=3, min_samples_leaf=8, l2_regularization=2.0, random_state=seed)
            hgb.fit(X[tr], y[tr])
            p_lr = lr.predict_proba(sc.transform(X[va]))[:, 1]
            p_hgb = hgb.predict_proba(X[va])[:, 1]
            oof[va] = 0.50 * p_lr + 0.50 * p_hgb

        self.scaler = StandardScaler().fit(X)
        self.log_reg = LogisticRegression(C=0.6, max_iter=1000, class_weight="balanced", random_state=seed)
        self.log_reg.fit(self.scaler.transform(X), y)
        self.hgb = HistGradientBoostingClassifier(max_iter=80, max_depth=3, min_samples_leaf=8, l2_regularization=2.0, random_state=seed)
        self.hgb.fit(X, y)
        self.oof_metrics = {
            "n": int(len(y)),
            "n_pos": int(np.sum(y == 1)),
            "oof_pos_mean": round(float(np.mean(oof[y == 1])), 4),
            "oof_neg_mean": round(float(np.mean(oof[y == 0])), 4),
        }
        return self.oof_metrics

    def predict_score(self, vec: np.ndarray) -> float:
        assert self.scaler is not None and self.log_reg is not None and self.hgb is not None
        p_lr = float(self.log_reg.predict_proba(self.scaler.transform(vec[None, :]))[:, 1][0])
        p_hgb = float(self.hgb.predict_proba(vec[None, :])[0, 1])
        return float(np.clip(0.50 * p_lr + 0.50 * p_hgb, 0.0, 1.0))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "log_reg": self.log_reg,
            "hgb": self.hgb,
            "t_known": self.t_known,
            "t_uncertain": self.t_uncertain,
            "oof_metrics": self.oof_metrics,
        }, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "SoftCaneVerifierV4":
        p = joblib.load(path)
        obj = cls()
        obj.scaler = p["scaler"]
        obj.log_reg = p["log_reg"]
        obj.hgb = p["hgb"]
        obj.t_known = float(p["t_known"])
        obj.t_uncertain = float(p["t_uncertain"])
        obj.oof_metrics = dict(p["oof_metrics"])
        return obj
