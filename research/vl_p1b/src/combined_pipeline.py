"""Method C: Combined Improved Proposer + Hard-Negative DINOv2 Verifier Pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .common_utils import OBJECT_IDS
from .hard_negative_research import HardNegativeRecognizer
from .proposal_research import ObjectProposerVL1b, ProposalBox


@dataclass
class E2EDecisionConfig:
    e2e_tau_offset: float = 0.0   # Optional multiple-proposal offset calibrated on calibration_v1b_full
    prefer_non_person_crops: bool = True


class CombinedE2EPipeline:
    """Method C end-to-end pipeline: Proposer -> DINOv2 + Hard-Negative Verification -> 3-state output."""

    def __init__(
        self,
        proposer: ObjectProposerVL1b,
        recognizer: HardNegativeRecognizer,
        e2e_cfg: E2EDecisionConfig | None = None,
    ) -> None:
        self.proposer = proposer
        self.recognizer = recognizer
        self.e2e_cfg = e2e_cfg or E2EDecisionConfig()

    def predict(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided to CombinedE2EPipeline")
        t_start = time.perf_counter()
        proposals, prop_ms = self.proposer.propose(image)
        if not proposals:
            return {
                "status": "PROPOSAL_MISSED",
                "label": None,
                "score": 0.0,
                "pos_score": 0.0,
                "neg_sim": 0.0,
                "margin": 0.0,
                "proposals": [],
                "proposal_sources": [],
                "selected_box": None,
                "selected_source": None,
                "num_embeddings": 0,
                "proposal_ms": prop_ms,
                "verify_ms": 0.0,
                "total_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
            }

        best_known: dict | None = None
        best_any: dict | None = None
        n_emb = 0
        verify_ms_total = 0.0
        offset = float(self.e2e_cfg.e2e_tau_offset)

        for prop in proposals:
            x, y, bw, bh = prop.bbox_xywh
            roi = image[y:y + bh, x:x + bw]
            if roi.size == 0:
                continue
            dec = self.recognizer.predict(roi)
            n_emb += 1
            verify_ms_total += float(dec.get("feat_ms", 0.0)) + float(dec.get("match_ms", 0.0))

            # Apply calibrated multiple-proposal offset when deciding KNOWN across K candidates
            eff_status = dec["status"]
            if eff_status == "KNOWN" and offset > 0.0:
                oid = dec["label"]
                base_tau = (
                    self.recognizer.cfg.tau_global
                    if self.recognizer.cfg.mode in ("A0_baseline_hybrid", "A1_neg_prototypes")
                    else float(self.recognizer.cfg.tau_per_class.get(oid, self.recognizer.cfg.tau_global))
                )
                if dec["score"] < (base_tau + offset):
                    eff_status = "UNKNOWN"

            # Mild rank boost for tight non-person / non-fullframe crops so cluttered person boxes don't override clean cane crops
            tight_bonus = 0.0
            if self.e2e_cfg.prefer_non_person_crops and prop.source not in ("yolox_coco_0", "full_frame"):
                tight_bonus = 0.015

            entry = {
                "bbox_xywh": prop.bbox_xywh,
                "source": prop.source,
                "eff_status": eff_status,
                "label": dec["label"] if eff_status == "KNOWN" else None,
                "decision": dec,
                "sort_key": (dec["score"] + tight_bonus, dec["margin"]),
            }
            if eff_status == "KNOWN":
                if best_known is None or entry["sort_key"] > best_known["sort_key"]:
                    best_known = entry
            if best_any is None or entry["decision"]["score"] > best_any["decision"]["score"]:
                best_any = entry

        chosen = best_known or best_any
        if chosen is None:
            return {
                "status": "PROPOSAL_MISSED",
                "label": None,
                "score": 0.0,
                "pos_score": 0.0,
                "neg_sim": 0.0,
                "margin": 0.0,
                "proposals": [],
                "proposal_sources": [],
                "selected_box": None,
                "selected_source": None,
                "num_embeddings": n_emb,
                "proposal_ms": prop_ms,
                "verify_ms": round(verify_ms_total, 3),
                "total_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
            }

        dec = chosen["decision"]
        return {
            "status": chosen["eff_status"],
            "label": chosen["label"],
            "score": float(dec["score"]),
            "pos_score": float(dec.get("pos_score", dec["score"])),
            "neg_sim": float(dec.get("neg_sim", 0.0)),
            "margin": float(dec["margin"]),
            "selected_box": chosen["bbox_xywh"],
            "selected_source": chosen["source"],
            "proposals": [p.bbox_xywh for p in proposals],
            "proposal_sources": [p.source for p in proposals],
            "num_embeddings": n_emb,
            "proposal_ms": prop_ms,
            "verify_ms": round(verify_ms_total, 3),
            "total_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }

    def calibrate_e2e_offset(self, cal_full_images: list[tuple[str, np.ndarray]], max_far: float = 0.05) -> float:
        """Calibrate multiple-proposal offset strictly on calibration_v1b_full (never on holdout)."""
        raw_preds = [(y, self.proposer.propose(img)[0], img) for y, img in cal_full_images]
        # Precompute per-proposal recognizer decisions on calibration split
        pre = []
        for y, props, img in raw_preds:
            p_decs = []
            for p in props:
                px, py, pw, ph = p.bbox_xywh
                roi = img[py:py + ph, px:px + pw]
                if roi.size > 0:
                    p_decs.append((p, self.recognizer.predict(roi)))
            pre.append((y, p_decs))

        best = (-1e9, 0.0)
        for off in (0.0, 0.015, 0.03, 0.045, 0.06):
            tp = fa = n_pos = n_neg = 0
            for y, p_decs in pre:
                best_k = None
                for p, dec in p_decs:
                    if dec["status"] != "KNOWN":
                        continue
                    oid = dec["label"]
                    base_tau = (
                        self.recognizer.cfg.tau_global
                        if self.recognizer.cfg.mode in ("A0_baseline_hybrid", "A1_neg_prototypes")
                        else float(self.recognizer.cfg.tau_per_class.get(oid, self.recognizer.cfg.tau_global))
                    )
                    if dec["score"] >= base_tau + off:
                        tb = 0.015 if (self.e2e_cfg.prefer_non_person_crops and p.source not in ("yolox_coco_0", "full_frame")) else 0.0
                        key = (dec["score"] + tb, dec["margin"])
                        if best_k is None or key > best_k[0]:
                            best_k = (key, oid)
                pred_lbl = best_k[1] if best_k is not None else "UNKNOWN"
                if y == "unknown":
                    n_neg += 1
                    if pred_lbl in OBJECT_IDS:
                        fa += 1
                else:
                    n_pos += 1
                    if pred_lbl == y:
                        tp += 1
            acc = tp / max(1, n_pos)
            far = fa / max(1, n_neg)
            obj = acc - max(0.0, far - max_far) * 5.0 - 0.8 * far
            if obj > best[0]:
                best = (obj, off)
        self.e2e_cfg.e2e_tau_offset = best[1]
        return float(best[1])
