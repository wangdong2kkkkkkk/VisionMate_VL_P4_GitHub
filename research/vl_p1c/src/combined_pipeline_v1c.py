"""Sections V & VI: Multi-Proposal False-Positive Control & Combined Pipeline V1c.

Addresses Problem 2 (FAR rises when taking max over K crops):
1. Source-Category Compatibility Gate: thin-line geometry crops may ONLY propose object_01
   (white cane) or narrow objects, NEVER wallets/slippers/shoes, and object_01 thin-line crops
   must meet a minimum geometric cane-evidence gate.
2. Non-Redundant Multi-Proposal Support: two proposals count as independent support only when
   both favor class c AND 0.10 <= IoU(p_i, p_j) <= 0.68 (not near-duplicate boxes!).
3. Image-Level Family-Wise Threshold Adjustment:
   tau_eff(c, source, K_eval, has_support) = tau_per_class[c] + source_penalty[source]
       + alpha_k * log(1 + K_eval / 4.0) + (singleton_penalty if not supported else 0.0) + e2e_tau_offset
4. Soft Oriented Corridor Masking via `extract_thin_aware_roi` for F_thin diagonal shafts.
5. Explicit 3-state image output: KNOWN | AMBIGUOUS | UNKNOWN | PROPOSAL_MISSED.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time

import numpy as np

from .calibration_research import CalibratedRecognizerV1c, CalibrationConfigV1c, extract_roi_geometry_features
from .common_utils import OBJECT_IDS
from .proposal_research_v1c import ObjectProposerV1c
from .proposal_types import ProposalBox, iou_xywh
from .thin_object_proposer import extract_thin_aware_roi

# Compatibility rules prevent slender door-frame/table edges from masquerading as wallets/shoes
THIN_ALLOWED_CLASSES = {"object_01"}
SMALL_PREFERRED_CLASSES = {"object_02", "object_03"}
DEFAULT_SOURCE_OFFSETS = {
    "yolox_non_person": 0.0,
    "fg_contour": 0.015,
    "ms_sal": 0.025,
    "context": 0.010,
    "yolox_person": 0.045,
    "small_detail": 0.025,
    "thin_cane": 0.020,
    "thin_legacy": 0.045,
}


def canonical_source_group(prop: ProposalBox) -> str:
    src = prop.source
    if src.startswith("yolox_coco_"):
        return "yolox_person" if prop.coco_class_id == 0 else "yolox_non_person"
    if src == "fg_contour":
        return "fg_contour"
    if src in ("ms_sal_mid", "ms_sal_tight"):
        return "ms_sal"
    if src in ("center_72", "center_55", "sal_peak_72", "sal_peak_55", "full_frame", "person_wing_left", "person_wing_right"):
        return "context"
    if src in ("person_cane_ray", "oriented_thin_cane"):
        return "thin_cane"
    if src in ("geom_thin_line", "person_lower_cane"):
        return "thin_legacy"
    if src in ("geom_small_detail", "person_hand_zone"):
        return "small_detail"
    return "context"


@dataclass
class MultiProposalControlConfig:
    enable_control: bool = True
    alpha_k: float = 0.02  # Family-wise error penalty wrt log(1 + K_eval / 4)
    singleton_penalty_heuristic: float = 0.030  # Extra margin needed if heuristic box has no independent multi-view support
    singleton_penalty_primary: float = 0.008
    min_thin_cane_evidence: float = 0.42
    support_min_pos_score: float = 0.41
    e2e_tau_offset: float = 0.0
    use_thin_corridor_mask: bool = True
    source_offsets: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SOURCE_OFFSETS))
    # Baseline v1b-style greedy max behavior when enable_control=False (for strict ablation)
    legacy_non_person_bonus: float = 0.0


class CombinedPipelineV1c:
    """End-to-end Proposer + DINOv2 + Multi-Proposal Control pipeline."""

    def __init__(
        self,
        proposer: ObjectProposerV1c,
        recognizer: CalibratedRecognizerV1c,
        mp_cfg: MultiProposalControlConfig | None = None,
    ) -> None:
        self.proposer = proposer
        self.recognizer = recognizer
        self.mp_cfg = mp_cfg or MultiProposalControlConfig()

    def _evaluate_proposals(self, image: np.ndarray, proposals: list[ProposalBox]) -> tuple[list[dict], float]:
        t_v = time.perf_counter()
        raw_entries = []
        for idx, prop in enumerate(proposals):
            bx, by, bw, bh = prop.bbox_xywh
            if self.mp_cfg.use_thin_corridor_mask and prop.source_family == "F_thin":
                roi = extract_thin_aware_roi(image, prop)
            else:
                roi = image[by:by + bh, bx:bx + bw]
            if roi.size == 0:
                continue
            dec = self.recognizer.predict(roi)
            src_grp = canonical_source_group(prop)
            raw_entries.append({
                "idx": idx,
                "prop": prop,
                "source_group": src_grp,
                "decision": dec,
                "top_class": dec.get("top_class") or (dec["candidates"][0]["id"] if dec.get("candidates") else None),
            })
        verify_ms = (time.perf_counter() - t_v) * 1000.0
        return raw_entries, verify_ms

    def _has_independent_support(self, target_entry: dict, all_entries: list[dict], target_cls: str) -> bool:
        """True only if another proposal predicts target_cls cleanly AND has non-redundant overlap 0.10 <= IoU <= 0.68."""
        b_t = target_entry["prop"].bbox_xywh
        for other in all_entries:
            if other is target_entry or ("idx" in other and "idx" in target_entry and other["idx"] == target_entry["idx"]):
                continue
            ov = iou_xywh(b_t, other["prop"].bbox_xywh)
            if ov < 0.10 or ov > 0.68:
                continue
            dec_o = other["decision"]
            # Check top-1 or runner-up candidate for target_cls above support floor
            top_o = other["top_class"]
            if top_o == target_cls and dec_o["pos_score"] >= self.mp_cfg.support_min_pos_score:
                return True
        return False

    def decide_from_evaluated(self, raw_entries: list[dict], prop_ms: float, verify_ms: float) -> dict:
        if not raw_entries:
            return {
                "status": "PROPOSAL_MISSED",
                "label": None,
                "score": 0.0,
                "pos_score": 0.0,
                "neg_sim": 0.0,
                "margin": 0.0,
                "selected_box": None,
                "selected_source": None,
                "proposals": [],
                "proposal_sources": [],
                "num_embeddings": 0,
                "proposal_ms": prop_ms,
                "verify_ms": round(verify_ms, 3),
                "total_ms": round(prop_ms + verify_ms, 3),
                "candidate_records": [],
            }

        cfg = self.recognizer.cfg
        mp = self.mp_cfg
        k_eval = len(raw_entries)
        k_pen = mp.alpha_k * math.log(1.0 + k_eval / 4.0)

        best_known = None
        best_ambig = None
        best_any = None
        cand_records = []

        for ent in raw_entries:
            prop: ProposalBox = ent["prop"]
            dec = ent["decision"]
            src_grp = ent["source_group"]
            top_cls = ent["top_class"]
            raw_status = dec["status"]

            if not mp.enable_control:
                # Legacy baseline style selection (for fair ablation: B_v1c + R0 etc.)
                eff_status = raw_status
                eff_lbl = dec["label"]
                bonus = mp.legacy_non_person_bonus if prop.source not in ("yolox_coco_0", "full_frame") else 0.0
                sort_key = (dec["pos_score"] + bonus, dec["margin"])
                supported = False
            else:
                # Apply Multi-Proposal Control Gates!
                eff_status = raw_status
                eff_lbl = dec["label"]
                supported = False

                if top_cls is not None:
                    # 1. Source-Category Compatibility Gate (prevents thin lines becoming wallets/slippers/shoes)
                    if src_grp in ("thin_cane", "thin_legacy"):
                        if top_cls not in THIN_ALLOWED_CLASSES:
                            eff_status, eff_lbl = "UNKNOWN", None
                        elif prop.source_family == "F_thin" and src_grp == "thin_cane":
                            if prop.thin_evidence_score < mp.min_thin_cane_evidence:
                                eff_status, eff_lbl = "UNKNOWN", None
                        elif src_grp == "thin_legacy" and prop.raw_confidence < 0.45:
                            eff_status, eff_lbl = "UNKNOWN", None

                    # 2. Image-level adjusted threshold + non-redundant multi-view consistency
                    if eff_status == "KNOWN":
                        base_tau = (
                            cfg.tau_global
                            if cfg.mode in ("R0_vl_p1_baseline", "R1_global_conservative")
                            else float(cfg.tau_per_class.get(top_cls, cfg.tau_global))
                        )
                        src_off = float(mp.source_offsets.get(src_grp, 0.02))
                        supported = self._has_independent_support(ent, raw_entries, top_cls)
                        is_primary = src_grp in ("yolox_non_person", "fg_contour", "context")
                        if supported:
                            sup_pen = 0.0
                        else:
                            sup_pen = mp.singleton_penalty_primary if is_primary else mp.singleton_penalty_heuristic

                        # Extra guard for handheld white cane that actually showed real thin+red/person evidence
                        cane_bonus = -0.025 if (src_grp == "thin_cane" and top_cls == "object_01" and prop.thin_evidence_score >= 0.56) else 0.0
                        eff_tau = base_tau + src_off + k_pen + sup_pen + mp.e2e_tau_offset + cane_bonus

                        if dec["pos_score"] < eff_tau:
                            # Downgrade marginal singleton or weakly supported box to AMBIGUOUS instead of hard FP
                            if dec["pos_score"] >= (base_tau - 0.02) and supported:
                                eff_status, eff_lbl = "AMBIGUOUS", None
                            else:
                                eff_status, eff_lbl = "AMBIGUOUS" if dec["pos_score"] >= (eff_tau - 0.03) else "UNKNOWN", None
                        elif dec["margin"] < cfg.delta_pos_ambig:
                            eff_status, eff_lbl = "AMBIGUOUS", None

                sort_key = (dec["pos_score"] - float(mp.source_offsets.get(src_grp, 0.0)), dec["margin"])

            row = {
                "prop": prop,
                "source_group": src_grp,
                "eff_status": eff_status,
                "label": eff_lbl,
                "supported": supported,
                "decision": dec,
                "sort_key": sort_key,
            }
            cand_records.append({
                "source": prop.source,
                "source_group": src_grp,
                "bbox_xywh": prop.bbox_xywh,
                "top_class": top_cls,
                "raw_status": raw_status,
                "eff_status": eff_status,
                "supported": supported,
                "pos_score": float(dec["pos_score"]),
                "neg_sim": float(dec.get("neg_sim", 0.0)),
                "pos_neg_margin": float(dec.get("pos_neg_margin", 0.0)),
                "verifier_score": float(dec.get("verifier_score", 0.0)),
                "margin": float(dec["margin"]),
            })

            if eff_status == "KNOWN":
                if best_known is None or row["sort_key"] > best_known["sort_key"]:
                    best_known = row
            elif eff_status == "AMBIGUOUS":
                if best_ambig is None or row["sort_key"] > best_ambig["sort_key"]:
                    best_ambig = row
            if best_any is None or row["sort_key"] > best_any["sort_key"]:
                best_any = row

        chosen = best_known or best_ambig or best_any
        dec = chosen["decision"]
        return {
            "status": chosen["eff_status"],
            "label": chosen["label"],
            "score": float(dec["pos_score"]),
            "pos_score": float(dec["pos_score"]),
            "neg_sim": float(dec.get("neg_sim", 0.0)),
            "pos_neg_margin": float(dec.get("pos_neg_margin", 0.0)),
            "verifier_score": float(dec.get("verifier_score", 0.0)),
            "margin": float(dec["margin"]),
            "selected_box": chosen["prop"].bbox_xywh,
            "selected_source": chosen["prop"].source,
            "selected_source_group": chosen["source_group"],
            "proposals": [e["prop"].bbox_xywh for e in raw_entries],
            "proposal_sources": [e["prop"].source for e in raw_entries],
            "num_embeddings": len(raw_entries),
            "proposal_ms": round(prop_ms, 3),
            "verify_ms": round(verify_ms, 3),
            "total_ms": round(prop_ms + verify_ms, 3),
            "candidate_records": cand_records,
        }

    def predict(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided to CombinedPipelineV1c")
        t0 = time.perf_counter()
        proposals, prop_ms = self.proposer.propose(image)
        if not proposals:
            return self.decide_from_evaluated([], prop_ms, 0.0)
        raw_entries, verify_ms = self._evaluate_proposals(image, proposals)
        out = self.decide_from_evaluated(raw_entries, prop_ms, verify_ms)
        out["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return out

    def calibrate_e2e_on_full_split(
        self,
        cal_full_records: list[dict],
        cal_full_images: list[np.ndarray],
        max_far_target: float = 0.05,
    ) -> dict:
        """Calibrate image-level (alpha_k, singleton_penalty_heuristic, e2e_tau_offset) strictly on calibration_v1c_full."""
        pre_eval = []
        src_stats: dict[str, dict[str, int]] = {}
        for rec, img in zip(cal_full_records, cal_full_images):
            proposals, prop_ms = self.proposer.propose(img)
            raw_entries, verify_ms = self._evaluate_proposals(img, proposals)
            pre_eval.append((rec, raw_entries, prop_ms, verify_ms))
            y = rec["label_id"]
            for ent in raw_entries:
                sg = ent["source_group"]
                st = src_stats.setdefault(sg, {"total": 0, "raw_known_on_pos_correct": 0, "raw_known_on_neg_fp": 0, "raw_known_wrong_cls": 0})
                st["total"] += 1
                dec = ent["decision"]
                if dec["status"] == "KNOWN":
                    if y == "unknown":
                        st["raw_known_on_neg_fp"] += 1
                    elif dec["label"] == y:
                        st["raw_known_on_pos_correct"] += 1
                    else:
                        st["raw_known_wrong_cls"] += 1

        n_pos = max(1, sum(1 for r in cal_full_records if r["label_id"] != "unknown"))
        n_neg = max(1, sum(1 for r in cal_full_records if r["label_id"] == "unknown"))

        best = (-1e9, 0.02, 0.03, 0.0, 0.0, 0.0)
        for ak in (0.01, 0.02, 0.035):
            for sp_h in (0.015, 0.03, 0.045):
                for off in (-0.01, 0.0, 0.015, 0.03, 0.045):
                    self.mp_cfg.alpha_k = ak
                    self.mp_cfg.singleton_penalty_heuristic = sp_h
                    self.mp_cfg.e2e_tau_offset = off
                    tp = fa = wc = 0
                    for rec, raw_entries, prop_ms, verify_ms in pre_eval:
                        dec = self.decide_from_evaluated(raw_entries, prop_ms, verify_ms)
                        y = rec["label_id"]
                        p = dec["label"] if dec["status"] == "KNOWN" else dec["status"]
                        if y == "unknown":
                            if p in OBJECT_IDS:
                                fa += 1
                        else:
                            if p == y:
                                tp += 1
                            elif p in OBJECT_IDS:
                                wc += 1
                    recall = tp / n_pos
                    far = fa / n_neg
                    wc_rate = wc / n_pos
                    obj = recall - max(0.0, far - max_far_target) * 5.0 - 1.2 * far - 0.8 * wc_rate
                    if obj > best[0]:
                        best = (obj, ak, sp_h, off, round(recall * 100.0, 2), round(far * 100.0, 2))

        self.mp_cfg.alpha_k = best[1]
        self.mp_cfg.singleton_penalty_heuristic = best[2]
        self.mp_cfg.e2e_tau_offset = best[3]
        return {
            "alpha_k": best[1],
            "singleton_penalty_heuristic": best[2],
            "e2e_tau_offset": best[3],
            "cal_full_recall_pct": best[4],
            "cal_full_far_pct": best[5],
            "proposal_source_reliability_cal": src_stats,
        }
