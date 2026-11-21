"""Method C / Experimental Matrix: End-to-End VL-P2 Pipelines (A0, A1, B1, B2, C1, C2).

Combines:
- Method A: OrientedCaneLocalizer (rotated OBB + collinear gap bridging + bilateral step profile)
- Method B1: CaneShaftSegmenter (rectified corridor dynamic-programming + bilateral shaft mask)
- Method B2: RepresentationEmbeddingBank (rep1..rep6, representation-aligned DINOv2-Small INT8)
- Method C: CaneVsConfuserVerifier (OOF 5-fold CV calibrated cane vs confuser verifier & 3-state output)
- Multi-class integration option for 5-class compatibility regression against VL-P1c R5.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .cane_verifier import (
    CaneVerifierConfig,
    CaneVsConfuserVerifier,
    FEATURE_NAMES,
    extract_cane_verification_features,
)
from .localization_research import (
    OrientedCaneLocalizer,
    OrientedCaneProposal,
    YOLOXPersonAndObjectDetector,
)
from .masked_embedding_research import (
    DINOv2EmbedderV2,
    RepresentationEmbeddingBank,
    build_representation_canvas,
)
from .segmentation_research import CaneShaftSegmenter, SegmentationResult


@dataclass
class PipelineV2Config:
    name: str
    representation: str = "rep4_rotated_masked_crop"
    top_k_proposals: int = 4
    use_structural_verifier: bool = True
    require_segmentation: bool = True
    early_accept_prob: float = 0.65


EXPERIMENT_PRESETS = {
    "A0_axis_baseline": PipelineV2Config(
        name="A0_axis_baseline",
        representation="rep1_axis_crop",
        top_k_proposals=4,
        use_structural_verifier=False,
        require_segmentation=False,
    ),
    "A1_oriented_rotated": PipelineV2Config(
        name="A1_oriented_rotated",
        representation="rep2_rotated_crop",
        top_k_proposals=4,
        use_structural_verifier=False,
        require_segmentation=True,
    ),
    "B1_seg_masked_axis": PipelineV2Config(
        name="B1_seg_masked_axis",
        representation="rep3_masked_crop",
        top_k_proposals=4,
        use_structural_verifier=False,
        require_segmentation=True,
    ),
    "B2_oriented_seg_rot_masked": PipelineV2Config(
        name="B2_oriented_seg_rot_masked",
        representation="rep4_rotated_masked_crop",
        top_k_proposals=4,
        use_structural_verifier=False,
        require_segmentation=True,
    ),
    "C1_best_rep_dinov2_r5": PipelineV2Config(
        name="C1_best_rep_dinov2_r5",
        representation="rep6_minimal_corridor_crop",
        top_k_proposals=4,
        use_structural_verifier=False,
        require_segmentation=True,
    ),
    "C2_full_cane_verified": PipelineV2Config(
        name="C2_full_cane_verified",
        representation="rep6_minimal_corridor_crop",
        top_k_proposals=4,
        use_structural_verifier=True,
        require_segmentation=True,
    ),
}


class CombinedPipelineV2:
    """End-to-End White Cane Recognition Pipeline for uncropped full images."""

    def __init__(
        self,
        localizer: OrientedCaneLocalizer,
        segmenter: CaneShaftSegmenter,
        embedder: DINOv2EmbedderV2,
        rep_bank: RepresentationEmbeddingBank,
        verifier: CaneVsConfuserVerifier,
        cfg: PipelineV2Config,
    ) -> None:
        self.localizer = localizer
        self.segmenter = segmenter
        self.embedder = embedder
        self.rep_bank = rep_bank
        self.verifier = verifier
        self.cfg = cfg

    def evaluate_candidate(
        self,
        image: np.ndarray,
        proposal: OrientedCaneProposal,
        rep_name: str | None = None,
        return_full_mask: bool = False,
    ) -> dict:
        """Segment, render representation, embed with DINOv2, extract features, and run verifier."""
        rep = rep_name or self.cfg.representation
        seg = self.segmenter.segment_proposal(
            image, proposal, return_full_mask=return_full_mask
        )
        canvas = build_representation_canvas(image, proposal, seg, rep)
        emb, emb_ms = self.embedder.embed_canvas(canvas)
        pos_sim, shot_sim = self.rep_bank.score_embedding_against_cane(emb, rep)
        neg_sim = self.verifier.max_neg_similarity(emb)
        fvec, fdict = extract_cane_verification_features(image, proposal, seg, pos_sim, neg_sim)
        dec = self.verifier.decide(
            fvec,
            fdict,
            dinov2_only_mode=(not self.cfg.use_structural_verifier),
        )
        return {
            "proposal": proposal,
            "seg_res": seg,
            "emb": emb,
            "feat_vec": fvec,
            "feat_dict": fdict,
            "decision": dec,
            "pos_sim": pos_sim,
            "shot_sim": shot_sim,
            "neg_sim": neg_sim,
            "seg_ms": seg.seg_ms,
            "emb_ms": emb_ms,
        }

    def predict(
        self,
        image: np.ndarray,
        precomputed_props: list[OrientedCaneProposal] | None = None,
        precomputed_ctx: dict | None = None,
    ) -> dict:
        """Predict KNOWN(object_01) / AMBIGUOUS / UNKNOWN on an uncropped image with timing."""
        t0 = time.perf_counter()
        if precomputed_props is not None and precomputed_ctx is not None:
            props = precomputed_props[: self.cfg.top_k_proposals]
            loc_ms = float(precomputed_ctx.get("localization_ms", 0.0))
        else:
            props, ctx = self.localizer.detect_with_context(image, top_k=self.cfg.top_k_proposals)
            loc_ms = float(ctx["localization_ms"])

        if not props:
            tot_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "UNKNOWN",
                "label": "UNKNOWN",
                "score": 0.0,
                "margin": 0.0,
                "verifier_prob": 0.0,
                "reason": "no_oriented_cane_proposal",
                "selected_proposal": None,
                "proposals": [],
                "localization_ms": round(loc_ms, 2),
                "segmentation_ms": 0.0,
                "embedding_ms": 0.0,
                "verify_ms": 0.0,
                "total_ms": round(loc_ms + tot_ms, 2),
                "num_candidates_evaluated": 0,
            }

        seg_ms_sum = 0.0
        emb_ms_sum = 0.0
        t_ver0 = time.perf_counter()

        best_known = None
        best_ambig = None
        best_any = None
        n_eval = 0

        for p in props[:min(3, self.cfg.top_k_proposals)]:
            cand = self.evaluate_candidate(image, p, return_full_mask=False)
            n_eval += 1
            seg_ms_sum += cand["seg_ms"]
            emb_ms_sum += cand["emb_ms"]
            dec = cand["decision"]
            rank_score = float(dec["score"]) + 0.15 * float(dec["margin"])

            if best_any is None or rank_score > best_any[0]:
                best_any = (rank_score, cand)
            if n_eval >= 2 and best_any is not None and best_any[1]["pos_sim"] >= 0.52:
                break
            if dec["status"] == "KNOWN":
                if best_known is None or rank_score > best_known[0]:
                    best_known = (rank_score, cand)
                if dec["score"] >= self.cfg.early_accept_prob:
                    break
            elif dec["status"] == "AMBIGUOUS":
                if best_ambig is None or rank_score > best_ambig[0]:
                    best_ambig = (rank_score, cand)

        chosen = best_known or best_ambig or best_any
        assert chosen is not None
        winner = chosen[1]
        wdec = winner["decision"]
        wprop: OrientedCaneProposal = winner["proposal"]

        verify_ms = (time.perf_counter() - t_ver0) * 1000.0 - seg_ms_sum - emb_ms_sum
        tot_ms = loc_ms + seg_ms_sum + emb_ms_sum + max(0.0, verify_ms)
        return {
            "status": wdec["status"],
            "label": wdec["label"],
            "score": float(wdec["score"]),
            "margin": float(wdec["margin"]),
            "verifier_prob": float(wdec["verifier_prob"]),
            "pos_sim": float(winner["pos_sim"]),
            "neg_sim": float(winner["neg_sim"]),
            "reason": wdec["reason"],
            "selected_box": list(wprop.bbox_xywh),
            "selected_obb": list(wprop.obb_cxcywh_angle),
            "selected_endpoints": list(wprop.endpoints_xyxy),
            "selected_proposal": wprop,
            "proposals": props,
            "feat_dict": winner["feat_dict"],
            "localization_ms": round(loc_ms, 2),
            "segmentation_ms": round(seg_ms_sum, 2),
            "embedding_ms": round(emb_ms_sum, 2),
            "verify_ms": round(max(0.0, verify_ms), 2),
            "total_ms": round(tot_ms, 2),
            "num_candidates_evaluated": n_eval,
        }
