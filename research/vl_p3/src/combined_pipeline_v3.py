"""Unified VL-P3 End-to-End Inference Pipeline Across Presets A0, A1, B1, B2, C1, and C2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .cane_verifier_v3 import (
    DeBiasedCaneVerifierV3,
    compute_foreground_patch_embedding,
    compute_masked_shaft_crop_embedding,
    extract_verifier_features_v3,
)
from .common_utils import VL_P3_DIR, WORKTREE_ROOT, encode_mask_rle, obb_from_endpoints, sample_centerline_points
from .keypoint_detector import LocalizationPrediction, SupervisedKeypointCenterlineDetector
from .shaft_segmenter import SegmentationPrediction, SupervisedShaftSegmenter
from .spatial_features import DINOv2SpatialBackbone, PatchSemanticHead, SpatialFeatureBundle, YOLOXContextExtractor, extract_spatial_features

PIPELINE_PRESETS = (
    "A0_v2_lsd_hough_baseline",
    "A1_supervised_kp_centerline",
    "B1_supervised_seg_only",
    "B2_kp_guided_supervised_seg",
    "C1_b2_plus_masked_dinov2",
    "C2_full_vl_p3_pipeline",
)


def _import_v2_localizer():
    p2_root = WORKTREE_ROOT.parent / "VisionMate_VL_P2_AG"
    if str(p2_root) not in sys.path:
        sys.path.insert(0, str(p2_root))
    from research.vl_p2.src.localization_research import OrientedCaneLocalizer, YOLOXPersonAndObjectDetector
    return OrientedCaneLocalizer, YOLOXPersonAndObjectDetector


@dataclass
class PipelineV3Config:
    preset: str = "C2_full_vl_p3_pipeline"
    c1_t_known: float = 0.06
    c1_t_uncertain: float = -0.02
    a_t_known: float = 0.56
    a_t_uncertain: float = 0.36
    b1_t_known: float = 0.52
    b1_t_uncertain: float = 0.34
    b2_t_known: float = 0.54
    b2_t_uncertain: float = 0.35
    apply_structural_vetoes: bool = True
    use_masked_crop_eval: bool = True


class CombinedPipelineV3:
    """Unified CPU-first white cane detection, segmentation, and verification pipeline."""

    def __init__(
        self,
        dino_backbone: DINOv2SpatialBackbone,
        yolox_ctx: YOLOXContextExtractor | None = None,
        patch_head: PatchSemanticHead | None = None,
        kp_detector: SupervisedKeypointCenterlineDetector | None = None,
        seg_b1: SupervisedShaftSegmenter | None = None,
        seg_b2: SupervisedShaftSegmenter | None = None,
        verifier: DeBiasedCaneVerifierV3 | None = None,
        config: PipelineV3Config | None = None,
    ) -> None:
        self.dino = dino_backbone
        self.yolox = yolox_ctx
        self.patch_head = patch_head or PatchSemanticHead()
        self.kp_detector = kp_detector
        self.seg_b1 = seg_b1
        self.seg_b2 = seg_b2
        self.verifier = verifier
        self.config = config or PipelineV3Config()
        self._v2_localizer = None

    def _get_v2_localizer(self):
        if self._v2_localizer is None:
            cls, yolox_det_cls = _import_v2_localizer()
            yolox_path = VL_P3_DIR / "models" / "yolox_nano.onnx"
            if not yolox_path.is_file():
                yolox_path = WORKTREE_ROOT.parent / "VisionMate_VL_P2_AG" / "research" / "vl_p2" / "models" / "yolox_nano.onnx"
            v2_yolox = yolox_det_cls(yolox_path) if yolox_path.is_file() else None
            self._v2_localizer = cls(v2_yolox, top_k=6)
        return self._v2_localizer

    def _run_a0_baseline(self, image: np.ndarray, bundle: SpatialFeatureBundle, t0: float) -> dict:
        v2_loc = self._get_v2_localizer()
        h, w = image.shape[:2]
        props, _ = v2_loc.detect_with_context(image)
        if not props:
            ep = [round(w * 0.45, 2), round(h * 0.25, 2), round(w * 0.55, 2), round(h * 0.75, 2)]
            empty_m = np.zeros((h, w), dtype=np.uint8)
            return {
                "preset": "A0_v2_lsd_hough_baseline",
                "decision": "UNKNOWN",
                "pred_label": "UNKNOWN",
                "score": 0.0,
                "endpoints_xyxy": ep,
                "centerline_points_xy": sample_centerline_points(ep, 11),
                "obb_cxcywh_angle": obb_from_endpoints(ep[0], ep[1], ep[2], ep[3], 10.0)[0],
                "mask_rle": encode_mask_rle(empty_m),
                "mask_orig": empty_m,
                "visible_ratio": 0.0,
                "occluded": False,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "veto_reason": "a0_no_proposals",
            }
        top = props[0]
        p1 = (float(top.endpoints_xyxy[0]), float(top.endpoints_xyxy[1]))
        p2 = (float(top.endpoints_xyxy[2]), float(top.endpoints_xyxy[3]))
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
        ep = [round(p1[0], 2), round(p1[1], 2), round(p2[0], 2), round(p2[1], 2)]
        mask_orig = np.zeros((h, w), dtype=np.uint8)
        cv2.line(mask_orig, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), 255, max(3, int(round(0.015 * min(h, w)))))

        # Score top proposal using masked DINOv2 crop bank + VL-P2 structural score
        x, y, bw, bh = top.bbox_xywh
        pad = 20
        cx1, cy1 = max(0, x - pad), max(0, y - pad)
        cx2, cy2 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = image[cy1:cy2, cx1:cx2]
        side = max(crop.shape[0], crop.shape[1]) if crop.size else 64
        sq = np.full((side, side, 3), 127, dtype=np.uint8)
        if crop.size:
            ox, oy = (side - crop.shape[1]) // 2, (side - crop.shape[0]) // 2
            sq[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        emb, _ = self.dino.embed_canvas_cls(sq)
        margin = 0.0
        if self.verifier is not None and self.verifier.pos_crop_bank is not None and self.verifier.neg_crop_bank is not None:
            pos_s = float(np.max(self.verifier.pos_crop_bank @ emb))
            neg_s = float(np.max(self.verifier.neg_crop_bank @ emb))
            margin = pos_s - neg_s
        score = float(np.clip(0.50 * top.confidence + 0.50 * (0.5 + 1.8 * margin), 0.0, 1.0))
        # Apply fair A0 decision threshold (preserving VL-P2 structural conservatism)
        if score >= self.config.a_t_known and top.confidence >= 0.45 and top.white_purity_score >= 0.25:
            dec, lbl = "KNOWN", "object_01"
        elif score >= self.config.a_t_uncertain:
            dec, lbl = "UNCERTAIN", "UNCERTAIN"
        else:
            dec, lbl = "UNKNOWN", "UNKNOWN"

        length_px = float(top.obb_cxcywh_angle[2]) if len(top.obb_cxcywh_angle) >= 3 else float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        vis_ratio = float(np.clip(top.bilateral_profile_score, 0.0, 1.0))
        return {
            "preset": "A0_v2_lsd_hough_baseline",
            "decision": dec,
            "pred_label": lbl,
            "score": round(score, 4),
            "endpoints_xyxy": ep,
            "centerline_points_xy": sample_centerline_points(ep, 11),
            "obb_cxcywh_angle": obb_from_endpoints(ep[0], ep[1], ep[2], ep[3], max(8.0, 0.028 * length_px))[0],
            "mask_rle": encode_mask_rle(mask_orig),
            "mask_orig": mask_orig,
            "visible_ratio": round(vis_ratio, 4),
            "occluded": bool(vis_ratio < 0.72),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "veto_reason": None,
        }

    def predict(
        self,
        image: np.ndarray,
        preset: str | None = None,
        bundle: SpatialFeatureBundle | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        active_preset = preset or self.config.preset
        if active_preset not in PIPELINE_PRESETS:
            raise ValueError(f"Unknown preset: {active_preset}")

        feat_bundle = bundle if bundle is not None else extract_spatial_features(image, self.dino, self.yolox, self.patch_head)
        h, w = image.shape[:2]

        if active_preset == "A0_v2_lsd_hough_baseline":
            return self._run_a0_baseline(image, feat_bundle, t0)

        # Localization stage (needed by A1, B2, C1, C2)
        loc_pred: LocalizationPrediction | None = None
        if active_preset != "B1_supervised_seg_only":
            assert self.kp_detector is not None
            loc_pred = self.kp_detector.predict(feat_bundle)

        # Segmentation stage (B1 standalone vs B2 kp-guided vs A1 OBB tube proxy)
        seg_pred: SegmentationPrediction | None = None
        if active_preset == "B1_supervised_seg_only":
            assert self.seg_b1 is not None
            seg_pred = self.seg_b1.predict(feat_bundle, loc_pred=None)
        elif active_preset in ("B2_kp_guided_supervised_seg", "C1_b2_plus_masked_dinov2", "C2_full_vl_p3_pipeline"):
            assert self.seg_b2 is not None and loc_pred is not None
            seg_pred = self.seg_b2.predict(feat_bundle, loc_pred=loc_pred)

        if active_preset == "A1_supervised_kp_centerline":
            assert loc_pred is not None
            ep = loc_pred.endpoints_xyxy
            mask_orig = np.zeros((h, w), dtype=np.uint8)
            if loc_pred.detected:
                cv2.line(
                    mask_orig,
                    (int(round(ep[0])), int(round(ep[1]))),
                    (int(round(ep[2])), int(round(ep[3]))),
                    255,
                    max(3, int(round(0.016 * min(h, w)))),
                )
            pemb = compute_foreground_patch_embedding(feat_bundle, None, loc_pred)
            pos_b = self.verifier.pos_patch_bank if self.verifier else None
            neg_b = self.verifier.neg_patch_bank if self.verifier else None
            pos_s = float(np.max(pos_b @ pemb)) if pos_b is not None else 0.0
            neg_s = float(np.max(neg_b @ pemb)) if neg_b is not None else 0.0
            score = float(np.clip(0.65 * loc_pred.confidence + 0.35 * (0.5 + 1.8 * (pos_s - neg_s)), 0.0, 1.0))
            dec = "KNOWN" if (loc_pred.detected and score >= self.config.a_t_known) else ("UNCERTAIN" if score >= self.config.a_t_uncertain else "UNKNOWN")
            return {
                "preset": active_preset,
                "decision": dec,
                "pred_label": "object_01" if dec == "KNOWN" else dec,
                "score": round(score, 4),
                "endpoints_xyxy": ep,
                "centerline_points_xy": loc_pred.centerline_points_xy,
                "obb_cxcywh_angle": loc_pred.obb_cxcywh_angle,
                "mask_rle": encode_mask_rle(mask_orig),
                "mask_orig": mask_orig,
                "visible_ratio": loc_pred.visible_ratio,
                "occluded": loc_pred.occluded,
                "orientation_deg": loc_pred.orientation_deg,
                "length_px": loc_pred.length_px,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "veto_reason": None,
                "loc_pred": loc_pred,
            }

        assert seg_pred is not None
        ep_out = loc_pred.endpoints_xyxy if loc_pred is not None else seg_pred.endpoints_xyxy
        cl_out = loc_pred.centerline_points_xy if loc_pred is not None else sample_centerline_points(ep_out, 11)
        obb_out = loc_pred.obb_cxcywh_angle if loc_pred is not None else seg_pred.obb_cxcywh_angle
        vis_ratio = loc_pred.visible_ratio if loc_pred is not None else seg_pred.visible_shaft_ratio
        occ_flag = loc_pred.occluded if loc_pred is not None else bool(seg_pred.visible_shaft_ratio < 0.72)

        # B1 or B2 standalone mask + bank similarity
        if active_preset in ("B1_supervised_seg_only", "B2_kp_guided_supervised_seg"):
            pemb = compute_foreground_patch_embedding(feat_bundle, seg_pred, loc_pred)
            pos_b = self.verifier.pos_patch_bank if self.verifier else None
            neg_b = self.verifier.neg_patch_bank if self.verifier else None
            pos_s = float(np.max(pos_b @ pemb)) if pos_b is not None else 0.0
            neg_s = float(np.max(neg_b @ pemb)) if neg_b is not None else 0.0
            score = float(np.clip(0.55 * seg_pred.confidence + 0.45 * (0.5 + 1.8 * (pos_s - neg_s)), 0.0, 1.0))
            tk = self.config.b1_t_known if active_preset == "B1_supervised_seg_only" else self.config.b2_t_known
            tu = self.config.b1_t_uncertain if active_preset == "B1_supervised_seg_only" else self.config.b2_t_uncertain
            dec = "KNOWN" if (seg_pred.fg_area_ratio > 1e-4 and score >= tk) else ("UNCERTAIN" if score >= tu else "UNKNOWN")
            return {
                "preset": active_preset,
                "decision": dec,
                "pred_label": "object_01" if dec == "KNOWN" else dec,
                "score": round(score, 4),
                "endpoints_xyxy": ep_out,
                "centerline_points_xy": cl_out,
                "obb_cxcywh_angle": obb_out,
                "mask_rle": seg_pred.mask_rle,
                "mask_orig": seg_pred.mask_orig,
                "visible_ratio": vis_ratio,
                "occluded": occ_flag,
                "white_purity_on_mask": seg_pred.white_purity_on_mask,
                "red_tip_presence": seg_pred.red_tip_presence,
                "width_regularity": seg_pred.width_regularity,
                "edge_straightness": seg_pred.edge_straightness,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "veto_reason": None,
                "loc_pred": loc_pred,
                "seg_pred": seg_pred,
            }

        # C1 & C2: masked crop + foreground patch DINOv2 (+ verifier for C2)
        assert loc_pred is not None and self.verifier is not None
        pemb = compute_foreground_patch_embedding(feat_bundle, seg_pred, loc_pred)
        if self.config.use_masked_crop_eval and loc_pred.detected:
            cemb, _ = compute_masked_shaft_crop_embedding(image, feat_bundle, seg_pred, loc_pred, self.dino)
        else:
            cemb = pemb

        vec, fdict = extract_verifier_features_v3(
            feat_bundle,
            loc_pred,
            seg_pred,
            cemb,
            pemb,
            self.verifier.pos_crop_bank,
            self.verifier.neg_crop_bank,
            self.verifier.pos_patch_bank,
            self.verifier.neg_patch_bank,
        )

        if active_preset == "C1_b2_plus_masked_dinov2":
            # Raw masked DINOv2 + patch margin linear probability without full verifier or structural vetoes
            raw_m = 0.55 * fdict["margin_crop"] + 0.45 * fdict["margin_patch"]
            score_c1 = float(np.clip(0.5 + 2.2 * raw_m + 0.20 * seg_pred.confidence, 0.0, 1.0))
            if score_c1 >= self.config.c1_t_known and seg_pred.fg_area_ratio > 1e-4:
                dec_c1 = "KNOWN"
            elif score_c1 >= self.config.c1_t_uncertain:
                dec_c1 = "UNCERTAIN"
            else:
                dec_c1 = "UNKNOWN"
            return {
                "preset": active_preset,
                "decision": dec_c1,
                "pred_label": "object_01" if dec_c1 == "KNOWN" else dec_c1,
                "score": round(score_c1, 4),
                "margin_score": round(raw_m, 4),
                "endpoints_xyxy": ep_out,
                "centerline_points_xy": cl_out,
                "obb_cxcywh_angle": obb_out,
                "mask_rle": seg_pred.mask_rle,
                "mask_orig": seg_pred.mask_orig,
                "visible_ratio": vis_ratio,
                "occluded": occ_flag,
                "white_purity_on_mask": seg_pred.white_purity_on_mask,
                "red_tip_presence": seg_pred.red_tip_presence,
                "width_regularity": seg_pred.width_regularity,
                "edge_straightness": seg_pred.edge_straightness,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "veto_reason": None,
                "loc_pred": loc_pred,
                "seg_pred": seg_pred,
                "verifier_features": fdict,
            }

        # C2_full_vl_p3_pipeline
        vout = self.verifier.score_and_decide(vec, fdict, apply_vetoes=self.config.apply_structural_vetoes)
        return {
            "preset": active_preset,
            "decision": vout["decision"],
            "pred_label": vout["pred_label"],
            "score": vout["score"],
            "raw_prob": vout["raw_prob"],
            "anchor_prob_delta": vout["anchor_prob_delta"],
            "veto_reason": vout["veto_reason"],
            "endpoints_xyxy": ep_out,
            "centerline_points_xy": cl_out,
            "obb_cxcywh_angle": obb_out,
            "mask_rle": seg_pred.mask_rle,
            "mask_orig": seg_pred.mask_orig,
            "visible_ratio": vis_ratio,
            "occluded": occ_flag,
            "orientation_deg": loc_pred.orientation_deg,
            "length_px": loc_pred.length_px,
            "white_purity_on_mask": seg_pred.white_purity_on_mask,
            "red_tip_presence": seg_pred.red_tip_presence,
            "width_regularity": seg_pred.width_regularity,
            "edge_straightness": seg_pred.edge_straightness,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "loc_pred": loc_pred,
            "seg_pred": seg_pred,
            "verifier_features": fdict,
        }
