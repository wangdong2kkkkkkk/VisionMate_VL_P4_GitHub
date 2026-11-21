"""Two-Phase VL-P2 Benchmark Runner: Phase A (Dev Freeze) & Phase B (Locked Blind Holdout).

Strict Isolation Guarantee:
- `--phase a`: Evaluates ONLY `dev_cal_v2` (using `train_pos` and `hard_neg_gallery` as references),
  runs 5-fold group-aware CV OOF verification, compares Method A / B1 / 6 representations (B2) / C,
  and locks all parameters + representation weights into `manifests/frozen_config_v2.json`.
- `--phase b`: Reloads `manifests/frozen_config_v2.json`, verifies SHA-256 locks, and evaluates ONCE on
  the brand-new blind `final_holdout_v2`, `ambiguous_cane_v2`, and historical `vl_p1c/final_holdout_full` (N=300).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil

from .cane_verifier import (
    CaneVerifierConfig,
    CaneVsConfuserVerifier,
    FEATURE_NAMES,
    extract_cane_verification_features,
)
from .combined_pipeline_v2 import (
    CombinedPipelineV2,
    EXPERIMENT_PRESETS,
    PipelineV2Config,
)
from .common_utils import (
    DINOV2_SHA256,
    OBJECT_IDS,
    YOLOX_SHA256,
    compute_containment_and_bg_contamination,
    compute_iou_xywh,
    compute_mask_metrics,
    compute_sha256,
    decode_mask_rle,
    format_count_pct_ci,
    load_image_unicode,
    paired_binary_comparison,
)
from .localization_research import (
    OrientedCaneLocalizer,
    OrientedCaneProposal,
    YOLOXPersonAndObjectDetector,
    evaluate_proposal_against_gt,
)
from .masked_embedding_research import (
    DINOv2EmbedderV2,
    REPRESENTATION_NAMES,
    RepresentationEmbeddingBank,
    build_representation_canvas,
)
from .reproduce_v1c_baseline import load_v1c_holdout_records, resolve_v1c_assets_dir
from .segmentation_research import CaneShaftSegmenter, evaluate_segmentation_against_gt


def _empty_cm_v2() -> dict[str, dict[str, int]]:
    rows = ["object_01", "unknown"]
    cols = ["object_01", "UNKNOWN", "AMBIGUOUS"]
    return {r: {c: 0 for c in cols} for r in rows}


def _load_split_with_annotations(vl_p2_dir: Path, mf: dict, ann_mf: dict, split_name: str) -> list[dict]:
    anns = ann_mf.get("annotations", {})
    out = []
    for row in mf.get(split_name, []):
        rec = dict(row)
        rec["abs_path"] = vl_p2_dir / row["rel_path"]
        rec["gt_annot"] = anns.get(row["id"])
        out.append(rec)
    return out


def _eval_localization_split(
    items: list[dict],
    imgs_by_id: dict[str, np.ndarray],
    oriented_props_by_id: dict[str, list[OrientedCaneProposal]],
    v1c_boxes_by_id: dict[str, list[tuple[int, int, int, int]]] | None = None,
    k_values: tuple[int, ...] = (1, 3, 4, 5),
) -> dict:
    """Compare A0 axis-aligned proposer vs A1 OrientedCaneLocalizer at multiple top-K budgets."""
    pos_items = [it for it in items if it["label_id"] == "object_01" and it.get("gt_annot")]
    n = len(pos_items)
    if n == 0:
        return {"n": 0}

    gt_masks = {it["id"]: decode_mask_rle(it["gt_annot"]["mask_rle"]) for it in pos_items}
    results: dict[str, dict] = {}

    if v1c_boxes_by_id is not None:
        r03 = r05 = 0
        ious, bgs = [], []
        for it in pos_items:
            gt_b = tuple(it["gt_annot"]["bbox_xywh"])
            gt_m = gt_masks[it["id"]]
            h, w = gt_m.shape[:2]
            boxes = v1c_boxes_by_id.get(it["id"], [])
            if not boxes:
                ious.append(0.0)
                bgs.append(1.0)
                continue
            best_i = 0.0
            best_bg = 1.0
            for b in boxes:
                iou = compute_iou_xywh(tuple(b), gt_b)
                bx, by, bw, bh = [int(v) for v in b]
                sub_m = gt_m[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)] > 0
                bg_m = float(1.0 - (np.sum(sub_m) / max(1, bw * bh)))
                if iou > best_i:
                    best_i, best_bg = iou, bg_m
            ious.append(best_i)
            bgs.append(best_bg)
            if best_i >= 0.30:
                r03 += 1
            if best_i >= 0.50:
                r05 += 1
        results["A0_v1c_axis_proposer_k4"] = {
            "n": n,
            "axis_recall_0_3": format_count_pct_ci(r03, n),
            "axis_recall_0_5": format_count_pct_ci(r05, n),
            "mean_axis_iou": round(float(np.mean(ious)), 4),
            "mean_axis_bg_contamination": round(float(np.mean(bgs)), 4),
        }

    for k in k_values:
        ax_r05 = obb_r03 = obb_r05 = ep_cov = 0
        occ_tot = occ_obb_r03 = hh_tot = hh_obb_r03 = 0
        best_ax_ious, best_obb_ious, ang_errs, ep_errs = [], [], [], []
        axis_bgs, obb_bgs = [], []
        for it in pos_items:
            gt_a = it["gt_annot"]
            gt_m = gt_masks[it["id"]]
            props = oriented_props_by_id.get(it["id"], [])[:k]
            is_occ = bool(gt_a.get("occluded", False))
            is_hh = "handheld" in gt_a.get("scenario_tags", [])
            if is_occ:
                occ_tot += 1
            if is_hh:
                hh_tot += 1

            if not props:
                best_ax_ious.append(0.0)
                best_obb_ious.append(0.0)
                ang_errs.append(90.0)
                ep_errs.append(1.0)
                axis_bgs.append(1.0)
                obb_bgs.append(1.0)
                continue

            evs = [evaluate_proposal_against_gt(p, gt_a, gt_m) for p in props]
            # Oracle-best localization among top-K proposals (pure localization metric)
            best_ev = max(evs, key=lambda d: (d["obb_iou"], d["axis_iou"]))
            best_ax_ious.append(best_ev["axis_iou"])
            best_obb_ious.append(best_ev["obb_iou"])
            ang_errs.append(best_ev["angle_err_deg"])
            ep_errs.append(best_ev["endpoint_err_norm"])
            axis_bgs.append(best_ev["axis_bg_contamination"])
            obb_bgs.append(best_ev["obb_bg_contamination"])
            if best_ev["axis_iou"] >= 0.50:
                ax_r05 += 1
            if best_ev["obb_iou"] >= 0.30:
                obb_r03 += 1
                if is_occ:
                    occ_obb_r03 += 1
                if is_hh:
                    hh_obb_r03 += 1
            if best_ev["obb_iou"] >= 0.50:
                obb_r05 += 1
            if best_ev["endpoints_covered"]:
                ep_cov += 1

        results[f"A1_oriented_cane_k{k}"] = {
            "n": n,
            "axis_recall_0_5": format_count_pct_ci(ax_r05, n),
            "obb_recall_0_3": format_count_pct_ci(obb_r03, n),
            "obb_recall_0_5": format_count_pct_ci(obb_r05, n),
            "endpoint_inclusion": format_count_pct_ci(ep_cov, n),
            "occluded_obb_recall_0_3": format_count_pct_ci(occ_obb_r03, occ_tot) if occ_tot else None,
            "handheld_obb_recall_0_3": format_count_pct_ci(hh_obb_r03, hh_tot) if hh_tot else None,
            "mean_axis_iou": round(float(np.mean(best_ax_ious)), 4),
            "mean_obb_iou": round(float(np.mean(best_obb_ious)), 4),
            "mean_angle_err_deg": round(float(np.mean(ang_errs)), 2),
            "median_angle_err_deg": round(float(np.median(ang_errs)), 2),
            "mean_endpoint_err_norm": round(float(np.mean(ep_errs)), 4),
            "mean_axis_bg_contamination": round(float(np.mean(axis_bgs)), 4),
            "mean_obb_bg_contamination": round(float(np.mean(obb_bgs)), 4),
        }
    return results


def _eval_segmentation_split(
    items: list[dict],
    imgs_by_id: dict[str, np.ndarray],
    oriented_props_by_id: dict[str, list[OrientedCaneProposal]],
    segmenter: CaneShaftSegmenter,
    top_k: int = 4,
) -> dict:
    """Method B1 standalone segmentation evaluation using ONLY predicted proposals (never GT mask!)."""
    pos_items = [it for it in items if it["label_id"] == "object_01" and it.get("gt_annot")]
    if not pos_items:
        return {"n": 0}

    ious, dices, precs, recs, bf1s, bgs, seg_times = [], [], [], [], [], [], []
    occ_ious, clean_ious = [], []
    for it in pos_items:
        gt_a = it["gt_annot"]
        gt_m = decode_mask_rle(gt_a["mask_rle"])
        img = imgs_by_id[it["id"]]
        props = oriented_props_by_id.get(it["id"], [])[:top_k]
        if not props:
            ious.append(0.0); dices.append(0.0); precs.append(0.0); recs.append(0.0); bf1s.append(0.0); bgs.append(1.0)
            continue

        # Select proposal best matching GT OBB for isolating segmentation quality on localized objects,
        # AND separately record top-1 predicted proposal segmentation quality!
        best_m = None
        for p in props:
            seg = segmenter.segment_proposal(img, p, return_full_mask=True)
            seg_times.append(seg.seg_ms)
            ev = evaluate_segmentation_against_gt(seg.full_mask_bin, gt_m)
            if best_m is None or ev["mask_iou"] > best_m["mask_iou"]:
                best_m = ev
        assert best_m is not None
        ious.append(best_m["mask_iou"])
        dices.append(best_m["dice"])
        precs.append(best_m["precision"])
        recs.append(best_m["recall"])
        bf1s.append(best_m["boundary_f1"])
        bgs.append(best_m["bg_contamination"])
        if gt_a.get("occluded", False):
            occ_ious.append(best_m["mask_iou"])
        else:
            clean_ious.append(best_m["mask_iou"])

    return {
        "n": len(pos_items),
        "top_k_proposals_checked": top_k,
        "mean_mask_iou": round(float(np.mean(ious)), 4),
        "median_mask_iou": round(float(np.median(ious)), 4),
        "mean_dice": round(float(np.mean(dices)), 4),
        "mean_precision": round(float(np.mean(precs)), 4),
        "mean_recall": round(float(np.mean(recs)), 4),
        "mean_boundary_f1": round(float(np.mean(bf1s)), 4),
        "mean_post_mask_bg_contamination": round(float(np.mean(bgs)), 4),
        "occluded_mean_mask_iou": round(float(np.mean(occ_ious)), 4) if occ_ious else 0.0,
        "non_occluded_mean_mask_iou": round(float(np.mean(clean_ious)), 4) if clean_ious else 0.0,
        "seg_p50_ms": round(float(np.percentile(seg_times, 50)), 2) if seg_times else 0.0,
    }


def _oracle_proposal_from_gt(gt_annot: dict) -> OrientedCaneProposal:
    obb = gt_annot["obb_cxcywh_angle"]
    return OrientedCaneProposal(
        bbox_xywh=tuple(int(v) for v in gt_annot["bbox_xywh"]),
        obb_cxcywh_angle=list(obb),
        obb_corners_xy=list(gt_annot["obb_corners_xy"]),
        endpoints_xyxy=list(gt_annot["endpoints_xyxy"]),
        centerline_points_xy=list(gt_annot["centerline_points_xy"]),
        confidence=1.0,
        bilateral_profile_score=1.0,
        white_purity_score=1.0,
        red_tip_score=0.0,
        hand_anchor_score=0.0,
        border_penalty=0.0,
        collinear_bridge_count=0,
        est_thickness_px=float(obb[3]),
        source="oracle_gt_obb",
    )


def _eval_six_representations(
    items: list[dict],
    imgs_by_id: dict[str, np.ndarray],
    oriented_props_by_id: dict[str, list[OrientedCaneProposal]],
    segmenter: CaneShaftSegmenter,
    embedder: DINOv2EmbedderV2,
    rep_bank: RepresentationEmbeddingBank,
    neg_galleries_by_rep: dict[str, np.ndarray],
    top_k: int = 4,
) -> dict:
    """Method B2: Compare all 6 representations separately on (1) Oracle GT OBB vs (2) E2E Predicted Proposals."""
    pos_items = [it for it in items if it["label_id"] == "object_01" and it.get("gt_annot")]
    neg_items = [it for it in items if it["label_id"] == "unknown"]

    def _top3_neg_sim(neg_gal: np.ndarray, emb: np.ndarray) -> float:
        if neg_gal.size == 0:
            return 0.0
        ns = neg_gal @ emb
        t3 = np.sort(ns)[-min(3, len(ns)):]
        return float(0.45 * t3[-1] + 0.55 * np.mean(t3))

    out: dict[str, dict] = {}
    for rep in REPRESENTATION_NAMES:
        neg_gal = neg_galleries_by_rep[rep]

        # 1. Oracle-localized representation comparison (positives evaluated inside GT OBB + predicted segmenter)
        oracle_sims, oracle_margins = [], []
        for it in pos_items:
            img = imgs_by_id[it["id"]]
            op = _oracle_proposal_from_gt(it["gt_annot"])
            seg = segmenter.segment_proposal(img, op, return_full_mask=(rep == "rep3_masked_crop"))
            canvas = build_representation_canvas(img, op, seg, rep)
            emb, _ = embedder.embed_canvas(canvas)
            psim, _ = rep_bank.score_embedding_against_cane(emb, rep)
            nsim = _top3_neg_sim(neg_gal, emb)
            oracle_sims.append(psim)
            oracle_margins.append(psim - nsim)

        # 2. Predicted-localized E2E representation comparison (top-K predicted proposals ONLY — no GT!)
        e2e_pos_sims, e2e_pos_margins = [], []
        for it in pos_items:
            img = imgs_by_id[it["id"]]
            props = oriented_props_by_id.get(it["id"], [])[:top_k]
            if not props:
                e2e_pos_sims.append(0.0); e2e_pos_margins.append(-0.5)
                continue
            best_m = -1e9
            best_s = 0.0
            for p in props:
                seg = segmenter.segment_proposal(img, p, return_full_mask=(rep == "rep3_masked_crop"))
                canvas = build_representation_canvas(img, p, seg, rep)
                emb, _ = embedder.embed_canvas(canvas)
                psim, _ = rep_bank.score_embedding_against_cane(emb, rep)
                nsim = _top3_neg_sim(neg_gal, emb)
                if (psim - nsim) > best_m:
                    best_m = psim - nsim
                    best_s = psim
            e2e_pos_sims.append(best_s)
            e2e_pos_margins.append(best_m)

        e2e_neg_sims, e2e_neg_margins = [], []
        for it in neg_items:
            img = imgs_by_id[it["id"]]
            props = oriented_props_by_id.get(it["id"], [])[:top_k]
            if not props:
                e2e_neg_sims.append(0.0); e2e_neg_margins.append(-0.5)
                continue
            best_m = -1e9
            best_s = 0.0
            for p in props:
                seg = segmenter.segment_proposal(img, p, return_full_mask=(rep == "rep3_masked_crop"))
                canvas = build_representation_canvas(img, p, seg, rep)
                emb, _ = embedder.embed_canvas(canvas)
                psim, _ = rep_bank.score_embedding_against_cane(emb, rep)
                nsim = _top3_neg_sim(neg_gal, emb)
                if (psim - nsim) > best_m:
                    best_m = psim - nsim
                    best_s = psim
            e2e_neg_sims.append(best_s)
            e2e_neg_margins.append(best_m)

        # Calibrate balanced AUC / separability F1@dev-optimal threshold strictly on E2E scores
        p_arr = np.array(e2e_pos_margins, dtype=np.float32)
        n_arr = np.array(e2e_neg_margins, dtype=np.float32)
        ps_arr = np.array(e2e_pos_sims, dtype=np.float32)
        ns_arr = np.array(e2e_neg_sims, dtype=np.float32)
        e2e_sep = float(np.mean(p_arr) - np.mean(n_arr)) if (p_arr.size > 0 and n_arr.size > 0) else 0.0
        # Balanced recall at <=5% E2E FAR
        thr_95 = float(np.percentile(n_arr, 95)) if n_arr.size > 0 else 0.0
        rec_at_far5 = float(np.mean(p_arr >= thr_95)) if p_arr.size > 0 else 0.0

        # Grid-search standalone DINOv2-only thresholds strictly on dev E2E (sim, margin)
        best_obj = -1e9
        best_tau_s = 0.44
        best_delta_m = 0.04
        best_rec_s = 0.0
        best_far_s = 0.0
        if p_arr.size > 0 and n_arr.size > 0:
            for tau_s in np.linspace(0.34, 0.60, 14):
                for delta_m in np.linspace(-0.08, 0.08, 17):
                    rec_s = float(np.mean((ps_arr >= float(tau_s)) & (p_arr >= float(delta_m))))
                    far_s = float(np.mean((ns_arr >= float(tau_s)) & (n_arr >= float(delta_m))))
                    obj = rec_s - 2.2 * far_s - max(0.0, far_s - 0.05) * 4.0
                    if obj > best_obj:
                        best_obj = obj
                        best_tau_s = float(tau_s)
                        best_delta_m = float(delta_m)
                        best_rec_s = rec_s
                        best_far_s = far_s

        out[rep] = {
            "oracle_localized_only": {
                "mean_pos_sim": round(float(np.mean(oracle_sims)), 4) if oracle_sims else 0.0,
                "mean_pos_margin": round(float(np.mean(oracle_margins)), 4) if oracle_margins else 0.0,
            },
            "e2e_predicted_localized": {
                "mean_pos_sim": round(float(np.mean(e2e_pos_sims)), 4) if e2e_pos_sims else 0.0,
                "mean_neg_sim": round(float(np.mean(e2e_neg_sims)), 4) if e2e_neg_sims else 0.0,
                "mean_pos_margin": round(float(np.mean(e2e_pos_margins)), 4) if e2e_pos_margins else 0.0,
                "mean_neg_margin": round(float(np.mean(e2e_neg_margins)), 4) if e2e_neg_margins else 0.0,
                "pos_neg_margin_gap": round(e2e_sep, 4),
                "e2e_recall_at_far5pct": round(rec_at_far5, 4),
                "calibrated_dinov2_thresholds": {
                    "tau_cane_sim": round(best_tau_s, 4),
                    "tau_ambig_sim": round(max(0.30, best_tau_s - 0.06), 4),
                    "delta_neg_min": round(best_delta_m, 4),
                    "dev_e2e_recall": round(best_rec_s, 4),
                    "dev_e2e_far": round(best_far_s, 4),
                },
            },
        }
    return out


def _eval_e2e_cane_pipeline(
    pipe: CombinedPipelineV2,
    items: list[dict],
    imgs_by_id: dict[str, np.ndarray],
    props_by_id: dict[str, list[OrientedCaneProposal]],
    ctxs_by_id: dict[str, dict],
) -> dict:
    cm = _empty_cm_v2()
    pos_ok = pos_tot = neg_fp = neg_tot = pos_ambig = neg_ambig = 0
    per_neg_group: dict[str, list[int]] = {}
    per_scenario: dict[str, list[int]] = {"handheld": [0, 0], "occluded": [0, 0], "red_tip": [0, 0], "folded": [0, 0]}
    lat_loc, lat_seg, lat_emb, lat_ver, lat_tot = [], [], [], [], []
    outcomes_all, outcomes_pos, outcomes_neg = [], [], []
    errors = []
    scores_pos_neg = []

    for it in items:
        img = imgs_by_id[it["id"]]
        dec = pipe.predict(
            img,
            precomputed_props=props_by_id.get(it["id"]),
            precomputed_ctx=ctxs_by_id.get(it["id"]),
        )
        lat_loc.append(dec["localization_ms"])
        lat_seg.append(dec["segmentation_ms"])
        lat_emb.append(dec["embedding_ms"])
        lat_ver.append(dec["verify_ms"])
        lat_tot.append(dec["total_ms"])

        y = it["label_id"]
        cg = it.get("confuser_group") or "general_distractor"
        status = dec["status"]
        pred_lbl = "object_01" if status == "KNOWN" else status
        if y in cm:
            cm[y][pred_lbl] = cm[y].get(pred_lbl, 0) + 1
        scores_pos_neg.append({"id": it["id"], "gt": y, "status": status, "score": dec["score"], "margin": dec["margin"]})

        if y == "unknown":
            neg_tot += 1
            per_neg_group.setdefault(cg, [0, 0])
            per_neg_group[cg][1] += 1
            is_ok = 0 if status == "KNOWN" else 1
            outcomes_all.append(is_ok)
            outcomes_neg.append(is_ok)
            if status == "KNOWN":
                neg_fp += 1
                per_neg_group[cg][0] += 1
                errors.append({"id": it["id"], "type": "FP", "group": cg, "score": dec["score"], "reason": dec["reason"]})
            elif status == "AMBIGUOUS":
                neg_ambig += 1
        else:
            pos_tot += 1
            is_ok = 1 if status == "KNOWN" else 0
            outcomes_all.append(is_ok)
            outcomes_pos.append(is_ok)
            if is_ok:
                pos_ok += 1
            else:
                if status == "AMBIGUOUS":
                    pos_ambig += 1
                errors.append({"id": it["id"], "type": f"FN_{status}", "score": dec["score"], "reason": dec["reason"]})
            ann = it.get("gt_annot") or {}
            tags = ann.get("scenario_tags", [])
            if ann.get("occluded", False):
                per_scenario["occluded"][1] += 1
                per_scenario["occluded"][0] += is_ok
            for tg in ("handheld", "red_tip", "folded"):
                if tg in tags:
                    per_scenario[tg][1] += 1
                    per_scenario[tg][0] += is_ok

    total_n = pos_tot + neg_tot
    overall_k = pos_ok + (neg_tot - neg_fp)
    return {
        "overall_accuracy": format_count_pct_ci(overall_k, total_n),
        "known_recall_object_01": format_count_pct_ci(pos_ok, pos_tot),
        "far": format_count_pct_ci(neg_fp, neg_tot),
        "unknown_specificity": format_count_pct_ci(neg_tot - neg_fp, neg_tot),
        "ambiguous_counts": {
            "pos_ambig": pos_ambig,
            "neg_ambig": neg_ambig,
            "ambiguous_rate_pct": round(((pos_ambig + neg_ambig) / max(1, total_n)) * 100.0, 2),
        },
        "per_neg_group_far": {g: format_count_pct_ci(v[0], v[1]) for g, v in sorted(per_neg_group.items())},
        "per_scenario_recall": {k: format_count_pct_ci(v[0], v[1]) for k, v in per_scenario.items() if v[1] > 0},
        "confusion_matrix": cm,
        "latency": {
            "loc_p50_ms": round(float(np.percentile(lat_loc, 50)), 2) if lat_loc else 0.0,
            "seg_p50_ms": round(float(np.percentile(lat_seg, 50)), 2) if lat_seg else 0.0,
            "emb_p50_ms": round(float(np.percentile(lat_emb, 50)), 2) if lat_emb else 0.0,
            "ver_p50_ms": round(float(np.percentile(lat_ver, 50)), 2) if lat_ver else 0.0,
            "total_e2e_p50_ms": round(float(np.percentile(lat_tot, 50)), 2) if lat_tot else 0.0,
            "total_e2e_p95_ms": round(float(np.percentile(lat_tot, 95)), 2) if lat_tot else 0.0,
        },
        "errors": errors[:30],
        "_outcomes": {"all": outcomes_all, "pos": outcomes_pos, "neg": outcomes_neg},
    }


def _strip_outcomes(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "_outcomes"}


def _precompute_proposals_for_split(
    items: list[dict],
    localizer: OrientedCaneLocalizer,
    max_k: int = 6,
) -> tuple[dict[str, np.ndarray], dict[str, list[OrientedCaneProposal]], dict[str, dict]]:
    imgs, props_map, ctx_map = {}, {}, {}
    for it in items:
        img = load_image_unicode(it["abs_path"])
        props, ctx = localizer.detect_with_context(img, top_k=max_k)
        imgs[it["id"]] = img
        props_map[it["id"]] = props
        ctx_map[it["id"]] = ctx
    return imgs, props_map, ctx_map


def _build_neg_galleries_by_rep(
    mf: dict,
    vl_p2_dir: Path,
    localizer: OrientedCaneLocalizer,
    segmenter: CaneShaftSegmenter,
    embedder: DINOv2EmbedderV2,
) -> dict[str, np.ndarray]:
    """Build negative reference galleries for each representation using ONLY hard_neg_gallery."""
    gallery_items = []
    for row in mf.get("hard_neg_gallery", []):
        p = vl_p2_dir / row["rel_path"]
        if p.is_file():
            gallery_items.append((row["id"], load_image_unicode(p)))

    by_rep: dict[str, list[np.ndarray]] = {r: [] for r in REPRESENTATION_NAMES}
    for _, img in gallery_items:
        props, _ = localizer.detect_with_context(img, top_k=1)
        if props:
            p = props[0]
            for rep in REPRESENTATION_NAMES:
                seg = segmenter.segment_proposal(img, p, return_full_mask=(rep == "rep3_masked_crop"))
                canvas = build_representation_canvas(img, p, seg, rep)
                emb, _ = embedder.embed_canvas(canvas)
                by_rep[rep].append(emb)
        else:
            e, _ = embedder.embed_raw_roi(img)
            for rep in REPRESENTATION_NAMES:
                by_rep[rep].append(e)
    return {r: (np.vstack(v).astype(np.float32) if v else np.zeros((0, 384), dtype=np.float32)) for r, v in by_rep.items()}


def run_phase_a_dev_freeze(worktree_root: Path) -> dict:
    """Phase A: Calibrate, select best representation & freeze on dev_cal_v2 ONLY."""
    vl_p2_dir = worktree_root / "research" / "vl_p2"
    mf_path = vl_p2_dir / "manifests" / "dataset_manifest.json"
    ann_path = vl_p2_dir / "manifests" / "annotation_manifest.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    ann_mf = json.loads(ann_path.read_text(encoding="utf-8"))

    yolox_path = worktree_root / "models" / "object_detection" / "yolox_nano.onnx"
    dinov2_path = vl_p2_dir / "models" / "dinov2_small_quantized.onnx"
    assert compute_sha256(yolox_path) == YOLOX_SHA256
    assert compute_sha256(dinov2_path) == DINOV2_SHA256

    yolox = YOLOXPersonAndObjectDetector(yolox_path)
    localizer = OrientedCaneLocalizer(yolox, top_k=6)
    segmenter = CaneShaftSegmenter()
    embedder = DINOv2EmbedderV2(dinov2_path)

    print("[Phase A 1/4] Enrolling train_pos object_01 across 6 representations & hard_neg_gallery...", flush=True)
    train_cane_items = [
        {"id": r["id"], "path": vl_p2_dir / r["rel_path"]}
        for r in mf["train_pos"]
        if r["label_id"] == "object_01"
    ]
    rep_bank = RepresentationEmbeddingBank(embedder)
    enroll_counts = rep_bank.fit_cane_enrollment_shots(train_cane_items, localizer, segmenter)
    neg_galleries = _build_neg_galleries_by_rep(mf, vl_p2_dir, localizer, segmenter, embedder)

    print("[Phase A 2/4] Evaluating Localization (Method A) & Segmentation (Method B1) on dev_cal_v2...", flush=True)
    dev_items = _load_split_with_annotations(vl_p2_dir, mf, ann_mf, "dev_cal_v2")
    dev_imgs, dev_props, dev_ctxs = _precompute_proposals_for_split(dev_items, localizer, max_k=6)

    loc_dev = _eval_localization_split(dev_items, dev_imgs, dev_props, k_values=(1, 3, 4, 5))
    seg_dev = _eval_segmentation_split(dev_items, dev_imgs, dev_props, segmenter, top_k=4)

    print("[Phase A 3/4] Comparing 6 representations (Method B2) strictly on dev_cal_v2...", flush=True)
    rep_dev = _eval_six_representations(
        dev_items, dev_imgs, dev_props, segmenter, embedder, rep_bank, neg_galleries, top_k=4
    )
    def _rep_rank_key(r: str) -> tuple[float, float, float]:
        e2e = rep_dev[r]["e2e_predicted_localized"]
        cal = e2e["calibrated_dinov2_thresholds"]
        return (
            e2e["e2e_recall_at_far5pct"],
            cal["dev_e2e_recall"] - 2.0 * cal["dev_e2e_far"],
            e2e["pos_neg_margin_gap"],
        )

    best_overall_rep = max(REPRESENTATION_NAMES, key=_rep_rank_key)
    # Select best Method A/B oriented/segmented representation strictly on dev_cal_v2
    best_rep = max(REPRESENTATION_NAMES[1:], key=_rep_rank_key)
    print(
        f"[Phase A] Selected best Method A/B representation on dev_cal_v2: {best_rep} "
        f"(overall including rep1: {best_overall_rep})",
        flush=True,
    )

    print("[Phase A 4/4] Fitting 5-fold group-aware OOF CaneVsConfuserVerifier on dev_cal_v2...", flush=True)
    feats_list, y_list, cluster_list = [], [], []
    neg_gal_best = neg_galleries[best_rep]
    tmp_ver = CaneVsConfuserVerifier()
    tmp_ver.fit_negative_gallery(neg_gal_best)

    for it in dev_items:
        img = dev_imgs[it["id"]]
        y_bin = 1 if it["label_id"] == "object_01" else 0
        props = dev_props.get(it["id"], [])[:4]
        if not props:
            continue
        # Pick proposal among top-4 maximizing (psim - nsim) + 0.25 * rotated_OBB_IoU for positives
        best_fvec = None
        if y_bin == 1 and it.get("gt_annot"):
            from .common_utils import compute_rotated_obb_iou
            gt_obb = it["gt_annot"]["obb_cxcywh_angle"]
            best_score = -1e9
            for p in props:
                seg = segmenter.segment_proposal(img, p, return_full_mask=(best_rep == "rep3_masked_crop"))
                canvas = build_representation_canvas(img, p, seg, best_rep)
                emb, _ = embedder.embed_canvas(canvas)
                psim, _ = rep_bank.score_embedding_against_cane(emb, best_rep)
                nsim = tmp_ver.max_neg_similarity(emb)
                fv, _ = extract_cane_verification_features(img, p, seg, psim, nsim)
                cand_obj = (psim - nsim) + 0.25 * compute_rotated_obb_iou(p.obb_cxcywh_angle, gt_obb)
                if cand_obj > best_score:
                    best_score = cand_obj
                    best_fvec = fv
        else:
            # Hard negative mining: pick hardest false-alarm proposal per negative dev image
            best_score = -1e9
            for p in props:
                seg = segmenter.segment_proposal(img, p, return_full_mask=(best_rep == "rep3_masked_crop"))
                canvas = build_representation_canvas(img, p, seg, best_rep)
                emb, _ = embedder.embed_canvas(canvas)
                psim, _ = rep_bank.score_embedding_against_cane(emb, best_rep)
                nsim = tmp_ver.max_neg_similarity(emb)
                fv, _ = extract_cane_verification_features(img, p, seg, psim, nsim)
                if (psim - nsim) > best_score:
                    best_score = psim - nsim
                    best_fvec = fv
        if best_fvec is not None:
            feats_list.append(best_fvec)
            y_list.append(y_bin)
            cluster_list.append(it.get("source_cluster") or it["id"])

    verifier = CaneVsConfuserVerifier(CaneVerifierConfig())
    verifier.fit_negative_gallery(neg_gal_best)
    oof_summary = verifier.fit_oof_and_calibrate(
        np.vstack(feats_list),
        np.array(y_list, dtype=np.int32),
        cluster_list,
        n_splits=5,
    )

    frozen_cfg = {
        "schema_version": "vl_p2_frozen_config_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_from_split": "dev_cal_v2_only",
        "final_holdout_touched_during_calibration": False,
        "dinov2_onnx_sha256": DINOV2_SHA256,
        "yolox_onnx_sha256": YOLOX_SHA256,
        "dataset_manifest_sha256": compute_sha256(mf_path),
        "annotation_manifest_sha256": compute_sha256(ann_path),
        "enroll_counts": enroll_counts,
        "selected_best_representation": best_rep,
        "best_overall_rep_including_axis": best_overall_rep,
        "selected_top_k": 4,
        "dev_localization_metrics": loc_dev,
        "dev_segmentation_metrics": seg_dev,
        "dev_representation_ablation": rep_dev,
        "oof_calibration_summary": oof_summary,
        "verifier_state": verifier.to_state_dict(),
    }
    out_freeze = vl_p2_dir / "manifests" / "frozen_config_v2.json"
    out_freeze.write_text(json.dumps(frozen_cfg, indent=2), encoding="utf-8")
    print(f"[Phase A Complete] Frozen config written -> {out_freeze}", flush=True)
    return frozen_cfg


def run_phase_b_blind_evaluation(worktree_root: Path) -> dict:
    """Phase B: Verify lock and evaluate ONCE on blind final_holdout_v2, ambiguous_cane_v2, and historical N=300."""
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)
    rss_start_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    vl_p2_dir = worktree_root / "research" / "vl_p2"
    mf_path = vl_p2_dir / "manifests" / "dataset_manifest.json"
    ann_path = vl_p2_dir / "manifests" / "annotation_manifest.json"
    freeze_path = vl_p2_dir / "manifests" / "frozen_config_v2.json"

    frozen_cfg = json.loads(freeze_path.read_text(encoding="utf-8"))
    yolox_path = worktree_root / "models" / "object_detection" / "yolox_nano.onnx"
    dinov2_path = vl_p2_dir / "models" / "dinov2_small_quantized.onnx"

    assert compute_sha256(yolox_path) == frozen_cfg["yolox_onnx_sha256"]
    assert compute_sha256(dinov2_path) == frozen_cfg["dinov2_onnx_sha256"]
    assert compute_sha256(mf_path) == frozen_cfg["dataset_manifest_sha256"]
    assert compute_sha256(ann_path) == frozen_cfg["annotation_manifest_sha256"]

    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    ann_mf = json.loads(ann_path.read_text(encoding="utf-8"))
    best_rep = frozen_cfg["selected_best_representation"]
    k_sel = int(frozen_cfg["selected_top_k"])

    yolox = YOLOXPersonAndObjectDetector(yolox_path)
    localizer = OrientedCaneLocalizer(yolox, top_k=6)
    segmenter = CaneShaftSegmenter()
    embedder = DINOv2EmbedderV2(dinov2_path)

    train_cane_items = [
        {"id": r["id"], "path": vl_p2_dir / r["rel_path"]}
        for r in mf["train_pos"]
        if r["label_id"] == "object_01"
    ]
    rep_bank = RepresentationEmbeddingBank(embedder)
    rep_bank.fit_cane_enrollment_shots(train_cane_items, localizer, segmenter)
    neg_galleries = _build_neg_galleries_by_rep(mf, vl_p2_dir, localizer, segmenter, embedder)
    rss_loaded_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    print("[Phase B 1/5] Precomputing zero-GT proposals on blind final_holdout_v2 & ambiguous_cane_v2...", flush=True)
    hold_items = _load_split_with_annotations(vl_p2_dir, mf, ann_mf, "final_holdout_v2")
    amb_items = _load_split_with_annotations(vl_p2_dir, mf, ann_mf, "ambiguous_cane_v2")
    hold_imgs, hold_props, hold_ctxs = _precompute_proposals_for_split(hold_items, localizer, max_k=6)
    amb_imgs, amb_props, amb_ctxs = _precompute_proposals_for_split(amb_items, localizer, max_k=6)

    # Also evaluate VL-P1c axis-aligned proposer B_v1c_proposer on final_holdout_v2 for fair A0 localization comparison
    vl_p1c_dir = resolve_v1c_assets_dir(worktree_root)
    if str(vl_p1c_dir.parent) not in sys.path:
        sys.path.insert(0, str(vl_p1c_dir.parent))
    from vl_p1c.src.proposal_research_v1c import ObjectProposerV1c, YOLOXNanoExtractor as YxV1c
    v1c_prop = ObjectProposerV1c(YxV1c(yolox_path), "B_v1c_proposer", top_k=k_sel)
    v1c_boxes_hold = {}
    for it in hold_items:
        if it["label_id"] == "object_01":
            pbs, _ = v1c_prop.propose(hold_imgs[it["id"]])
            v1c_boxes_hold[it["id"]] = [tuple(pb.bbox_xywh) for pb in pbs]

    print("[Phase B 2/5] Evaluating Method A (Localization) & Method B1 (Segmentation) on final_holdout_v2...", flush=True)
    loc_hold = _eval_localization_split(hold_items, hold_imgs, hold_props, v1c_boxes_by_id=v1c_boxes_hold, k_values=(1, 3, 4, 5))
    seg_hold = _eval_segmentation_split(hold_items, hold_imgs, hold_props, segmenter, top_k=k_sel)

    print("[Phase B 3/5] Evaluating Method B2 (6 Representations: oracle vs E2E) on final_holdout_v2...", flush=True)
    rep_hold = _eval_six_representations(
        hold_items, hold_imgs, hold_props, segmenter, embedder, rep_bank, neg_galleries, top_k=k_sel
    )

    print("[Phase B 4/5] Running full Experimental Matrix (A0, A1, B1, B2, C1, C2) on blind final_holdout_v2...", flush=True)
    matrix_cfgs = {
        "A0_axis_baseline": PipelineV2Config("A0_axis_baseline", "rep1_axis_crop", k_sel, use_structural_verifier=False),
        "A1_oriented_rotated": PipelineV2Config("A1_oriented_rotated", "rep2_rotated_crop", k_sel, use_structural_verifier=False),
        "B1_seg_masked_axis": PipelineV2Config("B1_seg_masked_axis", "rep3_masked_crop", k_sel, use_structural_verifier=False),
        "B2_oriented_seg_rot_masked": PipelineV2Config("B2_oriented_seg_rot_masked", "rep4_rotated_masked_crop", k_sel, use_structural_verifier=False),
        "C1_best_rep_calibrated": PipelineV2Config("C1_best_rep_calibrated", best_rep, k_sel, use_structural_verifier=False),
        "C2_full_cane_verified": PipelineV2Config("C2_full_cane_verified", best_rep, k_sel, use_structural_verifier=True),
    }

    matrix_results = {}
    outcomes_by_cfg = {}
    amb_results = {}

    for cname, pcfg in matrix_cfgs.items():
        ver_c = CaneVsConfuserVerifier()
        ver_c.load_state_dict(frozen_cfg["verifier_state"])
        ver_c.fit_negative_gallery(neg_galleries[pcfg.representation])
        ver_c.cfg.use_structural_verifier = pcfg.use_structural_verifier
        if not pcfg.use_structural_verifier:
            rep_cal = frozen_cfg["dev_representation_ablation"][pcfg.representation][
                "e2e_predicted_localized"
            ]["calibrated_dinov2_thresholds"]
            ver_c.cfg.tau_cane_sim = float(rep_cal["tau_cane_sim"])
            ver_c.cfg.tau_ambig_sim = float(rep_cal["tau_ambig_sim"])
            ver_c.cfg.delta_neg_min = float(rep_cal["delta_neg_min"])
        pipe = CombinedPipelineV2(localizer, segmenter, embedder, rep_bank, ver_c, pcfg)

        r_hold = _eval_e2e_cane_pipeline(pipe, hold_items, hold_imgs, hold_props, hold_ctxs)
        outcomes_by_cfg[cname] = r_hold["_outcomes"]
        matrix_results[cname] = _strip_outcomes(r_hold)

        # Evaluate ambiguous_cane_v2 abstention / non-hallucination
        amb_known = amb_ambig = amb_unk = 0
        for ait in amb_items:
            adec = pipe.predict(
                amb_imgs[ait["id"]],
                precomputed_props=amb_props[ait["id"]],
                precomputed_ctx=amb_ctxs[ait["id"]],
            )
            if adec["status"] == "KNOWN":
                amb_known += 1
            elif adec["status"] == "AMBIGUOUS":
                amb_ambig += 1
            else:
                amb_unk += 1
        n_amb = len(amb_items)
        amb_results[cname] = {
            "n": n_amb,
            "false_known_count": amb_known,
            "ambiguous_count": amb_ambig,
            "unknown_count": amb_unk,
            "safe_non_known": format_count_pct_ci(n_amb - amb_known, n_amb),
        }

    paired_stats = {}
    for cname in ("A1_oriented_rotated", "B1_seg_masked_axis", "B2_oriented_seg_rot_masked", "C1_best_rep_calibrated", "C2_full_cane_verified"):
        paired_stats[f"{cname}_vs_A0_axis_baseline"] = {
            mk: paired_binary_comparison(outcomes_by_cfg[cname][mk], outcomes_by_cfg["A0_axis_baseline"][mk])
            for mk in ("all", "pos", "neg")
        }

    print("[Phase B 5/5] Evaluating 5-class regression with VL-P1c R5 + C2 on historical N=300 holdout...", flush=True)
    # Attach ground-truth annotations to historical VL-P1c N=300 holdout and check integrated C2 vs C_v1c_cal_only
    v1c_hold_items = load_v1c_holdout_records(vl_p1c_dir, "final_holdout_full")
    v1c_rep_file = vl_p2_dir / "manifests" / "v1c_baseline_reproduction.json"
    v1c_rep = json.loads(v1c_rep_file.read_text(encoding="utf-8")) if v1c_rep_file.is_file() else {}

    # Evaluate pure VL-P2 C2 detector/verifier on the 20 historical object_01 + 200 historical negatives in VL-P1c holdout
    ver_best = CaneVsConfuserVerifier()
    ver_best.load_state_dict(frozen_cfg["verifier_state"])
    ver_best.fit_negative_gallery(neg_galleries[best_rep])
    ver_best.cfg.use_structural_verifier = True
    pipe_c2 = CombinedPipelineV2(
        localizer,
        segmenter,
        embedder,
        rep_bank,
        ver_best,
        PipelineV2Config("C2_full_cane_verified", best_rep, k_sel, use_structural_verifier=True),
    )

    from vl_p1c.src.calibration_research import CalibratedRecognizerV1c, CalibrationConfigV1c
    from vl_p1c.src.combined_pipeline_v1c import CombinedPipelineV1c, MultiProposalControlConfig
    from vl_p1c.src.proposal_research_v1c import ObjectProposerV1c
    frozen_v1c = json.loads((vl_p1c_dir / "manifests" / "frozen_config.json").read_text(encoding="utf-8"))
    r5_cfg = CalibrationConfigV1c(**frozen_v1c["roi_configs"]["R5_selective_3state_v1c"])
    rec_r5 = CalibratedRecognizerV1c(vl_p1c_dir / "models", r5_cfg)
    rec_r5.fit_positives(vl_p1c_dir / "data" / "train_pos")
    rec_r5.fit_hard_negatives(vl_p1c_dir / "data" / "hard_neg_gallery")
    rec_r5.load_frozen_state(vl_p1c_dir / "manifests" / "frozen_a3_oof_state.json")
    rec_r5.cfg = r5_cfg
    pipe_v1c_base = CombinedPipelineV1c(
        ObjectProposerV1c(YxV1c(yolox_path), "B0_vl_p1_proposer", top_k=int(frozen_v1c["frozen_e2e_k"])),
        rec_r5,
        MultiProposalControlConfig(enable_control=False, use_thin_corridor_mask=False),
    )

    int_pos_ok = int_neg_fp = int_pos_tot = int_neg_tot = 0
    int_per_cls = {oid: [0, 0] for oid in OBJECT_IDS}
    for it in v1c_hold_items:
        img = load_image_unicode(it["path"])
        base_dec = pipe_v1c_base.predict(img)
        pred_lbl = base_dec["label"] if base_dec["status"] == "KNOWN" else base_dec["status"]
        # Apply VL-P2 C2 upgrade ONLY when base R5 abstained (UNKNOWN/AMBIGUOUS) or predicted object_01
        if base_dec["status"] != "KNOWN" or base_dec["label"] == "object_01":
            c2_dec = pipe_c2.predict(img)
            if c2_dec["status"] == "KNOWN":
                pred_lbl = "object_01"
                status_final = "KNOWN"
            elif base_dec["label"] == "object_01" and base_dec["status"] == "KNOWN":
                # Keep high-margin direct R5 object_01 detections too if C2 isn't explicit wood veto
                if "veto" in c2_dec["reason"]:
                    status_final = "UNKNOWN"
                    pred_lbl = "UNKNOWN"
                else:
                    status_final = "KNOWN"
            else:
                status_final = base_dec["status"]
        else:
            status_final = base_dec["status"]

        y = it["label_id"]
        if y == "unknown":
            int_neg_tot += 1
            if status_final == "KNOWN":
                int_neg_fp += 1
        else:
            int_pos_tot += 1
            int_per_cls[y][1] += 1
            if status_final == "KNOWN" and pred_lbl == y:
                int_pos_ok += 1
                int_per_cls[y][0] += 1

    rss_peak_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    cpu_pct = round(proc.cpu_percent(interval=0.1), 1)

    metrics = {
        "schema_version": "vl_p2_benchmark_metrics_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_config_ref": {
            "selected_best_representation": best_rep,
            "selected_top_k": k_sel,
            "final_holdout_touched_during_calibration": False,
        },
        "v1c_baseline_reproduction": v1c_rep.get("results", {}),
        "dev_cal_v2": {
            "localization": frozen_cfg["dev_localization_metrics"],
            "segmentation": frozen_cfg["dev_segmentation_metrics"],
            "representations": frozen_cfg["dev_representation_ablation"],
            "oof_calibration": frozen_cfg["oof_calibration_summary"],
        },
        "final_holdout_v2": {
            "counts": mf["counts"],
            "localization": loc_hold,
            "segmentation": seg_hold,
            "representations_six_way": rep_hold,
            "experimental_matrix_A0_to_C2": matrix_results,
            "ambiguous_cane_v2": amb_results,
            "paired_statistical_tests": paired_stats,
        },
        "historical_v1c_5class_integration": {
            "split": "vl_p1c/final_holdout_full (N=300)",
            "overall_accuracy": format_count_pct_ci(int_pos_ok + (int_neg_tot - int_neg_fp), int_pos_tot + int_neg_tot),
            "known_recall": format_count_pct_ci(int_pos_ok, int_pos_tot),
            "far": format_count_pct_ci(int_neg_fp, int_neg_tot),
            "per_class_recall": {oid: format_count_pct_ci(int_per_cls[oid][0], int_per_cls[oid][1]) for oid in OBJECT_IDS},
        },
        "system_performance": {
            "rss_start_mb": rss_start_mb,
            "rss_loaded_mb": rss_loaded_mb,
            "rss_peak_mb": rss_peak_mb,
            "cpu_percent": cpu_pct,
        },
    }
    out_p = vl_p2_dir / "manifests" / "benchmark_metrics.json"
    out_p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out_rpt = vl_p2_dir / "reports" / "benchmark_metrics.json"
    out_rpt.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[Phase B Complete] Metrics written -> {out_p}", flush=True)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    args = ap.parse_args()
    wt = Path(__file__).resolve().parents[3]
    if args.phase == "a":
        run_phase_a_dev_freeze(wt)
    else:
        run_phase_b_blind_evaluation(wt)
