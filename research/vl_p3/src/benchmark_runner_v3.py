"""Two-Phase VL-P3 Benchmark Runner (Phase A Dev Freeze + Single-Shot Phase B Blind Evaluation).

Strictly enforces zero data leakage:
  - `--phase a`: Builds DINOv2 banks & trains DeBiasedCaneVerifierV3 on `train_v3`, calibrates
                 thresholds on `val_cal_v3` (prioritizing FAR <= 5% while maximizing Recall),
                 and writes locked `manifests/frozen_config.json`.
  - `--phase b`: Verifies `frozen_config.json` checksums and evaluates ONCE on:
                   1) Blind `final_holdout_v3` across A0, A1, B1, B2, C1, C2
                   2) `ambiguous_cane_v3`
                   3) Historical `final_holdout_v2` (N=180)
                   4) Historical `vl_p1c/final_holdout_full` (N=300)
                 Records full-holdout latency (P50/P95), RSS/Private Bytes/CPU%, McNemar tests,
                 per-confuser FAR, handheld confuser FAR, anchor de-biasing, and contact sheets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
from sklearn.metrics import average_precision_score, roc_auc_score

from .cane_verifier_v3 import (
    DeBiasedCaneVerifierV3,
    compute_foreground_patch_embedding,
    compute_masked_shaft_crop_embedding,
    extract_verifier_features_v3,
)
from .common_utils import (
    VL_P3_DIR,
    WORKTREE_ROOT,
    axial_angle_diff_deg,
    compute_endpoint_and_centerline_metrics,
    compute_mask_metrics,
    compute_rotated_obb_iou,
    compute_sha256,
    decode_mask_rle,
    load_image_unicode,
    make_contact_sheet,
    obb_from_endpoints,
    paired_binary_comparison,
    wilson_ci_95,
)
from .combined_pipeline_v3 import CombinedPipelineV3, PIPELINE_PRESETS, PipelineV3Config
from .keypoint_detector import SupervisedKeypointCenterlineDetector
from .shaft_segmenter import SupervisedShaftSegmenter
from .spatial_features import DINOv2SpatialBackbone, PatchSemanticHead, YOLOXContextExtractor, extract_spatial_features


def _draw_pred_overlay(img: np.ndarray, out: dict, gt: dict | None = None, tag: str = "") -> np.ndarray:
    vis = img.copy()
    mask = out.get("mask_orig")
    if mask is not None and mask.shape[:2] == vis.shape[:2]:
        fg = mask > 0
        if np.any(fg):
            tint = np.zeros_like(vis)
            tint[:, :, 1] = 220
            tint[:, :, 2] = 60
            vis[fg] = cv2.addWeighted(vis[fg], 0.55, tint[fg], 0.45, 0)
    if gt is not None and gt.get("endpoints_xyxy"):
        ge = [int(round(v)) for v in gt["endpoints_xyxy"]]
        cv2.line(vis, (ge[0], ge[1]), (ge[2], ge[3]), (255, 220, 40), 2)
    ep = out.get("endpoints_xyxy")
    if ep:
        e = [int(round(v)) for v in ep]
        cv2.line(vis, (e[0], e[1]), (e[2], e[3]), (0, 255, 0), 3)
        cv2.circle(vis, (e[0], e[1]), 6, (255, 0, 0), -1)  # Grip = blue
        cv2.circle(vis, (e[2], e[3]), 6, (0, 0, 255), -1)  # Tip = red
    for pt in out.get("centerline_points_xy", []):
        cv2.circle(vis, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 255, 255), -1)
    lbl = f"{tag} {out['decision']} s={out['score']:.2f}"
    if out.get("veto_reason"):
        lbl += f" [{out['veto_reason'][:12]}]"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (15, 15, 15), -1)
    cv2.putText(vis, lbl[:56], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _compute_cls_metrics(records: list[dict]) -> dict:
    pos = [r for r in records if r["true_pos"]]
    neg = [r for r in records if not r["true_pos"]]
    tp = sum(1 for r in pos if r["decision"] == "KNOWN")
    pos_unc = sum(1 for r in pos if r["decision"] == "UNCERTAIN")
    fn = sum(1 for r in pos if r["decision"] == "UNKNOWN")
    fp = sum(1 for r in neg if r["decision"] == "KNOWN")
    neg_unc = sum(1 for r in neg if r["decision"] == "UNCERTAIN")
    tn = sum(1 for r in neg if r["decision"] == "UNKNOWN")

    rec = tp / len(pos) if pos else 0.0
    far = fp / len(neg) if neg else 0.0
    prec = tp / max(1, tp + fp)
    acc = (tp + tn) / max(1, len(records))
    unc_rate = (pos_unc + neg_unc) / max(1, len(records))
    f1 = (2.0 * prec * rec) / max(1e-8, prec + rec)

    labels = [1 if r["true_pos"] else 0 for r in records]
    scores = [float(r["score"]) for r in records]
    auc_dict = (
        {
            "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
            "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        }
        if pos and neg
        else {"roc_auc": 0.0, "pr_auc": 0.0}
    )

    # Risk-coverage curve across sorted score confidence (evaluating KNOWN vs UNKNOWN at coverage levels)
    risk_cov = []
    if len(records) >= 4:
        sorted_r = sorted(records, key=lambda x: abs(x["score"] - 0.5), reverse=True)
        for cov_frac in (1.0, 0.9, 0.8, 0.7, 0.6):
            k_n = max(1, int(round(cov_frac * len(sorted_r))))
            sub = sorted_r[:k_n]
            errs = sum(1 for x in sub if ((x["score"] >= 0.5) != x["true_pos"]))
            risk_cov.append({"coverage": cov_frac, "n": k_n, "risk_error_rate": round(errs / k_n, 4)})

    return {
        "n_total": len(records),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "tp": tp,
        "pos_uncertain": pos_unc,
        "fn": fn,
        "fp": fp,
        "neg_uncertain": neg_unc,
        "tn": tn,
        "known_recall": round(rec, 4),
        "known_recall_ci95": [round(v, 4) for v in wilson_ci_95(tp, len(pos))],
        "far": round(far, 4),
        "far_ci95": [round(v, 4) for v in wilson_ci_95(fp, len(neg))],
        "known_precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "uncertain_rate": round(unc_rate, 4),
        "roc_auc": auc_dict["roc_auc"],
        "pr_auc": auc_dict["pr_auc"],
        "risk_coverage_curve": risk_cov,
    }


def _compute_geom_metrics(records: list[dict], ann: dict) -> dict:
    loc_rows, seg_rows, occ_rows, clr_rows = [], [], [], []
    for r in records:
        if not r["true_pos"]:
            continue
        gt = ann.get(r["id"])
        if gt is None:
            continue
        h, w = r["height"], r["width"]
        gt_ep = [float(v) for v in gt["endpoints_xyxy"]]
        if gt_ep[1] > gt_ep[3]:
            gt_ep = [gt_ep[2], gt_ep[3], gt_ep[0], gt_ep[1]]
        em = compute_endpoint_and_centerline_metrics(
            r["endpoints_xyxy"], gt_ep, r["centerline_points_xy"], gt["centerline_points_xy"], (h, w)
        )
        gt_len = max(10.0, math.hypot(gt_ep[2] - gt_ep[0], gt_ep[3] - gt_ep[1]))
        gt_obb, _ = obb_from_endpoints(gt_ep[0], gt_ep[1], gt_ep[2], gt_ep[3], thickness=max(8.0, 0.028 * gt_len))
        pr_obb, _ = obb_from_endpoints(
            r["endpoints_xyxy"][0], r["endpoints_xyxy"][1], r["endpoints_xyxy"][2], r["endpoints_xyxy"][3], thickness=max(8.0, 0.028 * gt_len)
        )
        em["rot_obb_iou"] = round(float(compute_rotated_obb_iou(pr_obb, gt_obb)), 4)
        em["angle_error_deg"] = round(float(axial_angle_diff_deg(
            math.degrees(math.atan2(r["endpoints_xyxy"][3] - r["endpoints_xyxy"][1], r["endpoints_xyxy"][2] - r["endpoints_xyxy"][0])),
            math.degrees(math.atan2(gt_ep[3] - gt_ep[1], gt_ep[2] - gt_ep[0])),
        )), 2)
        loc_rows.append(em)
        if gt.get("occluded"):
            occ_rows.append(em)
        else:
            clr_rows.append(em)

        if r.get("mask_orig") is not None:
            gt_m = decode_mask_rle(gt["mask_rle"]) > 0
            mm = compute_mask_metrics(r["mask_orig"], gt_m)
            seg_rows.append({
                "mask_iou": mm["mask_iou"],
                "dice_f1": mm["dice"],
                "pixel_precision": mm["precision"],
                "pixel_recall": mm["recall"],
                "boundary_f1": mm["boundary_f1"],
                "bg_contamination": mm["bg_contamination"],
            })

    def _agg(rows):
        if not rows:
            return {}
        return {f"mean_{k}": round(float(np.mean([x[k] for x in rows])), 4) for k in rows[0].keys()}

    return {
        "localization": _agg(loc_rows),
        "segmentation": _agg(seg_rows),
        "occluded_subset_loc": {"n": len(occ_rows), **_agg(occ_rows)},
        "clear_subset_loc": {"n": len(clr_rows), **_agg(clr_rows)},
    }


def _evaluate_split_with_cache(
    pipe: CombinedPipelineV3,
    items: list[dict],
    preset: str,
    cached_bundles: list | None = None,
    ann: dict | None = None,
    keep_masks: bool = False,
) -> tuple[list[dict], dict]:
    records = []
    latencies = []
    for idx, item in enumerate(items):
        img = load_image_unicode(Path(item["abs_path"]))
        bundle = cached_bundles[idx] if cached_bundles is not None else None
        out = pipe.predict(img, preset=preset, bundle=bundle)
        latencies.append(float(out["latency_ms"]))
        rec = {
            "id": item["id"],
            "label": item.get("label", "unknown"),
            "true_pos": item.get("label") == "object_01",
            "confuser_group": item.get("confuser_group", item.get("subtype", "none")),
            "person_holding_confuser": bool(item.get("person_holding_confuser", False)),
            "width": int(item.get("width", img.shape[1])),
            "height": int(item.get("height", img.shape[0])),
            "decision": out["decision"],
            "pred_label": out["pred_label"],
            "score": float(out["score"]),
            "veto_reason": out.get("veto_reason"),
            "endpoints_xyxy": out["endpoints_xyxy"],
            "centerline_points_xy": out["centerline_points_xy"],
            "visible_ratio": out.get("visible_ratio", 0.0),
            "occluded": out.get("occluded", False),
            "anchor_prob_delta": out.get("anchor_prob_delta", 0.0),
            "latency_ms": float(out["latency_ms"]),
            "mask_orig": out.get("mask_orig") if keep_masks else None,
        }
        records.append(rec)

    cls_m = _compute_cls_metrics(records)
    geom_m = _compute_geom_metrics(records, ann) if ann is not None else {}
    lat_arr = np.asarray(latencies, dtype=np.float32)
    cls_m["latency_ms"] = {
        "mean": round(float(np.mean(lat_arr)), 2),
        "p50": round(float(np.percentile(lat_arr, 50)), 2),
        "p95": round(float(np.percentile(lat_arr, 95)), 2),
        "min": round(float(np.min(lat_arr)), 2),
        "max": round(float(np.max(lat_arr)), 2),
        "n_measured": len(latencies),
    }
    return records, {"classification": cls_m, **geom_m}


def _calibrate_thresholds(scores: np.ndarray, is_pos: np.ndarray, target_far: float = 0.05) -> tuple[float, float]:
    """Select t_known maximizing Known Recall under FAR <= target_far on val_cal_v3."""
    best_tk, best_tu = 0.62, 0.38
    best_obj = (-1.0, -1.0, 0.0)
    for tk in np.linspace(0.40, 0.88, 49):
        fp_rate = float(np.mean(scores[~is_pos] >= tk)) if np.any(~is_pos) else 0.0
        tp_rate = float(np.mean(scores[is_pos] >= tk)) if np.any(is_pos) else 0.0
        within_far = 1.0 if fp_rate <= target_far else 0.0
        # Prioritize FAR <= target_far first, then max recall, then prefer lower FAR
        obj = (within_far, tp_rate if within_far else -fp_rate, -fp_rate)
        if obj > best_obj:
            best_obj = obj
            best_tk = float(tk)
    best_tu = round(max(0.22, best_tk - 0.18), 4)
    return round(best_tk, 4), best_tu


def run_phase_a() -> Path:
    manifests_dir = VL_P3_DIR / "manifests"
    models_dir = VL_P3_DIR / "models"
    ds = json.loads((manifests_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    ann = json.loads((manifests_dir / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]

    dino = DINOv2SpatialBackbone(models_dir / "dinov2_small_quantized.onnx")
    yolox = YOLOXContextExtractor(models_dir / "yolox_nano.onnx")
    patch_head = PatchSemanticHead.from_state_dict(json.loads((models_dir / "patch_semantic_head.json").read_text(encoding="utf-8")))
    kp_det = SupervisedKeypointCenterlineDetector.load(models_dir / "keypoint_detector_v3.joblib")
    seg_b1 = SupervisedShaftSegmenter.load(models_dir / "shaft_segmenter_b1_standalone.joblib")
    seg_b2 = SupervisedShaftSegmenter.load(models_dir / "shaft_segmenter_b2_kp_guided.joblib")

    print("Phase A: Building train_v3 banks & fitting DeBiasedCaneVerifierV3...", flush=True)
    pos_crop_embs, neg_crop_embs = [], []
    pos_patch_embs, neg_patch_embs = [], []
    train_packets = []

    for item in ds["train_v3"]:
        img = load_image_unicode(Path(item["abs_path"]))
        bundle = extract_spatial_features(img, dino, yolox, patch_head)
        loc_pred = kp_det.predict(bundle)
        seg_pred = seg_b2.predict(bundle, loc_pred=loc_pred)

        is_pos_item = item["label"] == "object_01"
        if is_pos_item:
            if item["id"] in ann:
                gt_m = decode_mask_rle(ann[item["id"]]["mask_rle"])
                ep = ann[item["id"]]["endpoints_xyxy"]
                # Use GT region for pure positive cane reference bank
                from dataclasses import replace
                from .keypoint_detector import LocalizationPrediction
                dummy_loc = LocalizationPrediction(
                    detected=True, endpoints_xyxy=ep, centerline_points_xy=ann[item["id"]]["centerline_points_xy"],
                    obb_cxcywh_angle=tuple(ann[item["id"]]["obb_cxcywh_angle"]), bbox_xywh=(0,0,10,10),
                    orientation_deg=0.0, length_px=100.0, length_ratio=0.5, visible_ratio=1.0,
                    occlusion_gap_ratio=0.0, occluded=False, confidence=1.0, kp_confidence=1.0,
                    cl_confidence=1.0, grip_score=1.0, tip_score=1.0, bridge_used=False,
                    grip_hm=loc_pred.grip_hm, tip_hm=loc_pred.tip_hm, cl_hm=loc_pred.cl_hm,
                    cos2_map=loc_pred.cos2_map, sin2_map=loc_pred.sin2_map,
                )
                ys, xs = np.nonzero(gt_m > 0)
                if len(xs) >= 8:
                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    dummy_loc.bbox_xywh = (x1, y1, max(4, x2 - x1 + 1), max(4, y2 - y1 + 1))
                    gt_seg = replace(seg_pred, mask_orig=gt_m)
                    c_emb, _ = compute_masked_shaft_crop_embedding(img, bundle, gt_seg, dummy_loc, dino)
                    p_emb = compute_foreground_patch_embedding(bundle, gt_seg, dummy_loc)
                else:
                    c_emb, _ = compute_masked_shaft_crop_embedding(img, bundle, seg_pred, loc_pred, dino)
                    p_emb = compute_foreground_patch_embedding(bundle, seg_pred, loc_pred)
            else:
                c_emb, _ = compute_masked_shaft_crop_embedding(img, bundle, seg_pred, loc_pred, dino)
                p_emb = compute_foreground_patch_embedding(bundle, seg_pred, loc_pred)
            pos_crop_embs.append(c_emb)
            pos_patch_embs.append(p_emb)
            bank_idx = len(pos_crop_embs) - 1
        else:
            c_emb, _ = compute_masked_shaft_crop_embedding(img, bundle, seg_pred, loc_pred, dino)
            p_emb = compute_foreground_patch_embedding(bundle, seg_pred, loc_pred)
            neg_crop_embs.append(c_emb)
            neg_patch_embs.append(p_emb)
            bank_idx = len(neg_crop_embs) - 1

        # Also embed predicted crop for verifier OOF training consistency
        pred_c, _ = compute_masked_shaft_crop_embedding(img, bundle, seg_pred, loc_pred, dino)
        pred_p = compute_foreground_patch_embedding(bundle, seg_pred, loc_pred)
        train_packets.append((item, bundle, loc_pred, seg_pred, pred_c, pred_p, bank_idx))

    pos_cb = np.stack(pos_crop_embs, axis=0).astype(np.float32)
    neg_cb = np.stack(neg_crop_embs, axis=0).astype(np.float32)
    pos_pb = np.stack(pos_patch_embs, axis=0).astype(np.float32)
    neg_pb = np.stack(neg_patch_embs, axis=0).astype(np.float32)

    verifier = DeBiasedCaneVerifierV3()
    verifier.set_banks(pos_cb, neg_cb, pos_pb, neg_pb)

    X_ver, y_ver = [], []
    # Leave-one-out self-exclusion when querying training banks so self-similarity=1.0 never leaks into verifier OOF features
    for item, bundle, loc_pred, seg_pred, c_emb, p_emb, bank_idx in train_packets:
        is_p = item["label"] == "object_01"
        q_pos_cb = np.delete(pos_cb, bank_idx, axis=0) if (is_p and pos_cb.shape[0] > 1) else pos_cb
        q_pos_pb = np.delete(pos_pb, bank_idx, axis=0) if (is_p and pos_pb.shape[0] > 1) else pos_pb
        q_neg_cb = np.delete(neg_cb, bank_idx, axis=0) if ((not is_p) and neg_cb.shape[0] > 1) else neg_cb
        q_neg_pb = np.delete(neg_pb, bank_idx, axis=0) if ((not is_p) and neg_pb.shape[0] > 1) else neg_pb
        vec, _ = extract_verifier_features_v3(
            bundle, loc_pred, seg_pred, c_emb, p_emb,
            q_pos_cb, q_neg_cb, q_pos_pb, q_neg_pb,
        )
        X_ver.append(vec)
        y_ver.append(1 if is_p else 0)

    oof_info = verifier.fit_oof(np.stack(X_ver, axis=0), np.asarray(y_ver, dtype=int))
    print(f"Verifier OOF training summary: {oof_info}", flush=True)

    # Calibrate preset thresholds strictly on val_cal_v3
    pipe = CombinedPipelineV3(dino, yolox, patch_head, kp_det, seg_b1, seg_b2, verifier, PipelineV3Config())
    val_items = ds["val_cal_v3"]
    val_bundles = [extract_spatial_features(load_image_unicode(Path(it["abs_path"])), dino, yolox, patch_head) for it in val_items]

    val_metrics: dict[str, dict] = {}
    is_pos_val = np.array([it["label"] == "object_01" for it in val_items], dtype=bool)
    for preset in PIPELINE_PRESETS:
        recs, _ = _evaluate_split_with_cache(pipe, val_items, preset, cached_bundles=val_bundles, ann=ann, keep_masks=False)
        scores = np.array([r["score"] for r in recs], dtype=np.float32)
        tk, tu = _calibrate_thresholds(scores, is_pos_val, target_far=0.05)
        if preset == "A1_supervised_kp_centerline":
            pipe.config.a_t_known, pipe.config.a_t_uncertain = tk, tu
        elif preset == "B1_supervised_seg_only":
            pipe.config.b1_t_known, pipe.config.b1_t_uncertain = tk, tu
        elif preset == "B2_kp_guided_supervised_seg":
            pipe.config.b2_t_known, pipe.config.b2_t_uncertain = tk, tu
        elif preset == "C1_b2_plus_masked_dinov2":
            pipe.config.c1_t_known, pipe.config.c1_t_uncertain = tk, tu
        elif preset == "C2_full_vl_p3_pipeline":
            verifier.t_known, verifier.t_uncertain = tk, tu
        _, eval_cal = _evaluate_split_with_cache(pipe, val_items, preset, cached_bundles=val_bundles, ann=ann, keep_masks=False)
        val_metrics[preset] = eval_cal
        print(f"Phase A val_cal_v3 [{preset}]: tk={tk} tu={tu} recall={eval_cal['classification']['known_recall']:.3f} far={eval_cal['classification']['far']:.3f}", flush=True)

    ver_path = models_dir / "cane_verifier_v3.joblib"
    ver_sha = verifier.save(ver_path)

    frozen_cfg = {
        "frozen": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "calibration_split": "val_cal_v3",
        "verifier_oof": oof_info,
        "config": {
            "preset": "C2_full_vl_p3_pipeline",
            "verifier_t_known": verifier.t_known,
            "verifier_t_uncertain": verifier.t_uncertain,
            "c1_t_known": pipe.config.c1_t_known,
            "c1_t_uncertain": pipe.config.c1_t_uncertain,
            "a_t_known": pipe.config.a_t_known,
            "a_t_uncertain": pipe.config.a_t_uncertain,
            "b1_t_known": pipe.config.b1_t_known,
            "b1_t_uncertain": pipe.config.b1_t_uncertain,
            "b2_t_known": pipe.config.b2_t_known,
            "b2_t_uncertain": pipe.config.b2_t_uncertain,
            "apply_structural_vetoes": True,
            "use_masked_crop_eval": True,
        },
        "model_sha256": {
            "dinov2_small_quantized": compute_sha256(models_dir / "dinov2_small_quantized.onnx"),
            "yolox_nano": compute_sha256(models_dir / "yolox_nano.onnx"),
            "patch_semantic_head": compute_sha256(models_dir / "patch_semantic_head.json"),
            "keypoint_detector_v3": compute_sha256(models_dir / "keypoint_detector_v3.joblib"),
            "shaft_segmenter_b1_standalone": compute_sha256(models_dir / "shaft_segmenter_b1_standalone.joblib"),
            "shaft_segmenter_b2_kp_guided": compute_sha256(models_dir / "shaft_segmenter_b2_kp_guided.joblib"),
            "cane_verifier_v3": ver_sha,
        },
        "val_cal_v3_metrics": val_metrics,
    }
    frozen_path = manifests_dir / "frozen_config.json"
    frozen_path.write_text(json.dumps(frozen_cfg, indent=2), encoding="utf-8")
    print(f"Phase A frozen config saved -> {frozen_path}", flush=True)
    return frozen_path


def _load_frozen_pipeline() -> tuple[CombinedPipelineV3, dict]:
    manifests_dir = VL_P3_DIR / "manifests"
    models_dir = VL_P3_DIR / "models"
    frozen_path = manifests_dir / "frozen_config.json"
    if not frozen_path.is_file():
        raise RuntimeError("Missing manifests/frozen_config.json — run --phase a first!")
    fcfg = json.loads(frozen_path.read_text(encoding="utf-8"))
    if not fcfg.get("frozen"):
        raise RuntimeError("frozen_config.json must have frozen=True before Phase B.")
    shas = fcfg["model_sha256"]

    dino = DINOv2SpatialBackbone(models_dir / "dinov2_small_quantized.onnx")
    yolox = YOLOXContextExtractor(models_dir / "yolox_nano.onnx")
    phead_path = models_dir / "patch_semantic_head.json"
    if compute_sha256(phead_path) != shas["patch_semantic_head"]:
        raise RuntimeError("patch_semantic_head SHA-256 mismatch")
    patch_head = PatchSemanticHead.from_state_dict(json.loads(phead_path.read_text(encoding="utf-8")))
    kp_det = SupervisedKeypointCenterlineDetector.load(models_dir / "keypoint_detector_v3.joblib", shas["keypoint_detector_v3"])
    seg_b1 = SupervisedShaftSegmenter.load(models_dir / "shaft_segmenter_b1_standalone.joblib", shas["shaft_segmenter_b1_standalone"])
    seg_b2 = SupervisedShaftSegmenter.load(models_dir / "shaft_segmenter_b2_kp_guided.joblib", shas["shaft_segmenter_b2_kp_guided"])
    verifier = DeBiasedCaneVerifierV3.load(models_dir / "cane_verifier_v3.joblib", shas["cane_verifier_v3"])

    c = fcfg["config"]
    verifier.t_known = float(c["verifier_t_known"])
    verifier.t_uncertain = float(c["verifier_t_uncertain"])
    pcfg = PipelineV3Config(
        preset=c["preset"],
        c1_t_known=float(c["c1_t_known"]),
        c1_t_uncertain=float(c["c1_t_uncertain"]),
        a_t_known=float(c["a_t_known"]),
        a_t_uncertain=float(c["a_t_uncertain"]),
        b1_t_known=float(c["b1_t_known"]),
        b1_t_uncertain=float(c["b1_t_uncertain"]),
        b2_t_known=float(c["b2_t_known"]),
        b2_t_uncertain=float(c["b2_t_uncertain"]),
    )
    return CombinedPipelineV3(dino, yolox, patch_head, kp_det, seg_b1, seg_b2, verifier, pcfg), fcfg


def run_phase_b() -> Path:
    manifests_dir = VL_P3_DIR / "manifests"
    reports_dir = VL_P3_DIR / "reports"
    cs_dir = VL_P3_DIR / "cache" / "contact_sheets"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cs_dir.mkdir(parents=True, exist_ok=True)

    pipe, fcfg = _load_frozen_pipeline()
    ds = json.loads((manifests_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    ann = json.loads((manifests_dir / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)
    rss_start = proc.memory_info().rss / (1024.0 * 1024.0)

    # 1. Blind final_holdout_v3 evaluation (all presets, true single-pass latency on C2 first!)
    fh_items = ds["final_holdout_v3"]
    print(f"Phase B: Running single-shot blind evaluation on final_holdout_v3 (N={len(fh_items)})...", flush=True)

    # Measure full-holdout end-to-end uncached C2 latency & resource telemetry across ALL holdout items
    c2_records, c2_metrics = _evaluate_split_with_cache(pipe, fh_items, "C2_full_vl_p3_pipeline", cached_bundles=None, ann=ann, keep_masks=True)
    rss_peak = proc.memory_info().rss / (1024.0 * 1024.0)
    priv_mb = getattr(proc.memory_info(), "private", proc.memory_info().rss) / (1024.0 * 1024.0)
    cpu_pct = proc.cpu_percent(interval=None)

    # Extract holdout feature bundles once for deterministic A0/A1/B1/B2/C1 comparisons + store standalone uncached timings on A0/A1/B1/B2/C1 too
    fh_bundles = [extract_spatial_features(load_image_unicode(Path(it["abs_path"])), pipe.dino, pipe.yolox, pipe.patch_head) for it in fh_items]
    preset_results: dict[str, dict] = {"C2_full_vl_p3_pipeline": c2_metrics}
    preset_records: dict[str, list[dict]] = {"C2_full_vl_p3_pipeline": c2_records}

    for preset in PIPELINE_PRESETS:
        if preset == "C2_full_vl_p3_pipeline":
            continue
        recs, met = _evaluate_split_with_cache(pipe, fh_items, preset, cached_bundles=fh_bundles, ann=ann, keep_masks=(preset == "B1_supervised_seg_only" or preset == "B2_kp_guided_supervised_seg"))
        preset_results[preset] = met
        preset_records[preset] = recs
        print(f"  Holdout [{preset}]: Recall={met['classification']['known_recall']:.3f}, FAR={met['classification']['far']:.3f}, Acc={met['classification']['accuracy']:.3f}", flush=True)

    # Detailed C2 analyses on final_holdout_v3
    c2_by_confuser: dict[str, dict] = {}
    neg_c2 = [r for r in c2_records if not r["true_pos"]]
    for cg in sorted({r["confuser_group"] for r in neg_c2}):
        sub = [r for r in neg_c2 if r["confuser_group"] == cg]
        fp_n = sum(1 for r in sub if r["decision"] == "KNOWN")
        unc_n = sum(1 for r in sub if r["decision"] == "UNCERTAIN")
        c2_by_confuser[cg] = {"n": len(sub), "fp": fp_n, "uncertain": unc_n, "tn": len(sub) - fp_n - unc_n, "far": round(fp_n / max(1, len(sub)), 4)}

    handheld_sub = [r for r in neg_c2 if r["person_holding_confuser"]]
    hh_fp = sum(1 for r in handheld_sub if r["decision"] == "KNOWN")
    handheld_summary = {
        "n_handheld_confusers": len(handheld_sub),
        "fp": hh_fp,
        "uncertain": sum(1 for r in handheld_sub if r["decision"] == "UNCERTAIN"),
        "far": round(hh_fp / max(1, len(handheld_sub)), 4),
        "far_ci95": [round(v, 4) for v in wilson_ci_95(hh_fp, len(handheld_sub))] if handheld_sub else [0.0, 0.0],
    }

    # McNemar exact tests comparing C2 against A0, A1, B1, B2, C1
    mcnemar_tests = {}
    for other_p in ("A0_v2_lsd_hough_baseline", "A1_supervised_kp_centerline", "B1_supervised_seg_only", "B2_kp_guided_supervised_seg", "C1_b2_plus_masked_dinov2"):
        c2_pos_ok = [1 if r["decision"] == "KNOWN" else 0 for r in c2_records if r["true_pos"]]
        ot_pos_ok = [1 if r["decision"] == "KNOWN" else 0 for r in preset_records[other_p] if r["true_pos"]]
        c2_all_ok = [1 if (r["decision"] == "KNOWN") == r["true_pos"] else 0 for r in c2_records]
        ot_all_ok = [1 if (r["decision"] == "KNOWN") == r["true_pos"] else 0 for r in preset_records[other_p]]
        mcnemar_tests[f"C2_vs_{other_p}"] = {
            "positive_recall": paired_binary_comparison(c2_pos_ok, ot_pos_ok),
            "overall_binary_correct": paired_binary_comparison(c2_all_ok, ot_all_ok),
        }

    # Qualitative contact sheets (TP, FN, FP, Hard TN)
    def _save_qual(name: str, sub_recs: list[dict], limit: int = 16):
        thumbs = []
        for r in sub_recs[:limit]:
            it = next(x for x in fh_items if x["id"] == r["id"])
            img = load_image_unicode(Path(it["abs_path"]))
            thumbs.append((r["id"], _draw_pred_overlay(img, r, ann.get(r["id"]), tag=it["label"][:6])))
        if thumbs:
            make_contact_sheet(thumbs, cs_dir / name, cols=4, cell_w=280, cell_h=210)

    _save_qual("qual_holdout_tp.jpg", [r for r in c2_records if r["true_pos"] and r["decision"] == "KNOWN"])
    _save_qual("qual_holdout_fn.jpg", [r for r in c2_records if r["true_pos"] and r["decision"] != "KNOWN"])
    _save_qual("qual_holdout_fp.jpg", [r for r in c2_records if not r["true_pos"] and r["decision"] == "KNOWN"])
    _save_qual("qual_holdout_tn_handheld.jpg", [r for r in c2_records if not r["true_pos"] and r["decision"] == "UNKNOWN" and r["person_holding_confuser"]])

    # 2. Ambiguous cane set (ambiguous_cane_v3)
    amb_items = ds["ambiguous_cane_v3"]
    amb_bundles = [extract_spatial_features(load_image_unicode(Path(it["abs_path"])), pipe.dino, pipe.yolox, pipe.patch_head) for it in amb_items]
    amb_results = {}
    for preset in PIPELINE_PRESETS:
        recs, _ = _evaluate_split_with_cache(pipe, amb_items, preset, cached_bundles=amb_bundles)
        amb_results[preset] = {
            "n": len(recs),
            "known": sum(1 for r in recs if r["decision"] == "KNOWN"),
            "uncertain": sum(1 for r in recs if r["decision"] == "UNCERTAIN"),
            "unknown": sum(1 for r in recs if r["decision"] == "UNKNOWN"),
            "non_false_accept_rate": round(sum(1 for r in recs if r["decision"] != "KNOWN") / max(1, len(recs)), 4),
        }

    # 3. Historical final_holdout_v2 (N=180)
    p2_dir = WORKTREE_ROOT.parent / "VisionMate_VL_P2_AG" / "research" / "vl_p2"
    v2_hist_summary = {"status": "NOT_FOUND"}
    v2_ds_path = p2_dir / "manifests" / "dataset_manifest.json"
    if v2_ds_path.is_file():
        ds_v2 = json.loads(v2_ds_path.read_text(encoding="utf-8"))
        v2_ann = json.loads((VL_P3_DIR / "manifests" / "vl_p2_repaired_annotations.json").read_text(encoding="utf-8"))["repaired_annotations"]
        fh_v2 = []
        for it in ds_v2["final_holdout_v2"]:
            row = dict(it)
            row["label"] = row.get("label") or row.get("label_id", "unknown")
            ap = Path(row.get("abs_path", ""))
            if not ap.is_file() and row.get("rel_path"):
                ap = p2_dir / row["rel_path"]
            row["abs_path"] = str(ap)
            fh_v2.append(row)
        v2_recs, v2_met = _evaluate_split_with_cache(pipe, fh_v2, "C2_full_vl_p3_pipeline", ann=v2_ann, keep_masks=True)
        v2_hist_summary = {
            "n": len(fh_v2),
            "c2_vl_p3": v2_met,
            "vs_vl_p2_reported_frozen": {"recall": 30 / 51, "far": 8 / 129, "acc": 151 / 180},
            "vs_vl_p2_post_opt_4a10113": {"recall": 10 / 51, "far": 6 / 129, "acc": 133 / 180},
        }

    # 4. Historical VL-P1c N=300 compatibility regression check
    p1c_dir = WORKTREE_ROOT.parent / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c"
    p1c_summary = {"status": "NOT_FOUND"}
    p1c_m_path = p1c_dir / "manifests" / "dataset_manifest.json"
    if p1c_m_path.is_file():
        ds_p1c = json.loads(p1c_m_path.read_text(encoding="utf-8"))
        fhold_300 = ds_p1c.get("final_holdout_full", [])
        # Build multi-class prototype gallery from VL-P1c `train_pos` + `calibration_v1c_full` ONLY for object_02..object_05 compatibility
        cls_embs: dict[str, list[np.ndarray]] = {f"object_{i:02d}": [] for i in range(2, 6)}
        for split_k in ("train_pos", "calibration_v1c_full"):
            for it in ds_p1c.get(split_k, []):
                lbl = it.get("label") or it.get("label_id", "")
                if lbl in cls_embs:
                    rel = it.get("rel_path") or it.get("full_rel_path", "")
                    ap = Path(it.get("abs_path", "")) if it.get("abs_path") and Path(it["abs_path"]).is_file() else (p1c_dir / rel)
                    im = load_image_unicode(ap)
                    h, w = im.shape[:2]
                    side = max(h, w)
                    sq = np.full((side, side, 3), 127, dtype=np.uint8)
                    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = im
                    emb, _ = pipe.dino.embed_canvas_cls(sq)
                    cls_embs[lbl].append(emb)
        cls_protos = {k: (np.mean(v, axis=0) / max(1e-8, float(np.linalg.norm(np.mean(v, axis=0))))) for k, v in cls_embs.items() if v}

        obj01_tp, obj01_n = 0, 0
        other_ok, other_n = 0, 0
        unk_fp, unk_n = 0, 0
        for it in fhold_300:
            rel = it.get("rel_path") or it.get("full_rel_path", "")
            ap = Path(it.get("abs_path", "")) if it.get("abs_path") and Path(it["abs_path"]).is_file() else (p1c_dir / rel)
            im = load_image_unicode(ap)
            lbl = it.get("label") or it.get("label_id", "unknown")
            out_c2 = pipe.predict(im, preset="C2_full_vl_p3_pipeline")
            if lbl == "object_01":
                obj01_n += 1
                if out_c2["decision"] == "KNOWN":
                    obj01_tp += 1
            elif lbl in cls_protos:
                other_n += 1
                if out_c2["decision"] != "KNOWN":
                    h, w = im.shape[:2]
                    side = max(h, w)
                    sq = np.full((side, side, 3), 127, dtype=np.uint8)
                    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = im
                    emb, _ = pipe.dino.embed_canvas_cls(sq)
                    best_k = max(cls_protos, key=lambda k: float(cls_protos[k] @ emb))
                    if best_k == lbl and float(cls_protos[best_k] @ emb) >= 0.45:
                        other_ok += 1
            else:
                unk_n += 1
                if out_c2["decision"] == "KNOWN":
                    unk_fp += 1
        p1c_summary = {
            "n_total": len(fhold_300),
            "object_01_recall": round(obj01_tp / max(1, obj01_n), 4),
            "object_01_tp": obj01_tp,
            "object_01_n": obj01_n,
            "object_02_to_05_preserved_rate": round(other_ok / max(1, other_n), 4),
            "object_02_to_05_ok": other_ok,
            "object_02_to_05_n": other_n,
            "unknown_object_01_far": round(unk_fp / max(1, unk_n), 4),
            "unknown_fp": unk_fp,
            "unknown_n": unk_n,
        }

    final_report = {
        "phase": "PHASE_B_BLIND_EVALUATION",
        "frozen_at": fcfg["frozen_at"],
        "final_holdout_v3_presets": preset_results,
        "c2_per_confuser_group": c2_by_confuser,
        "c2_handheld_confuser_subset": handheld_summary,
        "mcnemar_significance_tests": mcnemar_tests,
        "ambiguous_cane_v3": amb_results,
        "historical_final_holdout_v2_N180": v2_hist_summary,
        "historical_vl_p1c_N300": p1c_summary,
        "system_resources": {
            "rss_start_mb": round(rss_start, 2),
            "rss_peak_mb": round(rss_peak, 2),
            "private_bytes_mb": round(priv_mb, 2),
            "cpu_percent": round(cpu_pct, 2),
            "c2_full_holdout_latency_ms": c2_metrics["classification"]["latency_ms"],
        },
        "c2_per_image_predictions": [
            {k: v for k, v in r.items() if k != "mask_orig"} for r in c2_records
        ],
    }

    out_m = manifests_dir / "benchmark_metrics.json"
    out_r = reports_dir / "benchmark_metrics.json"
    payload = json.dumps(final_report, indent=2, ensure_ascii=False)
    out_m.write_text(payload, encoding="utf-8")
    out_r.write_text(payload, encoding="utf-8")
    print(f"Phase B complete! Saved benchmark metrics -> {out_m}", flush=True)
    return out_m


def main() -> None:
    parser = argparse.ArgumentParser(description="VL-P3 Benchmark Runner")
    parser.add_argument("--phase", required=True, choices=["a", "b"], help="Run phase a (dev freeze) or phase b (blind eval)")
    args = parser.parse_args()
    if args.phase == "a":
        run_phase_a()
    else:
        run_phase_b()


if __name__ == "__main__":
    main()
