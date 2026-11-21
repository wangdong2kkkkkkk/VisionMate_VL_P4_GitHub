"""Step 4: Two-Phase VL-P1c Calibration Freeze & Locked Blind Holdout Benchmark Runner.

Strict Blind-Holdout Isolation Protocol:
- Phase A (`--phase a`): Uses ONLY `train_pos`, `hard_neg_gallery`, `calibration_v1c_roi`, and
  `calibration_v1c_full`. Never opens `final_holdout_roi` or `final_holdout_full`.
  Runs 5-fold group-aware CV, sample-size sweep (N=50..200), selects `delta_floor` and E2E `K`
  strictly from calibration, checks CV gate, and writes `manifests/frozen_config.json`.
- Phase B (`--phase b`): Reloads `manifests/frozen_config.json` and `manifests/frozen_a3_oof_state.json`,
  verifies ONNX + manifest SHA-256 integrity, and evaluates unmodified baselines + VL-P1c across:
  * Historical `vl_p1_test` (N=75)
  * Historical `vl_p1b_holdout` (N=214)
  * Brand-new blind `final_holdout_v1c` (N=300)
  * `ambiguous_boundary_v1c` (N=15)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import psutil

from .calibration_research import (
    CalibratedRecognizerV1c,
    CalibrationConfigV1c,
    compute_risk_coverage_curve,
    evaluate_5fold_cv_and_sample_scaling,
    extract_roi_geometry_features,
    make_group_aware_stratified_folds,
)
from .combined_pipeline_v1c import CombinedPipelineV1c, MultiProposalControlConfig
from .common_utils import (
    OBJECT_IDS,
    LABELS_VI,
    compute_containment_and_bg_contamination,
    compute_iou_xywh,
    compute_sha256,
    format_count_pct_ci,
    load_image_unicode,
    paired_binary_comparison,
)
from .labeling_protocol import HARD_NEGATIVE_GROUPS
from .proposal_research_v1c import ObjectProposerV1c, YOLOXNanoExtractor
from .reproduce_vl_p1b_baseline import _load_vl_p1b_modules
from .thin_object_proposer import evaluate_thin_object_metrics


def _pct_ci(k: int, n: int) -> dict:
    d = format_count_pct_ci(k, n)
    d["formatted"] = f"{d['ratio']} ({d['pct']:.1f}%, 95% CI [{d['ci95_low_pct']:.2f}, {d['ci95_high_pct']:.2f}])"
    return d


def _empty_confusion(rows: list[str], cols: list[str]) -> dict[str, dict[str, int]]:
    return {r: {c: 0 for c in cols} for r in rows}


def _load_vl_p1_test(vl_p1_dir: Path) -> tuple[list[dict], list[dict]]:
    mf = json.loads((vl_p1_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    roi_items, full_items = [], []
    for item in mf["test"]:
        lbl = item["label"]
        cg = item.get("confuser_type") or ("shoe_distractor" if lbl == "unknown" else "positive")
        roi_items.append({
            "id": item["id"],
            "label_id": lbl,
            "confuser_group": cg,
            "path": vl_p1_dir / item["roi_rel_path"],
            "roi_bbox_xywh": tuple(item["roi_bbox_xywh"]) if item.get("roi_bbox_xywh") else None,
        })
        full_items.append({
            "id": item["id"],
            "label_id": lbl,
            "confuser_group": cg,
            "path": vl_p1_dir / item["full_rel_path"],
            "roi_bbox_xywh": tuple(item["roi_bbox_xywh"]) if item.get("roi_bbox_xywh") else None,
        })
    return roi_items, full_items


def _load_split_records(base_dir: Path, mf: dict, split_name: str, use_roi: bool) -> list[dict]:
    items = []
    for row in mf.get(split_name, []):
        rel = row["roi_rel_path"] if use_roi else row["full_rel_path"]
        items.append({
            "id": row["id"],
            "label_id": row["label_id"],
            "confuser_group": row.get("confuser_group") or "positive",
            "source_cluster": row.get("source_cluster") or row.get("source_title") or row["id"],
            "path": base_dir / rel,
            "roi_bbox_xywh": tuple(row["roi_bbox_xywh"]) if row.get("roi_bbox_xywh") else None,
            "is_inherited_cal_v1b": bool(row.get("is_inherited_cal_v1b", False)),
            "has_person": bool(row.get("has_person", False)),
        })
    return items


def _eval_roi_recognizer(rec, items: list[dict], score_cache: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    rows = list(OBJECT_IDS) + ["unknown"]
    cols = list(OBJECT_IDS) + ["UNKNOWN", "AMBIGUOUS"]
    cm = _empty_confusion(rows, cols)

    pos_correct = pos_total = pos_wrong_cls = 0
    neg_fp = neg_total = neg_ambig = pos_ambig = 0
    shoe_slipper_err = shoe_slipper_tot = 0
    per_class = {oid: [0, 0] for oid in OBJECT_IDS}
    per_neg_group: dict[str, list[int]] = {}
    lat_ms = []
    errors = []
    preds_rc: list[tuple[str, dict]] = []
    outcomes_all, outcomes_pos, outcomes_neg = [], [], []

    for idx, item in enumerate(items):
        if score_cache is not None and isinstance(rec, CalibratedRecognizerV1c):
            t0 = time.perf_counter()
            rk = rec.score_features(score_cache[idx][0], score_cache[idx][1])
            dec = rec.decide_with_config(rk)
            lat = (time.perf_counter() - t0) * 1000.0
        else:
            img = load_image_unicode(item["path"])
            dec = rec.predict(img)
            lat = float(dec["feat_ms"]) + float(dec["match_ms"])
        lat_ms.append(lat)

        y = item["label_id"]
        cg = item["confuser_group"]
        status = dec["status"]
        pred_lbl = dec["label"] if status == "KNOWN" else status
        cm[y][pred_lbl] = cm[y].get(pred_lbl, 0) + 1
        preds_rc.append((y, dec))

        if y == "unknown":
            neg_total += 1
            per_neg_group.setdefault(cg, [0, 0])
            per_neg_group[cg][1] += 1
            is_ok = 0 if status == "KNOWN" else 1
            outcomes_all.append(is_ok)
            outcomes_neg.append(is_ok)
            if status == "KNOWN":
                neg_fp += 1
                per_neg_group[cg][0] += 1
                errors.append({
                    "id": item["id"],
                    "type": "FP",
                    "gt": y,
                    "group": cg,
                    "pred": dec["label"],
                    "score": round(float(dec.get("pos_score", dec["score"])), 4),
                    "neg_sim": round(float(dec.get("neg_sim", 0.0)), 4),
                    "verifier_score": round(float(dec.get("verifier_score", 0.0)), 4),
                })
            elif status == "AMBIGUOUS":
                neg_ambig += 1
        else:
            pos_total += 1
            per_class[y][1] += 1
            if y in ("object_04", "object_05"):
                shoe_slipper_tot += 1
            is_ok = 1 if (status == "KNOWN" and pred_lbl == y) else 0
            outcomes_all.append(is_ok)
            outcomes_pos.append(is_ok)
            if is_ok:
                pos_correct += 1
                per_class[y][0] += 1
            else:
                if status == "KNOWN" and pred_lbl != y:
                    pos_wrong_cls += 1
                    if {y, pred_lbl} == {"object_04", "object_05"}:
                        shoe_slipper_err += 1
                    etype = "WRONG_CLASS"
                elif status == "AMBIGUOUS":
                    pos_ambig += 1
                    etype = "FN_AMBIGUOUS"
                else:
                    etype = "FN_UNKNOWN"
                errors.append({
                    "id": item["id"],
                    "type": etype,
                    "gt": y,
                    "pred": pred_lbl,
                    "score": round(float(dec.get("pos_score", dec["score"])), 4),
                    "neg_sim": round(float(dec.get("neg_sim", 0.0)), 4),
                    "top1": dec["candidates"][0]["id"] if dec.get("candidates") else None,
                })

    total_n = pos_total + neg_total
    overall_k = pos_correct + (neg_total - neg_fp)
    return {
        "overall_accuracy": _pct_ci(overall_k, total_n),
        "known_recall": _pct_ci(pos_correct, pos_total),
        "unknown_acc": _pct_ci(neg_total - neg_fp, neg_total),
        "far": _pct_ci(neg_fp, neg_total),
        "wrong_class_pos": f"{pos_wrong_cls}/{pos_total}",
        "shoe_slipper_confusion": f"{shoe_slipper_err}/{shoe_slipper_tot}",
        "ambiguous_counts": {
            "pos_ambig": pos_ambig,
            "neg_ambig": neg_ambig,
            "ambiguous_rate_pct": round(((pos_ambig + neg_ambig) / max(1, total_n)) * 100.0, 2),
        },
        "per_class_recall": {oid: _pct_ci(per_class[oid][0], per_class[oid][1]) for oid in OBJECT_IDS},
        "per_neg_group_far": {g: _pct_ci(v[0], v[1]) for g, v in sorted(per_neg_group.items())},
        "latency_p50_ms": round(float(np.percentile(lat_ms, 50)), 2) if lat_ms else 0.0,
        "latency_p95_ms": round(float(np.percentile(lat_ms, 95)), 2) if lat_ms else 0.0,
        "confusion_matrix": cm,
        "risk_coverage_curve": compute_risk_coverage_curve(preds_rc),
        "errors": errors,
        "_outcomes": {"all": outcomes_all, "pos": outcomes_pos, "neg": outcomes_neg},
    }


def _eval_proposer_standalone(yolox: YOLOXNanoExtractor, mode: str, top_k: int, items: list[dict]) -> dict:
    proposer = ObjectProposerV1c(yolox=yolox, mode=mode, top_k=top_k)
    r03 = r05 = n_pos = 0
    best_ious, best_conts, best_bgs, counts, lats = [], [], [], [], []
    fam_counts: dict[str, int] = {}
    per_cat: dict[str, dict] = {
        oid: {"r03": 0, "r05": 0, "n": 0, "ious": [], "conts": [], "bgs": []}
        for oid in OBJECT_IDS
    }
    cane_metrics_all: list[dict] = []
    cane_handheld_metrics: list[dict] = []
    misses = []

    for item in items:
        if item["label_id"] == "unknown" or not item.get("roi_bbox_xywh"):
            continue
        n_pos += 1
        y = item["label_id"]
        gt = tuple(item["roi_bbox_xywh"])
        img = load_image_unicode(item["path"])
        props, ms = proposer.propose(img)
        lats.append(ms)
        counts.append(len(props))
        for pb in props:
            fam_counts[pb.source_family] = fam_counts.get(pb.source_family, 0) + 1

        best_iou = -1.0
        best_cont = best_bg = 0.0
        best_src = "none"
        for pb in props:
            iou = compute_iou_xywh(tuple(pb.bbox_xywh), gt)
            cont, bg = compute_containment_and_bg_contamination(tuple(pb.bbox_xywh), gt)
            if iou > best_iou:
                best_iou, best_cont, best_bg, best_src = iou, cont, bg, pb.source
        if best_iou < 0:
            best_iou = 0.0

        best_ious.append(best_iou)
        best_conts.append(best_cont)
        best_bgs.append(best_bg)
        per_cat[y]["n"] += 1
        per_cat[y]["ious"].append(best_iou)
        per_cat[y]["conts"].append(best_cont)
        per_cat[y]["bgs"].append(best_bg)
        if best_iou >= 0.3:
            r03 += 1
            per_cat[y]["r03"] += 1
        if best_iou >= 0.5:
            r05 += 1
            per_cat[y]["r05"] += 1
        else:
            misses.append({
                "id": item["id"],
                "label": y,
                "best_iou": round(best_iou, 4),
                "containment": round(best_cont, 4),
                "best_source": best_src,
            })

        if y == "object_01":
            tm = evaluate_thin_object_metrics(props, gt)
            tm["id"] = item["id"]
            cane_metrics_all.append(tm)
            has_person = item.get("has_person", False) or any(pb.coco_class_id == 0 for pb in props)
            if has_person:
                cane_handheld_metrics.append(tm)

    def _summarize_cane(rows: list[dict]) -> dict:
        m = len(rows)
        if m == 0:
            return {"n": 0}
        return {
            "n": m,
            "recall_iou_0_3": _pct_ci(sum(1 for r in rows if r["best_iou"] >= 0.30), m),
            "recall_iou_0_5": _pct_ci(sum(1 for r in rows if r["best_iou"] >= 0.50), m),
            "mean_best_iou": round(float(np.mean([r["best_iou"] for r in rows])), 4),
            "mean_containment": round(float(np.mean([r["best_containment"] for r in rows])), 4),
            "mean_bg_contamination": round(float(np.mean([r["best_bg_contamination"] for r in rows])), 4),
            "endpoint_inclusion_rate": _pct_ci(sum(1 for r in rows if r["endpoint_included"]), m),
            "mean_axis_angle_error_deg": round(float(np.mean([r["axis_angle_error_deg"] for r in rows])), 2),
        }

    tot_fams = max(1, sum(fam_counts.values()))
    return {
        "mode": mode,
        "top_k": top_k,
        "recall_iou_0_3": _pct_ci(r03, n_pos),
        "recall_iou_0_5": _pct_ci(r05, n_pos),
        "mean_best_iou": round(float(np.mean(best_ious)) if best_ious else 0.0, 4),
        "mean_containment": round(float(np.mean(best_conts)) if best_conts else 0.0, 4),
        "mean_bg_contamination": round(float(np.mean(best_bgs)) if best_bgs else 0.0, 4),
        "avg_proposals_per_image": round(float(np.mean(counts)) if counts else 0.0, 2),
        "source_family_share_pct": {k: round((v / tot_fams) * 100.0, 1) for k, v in sorted(fam_counts.items())},
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 2) if lats else 0.0,
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
        "per_category": {
            oid: {
                "recall_iou_0_3": _pct_ci(per_cat[oid]["r03"], per_cat[oid]["n"]),
                "recall_iou_0_5": _pct_ci(per_cat[oid]["r05"], per_cat[oid]["n"]),
                "mean_iou": round(float(np.mean(per_cat[oid]["ious"])) if per_cat[oid]["ious"] else 0.0, 4),
                "mean_containment": round(float(np.mean(per_cat[oid]["conts"])) if per_cat[oid]["conts"] else 0.0, 4),
                "mean_bg_contamination": round(float(np.mean(per_cat[oid]["bgs"])) if per_cat[oid]["bgs"] else 0.0, 4),
            }
            for oid in OBJECT_IDS
        },
        "white_cane_specialized": {
            "all_object_01": _summarize_cane(cane_metrics_all),
            "handheld_person_subset": _summarize_cane(cane_handheld_metrics),
        },
        "misses_iou_0_5": misses[:25],
    }


def _eval_e2e_pipeline(pipe, items: list[dict], is_v1c_pipe: bool = True) -> dict:
    rows = list(OBJECT_IDS) + ["unknown"]
    cols = list(OBJECT_IDS) + ["UNKNOWN", "AMBIGUOUS", "PROPOSAL_MISSED"]
    cm = _empty_confusion(rows, cols)

    pos_correct = pos_total = pos_wrong_cls = pos_prop_missed = 0
    neg_fp = neg_total = neg_ambig = pos_ambig = 0
    tp_iou05 = 0
    shoe_slipper_err = shoe_slipper_tot = 0
    per_class = {oid: [0, 0] for oid in OBJECT_IDS}
    per_neg_group: dict[str, list[int]] = {}
    prop_ms, ver_ms, tot_ms, n_embs = [], [], [], []
    src_counts: dict[str, int] = {}
    fp_sources: dict[str, int] = {}
    errors = []
    preds_rc: list[tuple[str, dict]] = []
    outcomes_all, outcomes_pos, outcomes_neg = [], [], []

    for item in items:
        img = load_image_unicode(item["path"])
        dec = pipe.predict(img)
        prop_ms.append(float(dec["proposal_ms"]))
        ver_ms.append(float(dec["verify_ms"]))
        tot_ms.append(float(dec["total_ms"]))
        n_embs.append(int(dec["num_embeddings"]))
        src = dec.get("selected_source") or "none"
        src_counts[src] = src_counts.get(src, 0) + 1

        y = item["label_id"]
        cg = item["confuser_group"]
        status = dec["status"]
        pred_lbl = dec["label"] if status == "KNOWN" else status
        cm[y][pred_lbl] = cm[y].get(pred_lbl, 0) + 1
        preds_rc.append((y, {
            "status": "KNOWN" if status == "KNOWN" else status,
            "label": dec["label"],
            "pos_score": float(dec.get("pos_score", dec["score"])),
            "margin": float(dec.get("margin", 0.0)),
        }))

        if y == "unknown":
            neg_total += 1
            per_neg_group.setdefault(cg, [0, 0])
            per_neg_group[cg][1] += 1
            is_ok = 0 if status == "KNOWN" else 1
            outcomes_all.append(is_ok)
            outcomes_neg.append(is_ok)
            if status == "KNOWN":
                neg_fp += 1
                per_neg_group[cg][0] += 1
                fp_sources[src] = fp_sources.get(src, 0) + 1
                errors.append({
                    "id": item["id"],
                    "type": "E2E_FP",
                    "gt": y,
                    "group": cg,
                    "pred": dec["label"],
                    "score": round(float(dec["score"]), 4),
                    "selected_box": dec["selected_box"],
                    "selected_source": src,
                })
            elif status == "AMBIGUOUS":
                neg_ambig += 1
        else:
            pos_total += 1
            per_class[y][1] += 1
            if y in ("object_04", "object_05"):
                shoe_slipper_tot += 1
            gt = item.get("roi_bbox_xywh")
            best_prop_iou = (
                max((compute_iou_xywh(tuple(b), gt) for b in dec["proposals"]), default=0.0)
                if gt and dec.get("proposals")
                else 0.0
            )
            if gt and best_prop_iou < 0.30:
                pos_prop_missed += 1

            is_ok = 1 if (status == "KNOWN" and pred_lbl == y) else 0
            outcomes_all.append(is_ok)
            outcomes_pos.append(is_ok)
            if is_ok:
                pos_correct += 1
                per_class[y][0] += 1
                if gt and dec.get("selected_box") and compute_iou_xywh(tuple(dec["selected_box"]), gt) >= 0.50:
                    tp_iou05 += 1
            else:
                if status == "KNOWN" and pred_lbl != y:
                    pos_wrong_cls += 1
                    if {y, pred_lbl} == {"object_04", "object_05"}:
                        shoe_slipper_err += 1
                    etype = "E2E_WRONG_CLASS"
                elif status == "AMBIGUOUS":
                    pos_ambig += 1
                    etype = "E2E_AMBIGUOUS"
                else:
                    etype = "E2E_PROPOSAL_MISSED" if (gt and best_prop_iou < 0.30) else "E2E_VERIFIER_REJECT"
                errors.append({
                    "id": item["id"],
                    "type": etype,
                    "gt": y,
                    "pred": pred_lbl,
                    "score": round(float(dec["score"]), 4),
                    "best_prop_iou": round(best_prop_iou, 4),
                    "selected_source": src,
                })

    total_n = pos_total + neg_total
    overall_k = pos_correct + (neg_total - neg_fp)
    return {
        "overall_accuracy": _pct_ci(overall_k, total_n),
        "known_recall": _pct_ci(pos_correct, pos_total),
        "e2e_strict_iou05": _pct_ci(tp_iou05, pos_total),
        "unknown_acc": _pct_ci(neg_total - neg_fp, neg_total),
        "far": _pct_ci(neg_fp, neg_total),
        "proposal_missed_pos": _pct_ci(pos_prop_missed, pos_total),
        "wrong_class_pos": f"{pos_wrong_cls}/{pos_total}",
        "shoe_slipper_confusion": f"{shoe_slipper_err}/{shoe_slipper_tot}",
        "ambiguous_counts": {
            "pos_ambig": pos_ambig,
            "neg_ambig": neg_ambig,
            "ambiguous_rate_pct": round(((pos_ambig + neg_ambig) / max(1, total_n)) * 100.0, 2),
        },
        "per_class_recall": {oid: _pct_ci(per_class[oid][0], per_class[oid][1]) for oid in OBJECT_IDS},
        "per_neg_group_far": {g: _pct_ci(v[0], v[1]) for g, v in sorted(per_neg_group.items())},
        "proposal_ms_p50": round(float(np.percentile(prop_ms, 50)), 2) if prop_ms else 0.0,
        "verify_ms_p50": round(float(np.percentile(ver_ms, 50)), 2) if ver_ms else 0.0,
        "total_ms_p50": round(float(np.percentile(tot_ms, 50)), 2) if tot_ms else 0.0,
        "total_ms_p95": round(float(np.percentile(tot_ms, 95)), 2) if tot_ms else 0.0,
        "total_ms_mean": round(float(np.mean(tot_ms)) if tot_ms else 0.0, 2),
        "avg_num_embeddings": round(float(np.mean(n_embs)) if n_embs else 0.0, 2),
        "selected_source_distribution": src_counts,
        "fp_source_distribution": fp_sources,
        "confusion_matrix": cm,
        "risk_coverage_curve": compute_risk_coverage_curve(preds_rc),
        "errors": errors,
        "_outcomes": {"all": outcomes_all, "pos": outcomes_pos, "neg": outcomes_neg},
    }


def _eval_ambiguous_split(rec, pipe, items: list[dict]) -> dict:
    roi_status, e2e_status = {}, {}
    rows = []
    for item in items:
        img = load_image_unicode(item["path"])
        d_roi = rec.predict(img)
        d_e2e = pipe.predict(img)
        rs, es = d_roi["status"], d_e2e["status"]
        roi_status[rs] = roi_status.get(rs, 0) + 1
        e2e_status[es] = e2e_status.get(es, 0) + 1
        rows.append({
            "id": item["id"],
            "roi_status": rs,
            "roi_label": d_roi.get("label"),
            "roi_score": round(float(d_roi.get("pos_score", d_roi["score"])), 4),
            "e2e_status": es,
            "e2e_label": d_e2e.get("label"),
            "e2e_score": round(float(d_e2e["score"]), 4),
        })
    return {"n": len(items), "roi_status_counts": roi_status, "e2e_status_counts": e2e_status, "items": rows}


def _benchmark_gallery_scaling(embedder, hard_neg_dir: Path, train_pos_dir: Path) -> list[dict]:
    from .calibration_research import _l2_normalize

    pos_mats = []
    for oid in OBJECT_IDS:
        mats = [embedder.extract(load_image_unicode(p)) for p in sorted((train_pos_dir / oid).glob("*"))]
        pos_mats.append(np.vstack(mats).astype(np.float32))
    pos_proto = np.vstack([_l2_normalize(m.mean(axis=0)) for m in pos_mats])

    all_neg_embs = []
    for grp in HARD_NEGATIVE_GROUPS:
        for p in sorted((hard_neg_dir / grp).glob("*")):
            if p.is_file():
                all_neg_embs.append(embedder.extract(load_image_unicode(p)))
    base = np.vstack(all_neg_embs).astype(np.float32)

    q = pos_mats[0][0]
    results = []
    for target_n in (20, 50, 100, 150):
        idx = np.arange(target_n) % len(base)
        neg_mat = base[idx]
        t0 = time.perf_counter()
        for _ in range(500):
            _ = pos_proto @ q
            _ = neg_mat @ q
        match_ms = ((time.perf_counter() - t0) * 1000.0) / 500.0
        results.append({
            "neg_gallery_n": target_n,
            "match_only_ms": round(match_ms, 5),
            "vector_db_bytes": int(neg_mat.nbytes + pos_proto.nbytes + sum(m.nbytes for m in pos_mats)),
        })
    return results


def _strip_internal_outcomes(obj: dict) -> dict:
    """Copy summary without huge raw binary arrays for cleaner JSON output."""
    return {k: v for k, v in obj.items() if k != "_outcomes"}


def run_phase_a_calibration(vl_p1c_dir: Path, max_far_target: float = 0.05) -> tuple[dict, bool]:
    """Phase A: Strictly calibrate on calibration_v1c ONLY. Never reads final_holdout."""
    models_dir = vl_p1c_dir / "models"
    worktree_root = vl_p1c_dir.parents[1]
    yolox_path = worktree_root / "models" / "object_detection" / "yolox_nano.onnx"
    mf_path = vl_p1c_dir / "manifests" / "dataset_manifest.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))

    cal_roi = _load_split_records(vl_p1c_dir, mf, "calibration_v1c_roi", use_roi=True)
    cal_full = _load_split_records(vl_p1c_dir, mf, "calibration_v1c_full", use_roi=False)

    print(f"[Phase A] Enrolling train_pos + hard_neg_gallery; N_cal={len(cal_roi)} (blind holdout NOT touched)...", flush=True)
    rec_base = CalibratedRecognizerV1c(models_dir, CalibrationConfigV1c())
    enr_pos = rec_base.fit_positives(vl_p1c_dir / "data" / "train_pos")
    enr_neg = rec_base.fit_hard_negatives(vl_p1c_dir / "data" / "hard_neg_gallery")

    cal_embs, cal_geoms = [], []
    for rec in cal_roi:
        img = load_image_unicode(rec["path"])
        cal_embs.append(rec_base.embedder.extract(img))
        cal_geoms.append(extract_roi_geometry_features(img))

    # 1. Group-aware 5-fold CV + delta_floor selection + sample size scaling
    cv_report = evaluate_5fold_cv_and_sample_scaling(
        rec_base, cal_roi, cal_embs, cal_geoms,
        delta_floor_grid=(0.04, 0.06, 0.08),
        max_far_target=max_far_target,
    )
    selected_df = float(cv_report["selected_delta_floor"])

    # Also measure unfloored R2-equivalent instability across same 5 folds (for Problem 1 variance proof)
    folds = make_group_aware_stratified_folds(cal_roi, n_folds=5)
    fold_map = {idx: f_id for f_id, idxs in enumerate(folds) for idx in idxs}
    unfloored_taus = {oid: [] for oid in OBJECT_IDS}
    for f_id in range(len(folds)):
        tr_i = [i for i in range(len(cal_roi)) if fold_map[i] != f_id]
        cfg_u = rec_base.calibrate_mode_on_records(
            [cal_roi[i] for i in tr_i], [cal_embs[i] for i in tr_i], [cal_geoms[i] for i in tr_i],
            mode="R2_per_class_floored", delta_floor=0.45, max_far_target=max_far_target,
        )
        for oid in OBJECT_IDS:
            unfloored_taus[oid].append(cfg_u.tau_per_class[oid])
    cv_report["unfloored_diagnostic_tau_stability"] = {
        oid: {
            "mean": round(float(np.mean(unfloored_taus[oid])), 4),
            "std": round(float(np.std(unfloored_taus[oid])), 4),
            "min": round(float(np.min(unfloored_taus[oid])), 4),
            "max": round(float(np.max(unfloored_taus[oid])), 4),
        }
        for oid in OBJECT_IDS
    }

    # Check CV decision gate before freezing
    r1_r5_modes = [m for m in cv_report["modes_5fold_cv"] if m != "R0_vl_p1_baseline"]
    cv_gate_ok = any(
        cv_report["modes_5fold_cv"][m]["mean_recall_pct"] >= 55.0
        and cv_report["modes_5fold_cv"][m]["mean_far_pct"] <= 15.0
        for m in r1_r5_modes
    )

    # 2. Full calibration fit across R0..R5 on N=200 calibration_v1c_roi
    full_cal_configs: dict[str, dict] = {}
    rec_r5: CalibratedRecognizerV1c | None = None
    roi_cal_resync: dict[str, dict] = {}
    for mode in (
        "R0_vl_p1_baseline",
        "R1_global_conservative",
        "R2_per_class_floored",
        "R3_distributional_quantile",
        "R4_out_of_fold_verifier",
        "R5_selective_3state_v1c",
    ):
        rec_m = CalibratedRecognizerV1c(models_dir, CalibrationConfigV1c(mode=mode), embedder=rec_base.embedder)
        rec_m.pos_embeddings, rec_m.pos_prototypes, rec_m.pos_names, rec_m._pos_geoms = (
            rec_base.pos_embeddings, rec_base.pos_prototypes, rec_base.pos_names, rec_base._pos_geoms
        )
        rec_m.neg_embeddings, rec_m.neg_prototypes, rec_m.neg_names, rec_m._neg_geoms = (
            rec_base.neg_embeddings, rec_base.neg_prototypes, rec_base.neg_names, rec_base._neg_geoms
        )
        rec_m.fit_full_calibration(cal_roi, cal_embs, cal_geoms, delta_floor=selected_df, max_far_target=max_far_target)
        full_cal_configs[mode] = asdict(rec_m.cfg)
        roi_eval = _eval_roi_recognizer(rec_m, cal_roi, score_cache=list(zip(cal_embs, cal_geoms)))
        roi_cal_resync[mode] = {
            "overall_accuracy": roi_eval["overall_accuracy"],
            "known_recall": roi_eval["known_recall"],
            "far": roi_eval["far"],
            "ambiguous_counts": roi_eval["ambiguous_counts"],
        }
        if mode == "R5_selective_3state_v1c":
            rec_r5 = rec_m

    assert rec_r5 is not None

    cal_preview = {
        "final_holdout_touched_during_calibration": False,
        "enrollment": {**enr_pos, **enr_neg, "db_sizes": rec_r5.serialized_db_sizes()},
        "cv_report": cv_report,
        "cv_gate_passed": cv_gate_ok,
        "roi_modes_cal_fit": full_cal_configs,
        "roi_cal_eval": roi_cal_resync,
    }
    (vl_p1c_dir / "reports" / "calibration_cv_preview.json").write_text(
        json.dumps(cal_preview, indent=2), encoding="utf-8"
    )
    if not cv_gate_ok:
        print("[Phase A] CV gate FAILED (all R1..R5 < 55% Recall or > 15% FAR). Stopping before freeze!", flush=True)
        return cal_preview, False

    # 3. Calibrate E2E MultiProposalControlConfig strictly on calibration_v1c_full across K=4, 6, 8
    yolox = YOLOXNanoExtractor(yolox_path)
    cal_full_imgs = [load_image_unicode(r["path"]) for r in cal_full]
    e2e_cal_by_k: dict[str, dict] = {}
    e2e_summary_by_k: dict[str, dict] = {}
    for k in (4, 6, 8):
        prop_k = ObjectProposerV1c(yolox=yolox, mode="B_v1c_proposer", top_k=k)
        mp_cfg = MultiProposalControlConfig(enable_control=True)
        pipe_k = CombinedPipelineV1c(prop_k, rec_r5, mp_cfg)
        cal_res_k = pipe_k.calibrate_e2e_on_full_split(cal_full, cal_full_imgs, max_far_target=max_far_target)
        # Measure calibration latency & full cal metrics using frozen mp_cfg
        t0_k = time.perf_counter()
        eval_k = _eval_e2e_pipeline(pipe_k, cal_full)
        _ = time.perf_counter() - t0_k
        e2e_cal_by_k[str(k)] = asdict(pipe_k.mp_cfg)
        e2e_summary_by_k[str(k)] = {
            **cal_res_k,
            "cal_overall_accuracy": eval_k["overall_accuracy"],
            "cal_known_recall": eval_k["known_recall"],
            "cal_far": eval_k["far"],
            "cal_mean_ms": eval_k["total_ms_mean"],
            "cal_p50_ms": eval_k["total_ms_p50"],
            "cal_p95_ms": eval_k["total_ms_p95"],
        }

    # Select operating K in {4, 6} strictly from calibration_v1c_full (latency/recall/FAR rule)
    k4_s, k6_s = e2e_summary_by_k["4"], e2e_summary_by_k["6"]
    def _k_score(s: dict) -> float:
        r = float(s["cal_known_recall"]["pct"])
        f = float(s["cal_far"]["pct"])
        ms = float(s["cal_mean_ms"])
        return r - 3.0 * max(0.0, f - 5.0) - 1.0 * f - 0.02 * max(0.0, ms - 75.0)

    frozen_e2e_k = 6 if _k_score(k6_s) > _k_score(k4_s) else 4

    # 4. Write immutable frozen artifacts BEFORE Phase B ever sees final_holdout
    frozen_a3_path = vl_p1c_dir / "manifests" / "frozen_a3_oof_state.json"
    rec_r5.export_frozen_state(frozen_a3_path)

    cal_full_payload = {
        **cal_preview,
        "e2e_calibration_by_k": e2e_summary_by_k,
        "selected_frozen_e2e_k": frozen_e2e_k,
    }
    cal_json = json.dumps(cal_full_payload, indent=2)
    (vl_p1c_dir / "reports" / "calibration_metrics.json").write_text(cal_json, encoding="utf-8")
    (vl_p1c_dir / "calibration_metrics.json").write_text(cal_json, encoding="utf-8")

    frozen_cfg = {
        "schema_version": "vl_p1c_frozen_config_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_from_split": "calibration_v1c_roi + calibration_v1c_full (N=200)",
        "final_holdout_touched_during_calibration": False,
        "dinov2_onnx_sha256": compute_sha256(models_dir / "dinov2_small_quantized.onnx"),
        "yolox_onnx_sha256": compute_sha256(yolox_path),
        "dataset_manifest_sha256": compute_sha256(mf_path),
        "frozen_a3_oof_state_sha256": compute_sha256(frozen_a3_path),
        "selected_delta_floor": selected_df,
        "frozen_e2e_k": frozen_e2e_k,
        "roi_configs": full_cal_configs,
        "e2e_configs_by_k": e2e_cal_by_k,
        "db_sizes": rec_r5.serialized_db_sizes(),
    }
    f_json = json.dumps(frozen_cfg, indent=2)
    (vl_p1c_dir / "manifests" / "frozen_config.json").write_text(f_json, encoding="utf-8")
    (vl_p1c_dir / "frozen_config.json").write_text(f_json, encoding="utf-8")
    print(f"[Phase A Done] selected_delta_floor={selected_df}, frozen_e2e_k={frozen_e2e_k}, frozen_config written.", flush=True)
    return cal_full_payload, True


def _format_confusion_md(title: str, cm: dict[str, dict[str, int]], cols: list[str]) -> str:
    lines = [f"### {title}", "", "| GT \\\\ Pred | " + " | ".join(cols) + " |", "|---|" + "|".join(["---:"] * len(cols)) + "|"]
    for row_k in list(OBJECT_IDS) + ["unknown"]:
        vals = " | ".join(str(cm.get(row_k, {}).get(c, 0)) for c in cols)
        lines.append(f"| `{row_k}` | {vals} |")
    lines.append("")
    return "\n".join(lines)


def run_phase_b_locked_evaluation(vl_p1c_dir: Path) -> dict:
    """Phase B: Reload frozen_config.json & evaluate locked models on historical + final_holdout_v1c."""
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)
    rss_start_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    worktree_root = vl_p1c_dir.parents[1]
    vl_p1_dir = worktree_root / "research" / "vl_p1"
    vl_p1b_dir = worktree_root / "research" / "vl_p1b"
    models_dir = vl_p1c_dir / "models"
    yolox_path = worktree_root / "models" / "object_detection" / "yolox_nano.onnx"

    frozen_cfg = json.loads((vl_p1c_dir / "manifests" / "frozen_config.json").read_text(encoding="utf-8"))
    mf_path = vl_p1c_dir / "manifests" / "dataset_manifest.json"
    mf_v1c = json.loads(mf_path.read_text(encoding="utf-8"))
    mf_v1b = json.loads((vl_p1b_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))

    # Verify checksums match Phase A freeze
    assert compute_sha256(models_dir / "dinov2_small_quantized.onnx") == frozen_cfg["dinov2_onnx_sha256"]
    assert compute_sha256(yolox_path) == frozen_cfg["yolox_onnx_sha256"]
    assert compute_sha256(mf_path) == frozen_cfg["dataset_manifest_sha256"]
    a3_state_path = vl_p1c_dir / "manifests" / "frozen_a3_oof_state.json"
    assert compute_sha256(a3_state_path) == frozen_cfg["frozen_a3_oof_state_sha256"]

    # Load evaluation splits
    vl_p1_roi, vl_p1_full = _load_vl_p1_test(vl_p1_dir)
    v1b_hold_roi = _load_split_records(vl_p1b_dir, mf_v1b, "holdout_roi", use_roi=True)
    v1b_hold_full = _load_split_records(vl_p1b_dir, mf_v1b, "holdout_full", use_roi=False)
    v1c_hold_roi = _load_split_records(vl_p1c_dir, mf_v1c, "final_holdout_roi", use_roi=True)
    v1c_hold_full = _load_split_records(vl_p1c_dir, mf_v1c, "final_holdout_full", use_roi=False)
    v1c_ambig = _load_split_records(vl_p1c_dir, mf_v1c, "ambiguous_boundary_v1c", use_roi=False)

    print("[Phase B 1/5] Loading locked recognizers + unmodified VL-P1/VL-P1b baselines...", flush=True)
    yolox = YOLOXNanoExtractor(yolox_path)
    base_v1c = CalibratedRecognizerV1c(models_dir, CalibrationConfigV1c())
    base_v1c.fit_positives(vl_p1c_dir / "data" / "train_pos")
    base_v1c.fit_hard_negatives(vl_p1c_dir / "data" / "hard_neg_gallery")

    # Build locked R0..R5 using frozen configs + frozen OOF verifier weights
    roi_models: dict[str, CalibratedRecognizerV1c] = {}
    for mode, cfg_dict in frozen_cfg["roi_configs"].items():
        rec_m = CalibratedRecognizerV1c(models_dir, CalibrationConfigV1c(**cfg_dict), embedder=base_v1c.embedder)
        rec_m.pos_embeddings, rec_m.pos_prototypes, rec_m.pos_names, rec_m._pos_geoms = (
            base_v1c.pos_embeddings, base_v1c.pos_prototypes, base_v1c.pos_names, base_v1c._pos_geoms
        )
        rec_m.neg_embeddings, rec_m.neg_prototypes, rec_m.neg_names, rec_m._neg_geoms = (
            base_v1c.neg_embeddings, base_v1c.neg_prototypes, base_v1c.neg_names, base_v1c._neg_geoms
        )
        # Load OOF verifier weights for R4/R5 directly from frozen state (preserving exact mode's thresholds!)
        if mode in ("R4_out_of_fold_verifier", "R5_selective_3state_v1c"):
            rec_m.load_frozen_state(a3_state_path)
            rec_m.cfg = CalibrationConfigV1c(**cfg_dict)
        roi_models[mode] = rec_m

    # Unmodified historical VL-P1 & VL-P1b recognizers/proposers
    HardNegConfig, HardNegativeRecognizer, YxV1b, PropV1b, PipeV1b, CfgV1b = _load_vl_p1b_modules()
    v1b_fr = json.loads((vl_p1b_dir / "manifests" / "frozen_thresholds.json").read_text(encoding="utf-8"))
    v1b_cal = v1b_fr["calibration_summaries"]

    def _make_v1b_rec(mkey: str):
        rec = HardNegativeRecognizer(vl_p1b_dir / "models", HardNegConfig(**dict(v1b_cal[mkey]["calibration_params"])))
        rec.fit_positives(vl_p1b_dir / "data" / "train_pos")
        rec.fit_hard_negatives(vl_p1b_dir / "data" / "hard_neg_gallery")
        if mkey == "A3_full_verifier":
            rec.load_frozen_state(vl_p1b_dir / "manifests" / "frozen_a3_state.json")
        return rec

    rec_v1b_a0 = _make_v1b_rec("A0_baseline_hybrid")
    rec_v1b_a1 = _make_v1b_rec("A1_neg_prototypes")
    rec_v1b_a3 = _make_v1b_rec("A3_full_verifier")
    rss_loaded_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    print("[Phase B 2/5] Evaluating ROI models (R0..R5 + VL-P1b A1/A3) on vl_p1_test, vl_p1b_holdout, final_holdout_roi...", flush=True)
    roi_results = {}
    roi_outcomes_fhold: dict[str, dict] = {}
    all_roi_eval_models = {
        **roi_models,
        "A1_vl_p1b_ref": rec_v1b_a1,
        "A3_vl_p1b_ref": rec_v1b_a3,
    }
    for mname, rec in all_roi_eval_models.items():
        r_p1 = _eval_roi_recognizer(rec, vl_p1_roi)
        r_1b = _eval_roi_recognizer(rec, v1b_hold_roi)
        r_1c = _eval_roi_recognizer(rec, v1c_hold_roi)
        roi_outcomes_fhold[mname] = r_1c["_outcomes"]
        roi_results[mname] = {
            "vl_p1_test": _strip_internal_outcomes(r_p1),
            "vl_p1b_holdout": _strip_internal_outcomes(r_1b),
            "final_holdout_roi": _strip_internal_outcomes(r_1c),
        }

    print("[Phase B 3/5] Evaluating 3x3 Fair Proposal Matrix + YOLOX baselines + white-cane metrics...", flush=True)
    prop_results = {}
    for bmode in ("yolox_strict", "yolox_low"):
        prop_results[f"{bmode}__k4"] = {
            "vl_p1b_holdout": _eval_proposer_standalone(yolox, bmode, 4, v1b_hold_full),
            "final_holdout_full": _eval_proposer_standalone(yolox, bmode, 4, v1c_hold_full),
        }
    for bmode in ("B0_vl_p1_proposer", "B_full_v1b", "B_v1c_proposer"):
        for k in (4, 6, 8):
            prop_results[f"{bmode}__k{k}"] = {
                "vl_p1b_holdout": _eval_proposer_standalone(yolox, bmode, k, v1b_hold_full),
                "final_holdout_full": _eval_proposer_standalone(yolox, bmode, k, v1c_hold_full),
            }

    print("[Phase B 4/5] Evaluating End-to-End pipelines across vl_p1_test, vl_p1b_holdout, final_holdout_full, ambiguous...", flush=True)
    frozen_k = int(frozen_cfg["frozen_e2e_k"])
    yolox_v1b = YxV1b(yolox_path)
    pipe_c1 = PipeV1b(PropV1b(yolox_v1b, mode="B0_vl_p1_proposer", top_k=4), rec_v1b_a0, CfgV1b(0.0, False))
    pipe_c4 = PipeV1b(
        PropV1b(yolox_v1b, mode="B_full_v1b", top_k=4),
        rec_v1b_a3,
        CfgV1b(float(v1b_fr.get("e2e_tau_offset_c4", 0.0)), True),
    )

    e2e_pipelines: dict[str, tuple[object, bool]] = {
        "C1_vl_p1_baseline": (pipe_c1, False),
        "C4_vl_p1b_combined": (pipe_c4, False),
        "C_v1c_cal_only": (
            CombinedPipelineV1c(
                ObjectProposerV1c(yolox, "B0_vl_p1_proposer", top_k=frozen_k),
                roi_models["R5_selective_3state_v1c"],
                MultiProposalControlConfig(enable_control=False, use_thin_corridor_mask=False),
            ),
            True,
        ),
        "C_v1c_prop_only": (
            CombinedPipelineV1c(
                ObjectProposerV1c(yolox, "B_v1c_proposer", top_k=frozen_k),
                roi_models["R0_vl_p1_baseline"],
                MultiProposalControlConfig(enable_control=False, use_thin_corridor_mask=True),
            ),
            True,
        ),
    }
    for k in (4, 6, 8):
        mp_k = MultiProposalControlConfig(**frozen_cfg["e2e_configs_by_k"][str(k)])
        e2e_pipelines[f"C_v1c_full__k{k}"] = (
            CombinedPipelineV1c(
                ObjectProposerV1c(yolox, "B_v1c_proposer", top_k=k),
                roi_models["R5_selective_3state_v1c"],
                mp_k,
            ),
            True,
        )

    e2e_results = {}
    e2e_outcomes_1b: dict[str, dict] = {}
    e2e_outcomes_1c: dict[str, dict] = {}
    rec_r5 = roi_models["R5_selective_3state_v1c"]

    for pname, (pipe_obj, is_v1c) in e2e_pipelines.items():
        r_p1 = _eval_e2e_pipeline(pipe_obj, vl_p1_full, is_v1c)
        r_1b = _eval_e2e_pipeline(pipe_obj, v1b_hold_full, is_v1c)
        r_1c = _eval_e2e_pipeline(pipe_obj, v1c_hold_full, is_v1c)
        amb_1c = _eval_ambiguous_split(rec_r5 if is_v1c else rec_v1b_a3, pipe_obj, v1c_ambig)
        e2e_outcomes_1b[pname] = r_1b["_outcomes"]
        e2e_outcomes_1c[pname] = r_1c["_outcomes"]
        e2e_results[pname] = {
            "vl_p1_test_full": _strip_internal_outcomes(r_p1),
            "vl_p1b_holdout": _strip_internal_outcomes(r_1b),
            "final_holdout_full": _strip_internal_outcomes(r_1c),
            "ambiguous_boundary_v1c": amb_1c,
        }

    # Explicit alias key `C_v1c_full` pointing to the frozen calibration K
    e2e_results["C_v1c_full"] = e2e_results[f"C_v1c_full__k{frozen_k}"]

    # Paired statistical tests (McNemar exact p + paired bootstrap 95% CI)
    frozen_name = f"C_v1c_full__k{frozen_k}"
    paired_stats = {}
    for ref_name in ("C1_vl_p1_baseline", "C4_vl_p1b_combined"):
        paired_stats[f"{frozen_name}_vs_{ref_name}"] = {
            "vl_p1b_holdout_214": {
                metric_k: paired_binary_comparison(e2e_outcomes_1b[frozen_name][metric_k], e2e_outcomes_1b[ref_name][metric_k])
                for metric_k in ("all", "pos", "neg")
            },
            "final_holdout_300": {
                metric_k: paired_binary_comparison(e2e_outcomes_1c[frozen_name][metric_k], e2e_outcomes_1c[ref_name][metric_k])
                for metric_k in ("all", "pos", "neg")
            },
        }
    paired_stats["R5_vs_R0_roi_final_holdout_300"] = {
        metric_k: paired_binary_comparison(
            roi_outcomes_fhold["R5_selective_3state_v1c"][metric_k],
            roi_outcomes_fhold["R0_vl_p1_baseline"][metric_k],
        )
        for metric_k in ("all", "pos", "neg")
    }
    paired_stats["R5_vs_A3_v1b_roi_final_holdout_300"] = {
        metric_k: paired_binary_comparison(
            roi_outcomes_fhold["R5_selective_3state_v1c"][metric_k],
            roi_outcomes_fhold["A3_vl_p1b_ref"][metric_k],
        )
        for metric_k in ("all", "pos", "neg")
    }

    print("[Phase B 5/5] Profiling uncontended CPU, RAM/PrivateBytes & gallery scalability...", flush=True)
    scaling = _benchmark_gallery_scaling(base_v1c.embedder, vl_p1c_dir / "data" / "hard_neg_gallery", vl_p1c_dir / "data" / "train_pos")
    mem_full = proc.memory_full_info()
    hw_telemetry = {
        "rss_start_mb": rss_start_mb,
        "rss_loaded_mb": rss_loaded_mb,
        "rss_end_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
        "private_bytes_mb": round(mem_full.private / (1024 * 1024), 2),
        "cpu_percent_avg": round(proc.cpu_percent(interval=0.1), 2),
        "gallery_scaling": scaling,
    }

    # Build proposal_ablation.md + confusion_matrices.md
    cm_cols_roi = list(OBJECT_IDS) + ["UNKNOWN", "AMBIGUOUS"]
    cm_cols_e2e = list(OBJECT_IDS) + ["UNKNOWN", "AMBIGUOUS", "PROPOSAL_MISSED"]
    cm_md = [
        "# Confusion Matrices — VL-P1c (Locked Blind Evaluation)",
        "",
        "## 1. ROI Classification on `final_holdout_roi` (`N=300`)",
        "",
        _format_confusion_md("R0_vl_p1_baseline", roi_results["R0_vl_p1_baseline"]["final_holdout_roi"]["confusion_matrix"], cm_cols_roi),
        _format_confusion_md("A3_vl_p1b_ref (VL-P1b reference)", roi_results["A3_vl_p1b_ref"]["final_holdout_roi"]["confusion_matrix"], cm_cols_roi),
        _format_confusion_md("R5_selective_3state_v1c (VL-P1c Locked)", roi_results["R5_selective_3state_v1c"]["final_holdout_roi"]["confusion_matrix"], cm_cols_roi),
        "## 2. End-to-End Pipelines on `final_holdout_full` (`N=300`)",
        "",
        _format_confusion_md("C1_vl_p1_baseline", e2e_results["C1_vl_p1_baseline"]["final_holdout_full"]["confusion_matrix"], cm_cols_e2e),
        _format_confusion_md("C4_vl_p1b_combined", e2e_results["C4_vl_p1b_combined"]["final_holdout_full"]["confusion_matrix"], cm_cols_e2e),
        _format_confusion_md(f"C_v1c_full (Frozen K={frozen_k})", e2e_results["C_v1c_full"]["final_holdout_full"]["confusion_matrix"], cm_cols_e2e),
    ]
    cm_txt = "\n".join(cm_md)
    (vl_p1c_dir / "reports" / "confusion_matrices.md").write_text(cm_txt, encoding="utf-8")
    (vl_p1c_dir / "confusion_matrices.md").write_text(cm_txt, encoding="utf-8")

    # Build proposal_ablation.md directly from observed numbers
    prop_md = [
        "# Object Proposal 3x3 Fair Budget & White-Cane Ablation (VL-P1c)",
        "",
        "## 1. Fair Budget 3x3 Matrix on `final_holdout_full` (`N=100` positive images)",
        "",
        "| Proposer | Top-K | R@0.3 | R@0.5 | Mean IoU | Containment | BG Contam | `object_01` R@0.5 | Prop Latency p50 (ms) |",
        "|---|---:|---|---|---:|---:|---:|---|---:|",
    ]
    for pkey in (
        "yolox_strict__k4", "yolox_low__k4",
        "B0_vl_p1_proposer__k4", "B0_vl_p1_proposer__k6", "B0_vl_p1_proposer__k8",
        "B_full_v1b__k4", "B_full_v1b__k6", "B_full_v1b__k8",
        "B_v1c_proposer__k4", "B_v1c_proposer__k6", "B_v1c_proposer__k8",
    ):
        d = prop_results[pkey]["final_holdout_full"]
        prop_md.append(
            f"| `{d['mode']}` | {d['top_k']} | {d['recall_iou_0_3']['ratio']} ({d['recall_iou_0_3']['pct']}%) | "
            f"{d['recall_iou_0_5']['ratio']} ({d['recall_iou_0_5']['pct']}%) | {d['mean_best_iou']} | "
            f"{d['mean_containment']} | {d['mean_bg_contamination']} | "
            f"{d['per_category']['object_01']['recall_iou_0_5']['ratio']} | {d['latency_p50_ms']} |"
        )
    prop_md.extend([
        "",
        "## 2. Dedicated White-Cane (`object_01`, N=20) & Handheld Subset Telemetry (`final_holdout_full`)",
        "",
        "| Proposer @ K | `object_01` R@0.3 | `object_01` R@0.5 | Mean IoU | Containment | BG Contam | Endpoint Incl | Axis Err (deg) | Handheld Subset R@0.5 |",
        "|---|---|---|---|---:|---:|---|---:|---|",
    ])
    for pkey in ("B0_vl_p1_proposer__k4", "B_full_v1b__k4", "B_full_v1b__k8", "B_v1c_proposer__k4", "B_v1c_proposer__k6", "B_v1c_proposer__k8"):
        wc = prop_results[pkey]["final_holdout_full"]["white_cane_specialized"]["all_object_01"]
        hh = prop_results[pkey]["final_holdout_full"]["white_cane_specialized"]["handheld_person_subset"]
        hh_str = hh["recall_iou_0_5"]["ratio"] if hh.get("n", 0) > 0 else "0/0"
        prop_md.append(
            f"| `{pkey}` | {wc['recall_iou_0_3']['ratio']} | {wc['recall_iou_0_5']['ratio']} | {wc['mean_best_iou']} | "
            f"{wc['mean_containment']} | {wc['mean_bg_contamination']} | {wc['endpoint_inclusion_rate']['ratio']} | "
            f"{wc['mean_axis_angle_error_deg']} | {hh_str} |"
        )
    prop_md.append("")
    prop_txt = "\n".join(prop_md)
    (vl_p1c_dir / "reports" / "proposal_ablation.md").write_text(prop_txt, encoding="utf-8")
    (vl_p1c_dir / "proposal_ablation.md").write_text(prop_txt, encoding="utf-8")

    benchmark_payload = {
        "schema_version": "vl_p1c_benchmark_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_config_ref": {
            "selected_delta_floor": frozen_cfg["selected_delta_floor"],
            "frozen_e2e_k": frozen_k,
            "final_holdout_touched_during_calibration": frozen_cfg["final_holdout_touched_during_calibration"],
        },
        "roi_results": roi_results,
        "proposal_results": prop_results,
        "e2e_results": e2e_results,
        "paired_statistical_tests": paired_stats,
        "hardware_telemetry": hw_telemetry,
    }
    bm_json = json.dumps(benchmark_payload, indent=2)
    (vl_p1c_dir / "reports" / "benchmark_metrics.json").write_text(bm_json, encoding="utf-8")
    (vl_p1c_dir / "benchmark_metrics.json").write_text(bm_json, encoding="utf-8")
    print("[Phase B Done] benchmark_metrics.json, confusion_matrices.md, proposal_ablation.md written.", flush=True)
    return benchmark_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="VL-P1c Calibration Freeze & Locked Benchmark Runner")
    parser.add_argument("--phase", choices=["a", "b", "all"], default="all")
    parser.add_argument("--max-far", type=float, default=0.05)
    args = parser.parse_args()

    vl_p1c_dir = Path(__file__).resolve().parent.parent
    if args.phase in ("a", "all"):
        _, ok = run_phase_a_calibration(vl_p1c_dir, max_far_target=args.max_far)
        if not ok:
            raise SystemExit(2)
    if args.phase in ("b", "all"):
        run_phase_b_locked_evaluation(vl_p1c_dir)


if __name__ == "__main__":
    main()
