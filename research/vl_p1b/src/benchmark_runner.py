"""Step 4: Reproducible VL-P1b benchmark & ablation runner.

Executes offline (CPU-only ONNX Runtime) on curated manifest splits:
1. Calibrates A0..A3 and C4 strictly on calibration_v1b / calibration_v1b_full
   and freezes thresholds to manifests/frozen_thresholds.json BEFORE seeing holdout.
2. Evaluates Method A (A0..A3) on historical vl_p1_test & independent holdout_roi.
3. Evaluates Method B (yolox_strict, yolox_low, B0, B1, B2, B_full_v1b K=3,4,6,8)
   with per-category IoU@0.3, IoU@0.5, containment, and bg_contamination.
4. Evaluates Method C (C1..C4) end-to-end on vl_p1_test_full & holdout_full + ambiguous.
5. Collects uncontended CPU/RAM/PrivateBytes & N=20/50/100 scaling telemetry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import psutil

from .common_utils import (
    OBJECT_IDS,
    LABELS_VI,
    compute_containment_and_bg_contamination,
    compute_iou_xywh,
    compute_sha256,
    format_count_pct_ci,
    load_image_unicode,
    wilson_ci_95,
)
from .combined_pipeline import CombinedE2EPipeline, E2EDecisionConfig
from .hard_negative_research import (
    DINOv2Embedder,
    HardNegConfig,
    HardNegativeRecognizer,
    _l2_normalize,
)
from .labeling_protocol import HARD_NEGATIVE_GROUPS
from .proposal_research import ObjectProposerVL1b, YOLOXNanoExtractor


def _pct_ci(k: int, n: int) -> dict:
    d = format_count_pct_ci(k, n)
    d["formatted"] = f"{d['ratio']} ({d['pct']:.1f}%, 95% CI [{d['ci95_low_pct']:.2f}, {d['ci95_high_pct']:.2f}])"
    return d


def _load_vl_p1_test(vl_p1_dir: Path) -> tuple[list[dict], list[dict]]:
    mf = json.loads((vl_p1_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    roi_items, full_items = [], []
    for item in mf["test"]:
        lbl = item["label"]
        cg = item.get("confuser_type") or ("shoe_distractor" if lbl == "unknown" else "positive")
        roi_p = vl_p1_dir / item["roi_rel_path"]
        roi_items.append({
            "id": item["id"],
            "label_id": lbl,
            "confuser_group": cg,
            "path": roi_p,
        })
        full_p = vl_p1_dir / item["full_rel_path"]
        full_items.append({
            "id": item["id"],
            "label_id": lbl,
            "confuser_group": cg,
            "path": full_p,
            "roi_bbox_xywh": tuple(item["roi_bbox_xywh"]) if item.get("roi_bbox_xywh") else None,
        })
    return roi_items, full_items


def _load_vl_p1b_split(vl_p1b_dir: Path, mf: dict, split_name: str, use_roi: bool) -> list[dict]:
    items = []
    for row in mf.get(split_name, []):
        rel = row["roi_rel_path"] if use_roi else row["full_rel_path"]
        items.append({
            "id": row["id"],
            "label_id": row["label_id"],
            "confuser_group": row.get("confuser_group") or "positive",
            "path": vl_p1b_dir / rel,
            "roi_bbox_xywh": tuple(row["roi_bbox_xywh"]) if row.get("roi_bbox_xywh") else None,
        })
    return items


def _empty_confusion(labels: list[str], cols: list[str]) -> dict[str, dict[str, int]]:
    return {r: {c: 0 for c in cols} for r in labels}


def evaluate_roi_recognizer(
    recognizer: HardNegativeRecognizer,
    items: list[dict],
) -> dict:
    row_labels = list(OBJECT_IDS) + ["unknown"]
    col_labels = list(OBJECT_IDS) + ["UNKNOWN", "AMBIGUOUS"]
    cm = _empty_confusion(row_labels, col_labels)

    pos_correct = pos_total = pos_wrong_cls = 0
    neg_fp = neg_total = neg_ambig = pos_ambig = 0
    shoe_slipper_err = shoe_slipper_tot = 0
    per_class = {oid: [0, 0] for oid in OBJECT_IDS}
    per_neg_group: dict[str, list[int]] = {}
    lat_ms = []
    errors = []

    for item in items:
        img = load_image_unicode(item["path"])
        dec = recognizer.predict(img)
        lat_ms.append(float(dec["feat_ms"]) + float(dec["match_ms"]))
        y = item["label_id"]
        cg = item["confuser_group"]
        status = dec["status"]
        pred_lbl = dec["label"] if status == "KNOWN" else status
        cm[y][pred_lbl] = cm[y].get(pred_lbl, 0) + 1

        if y == "unknown":
            neg_total += 1
            per_neg_group.setdefault(cg, [0, 0])
            per_neg_group[cg][1] += 1
            if status == "KNOWN":
                neg_fp += 1
                per_neg_group[cg][0] += 1
                errors.append({
                    "id": item["id"],
                    "type": "FP",
                    "gt": y,
                    "group": cg,
                    "pred": dec["label"],
                    "score": dec["score"],
                    "pos_score": dec.get("pos_score"),
                    "neg_sim": dec.get("neg_sim"),
                })
            elif status == "AMBIGUOUS":
                neg_ambig += 1
        else:
            pos_total += 1
            per_class[y][1] += 1
            if y in ("object_04", "object_05"):
                shoe_slipper_tot += 1
            if status == "KNOWN" and pred_lbl == y:
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
                    "score": dec["score"],
                    "pos_score": dec.get("pos_score"),
                    "neg_sim": dec.get("neg_sim"),
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
        "ambiguous_counts": {"pos_ambig": pos_ambig, "neg_ambig": neg_ambig},
        "per_class_recall": {oid: _pct_ci(per_class[oid][0], per_class[oid][1]) for oid in OBJECT_IDS},
        "per_neg_group_far": {
            g: _pct_ci(v[0], v[1]) for g, v in sorted(per_neg_group.items())
        },
        "latency_p50_ms": round(float(np.percentile(lat_ms, 50)), 2) if lat_ms else 0.0,
        "latency_p95_ms": round(float(np.percentile(lat_ms, 95)), 2) if lat_ms else 0.0,
        "confusion_matrix": cm,
        "errors": errors,
    }


def evaluate_proposer_standalone(
    yolox: YOLOXNanoExtractor,
    mode: str,
    top_k: int,
    items: list[dict],
) -> dict:
    """Evaluate proposal quality (GT bbox NEVER passed to proposer)."""
    proposer = ObjectProposerVL1b(yolox=yolox, mode=mode, top_k=top_k)
    r03 = r05 = n_pos = 0
    best_ious, best_conts, best_bgs, counts, lats = [], [], [], [], []
    per_cat: dict[str, dict] = {
        oid: {"r03": 0, "r05": 0, "n": 0, "ious": [], "conts": [], "bgs": []}
        for oid in OBJECT_IDS
    }
    misses = []

    for item in items:
        img = load_image_unicode(item["path"])
        props, ms = proposer.propose(img)  # Strictly image-only!
        counts.append(len(props))
        lats.append(ms)
        if item["label_id"] == "unknown" or item["roi_bbox_xywh"] is None:
            continue
        gt = item["roi_bbox_xywh"]
        oid = item["label_id"]
        n_pos += 1
        per_cat[oid]["n"] += 1
        if props:
            scored = [(compute_iou_xywh(p.bbox_xywh, gt), p) for p in props]
            best_iou, best_prop = max(scored, key=lambda t: t[0])
            cont, bg_cont = compute_containment_and_bg_contamination(best_prop.bbox_xywh, gt)
        else:
            best_iou, cont, bg_cont, best_prop = 0.0, 0.0, 1.0, None
        best_ious.append(best_iou)
        best_conts.append(cont)
        best_bgs.append(bg_cont)
        per_cat[oid]["ious"].append(best_iou)
        per_cat[oid]["conts"].append(cont)
        per_cat[oid]["bgs"].append(bg_cont)
        if best_iou >= 0.3:
            r03 += 1
            per_cat[oid]["r03"] += 1
        if best_iou >= 0.5:
            r05 += 1
            per_cat[oid]["r05"] += 1
        else:
            misses.append({
                "id": item["id"],
                "gt_class": oid,
                "best_iou": round(best_iou, 4),
                "containment": round(cont, 4),
                "bg_contamination": round(bg_cont, 4),
                "best_source": best_prop.source if best_prop else None,
            })

    per_category_summary = {}
    for oid in OBJECT_IDS:
        c = per_cat[oid]
        per_category_summary[oid] = {
            "recall_iou_0_3": _pct_ci(c["r03"], c["n"]),
            "recall_iou_0_5": _pct_ci(c["r05"], c["n"]),
            "mean_best_iou": round(float(np.mean(c["ious"])) if c["ious"] else 0.0, 4),
            "mean_containment": round(float(np.mean(c["conts"])) if c["conts"] else 0.0, 4),
            "mean_bg_contamination": round(float(np.mean(c["bgs"])) if c["bgs"] else 0.0, 4),
        }

    return {
        "mode": mode,
        "top_k": top_k,
        "recall_iou_0_3": _pct_ci(r03, n_pos),
        "recall_iou_0_5": _pct_ci(r05, n_pos),
        "mean_best_iou": round(float(np.mean(best_ious)) if best_ious else 0.0, 4),
        "mean_containment": round(float(np.mean(best_conts)) if best_conts else 0.0, 4),
        "mean_bg_contamination": round(float(np.mean(best_bgs)) if best_bgs else 0.0, 4),
        "avg_proposals_per_image": round(float(np.mean(counts)) if counts else 0.0, 2),
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 2) if lats else 0.0,
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
        "per_category": per_category_summary,
        "misses_iou_0_5": misses[:25],
    }


def evaluate_e2e_pipeline(
    pipeline: CombinedE2EPipeline,
    items: list[dict],
) -> dict:
    row_labels = list(OBJECT_IDS) + ["unknown"]
    col_labels = list(OBJECT_IDS) + ["UNKNOWN", "AMBIGUOUS", "PROPOSAL_MISSED"]
    cm = _empty_confusion(row_labels, col_labels)

    pos_correct = pos_total = pos_wrong_cls = pos_prop_missed = 0
    neg_fp = neg_total = neg_ambig = pos_ambig = 0
    tp_iou05 = 0
    shoe_slipper_err = shoe_slipper_tot = 0
    per_class = {oid: [0, 0] for oid in OBJECT_IDS}
    per_neg_group: dict[str, list[int]] = {}
    prop_ms, ver_ms, tot_ms, n_embs = [], [], [], []
    src_counts: dict[str, int] = {}
    errors = []

    for item in items:
        img = load_image_unicode(item["path"])
        dec = pipeline.predict(img)
        prop_ms.append(float(dec["proposal_ms"]))
        ver_ms.append(float(dec["verify_ms"]))
        tot_ms.append(float(dec["total_ms"]))
        n_embs.append(int(dec["num_embeddings"]))
        if dec.get("selected_source"):
            src = dec["selected_source"]
            src_counts[src] = src_counts.get(src, 0) + 1

        y = item["label_id"]
        cg = item["confuser_group"]
        status = dec["status"]
        pred_lbl = dec["label"] if status == "KNOWN" else status
        cm[y][pred_lbl] = cm[y].get(pred_lbl, 0) + 1

        if y == "unknown":
            neg_total += 1
            per_neg_group.setdefault(cg, [0, 0])
            per_neg_group[cg][1] += 1
            if status == "KNOWN":
                neg_fp += 1
                per_neg_group[cg][0] += 1
                errors.append({
                    "id": item["id"],
                    "type": "E2E_FP",
                    "gt": y,
                    "group": cg,
                    "pred": dec["label"],
                    "score": dec["score"],
                    "selected_box": dec["selected_box"],
                    "selected_source": dec["selected_source"],
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
                max((compute_iou_xywh(b, gt) for b in dec["proposals"]), default=0.0)
                if gt and dec["proposals"]
                else 0.0
            )
            if gt and best_prop_iou < 0.3:
                pos_prop_missed += 1

            if status == "KNOWN" and pred_lbl == y:
                pos_correct += 1
                per_class[y][0] += 1
                if gt and dec["selected_box"] and compute_iou_xywh(tuple(dec["selected_box"]), gt) >= 0.5:
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
                    etype = "E2E_FN_UNKNOWN"
                errors.append({
                    "id": item["id"],
                    "type": etype,
                    "gt": y,
                    "pred": pred_lbl,
                    "score": dec["score"],
                    "best_proposal_iou": round(best_prop_iou, 4),
                    "selected_source": dec["selected_source"],
                    "num_proposals": len(dec["proposals"]),
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
        "proposal_missed_iou_lt_0_3": _pct_ci(pos_prop_missed, pos_total),
        "tp_with_selected_iou_ge_0_5": _pct_ci(tp_iou05, max(1, pos_correct)),
        "ambiguous_counts": {"pos_ambig": pos_ambig, "neg_ambig": neg_ambig},
        "per_class_recall": {oid: _pct_ci(per_class[oid][0], per_class[oid][1]) for oid in OBJECT_IDS},
        "per_neg_group_far": {g: _pct_ci(v[0], v[1]) for g, v in sorted(per_neg_group.items())},
        "selected_source_counts": src_counts,
        "avg_num_embeddings": round(float(np.mean(n_embs)) if n_embs else 0.0, 2),
        "proposal_ms_p50": round(float(np.percentile(prop_ms, 50)), 2) if prop_ms else 0.0,
        "verify_ms_p50": round(float(np.percentile(ver_ms, 50)), 2) if ver_ms else 0.0,
        "total_ms_p50": round(float(np.percentile(tot_ms, 50)), 2) if tot_ms else 0.0,
        "total_ms_p95": round(float(np.percentile(tot_ms, 95)), 2) if tot_ms else 0.0,
        "confusion_matrix": cm,
        "errors": errors,
    }


def _evaluate_ambiguous_split(recognizer: HardNegativeRecognizer, pipeline: CombinedE2EPipeline, items: list[dict]) -> dict:
    roi_counts = {"KNOWN": 0, "AMBIGUOUS": 0, "UNKNOWN": 0}
    e2e_counts = {"KNOWN": 0, "AMBIGUOUS": 0, "UNKNOWN": 0, "PROPOSAL_MISSED": 0}
    details = []
    for item in items:
        img = load_image_unicode(item["path"])
        dec_r = recognizer.predict(img)
        roi_counts[dec_r["status"]] = roi_counts.get(dec_r["status"], 0) + 1
        dec_e = pipeline.predict(img)
        e2e_counts[dec_e["status"]] = e2e_counts.get(dec_e["status"], 0) + 1
        details.append({
            "id": item["id"],
            "roi_status": dec_r["status"],
            "roi_label": dec_r["label"],
            "roi_score": dec_r["score"],
            "e2e_status": dec_e["status"],
            "e2e_label": dec_e["label"],
            "e2e_score": dec_e["score"],
        })
    return {"n": len(items), "roi_counts": roi_counts, "e2e_counts": e2e_counts, "details": details}


def _benchmark_gallery_scaling(embedder: DINOv2Embedder, hard_neg_dir: Path, train_pos_dir: Path) -> list[dict]:
    """Measure matching RAM and p50/p95 match_ms for negative gallery sizes N=20, 50, 100."""
    # Collect available negative images (repeat deterministically only if physical < N for stress test)
    neg_paths = sorted(p for p in hard_neg_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    neg_embs = [embedder.extract(load_image_unicode(p)) for p in neg_paths]
    pos_rec = HardNegativeRecognizer(embedder.model_path.parent, HardNegConfig(mode="A3_full_verifier"))
    pos_rec.fit_positives(train_pos_dir)
    if neg_paths:
        pos_rec.fit_hard_negatives(hard_neg_dir)
    pos_mat = np.vstack(list(pos_rec.pos_embeddings.values())).astype(np.float32)
    sample_emb = neg_embs[0] if neg_embs else np.zeros(768, dtype=np.float32)

    results = []
    for target_n in (20, 50, 100):
        if neg_embs:
            idx = [i % len(neg_embs) for i in range(target_n)]
            mat = np.stack([neg_embs[i] for i in idx], axis=0)
        else:
            mat = np.zeros((target_n, 768), dtype=np.float32)
        times_ms = []
        for _ in range(40):
            t0 = time.perf_counter()
            _ = mat @ sample_emb
            _ = pos_mat @ sample_emb
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        results.append({
            "neg_gallery_n": target_n,
            "matrix_bytes": int(mat.nbytes),
            "match_p50_ms": round(float(np.percentile(times_ms, 50)), 4),
            "match_p95_ms": round(float(np.percentile(times_ms, 95)), 4),
        })
    return results


def _format_cm_markdown(title: str, cm: dict[str, dict[str, int]]) -> str:
    cols = list(next(iter(cm.values())).keys())
    lines = [f"### {title}", "", "| GT \\ Pred | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for row_k, row_v in cm.items():
        vals = " | ".join(str(row_v.get(c, 0)) for c in cols)
        lines.append(f"| `{row_k}` | {vals} |")
    lines.append("")
    return "\n".join(lines)


def run_all_benchmarks(vl_p1_dir: Path, vl_p1b_dir: Path) -> dict:
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)
    rss_start_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    models_dir = vl_p1b_dir / "models"
    yolox_path = vl_p1b_dir.parent.parent / "models" / "object_detection" / "yolox_nano.onnx"
    mf = json.loads((vl_p1b_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))

    train_pos_dir = vl_p1b_dir / "data" / "train_pos"
    hard_neg_dir = vl_p1b_dir / "data" / "hard_neg_gallery"

    cal_roi = _load_vl_p1b_split(vl_p1b_dir, mf, "calibration_v1b_roi", use_roi=True)
    cal_full = _load_vl_p1b_split(vl_p1b_dir, mf, "calibration_v1b_full", use_roi=False)
    hold_roi = _load_vl_p1b_split(vl_p1b_dir, mf, "holdout_roi", use_roi=True)
    hold_full = _load_vl_p1b_split(vl_p1b_dir, mf, "holdout_full", use_roi=False)
    amb_items = _load_vl_p1b_split(vl_p1b_dir, mf, "ambiguous_boundary", use_roi=False)
    vl_p1_roi, vl_p1_full = _load_vl_p1_test(vl_p1_dir)

    print("[1/5] Loading models & calibrating strictly on calibration_v1b...", flush=True)
    yolox = YOLOXNanoExtractor(yolox_path)
    cal_items_loaded = [(r["label_id"], r["confuser_group"], load_image_unicode(r["path"])) for r in cal_roi]

    a_modes = ("A0_baseline_hybrid", "A1_neg_prototypes", "A2_adaptive_thresholds", "A3_full_verifier")
    a_models: dict[str, HardNegativeRecognizer] = {}
    cal_summaries: dict[str, dict] = {}
    for mode in a_modes:
        rec = HardNegativeRecognizer(models_dir, HardNegConfig(mode=mode))
        rec.fit_positives(train_pos_dir)
        rec.fit_hard_negatives(hard_neg_dir)
        cal_info = rec.calibrate_on_split(cal_items_loaded, max_far_target=0.05)
        a_models[mode] = rec
        cal_summaries[mode] = {
            "calibration_params": cal_info,
            "calibration_roi_check": evaluate_roi_recognizer(rec, cal_roi)["known_recall"],
            "calibration_roi_far": evaluate_roi_recognizer(rec, cal_roi)["far"],
            "db_sizes": rec.serialized_db_sizes(),
        }

    # Export frozen A3 state + freeze E2E offset on calibration_v1b_full BEFORE touching holdout!
    proposer_b_full = ObjectProposerVL1b(yolox=yolox, mode="B_full_v1b", top_k=4)
    cal_full_loaded = [(r["label_id"], load_image_unicode(r["path"])) for r in cal_full]
    c4_pipe_cal = CombinedE2EPipeline(proposer_b_full, a_models["A3_full_verifier"], E2EDecisionConfig(e2e_tau_offset=0.0))
    e2e_offset = c4_pipe_cal.calibrate_e2e_offset(cal_full_loaded, max_far=0.05)

    frozen_a3_path = vl_p1b_dir / "manifests" / "frozen_a3_state.json"
    a_models["A3_full_verifier"].export_frozen_state(frozen_a3_path)
    frozen_thresh_path = vl_p1b_dir / "manifests" / "frozen_thresholds.json"
    frozen_payload = {
        "frozen_from_split": "calibration_v1b + calibration_v1b_full",
        "holdout_touched_during_calibration": False,
        "e2e_tau_offset_c4": e2e_offset,
        "calibration_summaries": cal_summaries,
        "frozen_a3_state_sha256": compute_sha256(frozen_a3_path),
    }
    frozen_thresh_path.write_text(json.dumps(frozen_payload, indent=2), encoding="utf-8")
    frozen_sha = compute_sha256(frozen_thresh_path)
    rss_loaded_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    print(f"[Frozen] thresholds sha256={frozen_sha[:16]} e2e_offset={e2e_offset}", flush=True)

    print("[2/5] Evaluating Method A (ROI A0..A3) on vl_p1_test & independent holdout...", flush=True)
    method_a_results = {}
    for mode, rec in a_models.items():
        method_a_results[mode] = {
            "vl_p1_test": evaluate_roi_recognizer(rec, vl_p1_roi),
            "holdout_roi": evaluate_roi_recognizer(rec, hold_roi),
        }

    print("[3/5] Evaluating Method B (proposals standalone & K=3,4,6,8)...", flush=True)
    method_b_results = {}
    for b_mode in ("yolox_strict", "yolox_low", "B0_vl_p1_proposer", "B1_multiscale", "B2_geometry_thin", "B_full_v1b"):
        method_b_results[f"{b_mode}__k4"] = {
            "vl_p1_test_full": evaluate_proposer_standalone(yolox, b_mode, 4, vl_p1_full),
            "holdout_full": evaluate_proposer_standalone(yolox, b_mode, 4, hold_full),
        }
    for k in (3, 6, 8):
        method_b_results[f"B_full_v1b__k{k}"] = {
            "vl_p1_test_full": evaluate_proposer_standalone(yolox, "B_full_v1b", k, vl_p1_full),
            "holdout_full": evaluate_proposer_standalone(yolox, "B_full_v1b", k, hold_full),
        }

    print("[4/5] Evaluating Method C (C1..C4 E2E ablation) on vl_p1_test, holdout, & ambiguous...", flush=True)
    proposer_b0 = ObjectProposerVL1b(yolox=yolox, mode="B0_vl_p1_proposer", top_k=4)
    c_configs = {
        "C1_vl_p1_baseline": CombinedE2EPipeline(proposer_b0, a_models["A0_baseline_hybrid"], E2EDecisionConfig(0.0, False)),
        "C2_hard_neg_only": CombinedE2EPipeline(proposer_b0, a_models["A3_full_verifier"], E2EDecisionConfig(0.0, True)),
        "C3_proposal_only": CombinedE2EPipeline(proposer_b_full, a_models["A0_baseline_hybrid"], E2EDecisionConfig(0.0, True)),
        "C4_combined_v1b": CombinedE2EPipeline(proposer_b_full, a_models["A3_full_verifier"], E2EDecisionConfig(e2e_offset, True)),
    }
    method_c_results = {}
    for cname, pipe in c_configs.items():
        method_c_results[cname] = {
            "vl_p1_test_full": evaluate_e2e_pipeline(pipe, vl_p1_full),
            "holdout_full": evaluate_e2e_pipeline(pipe, hold_full),
            "ambiguous_boundary": _evaluate_ambiguous_split(a_models["A3_full_verifier"], pipe, amb_items),
        }

    print("[5/5] Profiling uncontended CPU/RAM/PrivateBytes & gallery scaling...", flush=True)
    scaling = _benchmark_gallery_scaling(a_models["A3_full_verifier"].embedder, hard_neg_dir, train_pos_dir)
    mem_full = proc.memory_full_info()
    hw_telemetry = {
        "rss_start_mb": rss_start_mb,
        "rss_loaded_mb": rss_loaded_mb,
        "rss_end_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
        "private_bytes_mb": round(mem_full.private / (1024 * 1024), 2),
        "cpu_percent_avg": round(proc.cpu_percent(interval=0.1), 2),
        "gallery_scaling_n_20_50_100": scaling,
    }

    report = {
        "frozen_thresholds_sha256": frozen_sha,
        "dataset_manifest_sha256": compute_sha256(vl_p1b_dir / "manifests" / "dataset_manifest.json"),
        "calibration": frozen_payload,
        "method_a_roi": method_a_results,
        "method_b_proposals": method_b_results,
        "method_c_e2e_ablation": method_c_results,
        "hardware_and_scalability": hw_telemetry,
    }

    reports_dir = vl_p1b_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "benchmark_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write confusion matrices Markdown
    cm_lines = ["# VL-P1b Confusion Matrices", ""]
    for m in ("A0_baseline_hybrid", "A3_full_verifier"):
        cm_lines.append(_format_cm_markdown(f"ROI Holdout — {m}", method_a_results[m]["holdout_roi"]["confusion_matrix"]))
    for c in ("C1_vl_p1_baseline", "C2_hard_neg_only", "C3_proposal_only", "C4_combined_v1b"):
        cm_lines.append(_format_cm_markdown(f"E2E Holdout — {c}", method_c_results[c]["holdout_full"]["confusion_matrix"]))
    (reports_dir / "confusion_matrices.md").write_text("\n".join(cm_lines), encoding="utf-8")

    print("[Done] Benchmark reports written to", reports_dir, flush=True)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run_all_benchmarks(root / "research" / "vl_p1", root / "research" / "vl_p1b")
