"""Two-Phase VL-P4 Benchmark Runner (Phase A Calibrate/Freeze + Single-Shot Phase B Blind Evaluation)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import psutil
from sklearn.metrics import average_precision_score, roc_auc_score

from .cnn_cane_net import INPUT_SIZE, ONNXCaneRunner
from .common_utils import (
    VL_P4_DIR,
    WORKSPACE_ROOT,
    compute_endpoint_and_centerline_metrics,
    compute_mask_metrics,
    compute_sha256,
    decode_mask_rle,
    load_image_unicode,
    paired_binary_comparison,
    save_image_unicode,
    wilson_ci_95,
)
from .soft_verifier_v4 import SoftCaneVerifierV4, extract_soft_line_features


def _load_v3_a1_pipeline():
    p3_root = WORKSPACE_ROOT / "VisionMate_VL_P3_AG"
    if str(p3_root) not in sys.path:
        sys.path.insert(0, str(p3_root))
    from research.vl_p3.src.benchmark_runner_v3 import _load_frozen_pipeline
    pipe, _ = _load_frozen_pipeline()
    return pipe


def _calibrate_thresholds(scores: np.ndarray, is_pos: np.ndarray, target_far: float = 0.05) -> tuple[float, float]:
    best_tk, best_obj = 0.55, (-1.0, -1.0, 0.0)
    for tk in np.linspace(0.25, 0.92, 68):
        fp_rate = float(np.mean(scores[~is_pos] >= tk)) if np.any(~is_pos) else 0.0
        tp_rate = float(np.mean(scores[is_pos] >= tk)) if np.any(is_pos) else 0.0
        ok = 1.0 if fp_rate <= target_far else 0.0
        obj = (ok, tp_rate if ok else -fp_rate, -fp_rate)
        if obj > best_obj:
            best_obj = obj
            best_tk = float(tk)
    best_tu = round(max(0.18, best_tk - 0.18), 4)
    return round(best_tk, 4), best_tu


def _compute_cls_metrics(records: list[dict]) -> dict:
    pos = [r for r in records if r["true_pos"]]
    neg = [r for r in records if not r["true_pos"]]
    tp = sum(1 for r in pos if r["decision"] == "KNOWN")
    pos_unc = sum(1 for r in pos if r["decision"] == "UNCERTAIN")
    fn = len(pos) - tp - pos_unc
    fp = sum(1 for r in neg if r["decision"] == "KNOWN")
    neg_unc = sum(1 for r in neg if r["decision"] == "UNCERTAIN")
    tn = len(neg) - fp - neg_unc

    n_pos, n_neg, n_tot = len(pos), len(neg), len(records)
    rec = tp / max(1, n_pos)
    far = fp / max(1, n_neg)
    prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(1, n_tot)
    unc_rate = (pos_unc + neg_unc) / max(1, n_tot)

    y_true = np.array([1 if r["true_pos"] else 0 for r in records], dtype=np.int32)
    y_score = np.array([r["score"] for r in records], dtype=np.float32)
    roc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else 0.5
    pr = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else 0.0

    # Risk-coverage curve (sorted by margin from 0.5 or score confidence)
    ranked = sorted(records, key=lambda r: abs(r["score"] - 0.5), reverse=True)
    rc_curve = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6):
        k = max(1, int(round(cov * n_tot)))
        sub = ranked[:k]
        errs = sum(1 for r in sub if (r["score"] >= 0.5) != r["true_pos"])
        rc_curve.append({"coverage": cov, "n": k, "risk_error_rate": round(errs / k, 4)})

    return {
        "n_total": n_tot, "n_pos": n_pos, "n_neg": n_neg,
        "tp": tp, "pos_uncertain": pos_unc, "fn": fn,
        "fp": fp, "neg_uncertain": neg_unc, "tn": tn,
        "known_recall": round(rec, 4),
        "known_recall_ci95": [round(v, 4) for v in wilson_ci_95(tp, n_pos)],
        "far": round(far, 4),
        "far_ci95": [round(v, 4) for v in wilson_ci_95(fp, n_neg)],
        "known_precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "uncertain_rate": round(unc_rate, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "risk_coverage_curve": rc_curve,
    }


def _eval_cnn_variant(
    items: list[dict],
    ann: dict,
    runner: ONNXCaneRunner,
    tk: float,
    tu: float,
    ver: SoftCaneVerifierV4 | None = None,
    keep_masks: bool = False,
) -> tuple[list[dict], dict]:
    records, lats, loc_rows, occ_rows, clr_rows, seg_rows = [], [], [], [], [], []
    t_dec_list, t_pre_list, t_cnn_list, t_post_list = [], [], [], []

    # Warmup 1st sample if needed (discard warmup from latency stats only? Keep full measured!)
    for it in items:
        t0 = time.perf_counter()
        img = load_image_unicode(Path(it["abs_path"]))
        t_img = time.perf_counter()
        pred = runner.predict(img)
        t_ver0 = time.perf_counter()
        if ver is not None:
            vec, _ = extract_soft_line_features(img, pred)
            score = ver.predict_score(vec)
        else:
            score = float(pred["obj_prob"])
        t_end = time.perf_counter()

        dec = "KNOWN" if score >= tk else ("UNCERTAIN" if score >= tu else "UNKNOWN")
        e2e_ms = (t_end - t0) * 1000.0
        lats.append(e2e_ms)
        t_dec_list.append((t_img - t0) * 1000.0)
        t_pre_list.append(pred["pre_ms"])
        t_cnn_list.append(pred["cnn_ms"])
        t_post_list.append(pred["decode_ms"] + (t_end - t_ver0) * 1000.0)

        rec = {
            "id": it["id"],
            "true_pos": it.get("label") == "object_01",
            "confuser_group": it.get("confuser_group", "none"),
            "person_holding_confuser": bool(it.get("person_holding_confuser", False)),
            "decision": dec,
            "score": round(float(score), 4),
            "endpoints_xyxy": pred["endpoints_xyxy"],
            "centerline_points_xy": pred["centerline_points_xy"],
            "e2e_ms": round(e2e_ms, 2),
        }
        records.append(rec)

        if rec["true_pos"] and it["id"] in ann:
            gt = ann[it["id"]]
            em = compute_endpoint_and_centerline_metrics(
                pred["endpoints_xyxy"], gt["endpoints_xyxy"],
                pred["centerline_points_xy"], gt["centerline_points_xy"],
                img.shape[:2],
            )
            loc_rows.append(em)
            (occ_rows if gt.get("occluded") else clr_rows).append(em)
            if keep_masks and runner.mode != "A1_CNN_Keypoint" and gt.get("mask_rle"):
                gt_m = decode_mask_rle(gt["mask_rle"]) > 0
                seg_rows.append(compute_mask_metrics(pred["mask_orig"], gt_m))

    cls_m = _compute_cls_metrics(records)
    cls_m["latency_ms"] = {
        "mean": round(float(np.mean(lats)), 2),
        "p50": round(float(np.percentile(lats, 50)), 2),
        "p95": round(float(np.percentile(lats, 95)), 2),
        "min": round(float(np.min(lats)), 2),
        "max": round(float(np.max(lats)), 2),
        "n_measured": len(lats),
        "breakdown_mean_ms": {
            "img_decode": round(float(np.mean(t_dec_list)), 2),
            "preprocess": round(float(np.mean(t_pre_list)), 2),
            "cnn_onnx": round(float(np.mean(t_cnn_list)), 2),
            "post_verify": round(float(np.mean(t_post_list)), 2),
        },
    }

    def _agg(rows):
        if not rows:
            return {}
        return {f"mean_{k}": round(float(np.mean([x[k] for x in rows])), 4) for k in rows[0]}

    return records, {
        "classification": cls_m,
        "localization": _agg(loc_rows),
        "segmentation": _agg(seg_rows),
        "occluded_subset_loc": {"n": len(occ_rows), **_agg(occ_rows)},
        "clear_subset_loc": {"n": len(clr_rows), **_agg(clr_rows)},
    }


def _eval_v3_a1_uncached(items: list[dict], ann: dict, pipe_v3) -> tuple[list[dict], dict]:
    records, lats, loc_rows, occ_rows, clr_rows = [], [], [], [], []
    for it in items:
        t0 = time.perf_counter()
        img = load_image_unicode(Path(it["abs_path"]))
        out = pipe_v3.predict(img, preset="A1_supervised_kp_centerline", bundle=None)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        lats.append(e2e_ms)
        rec = {
            "id": it["id"],
            "true_pos": it.get("label") == "object_01",
            "confuser_group": it.get("confuser_group", "none"),
            "person_holding_confuser": bool(it.get("person_holding_confuser", False)),
            "decision": out["decision"],
            "score": float(out["score"]),
            "endpoints_xyxy": out["endpoints_xyxy"],
            "centerline_points_xy": out["centerline_points_xy"],
            "e2e_ms": round(e2e_ms, 2),
        }
        records.append(rec)
        if rec["true_pos"] and it["id"] in ann:
            gt = ann[it["id"]]
            em = compute_endpoint_and_centerline_metrics(
                out["endpoints_xyxy"], gt["endpoints_xyxy"],
                out["centerline_points_xy"], gt["centerline_points_xy"],
                img.shape[:2],
            )
            loc_rows.append(em)
            (occ_rows if gt.get("occluded") else clr_rows).append(em)

    cls_m = _compute_cls_metrics(records)
    cls_m["latency_ms"] = {
        "mean": round(float(np.mean(lats)), 2),
        "p50": round(float(np.percentile(lats, 50)), 2),
        "p95": round(float(np.percentile(lats, 95)), 2),
        "min": round(float(np.min(lats)), 2),
        "max": round(float(np.max(lats)), 2),
        "n_measured": len(lats),
    }

    def _agg(rows):
        return {f"mean_{k}": round(float(np.mean([x[k] for x in rows])), 4) for k in rows[0]} if rows else {}

    return records, {
        "classification": cls_m,
        "localization": _agg(loc_rows),
        "segmentation": {},
        "occluded_subset_loc": {"n": len(occ_rows), **_agg(occ_rows)},
        "clear_subset_loc": {"n": len(clr_rows), **_agg(clr_rows)},
    }


def run_phase_a() -> Path:
    manifests_dir = VL_P4_DIR / "manifests"
    models_dir = VL_P4_DIR / "models"
    ds = json.loads((manifests_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    ann = json.loads((manifests_dir / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]
    tm = json.loads((manifests_dir / "training_metrics.json").read_text(encoding="utf-8"))

    val_items = ds["val_cal_v4"]
    is_pos = np.array([it["label"] == "object_01" for it in val_items], dtype=bool)

    ver = SoftCaneVerifierV4.load(models_dir / "soft_verifier_v4.joblib")
    base_mode = tm["verifier_base_mode"]

    calibrated = {}
    val_metrics = {}
    # Calibrate A1_CNN_Keypoint, B1_CNN_Segmentation, B2_CNN_MultiTask, C1_CNN_WithVerifier
    variants_spec = {
        "A1_CNN_Keypoint": (models_dir / tm["variants"]["A1_CNN_Keypoint"]["onnx_parity"]["recommended_onnx"], "A1_CNN_Keypoint", None),
        "B1_CNN_Segmentation": (models_dir / tm["variants"]["B1_CNN_Segmentation"]["onnx_parity"]["recommended_onnx"], "B1_CNN_Segmentation", None),
        "B2_CNN_MultiTask": (models_dir / tm["variants"]["B2_CNN_MultiTask"]["onnx_parity"]["recommended_onnx"], "B2_CNN_MultiTask", None),
        "C1_CNN_WithVerifier": (models_dir / tm["verifier_base_onnx"], base_mode, ver),
    }

    for vname, (onnx_p, mode, v_obj) in variants_spec.items():
        runner = ONNXCaneRunner(onnx_p, mode=mode)
        # Warmup
        _ = runner.predict(load_image_unicode(Path(val_items[0]["abs_path"])))
        raw_recs, _ = _eval_cnn_variant(val_items, ann, runner, 0.5, 0.3, ver=v_obj, keep_masks=False)
        scores = np.array([r["score"] for r in raw_recs], dtype=np.float32)
        tk, tu = _calibrate_thresholds(scores, is_pos, target_far=0.05)
        recs, met = _eval_cnn_variant(val_items, ann, runner, tk, tu, ver=v_obj, keep_masks=True)
        calibrated[vname] = {"onnx": onnx_p.name, "mode": mode, "use_verifier": v_obj is not None, "t_known": tk, "t_uncertain": tu, "sha256": compute_sha256(onnx_p)}
        val_metrics[vname] = met
        c = met["classification"]
        print(f"Phase A val_cal_v4 [{vname}]: tk={tk} tu={tu} recall={c['known_recall']:.3f} far={c['far']:.3f} roc={c['roc_auc']:.3f} p50={c['latency_ms']['p50']:.1f}ms", flush=True)

    # Select C2_FinalCandidate strictly on val_cal_v4 (highest recall under FAR<=5%, tiebreak PR-AUC)
    c2_pick = max(
        ("A1_CNN_Keypoint", "B2_CNN_MultiTask", "C1_CNN_WithVerifier"),
        key=lambda k: (
            1 if val_metrics[k]["classification"]["far"] <= 0.05 else 0,
            val_metrics[k]["classification"]["known_recall"],
            val_metrics[k]["classification"]["pr_auc"],
        ),
    )
    calibrated["C2_FinalCandidate"] = dict(calibrated[c2_pick], selected_from=c2_pick)

    frozen = {
        "frozen": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "calibration_split": "val_cal_v4",
        "input_size": INPUT_SIZE,
        "selected_c2_source": c2_pick,
        "variants": calibrated,
        "verifier_sha256": compute_sha256(models_dir / "soft_verifier_v4.joblib"),
        "val_cal_v4_metrics": val_metrics,
    }
    out_path = manifests_dir / "frozen_config.json"
    out_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Phase A frozen! Selected C2_FinalCandidate={c2_pick} -> {out_path}", flush=True)
    return out_path


def _save_qual_sheets(records: list[dict], items: list[dict], ann: dict, out_dir: Path) -> None:
    item_map = {r["id"]: r for r in items}

    def _render(name: str, sub: list[dict]):
        if not sub:
            return
        cols, cw, ch = 4, 280, 210
        rows = math.ceil(min(16, len(sub)) / cols)
        canvas = np.full((rows * (ch + 24), cols * cw, 3), 240, dtype=np.uint8)
        for idx, r in enumerate(sub[:16]):
            it = item_map[r["id"]]
            img = load_image_unicode(Path(it["abs_path"]))
            vis = img.copy()
            if r["id"] in ann:
                ge = [int(round(v)) for v in ann[r["id"]]["endpoints_xyxy"]]
                cv2.line(vis, (ge[0], ge[1]), (ge[2], ge[3]), (255, 220, 40), 2)
            ep = [int(round(v)) for v in r["endpoints_xyxy"]]
            cv2.line(vis, (ep[0], ep[1]), (ep[2], ep[3]), (0, 255, 0), 3)
            cv2.circle(vis, (ep[0], ep[1]), 5, (255, 0, 0), -1)
            cv2.circle(vis, (ep[2], ep[3]), 5, (0, 0, 255), -1)
            row_i, col_i = divmod(idx, cols)
            x0, y0 = col_i * cw, row_i * (ch + 24) + 24
            sc = min((cw - 4) / vis.shape[1], (ch - 4) / vis.shape[0])
            rw, rh = max(1, int(round(vis.shape[1] * sc))), max(1, int(round(vis.shape[0] * sc)))
            res = cv2.resize(vis, (rw, rh), interpolation=cv2.INTER_AREA)
            canvas[y0 : y0 + rh, x0 : x0 + rw] = res
            cv2.putText(canvas, f"{r['id'][-10:]} {r['decision']} {r['score']:.2f}", (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        save_image_unicode(out_dir / name, canvas, quality=92)

    _render("qual_v4_tp.jpg", [r for r in records if r["true_pos"] and r["decision"] == "KNOWN"])
    _render("qual_v4_fn.jpg", [r for r in records if r["true_pos"] and r["decision"] != "KNOWN"])
    _render("qual_v4_fp.jpg", [r for r in records if not r["true_pos"] and r["decision"] == "KNOWN"])
    _render("qual_v4_tn_handheld.jpg", [r for r in records if not r["true_pos"] and r["decision"] == "UNKNOWN" and r["person_holding_confuser"]])


def run_phase_b() -> Path:
    manifests_dir = VL_P4_DIR / "manifests"
    models_dir = VL_P4_DIR / "models"
    cs_dir = VL_P4_DIR / "cache" / "contact_sheets"
    cs_dir.mkdir(parents=True, exist_ok=True)

    fcfg = json.loads((manifests_dir / "frozen_config.json").read_text(encoding="utf-8"))
    assert fcfg.get("frozen") is True
    assert compute_sha256(models_dir / "soft_verifier_v4.joblib") == fcfg["verifier_sha256"]

    ds = json.loads((manifests_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    ann = json.loads((manifests_dir / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]

    p3_dir = WORKSPACE_ROOT / "VisionMate_VL_P3_AG" / "research" / "vl_p3" / "manifests"
    p3_ds = json.loads((p3_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    p3_ann = json.loads((p3_dir / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]

    ver = SoftCaneVerifierV4.load(models_dir / "soft_verifier_v4.joblib")
    pipe_v3 = _load_v3_a1_pipeline()

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)
    rss_start = proc.memory_info().rss / (1024.0 * 1024.0)

    def _eval_all_presets_on_split(split_items: list[dict], split_ann: dict, label: str) -> tuple[dict, dict]:
        print(f"\nPhase B: Uncached E2E evaluation on {label} (N={len(split_items)})...", flush=True)
        res_met, res_rec = {}, {}
        recs_a0, met_a0 = _eval_v3_a1_uncached(split_items, split_ann, pipe_v3)
        res_met["A0_VL_P3_A1"] = met_a0
        res_rec["A0_VL_P3_A1"] = recs_a0
        print(f"  [A0_VL_P3_A1] Recall={met_a0['classification']['known_recall']:.3f} FAR={met_a0['classification']['far']:.3f} EP_err={met_a0['localization']['mean_endpoint_err_px']:.1f} P50={met_a0['classification']['latency_ms']['p50']:.1f}ms", flush=True)

        for vname in ("A1_CNN_Keypoint", "B1_CNN_Segmentation", "B2_CNN_MultiTask", "C1_CNN_WithVerifier", "C2_FinalCandidate"):
            vc = fcfg["variants"][vname]
            onnx_p = models_dir / vc["onnx"]
            assert compute_sha256(onnx_p) == vc["sha256"], f"Checksum mismatch {vname}"
            runner = ONNXCaneRunner(onnx_p, mode=vc["mode"])
            # Warmup 1 image before timing split
            _ = runner.predict(load_image_unicode(Path(split_items[0]["abs_path"])))
            recs, met = _eval_cnn_variant(
                split_items, split_ann, runner, vc["t_known"], vc["t_uncertain"],
                ver=(ver if vc["use_verifier"] else None), keep_masks=True,
            )
            met["model_load_ms"] = runner.load_ms
            met["model_size_mb"] = round(onnx_p.stat().st_size / (1024.0 * 1024.0), 3)
            res_met[vname] = met
            res_rec[vname] = recs
            c = met["classification"]
            print(f"  [{vname}] Recall={c['known_recall']:.3f} FAR={c['far']:.3f} EP_err={met['localization'].get('mean_endpoint_err_px', 0):.1f} P50={c['latency_ms']['p50']:.1f}ms", flush=True)
        return res_met, res_rec

    fh4_met, fh4_rec = _eval_all_presets_on_split(ds["final_holdout_v4"], ann, "final_holdout_v4 (Brand-New)")
    fh3_met, fh3_rec = _eval_all_presets_on_split(p3_ds["final_holdout_v3"], p3_ann, "final_holdout_v3 (VL-P3 Unseen Benchmark)")

    rss_peak = proc.memory_info().rss / (1024.0 * 1024.0)
    priv_mb = getattr(proc.memory_info(), "private", proc.memory_info().rss) / (1024.0 * 1024.0)
    cpu_pct = proc.cpu_percent(interval=None)

    # Save qualitative contact sheets on both holdouts for C2_FinalCandidate
    _save_qual_sheets(fh4_rec["C2_FinalCandidate"] + fh3_rec["C2_FinalCandidate"], ds["final_holdout_v4"] + p3_ds["final_holdout_v3"], {**ann, **p3_ann}, cs_dir)

    # Per-confuser FAR & McNemar C2 vs A0 on final_holdout_v4 & final_holdout_v3
    def _per_confuser(recs: list[dict]) -> dict:
        out = {}
        negs = [r for r in recs if not r["true_pos"]]
        for cg in sorted({r["confuser_group"] for r in negs}):
            sub = [r for r in negs if r["confuser_group"] == cg]
            fp = sum(1 for r in sub if r["decision"] == "KNOWN")
            out[cg] = {"n": len(sub), "fp": fp, "far": round(fp / max(1, len(sub)), 4)}
        return out

    def _mcnemar(recs_new: list[dict], recs_ref: list[dict]) -> dict:
        pos_new = [1 if r["decision"] == "KNOWN" else 0 for r in recs_new if r["true_pos"]]
        pos_ref = [1 if r["decision"] == "KNOWN" else 0 for r in recs_ref if r["true_pos"]]
        all_new = [1 if (r["decision"] == "KNOWN") == r["true_pos"] else 0 for r in recs_new]
        all_ref = [1 if (r["decision"] == "KNOWN") == r["true_pos"] else 0 for r in recs_ref]
        return {
            "positive_recall": paired_binary_comparison(pos_new, pos_ref),
            "overall_binary_correct": paired_binary_comparison(all_new, all_ref),
        }

    # Ambiguous boundary check (N=26)
    amb_items = ds["ambiguous_cane_v4"]
    c2_cfg = fcfg["variants"]["C2_FinalCandidate"]
    c2_runner = ONNXCaneRunner(models_dir / c2_cfg["onnx"], mode=c2_cfg["mode"])
    amb_recs, _ = _eval_cnn_variant(amb_items, {}, c2_runner, c2_cfg["t_known"], c2_cfg["t_uncertain"], ver=(ver if c2_cfg["use_verifier"] else None))
    amb_summary = {
        "n": len(amb_recs),
        "known": sum(1 for r in amb_recs if r["decision"] == "KNOWN"),
        "uncertain": sum(1 for r in amb_recs if r["decision"] == "UNCERTAIN"),
        "unknown": sum(1 for r in amb_recs if r["decision"] == "UNKNOWN"),
        "non_false_accept_rate": round(sum(1 for r in amb_recs if r["decision"] != "KNOWN") / max(1, len(amb_recs)), 4),
    }

    out = {
        "phase": "PHASE_B_BLIND_EVALUATION",
        "frozen_at": fcfg["frozen_at"],
        "selected_c2_source": fcfg["selected_c2_source"],
        "final_holdout_v4_metrics": fh4_met,
        "final_holdout_v3_metrics": fh3_met,
        "c2_per_confuser_v4": _per_confuser(fh4_rec["C2_FinalCandidate"]),
        "c2_per_confuser_v3": _per_confuser(fh3_rec["C2_FinalCandidate"]),
        "mcnemar_c2_vs_a0_v4": _mcnemar(fh4_rec["C2_FinalCandidate"], fh4_rec["A0_VL_P3_A1"]),
        "mcnemar_c2_vs_a0_v3": _mcnemar(fh3_rec["C2_FinalCandidate"], fh3_rec["A0_VL_P3_A1"]),
        "ambiguous_cane_v4": amb_summary,
        "system_resources": {
            "rss_start_mb": round(rss_start, 2),
            "rss_peak_mb": round(rss_peak, 2),
            "private_bytes_mb": round(priv_mb, 2),
            "cpu_percent": round(cpu_pct, 1),
        },
    }
    bm_path = manifests_dir / "benchmark_metrics.json"
    bm_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nPhase B complete! Saved -> {bm_path}", flush=True)
    return bm_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    args = ap.parse_args()
    if args.phase == "a":
        run_phase_a()
    else:
        run_phase_b()


if __name__ == "__main__":
    main()
