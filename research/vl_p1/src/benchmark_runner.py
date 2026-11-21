"""Comprehensive Benchmark & Failure Analysis Runner for VisionMate VL-P1."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import psutil

from .embedding_recognizer import EmbeddingConfig, EmbeddingRecognizer
from .full_image_pipeline import FullImagePipeline, YOLOXProposer, compute_iou_xywh
from .image_utils import OBJECT_IDS, list_images, load_image_unicode
from .orb_recognizer import ORBConfig, ORBRecognizer

PRED_COLS = [*OBJECT_IDS, "UNKNOWN", "AMBIGUOUS", "PROPOSAL_MISSED"]


def _load_split_samples(split_dir: Path) -> list[tuple[str, Path, np.ndarray]]:
    out = []
    for label in [*OBJECT_IDS, "unknown"]:
        for p in list_images(split_dir / label):
            out.append((label, p, load_image_unicode(p)))
    return out


def _compute_quality_metrics(
    records: list[dict],
    manifest_meta: dict[str, dict] | None = None,
) -> dict:
    pos_recs = [r for r in records if r["true_label"] != "unknown"]
    neg_recs = [r for r in records if r["true_label"] == "unknown"]

    n_pos = len(pos_recs)
    n_neg = len(neg_recs)
    correct_pos = sum(1 for r in pos_recs if r["pred_col"] == r["true_label"])
    pos_unknown = sum(1 for r in pos_recs if r["pred_col"] == "UNKNOWN")
    pos_ambig = sum(1 for r in pos_recs if r["pred_col"] == "AMBIGUOUS")
    pos_missed = sum(1 for r in pos_recs if r["pred_col"] == "PROPOSAL_MISSED")
    pos_wrong_cls = n_pos - correct_pos - pos_unknown - pos_ambig - pos_missed

    neg_false_accept = sum(1 for r in neg_recs if r["pred_col"] in OBJECT_IDS)
    neg_unknown = sum(1 for r in neg_recs if r["pred_col"] == "UNKNOWN")
    neg_ambig = sum(1 for r in neg_recs if r["pred_col"] == "AMBIGUOUS")
    neg_missed = sum(1 for r in neg_recs if r["pred_col"] == "PROPOSAL_MISSED")

    per_class = {}
    for oid in OBJECT_IDS:
        cls_recs = [r for r in pos_recs if r["true_label"] == oid]
        tp = sum(1 for r in cls_recs if r["pred_col"] == oid)
        per_class[oid] = {
            "total": len(cls_recs),
            "correct": tp,
            "recall": round(tp / max(1, len(cls_recs)), 4),
        }
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))

    cm: dict[str, dict[str, int]] = {
        t: {p: 0 for p in PRED_COLS} for t in [*OBJECT_IDS, "unknown"]
    }
    for r in records:
        cm[r["true_label"]][r["pred_col"]] += 1

    # Hard confuser pair diagnostics
    shoe_as_slipper = cm["object_05"]["object_04"]
    slipper_as_shoe = cm["object_04"]["object_05"]
    confuser_far = {}
    if manifest_meta:
        by_ctype: dict[str, list[bool]] = {}
        for r in neg_recs:
            meta = manifest_meta.get(f"{r['true_label']}/{r['filename']}", {})
            ctype = meta.get("confuser_type", "general")
            by_ctype.setdefault(ctype, []).append(r["pred_col"] in OBJECT_IDS)
        for ctype, vals in sorted(by_ctype.items()):
            confuser_far[ctype] = {
                "total": len(vals),
                "false_accepted": sum(vals),
                "far": round(sum(vals) / max(1, len(vals)), 4),
            }

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "top1_pos_accuracy": round(correct_pos / max(1, n_pos), 4),
        "macro_recall": round(macro_recall, 4),
        "per_class": per_class,
        "pos_unknown_rate": round(pos_unknown / max(1, n_pos), 4),
        "pos_ambiguous_rate": round(pos_ambig / max(1, n_pos), 4),
        "pos_proposal_missed_rate": round(pos_missed / max(1, n_pos), 4),
        "pos_wrong_class_count": pos_wrong_cls,
        "neg_far": round(neg_false_accept / max(1, n_neg), 4),
        "neg_unknown_rate": round(neg_unknown / max(1, n_neg), 4),
        "neg_ambiguous_rate": round(neg_ambig / max(1, n_neg), 4),
        "neg_proposal_missed_rate": round(neg_missed / max(1, n_neg), 4),
        "confusion_matrix": cm,
        "hard_pairs": {
            "shoe_predicted_as_slipper": shoe_as_slipper,
            "slipper_predicted_as_shoe": slipper_as_shoe,
            "neg_confuser_breakdown": confuser_far,
        },
        "failure_cases": [
            r for r in records
            if (r["true_label"] != "unknown" and r["pred_col"] != r["true_label"])
            or (r["true_label"] == "unknown" and r["pred_col"] in OBJECT_IDS)
        ],
    }


def _calibrate_orb_on_cal(rec: ORBRecognizer, cal_samples: list[tuple[str, Path, np.ndarray]]) -> dict:
    pre = [(y, p, rec.predict(img)) for y, p, img in cal_samples]
    best = (0.0, 10, 0.45, 4)
    n_pos = sum(1 for y, _, _ in pre if y != "unknown")
    n_neg = sum(1 for y, _, _ in pre if y == "unknown")
    for min_inl in (4, 6, 8, 10, 12, 15):
        for min_ratio in (0.15, 0.25, 0.35, 0.45):
            for margin in (2, 3, 4):
                tp = fa = 0
                for y, _, res in pre:
                    cands = res["candidates"]
                    top = cands[0] if cands else {"inliers": 0, "ratio": 0.0, "id": None}
                    sec = cands[1]["inliers"] if len(cands) > 1 else 0
                    if top["inliers"] < min_inl or top["ratio"] < min_ratio:
                        pred = "UNKNOWN"
                    elif (top["inliers"] - sec) < margin:
                        pred = "AMBIGUOUS"
                    else:
                        pred = top["id"]
                    if y == "unknown" and pred in OBJECT_IDS:
                        fa += 1
                    elif y != "unknown" and pred == y:
                        tp += 1
                far = fa / max(1, n_neg)
                acc = tp / max(1, n_pos)
                obj = acc - max(0.0, far - 0.12) * 3.0 - 0.5 * far
                if obj > best[0]:
                    best = (obj, min_inl, min_ratio, margin)
    rec.cfg.min_inliers = best[1]
    rec.cfg.min_inlier_ratio = best[2]
    rec.cfg.min_margin_inliers = best[3]
    return {"min_inliers": best[1], "min_inlier_ratio": best[2], "min_margin_inliers": best[3]}


def _benchmark_perf_orb(train_dir: Path, sample_img: np.ndarray, adaptive: bool, n_runs: int = 10) -> dict:
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / (1024 * 1024)
    rec = ORBRecognizer(ORBConfig(adaptive_narrow_roi=adaptive))
    enroll_info = rec.fit_directory(train_dir)
    # Warmup
    for _ in range(2):
        rec.predict(sample_img)
    feat_times, match_times = [], []
    peak_rss = rss_before
    proc.cpu_percent(interval=None)
    for _ in range(n_runs):
        out = rec.predict(sample_img)
        feat_times.append(out["feat_ms"])
        match_times.append(out["match_ms"])
        peak_rss = max(peak_rss, proc.memory_info().rss / (1024 * 1024))
    cpu_pct = proc.cpu_percent(interval=None)
    return {
        "model_load_ms": 0.0,
        "model_size_mb": 0.0,
        "enroll_total_ms": enroll_info["enroll_total_ms"],
        "enroll_per_sample_ms": round(enroll_info["enroll_total_ms"] / 25.0, 3),
        "gallery_db_kb": 0.0,
        "feat_p50_ms": round(float(np.percentile(feat_times, 50)), 3),
        "feat_p95_ms": round(float(np.percentile(feat_times, 95)), 3),
        "match_p50_ms": round(float(np.percentile(match_times, 50)), 3),
        "match_p95_ms": round(float(np.percentile(match_times, 95)), 3),
        "peak_ram_mb": round(peak_rss, 2),
        "ram_delta_mb": round(peak_rss - rss_before, 2),
        "cpu_percent": round(cpu_pct, 1),
    }


def _benchmark_perf_embedding(
    models_dir: Path,
    train_dir: Path,
    backbone: str,
    scoring_mode: str,
    sample_img: np.ndarray,
    n_runs: int = 30,
) -> dict:
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / (1024 * 1024)
    rec = EmbeddingRecognizer(models_dir, EmbeddingConfig(backbone=backbone, scoring_mode=scoring_mode))
    enroll_info = rec.fit_directory(train_dir)
    for _ in range(5):
        rec.predict(sample_img)
    feat_times, match_times = [], []
    peak_rss = proc.memory_info().rss / (1024 * 1024)
    proc.cpu_percent(interval=None)
    for _ in range(n_runs):
        out = rec.predict(sample_img)
        feat_times.append(out["feat_ms"])
        match_times.append(out["match_ms"])
        peak_rss = max(peak_rss, proc.memory_info().rss / (1024 * 1024))
    cpu_pct = proc.cpu_percent(interval=None)
    return {
        "model_load_ms": round(rec.load_time_ms, 2),
        "model_size_mb": round(rec.model_disk_size_bytes() / (1024 * 1024), 3),
        "enroll_total_ms": enroll_info["enroll_total_ms"],
        "enroll_per_sample_ms": enroll_info["enroll_per_sample_ms"],
        "gallery_db_kb": round(enroll_info["gallery_db_bytes"] / 1024.0, 2),
        "feat_p50_ms": round(float(np.percentile(feat_times, 50)), 3),
        "feat_p95_ms": round(float(np.percentile(feat_times, 95)), 3),
        "match_p50_ms": round(float(np.percentile(match_times, 50)), 4),
        "match_p95_ms": round(float(np.percentile(match_times, 95)), 4),
        "peak_ram_mb": round(peak_rss, 2),
        "ram_delta_mb": round(peak_rss - rss_before, 2),
        "cpu_percent": round(cpu_pct, 1),
    }


def run_all_benchmarks(base_dir: Path) -> dict:
    data_dir = base_dir / "data"
    train_dir = data_dir / "train"
    cal_dir = data_dir / "calibration"
    test_roi_dir = data_dir / "test"
    test_full_dir = data_dir / "test_full"
    models_dir = base_dir / "models"
    yolox_path = base_dir.parent.parent / "models" / "object_detection" / "yolox_nano.onnx"

    manifest_path = base_dir / "manifests" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    test_meta_by_key = {
        f"{item['label']}/{Path(item['roi_rel_path']).name}": item
        for item in manifest.get("test", [])
    }
    full_meta_by_key = {
        f"{item['label']}/{Path(item['full_rel_path']).name}": item
        for item in manifest.get("test", [])
    }

    cal_samples = _load_split_samples(cal_dir)
    test_roi_samples = _load_split_samples(test_roi_dir)
    test_full_samples = _load_split_samples(test_full_dir)
    sample_img = test_roi_samples[0][2]

    task1_results = []
    perf_results = []
    trained_models: dict[str, EmbeddingRecognizer] = {}

    # 1. Method A: ORB Raw & ORB Adaptive
    for name, adaptive in [("orb_raw", False), ("orb_adaptive", True)]:
        print(f"[Benchmark] Running Method A: {name}...", flush=True)
        orb = ORBRecognizer(ORBConfig(adaptive_narrow_roi=adaptive))
        orb.fit_directory(train_dir)
        cal_info = _calibrate_orb_on_cal(orb, cal_samples)
        recs = []
        for y, p, img in test_roi_samples:
            pred = orb.predict(img)
            col = pred["label"] or pred["status"]
            recs.append({
                "filename": p.name,
                "true_label": y,
                "pred_col": col,
                "score": pred["score"],
                "margin": pred["margin"],
                "top_candidates": pred["candidates"],
            })
        q = _compute_quality_metrics(recs, test_meta_by_key)
        task1_results.append({"method": name, "calibration": cal_info, "quality": q})
        perf = _benchmark_perf_orb(train_dir, sample_img, adaptive)
        perf["method"] = name
        perf_results.append(perf)
        print(f"  -> {name}: pos_acc={q['top1_pos_accuracy']*100:.1f}%, far={q['neg_far']*100:.1f}%", flush=True)

    # Also test ORB on dev_probes/object_01/probe01.png specifically
    probe_path = base_dir / "dev_probes" / "object_01" / "probe01.png"
    probe_results = {}
    if probe_path.is_file():
        probe_img = load_image_unicode(probe_path)
        for name, adaptive in [("orb_raw", False), ("orb_adaptive", True)]:
            o = ORBRecognizer(ORBConfig(adaptive_narrow_roi=adaptive))
            o.fit_directory(train_dir)
            probe_results[name] = o.predict(probe_img)

    # 2. Method B: Embeddings across backbones and scoring modes
    cal_pairs = [(y, img) for y, _, img in cal_samples]
    model_grid = [
        ("mobilenetv3_small", "1nn_max"),
        ("mobilenetv3_small", "prototype"),
        ("mobilenetv3_small", "hybrid"),
        ("efficientnet_b0", "1nn_max"),
        ("efficientnet_b0", "prototype"),
        ("efficientnet_b0", "hybrid"),
        ("dinov2_small_int8", "1nn_max"),
        ("dinov2_small_int8", "prototype"),
        ("dinov2_small_int8", "hybrid"),
        ("ensemble_mv3_dino", "hybrid"),
    ]
    for backbone, mode in model_grid:
        tag = f"{backbone}__{mode}"
        rec = EmbeddingRecognizer(models_dir, EmbeddingConfig(backbone=backbone, scoring_mode=mode))
        rec.fit_directory(train_dir)
        cal_info = rec.calibrate_thresholds(cal_pairs, max_far=0.12)
        recs = []
        for y, p, img in test_roi_samples:
            pred = rec.predict(img)
            col = pred["label"] or pred["status"]
            recs.append({
                "filename": p.name,
                "true_label": y,
                "pred_col": col,
                "score": pred["score"],
                "margin": pred["margin"],
                "top_candidates": pred["candidates"],
            })
        q = _compute_quality_metrics(recs, test_meta_by_key)
        task1_results.append({"method": tag, "backbone": backbone, "scoring_mode": mode, "calibration": cal_info, "quality": q})
        print(f"  [Method B] {tag}: pos_acc={q['top1_pos_accuracy']*100:.1f}%, far={q['neg_far']*100:.1f}%", flush=True)
        trained_models[tag] = rec
        if probe_path.is_file() and mode == "hybrid":
            probe_results[tag] = rec.predict(load_image_unicode(probe_path))

    for backbone in ("mobilenetv3_small", "efficientnet_b0", "dinov2_small_int8", "ensemble_mv3_dino"):
        perf = _benchmark_perf_embedding(models_dir, train_dir, backbone, "hybrid", sample_img)
        perf["method"] = f"{backbone}__hybrid"
        perf_results.append(perf)
        print(f"  [Perf] {backbone}: feat_p50={perf['feat_p50_ms']}ms, peak_ram={perf['peak_ram_mb']}MB", flush=True)

    # 3. Task 2: Full-Image End-to-End Evaluation (YOLOX Strict/Low vs Hybrid Region Proposer)
    yolox = YOLOXProposer(yolox_path) if yolox_path.is_file() else None
    task2_configs = [
        ("yolox_strict__dinov2_hybrid", "yolox_strict", trained_models["dinov2_small_int8__hybrid"]),
        ("yolox_low__dinov2_hybrid", "yolox_low", trained_models["dinov2_small_int8__hybrid"]),
        ("hybrid_prop__mobilenetv3_hybrid", "hybrid", trained_models["mobilenetv3_small__hybrid"]),
        ("hybrid_prop__dinov2_hybrid", "hybrid", trained_models["dinov2_small_int8__hybrid"]),
        ("hybrid_prop__ensemble_hybrid", "hybrid", trained_models["ensemble_mv3_dino__hybrid"]),
    ]
    task2_results = []
    for tag, prop_mode, rec in task2_configs:
        pipe = FullImagePipeline(rec, yolox_proposer=yolox, proposer_mode=prop_mode)
        recs = []
        ious_pos = []
        hit_iou30 = hit_iou50 = missed_pos = 0
        lat_ms = []
        for y, p, img in test_full_samples:
            out = pipe.predict(img)
            lat_ms.append(out["total_ms"])
            col = out["label"] or out["status"]
            gt_box = full_meta_by_key.get(f"{y}/{p.name}", {}).get("roi_bbox_xywh")
            best_iou = 0.0
            if y != "unknown" and gt_box and out["proposals"]:
                best_iou = max(compute_iou_xywh(tuple(gt_box), tuple(pr)) for pr in out["proposals"])
                ious_pos.append(best_iou)
                if best_iou >= 0.30:
                    hit_iou30 += 1
                if best_iou >= 0.50:
                    hit_iou50 += 1
            elif y != "unknown" and not out["proposals"]:
                missed_pos += 1
            recs.append({
                "filename": p.name,
                "true_label": y,
                "pred_col": col,
                "score": out["score"],
                "margin": out["margin"],
                "selected_source": out["selected_source"],
                "best_proposal_iou": round(best_iou, 4),
            })
        q = _compute_quality_metrics(recs, full_meta_by_key)
        print(f"  [Task 2] {tag}: recall@0.3={hit_iou30}/{max(1, sum(1 for r in recs if r['true_label']!='unknown'))}, pos_acc={q['top1_pos_accuracy']*100:.1f}%, far={q['neg_far']*100:.1f}%", flush=True)
        n_pos_with_gt = max(1, sum(1 for r in recs if r["true_label"] != "unknown"))
        task2_results.append({
            "pipeline": tag,
            "proposer_mode": prop_mode,
            "proposal_recall_iou30": round(hit_iou30 / n_pos_with_gt, 4),
            "proposal_recall_iou50": round(hit_iou50 / n_pos_with_gt, 4),
            "proposal_missed_pos": missed_pos,
            "mean_best_proposal_iou": round(float(np.mean(ious_pos)) if ious_pos else 0.0, 4),
            "e2e_p50_ms": round(float(np.percentile(lat_ms, 50)), 2),
            "e2e_p95_ms": round(float(np.percentile(lat_ms, 95)), 2),
            "quality": q,
        })

    report = {
        "task1_roi_results": task1_results,
        "task2_full_image_results": task2_results,
        "performance_cpu": perf_results,
        "dev_probe_01_results": probe_results,
    }
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "benchmark_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown_reports(base_dir, report)
    return report


def _write_markdown_reports(base_dir: Path, report: dict) -> None:
    reports_dir = base_dir / "reports"

    # 1. Summary Benchmark Markdown
    lines = [
        "# Kết Quả Benchmark Định Lượng — VisionMate VL-P1",
        "",
        "## 1. Task 1 — Nhận diện trên ROI đã cắt (`data/test/`, 50 dương + 25 âm)",
        "| Phương pháp | Threshold $(\\tau, \\delta)$ | Top-1 Pos Acc | Macro Recall | Obj01 Gậy | Obj02 Chìa | Obj03 Ví | Obj04 Dép | Obj05 Giày | FAR (Âm tính) | Pos UNKNOWN | Pos AMBIGUOUS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["task1_roi_results"]:
        q = row["quality"]
        pc = q["per_class"]
        cal = row["calibration"]
        thr = f"inliers={cal.get('min_inliers')}, r={cal.get('min_inlier_ratio')}" if "min_inliers" in cal else f"{cal['tau_known']:.3f}, {cal['delta_margin']:.3f}"
        lines.append(
            f"| `{row['method']}` | {thr} | {q['top1_pos_accuracy']*100:.1f}% | {q['macro_recall']*100:.1f}% | "
            f"{pc['object_01']['recall']*100:.0f}% | {pc['object_02']['recall']*100:.0f}% | {pc['object_03']['recall']*100:.0f}% | "
            f"{pc['object_04']['recall']*100:.0f}% | {pc['object_05']['recall']*100:.0f}% | "
            f"{q['neg_far']*100:.1f}% | {q['pos_unknown_rate']*100:.1f}% | {q['pos_ambiguous_rate']*100:.1f}% |"
        )

    lines.extend([
        "",
        "## 2. Task 2 — Nhận diện trên Ảnh Gốc Chưa Cắt (`data/test_full/`, 50 dương + 25 âm)",
        "| Pipeline (Proposer + Classifier) | Proposal Recall@0.3 | Proposal Recall@0.5 | Miss Proposal | Mean IoU | Top-1 Pos Acc | Macro Recall | FAR (Âm) | E2E P50 (ms) | E2E P95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["task2_full_image_results"]:
        q = row["quality"]
        lines.append(
            f"| `{row['pipeline']}` | {row['proposal_recall_iou30']*100:.1f}% | {row['proposal_recall_iou50']*100:.1f}% | "
            f"{row['proposal_missed_pos']}/{q['n_pos']} | {row['mean_best_proposal_iou']:.3f} | "
            f"{q['top1_pos_accuracy']*100:.1f}% | {q['macro_recall']*100:.1f}% | {q['neg_far']*100:.1f}% | "
            f"{row['e2e_p50_ms']:.1f} | {row['e2e_p95_ms']:.1f} |"
        )

    lines.extend([
        "",
        "## 3. Hiệu Năng Phần Cứng CPU (Windows 11, Intel i5-1135G7)",
        "| Mô hình / Phương pháp | Model (MB) | Load (ms) | Feat P50 (ms) | Feat P95 (ms) | Match P50 (ms) | Enroll/ảnh (ms) | DB 25 ảnh (KB) | Peak RAM (MB) | ΔRAM (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for p in report["performance_cpu"]:
        lines.append(
            f"| `{p['method']}` | {p['model_size_mb']:.2f} | {p['model_load_ms']:.1f} | "
            f"{p['feat_p50_ms']:.2f} | {p['feat_p95_ms']:.2f} | {p['match_p50_ms']:.3f} | "
            f"{p['enroll_per_sample_ms']:.2f} | {p['gallery_db_kb']:.1f} | {p['peak_ram_mb']:.1f} | {p['ram_delta_mb']:.1f} |"
        )
    (reports_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2. Confusion matrices & failure analysis Markdown
    cm_lines = ["# Ma Trận Nhầm Lẫn & Phân Tích Lỗi — VL-P1", ""]
    for row in report["task1_roi_results"]:
        cm = row["quality"]["confusion_matrix"]
        hp = row["quality"]["hard_pairs"]
        cm_lines.extend([
            f"## {row['method']}",
            "| GT \\ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for t in [*OBJECT_IDS, "unknown"]:
            r = cm[t]
            cm_lines.append(
                f"| `{t}` | {r['object_01']} | {r['object_02']} | {r['object_03']} | {r['object_04']} | {r['object_05']} | {r['UNKNOWN']} | {r['AMBIGUOUS']} |"
            )
        cm_lines.append(
            f"- Giày nhầm thành Dép: {hp['shoe_predicted_as_slipper']} | Dép nhầm thành Giày: {hp['slipper_predicted_as_shoe']}"
        )
        if hp["neg_confuser_breakdown"]:
            cm_lines.append(f"- FAR theo nhóm confuser âm tính: `{json.dumps(hp['neg_confuser_breakdown'], ensure_ascii=False)}`")
        cm_lines.append("")
    (reports_dir / "confusion_matrices.md").write_text("\n".join(cm_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    run_all_benchmarks(base)
    print("Benchmark finished cleanly.")
