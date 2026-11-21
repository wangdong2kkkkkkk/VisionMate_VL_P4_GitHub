"""Step 1/Section III: Reproduce and freeze exact VL-P1 baseline metrics with extended VL-P1b diagnostics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

# Load sibling research/vl_p1/src under a distinct package name without modifying vl_p1
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
_VL_P1_ROOT = _WORKTREE_ROOT / "research" / "vl_p1"
_VL_P1B_ROOT = _WORKTREE_ROOT / "research" / "vl_p1b"


def _load_vl_p1_modules():
    pkg_name = "_vl_p1_src"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_VL_P1_ROOT / "src")]
        sys.modules[pkg_name] = pkg
    er_mod = importlib.import_module(f"{pkg_name}.embedding_recognizer")
    fp_mod = importlib.import_module(f"{pkg_name}.full_image_pipeline")
    return er_mod.EmbeddingConfig, er_mod.EmbeddingRecognizer, fp_mod.FullImagePipeline, fp_mod.YOLOXProposer


EmbeddingConfig, EmbeddingRecognizer, FullImagePipeline, YOLOXProposer = _load_vl_p1_modules()

from .common_utils import (  # noqa: E402
    OBJECT_IDS,
    compute_containment_and_bg_contamination,
    compute_iou_xywh,
    format_count_pct_ci,
    list_images,
    load_image_unicode,
)


def _load_split(split_dir: Path) -> list[tuple[str, Path, np.ndarray]]:
    out = []
    for lbl in [*OBJECT_IDS, "unknown"]:
        for p in list_images(split_dir / lbl):
            out.append((lbl, p, load_image_unicode(p)))
    return out


def reproduce_and_freeze() -> dict:
    data_dir = _VL_P1_ROOT / "data"
    models_dir = _VL_P1_ROOT / "models"
    yolox_path = _WORKTREE_ROOT / "models" / "object_detection" / "yolox_nano.onnx"
    manifest = json.loads((_VL_P1_ROOT / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    full_meta = {f"{it['label']}/{Path(it['full_rel_path']).name}": it for it in manifest["test"]}

    cal_samples = _load_split(data_dir / "calibration")
    test_roi = _load_split(data_dir / "test")
    test_full = _load_split(data_dir / "test_full")

    # 1. Baseline ROI: dinov2_small_int8__hybrid
    rec = EmbeddingRecognizer(models_dir, EmbeddingConfig(backbone="dinov2_small_int8", scoring_mode="hybrid"))
    rec.fit_directory(data_dir / "train")
    cal_info = rec.calibrate_thresholds([(y, img) for y, _, img in cal_samples], max_far=0.12)

    roi_pos_correct = roi_neg_fa = roi_wrong_cls = roi_shoe_slip = 0
    roi_per_class = {oid: [0, 0] for oid in OBJECT_IDS}
    for y, _, img in test_roi:
        pred = rec.predict(img)
        pcol = pred["label"] or pred["status"]
        if y == "unknown":
            if pcol in OBJECT_IDS:
                roi_neg_fa += 1
        else:
            roi_per_class[y][1] += 1
            if pcol == y:
                roi_pos_correct += 1
                roi_per_class[y][0] += 1
            elif pcol in OBJECT_IDS:
                roi_wrong_cls += 1
                if {y, pcol} == {"object_04", "object_05"}:
                    roi_shoe_slip += 1

    # 2. Baseline E2E: hybrid_prop__dinov2_hybrid
    yolox = YOLOXProposer(yolox_path)
    pipe = FullImagePipeline(rec, yolox_proposer=yolox, proposer_mode="hybrid")
    e2e_pos_correct = e2e_neg_fa = e2e_wrong_cls = e2e_shoe_slip = 0
    hit30 = hit50 = missed = 0
    ious_all = []
    contain_all = []
    bg_contam_all = []
    full_contain_85 = 0
    prop_counts = []
    per_class_e2e = {
        oid: {"correct": 0, "total": 0, "hit30": 0, "hit50": 0, "ious": [], "containments": [], "bg_contams": []}
        for oid in OBJECT_IDS
    }

    for y, p, img in test_full:
        out = pipe.predict(img)
        prop_counts.append(len(out["proposals"]))
        pcol = out["label"] or out["status"]
        gt_box = full_meta.get(f"{y}/{p.name}", {}).get("roi_bbox_xywh")
        if y == "unknown":
            if pcol in OBJECT_IDS:
                e2e_neg_fa += 1
        else:
            cstats = per_class_e2e[y]
            cstats["total"] += 1
            if pcol == y:
                e2e_pos_correct += 1
                cstats["correct"] += 1
            elif pcol in OBJECT_IDS:
                e2e_wrong_cls += 1
                if {y, pcol} == {"object_04", "object_05"}:
                    e2e_shoe_slip += 1

            if gt_box and out["proposals"]:
                gt_t = tuple(gt_box)
                scored = [(compute_iou_xywh(gt_t, tuple(pr)), tuple(pr)) for pr in out["proposals"]]
                best_iou, best_box = max(scored, key=lambda x: x[0])
                max_contain = max(
                    compute_containment_and_bg_contamination(tuple(pr), gt_t)[0]
                    for pr in out["proposals"]
                )
                _, bg_contam_best = compute_containment_and_bg_contamination(best_box, gt_t)
                ious_all.append(best_iou)
                contain_all.append(max_contain)
                bg_contam_all.append(bg_contam_best)
                cstats["ious"].append(best_iou)
                cstats["containments"].append(max_contain)
                cstats["bg_contams"].append(bg_contam_best)
                if best_iou >= 0.30:
                    hit30 += 1
                    cstats["hit30"] += 1
                if best_iou >= 0.50:
                    hit50 += 1
                    cstats["hit50"] += 1
                if max_contain >= 0.85:
                    full_contain_85 += 1
            else:
                missed += 1

    per_class_summary = {}
    for oid, st in per_class_e2e.items():
        per_class_summary[oid] = {
            "e2e_acc": format_count_pct_ci(st["correct"], st["total"]),
            "recall_iou30": format_count_pct_ci(st["hit30"], st["total"]),
            "recall_iou50": format_count_pct_ci(st["hit50"], st["total"]),
            "mean_iou": round(float(np.mean(st["ious"])), 4),
            "mean_containment": round(float(np.mean(st["containments"])), 4),
            "mean_bg_contamination": round(float(np.mean(st["bg_contams"])), 4),
        }

    frozen = {
        "source_commit": "988fdf1",
        "backbone": "dinov2_small_int8",
        "scoring_mode": "hybrid",
        "calibrated_thresholds": cal_info,
        "roi_baseline_vl_p1": {
            "pos_top1_accuracy": format_count_pct_ci(roi_pos_correct, 50),
            "far": format_count_pct_ci(roi_neg_fa, 25),
            "wrong_class_pos": f"{roi_wrong_cls}/50",
            "shoe_slipper_errors": f"{roi_shoe_slip}/20",
            "per_class_recall": {k: format_count_pct_ci(v[0], v[1]) for k, v in roi_per_class.items()},
        },
        "e2e_baseline_vl_p1": {
            "pos_top1_accuracy": format_count_pct_ci(e2e_pos_correct, 50),
            "far": format_count_pct_ci(e2e_neg_fa, 25),
            "wrong_class_pos": f"{e2e_wrong_cls}/50",
            "shoe_slipper_errors": f"{e2e_shoe_slip}/20",
            "proposal_recall_iou30": format_count_pct_ci(hit30, 50),
            "proposal_recall_iou50": format_count_pct_ci(hit50, 50),
            "no_proposal_count": f"{missed}/50",
            "unmatched_iou30": format_count_pct_ci(50 - hit30, 50),
            "unmatched_iou50": format_count_pct_ci(50 - hit50, 50),
            "full_containment_ge_85pct": format_count_pct_ci(full_contain_85, 50),
            "mean_iou": round(float(np.mean(ious_all)), 4),
            "mean_containment": round(float(np.mean(contain_all)), 4),
            "mean_bg_contamination": round(float(np.mean(bg_contam_all)), 4),
            "avg_proposals_per_image": round(float(np.mean(prop_counts)), 2),
            "per_class": per_class_summary,
        },
    }

    # Verify exact numerical parity against Section III specification
    assert roi_pos_correct == 39, f"Expected 39/50 ROI correct, got {roi_pos_correct}"
    assert roi_neg_fa == 3, f"Expected 3/25 ROI FA, got {roi_neg_fa}"
    assert e2e_pos_correct == 38, f"Expected 38/50 E2E correct, got {e2e_pos_correct}"
    assert e2e_neg_fa == 4, f"Expected 4/25 E2E FA, got {e2e_neg_fa}"
    assert hit30 == 46 and hit50 == 43, f"Expected 46/50 @0.3 & 43/50 @0.5, got {hit30}, {hit50}"

    out_path = _VL_P1B_ROOT / "reports" / "vl_p1_frozen_baseline.json"
    out_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    return frozen


if __name__ == "__main__":
    res = reproduce_and_freeze()
    print("FROZEN BASELINE REPRODUCED & VERIFIED:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
