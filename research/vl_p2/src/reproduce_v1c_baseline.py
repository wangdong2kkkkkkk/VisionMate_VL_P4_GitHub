"""Step 1: Exact reproduction of the VL-P1c baseline (Section III).

Reproduces C_v1c_cal_only (B0@K4 + R5) and C_v1c_full (B_v1c_proposer + R5 + P5 at K=4, 6, 8)
on the historical VL-P1c `final_holdout_full` (N=300, 100 positives incl. 20 object_01, 200 negatives).
Never modifies any file inside `research/vl_p1c/`.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

from .common_utils import (
    DINOV2_SHA256,
    OBJECT_IDS,
    YOLOX_SHA256,
    compute_containment_and_bg_contamination,
    compute_iou_xywh,
    compute_sha256,
    format_count_pct_ci,
    load_image_unicode,
)


def resolve_v1c_assets_dir(worktree_root: Path) -> Path:
    """Locate the sibling VL-P1c research directory containing frozen data & manifests."""
    local_dir = worktree_root / "research" / "vl_p1c"
    if (local_dir / "data" / "train_pos").exists():
        return local_dir
    sibling_dir = worktree_root.parent / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c"
    if (sibling_dir / "data" / "train_pos").exists():
        return sibling_dir
    raise FileNotFoundError(
        f"Cannot locate VL-P1c dataset directory at {local_dir} or {sibling_dir}"
    )


def _import_v1c_package(vl_p1c_dir: Path):
    pkg_parent = str(vl_p1c_dir.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    from vl_p1c.src.calibration_research import CalibratedRecognizerV1c, CalibrationConfigV1c
    from vl_p1c.src.combined_pipeline_v1c import CombinedPipelineV1c, MultiProposalControlConfig
    from vl_p1c.src.proposal_research_v1c import ObjectProposerV1c, YOLOXNanoExtractor
    return (
        CalibratedRecognizerV1c,
        CalibrationConfigV1c,
        CombinedPipelineV1c,
        MultiProposalControlConfig,
        ObjectProposerV1c,
        YOLOXNanoExtractor,
    )


def load_v1c_holdout_records(vl_p1c_dir: Path, split_name: str = "final_holdout_full") -> list[dict]:
    """Load historical VL-P1c full-image split records."""
    mf = json.loads((vl_p1c_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    items = []
    for row in mf.get(split_name, []):
        items.append({
            "id": row["id"],
            "label_id": row["label_id"],
            "confuser_group": row.get("confuser_group") or "positive",
            "source_cluster": row.get("source_cluster") or row.get("source_title") or row["id"],
            "path": vl_p1c_dir / row["full_rel_path"],
            "roi_path": vl_p1c_dir / row["roi_rel_path"],
            "roi_bbox_xywh": tuple(row["roi_bbox_xywh"]) if row.get("roi_bbox_xywh") else None,
            "has_person": bool(row.get("has_person", False)),
            "source_title": row.get("source_title", ""),
            "source_url": row.get("source_url", ""),
            "license": row.get("license", ""),
            "author": row.get("author", ""),
            "sha256": row.get("sha256", ""),
            "ahash": row.get("ahash", ""),
            "dhash": row.get("dhash", ""),
        })
    return items


def reproduce_v1c_baselines(
    worktree_root: Path,
    out_json_path: Path | None = None,
    configs_to_run: tuple[str, ...] = ("C_v1c_cal_only", "C_v1c_full__k4", "C_v1c_full__k6", "C_v1c_full__k8"),
) -> dict:
    """Run exact VL-P1c baseline configs on historical N=300 `final_holdout_full`."""
    vl_p1c_dir = resolve_v1c_assets_dir(worktree_root)
    yolox_path = worktree_root / "models" / "object_detection" / "yolox_nano.onnx"
    dinov2_path = worktree_root / "research" / "vl_p2" / "models" / "dinov2_small_quantized.onnx"
    if not dinov2_path.is_file():
        dinov2_path = vl_p1c_dir / "models" / "dinov2_small_quantized.onnx"

    assert compute_sha256(dinov2_path) == DINOV2_SHA256, "DINOv2 INT8 SHA-256 mismatch"
    assert compute_sha256(yolox_path) == YOLOX_SHA256, "YOLOX-Nano SHA-256 mismatch"

    (
        CalibratedRecognizerV1c,
        CalibrationConfigV1c,
        CombinedPipelineV1c,
        MultiProposalControlConfig,
        ObjectProposerV1c,
        YOLOXNanoExtractor,
    ) = _import_v1c_package(vl_p1c_dir)

    frozen_cfg = json.loads((vl_p1c_dir / "manifests" / "frozen_config.json").read_text(encoding="utf-8"))
    a3_state_path = vl_p1c_dir / "manifests" / "frozen_a3_oof_state.json"

    r5_cfg = CalibrationConfigV1c(**frozen_cfg["roi_configs"]["R5_selective_3state_v1c"])
    rec_r5 = CalibratedRecognizerV1c(vl_p1c_dir / "models", r5_cfg)
    rec_r5.fit_positives(vl_p1c_dir / "data" / "train_pos")
    rec_r5.fit_hard_negatives(vl_p1c_dir / "data" / "hard_neg_gallery")
    rec_r5.load_frozen_state(a3_state_path)
    rec_r5.cfg = r5_cfg

    yolox = YOLOXNanoExtractor(yolox_path)
    holdout_items = load_v1c_holdout_records(vl_p1c_dir, "final_holdout_full")

    pipelines = {}
    if "C_v1c_cal_only" in configs_to_run:
        k_cal = int(frozen_cfg.get("frozen_e2e_k", 4))
        pipelines["C_v1c_cal_only"] = CombinedPipelineV1c(
            ObjectProposerV1c(yolox, "B0_vl_p1_proposer", top_k=k_cal),
            rec_r5,
            MultiProposalControlConfig(enable_control=False, use_thin_corridor_mask=False),
        )
    for k in (4, 6, 8):
        key = f"C_v1c_full__k{k}"
        if key in configs_to_run:
            mp_k = MultiProposalControlConfig(**frozen_cfg["e2e_configs_by_k"][str(k)])
            pipelines[key] = CombinedPipelineV1c(
                ObjectProposerV1c(yolox, "B_v1c_proposer", top_k=k),
                rec_r5,
                mp_k,
            )

    loaded_imgs = {it["id"]: load_image_unicode(it["path"]) for it in holdout_items}
    results = {}
    for pname, pipe in pipelines.items():
        pos_ok = pos_tot = neg_fp = neg_tot = 0
        per_cls = {oid: [0, 0] for oid in OBJECT_IDS}
        obj01_prop_bgs = []
        tot_ms = []
        for it in holdout_items:
            dec = pipe.predict(loaded_imgs[it["id"]])
            tot_ms.append(float(dec["total_ms"]))
            y = it["label_id"]
            if y == "unknown":
                neg_tot += 1
                if dec["status"] == "KNOWN":
                    neg_fp += 1
            else:
                pos_tot += 1
                per_cls[y][1] += 1
                if dec["status"] == "KNOWN" and dec["label"] == y:
                    pos_ok += 1
                    per_cls[y][0] += 1
                if y == "object_01" and it["roi_bbox_xywh"] and dec.get("proposals"):
                    gt = it["roi_bbox_xywh"]
                    best_bg = min(
                        compute_containment_and_bg_contamination(tuple(b), gt)[1]
                        for b in dec["proposals"]
                    )
                    obj01_prop_bgs.append(best_bg)

        overall_k = pos_ok + (neg_tot - neg_fp)
        results[pname] = {
            "n": len(holdout_items),
            "overall_accuracy": format_count_pct_ci(overall_k, len(holdout_items)),
            "known_recall": format_count_pct_ci(pos_ok, pos_tot),
            "far": format_count_pct_ci(neg_fp, neg_tot),
            "per_class_recall": {oid: format_count_pct_ci(per_cls[oid][0], per_cls[oid][1]) for oid in OBJECT_IDS},
            "object_01_mean_min_bg_contamination": round(float(np.mean(obj01_prop_bgs)), 4) if obj01_prop_bgs else 0.0,
            "latency_p50_ms": round(float(np.percentile(tot_ms, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(tot_ms, 95)), 2),
        }

    payload = {
        "schema_version": "vl_p2_v1c_baseline_reproduction_v1",
        "vl_p1c_source_dir": str(vl_p1c_dir),
        "dinov2_sha256": DINOV2_SHA256,
        "yolox_sha256": YOLOX_SHA256,
        "historical_split": "vl_p1c/final_holdout_full (N=300)",
        "results": results,
    }
    if out_json_path is not None:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    wt = Path(__file__).resolve().parents[3]
    out_p = wt / "research" / "vl_p2" / "manifests" / "v1c_baseline_reproduction.json"
    t0 = time.perf_counter()
    res = reproduce_v1c_baselines(wt, out_p)
    elapsed = time.perf_counter() - t0
    for k, v in res["results"].items():
        print(
            f"{k}: Acc={v['overall_accuracy']['formatted']} | "
            f"Rec={v['known_recall']['formatted']} | "
            f"FAR={v['far']['formatted']} | "
            f"obj01={v['per_class_recall']['object_01']['ratio']} | "
            f"P50={v['latency_p50_ms']}ms P95={v['latency_p95_ms']}ms"
        )
    print(f"Elapsed: {elapsed:.1f}s -> saved {out_p}")
