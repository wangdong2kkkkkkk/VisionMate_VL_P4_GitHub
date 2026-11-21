"""Step 2 / Section III: Reproduce and freeze exact VL-P1b historical baseline numbers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
_VL_P1B_ROOT = _WORKTREE_ROOT / "research" / "vl_p1b"
_VL_P1C_ROOT = _WORKTREE_ROOT / "research" / "vl_p1c"

from .common_utils import OBJECT_IDS, compute_sha256, format_count_pct_ci, load_image_unicode  # noqa: E402


def _load_vl_p1b_modules():
    pkg_name = "_vl_p1b_src"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_VL_P1B_ROOT / "src")]
        sys.modules[pkg_name] = pkg
    hn_mod = importlib.import_module(f"{pkg_name}.hard_negative_research")
    pr_mod = importlib.import_module(f"{pkg_name}.proposal_research")
    cp_mod = importlib.import_module(f"{pkg_name}.combined_pipeline")
    return (
        hn_mod.HardNegConfig,
        hn_mod.HardNegativeRecognizer,
        pr_mod.YOLOXNanoExtractor,
        pr_mod.ObjectProposerVL1b,
        cp_mod.CombinedE2EPipeline,
        cp_mod.E2EDecisionConfig,
    )


def reproduce_and_freeze_vl_p1b() -> dict:
    """Re-run frozen VL-P1b recognizers, proposers, and E2E pipelines on vl_p1b holdout (N=214)."""
    HardNegConfig, HardNegativeRecognizer, YOLOXNanoExtractor, ObjectProposerVL1b, CombinedE2EPipeline, E2EDecisionConfig = (
        _load_vl_p1b_modules()
    )
    manifest_p = _VL_P1B_ROOT / "manifests" / "dataset_manifest.json"
    frozen_p = _VL_P1B_ROOT / "manifests" / "frozen_thresholds.json"
    a3_state_p = _VL_P1B_ROOT / "manifests" / "frozen_a3_state.json"

    mf = json.loads(manifest_p.read_text(encoding="utf-8"))
    fr = json.loads(frozen_p.read_text(encoding="utf-8"))
    holdout_items = mf["holdout_full"]

    data_root = _VL_P1B_ROOT
    models_dir = _VL_P1B_ROOT / "models"
    yolox_path = _WORKTREE_ROOT / "models" / "object_detection" / "yolox_nano.onnx"

    roi_imgs = [(it["label_id"], it, load_image_unicode(data_root / it["roi_rel_path"])) for it in holdout_items]
    full_imgs = [(it["label_id"], it, load_image_unicode(data_root / it["full_rel_path"])) for it in holdout_items]

    # Build A0, A1, A3 using exact frozen VL-P1b configs
    cal = fr["calibration_summaries"]

    def _make_rec(mode_key: str) -> HardNegativeRecognizer:
        params = dict(cal[mode_key]["calibration_params"])
        cfg = HardNegConfig(**params)
        rec = HardNegativeRecognizer(models_dir, cfg)
        rec.fit_positives(data_root / "data" / "train_pos")
        rec.fit_hard_negatives(data_root / "data" / "hard_neg_gallery")
        if mode_key == "A3_full_verifier":
            rec.load_frozen_state(a3_state_p)
        return rec

    rec_a0 = _make_rec("A0_baseline_hybrid")
    rec_a1 = _make_rec("A1_neg_prototypes")
    rec_a3 = _make_rec("A3_full_verifier")

    def _eval_roi(rec: HardNegativeRecognizer) -> dict:
        tp = fa = pos_n = neg_n = wc = 0
        for y, _, img in roi_imgs:
            dec = rec.predict(img)
            p = dec["label"] if dec["status"] == "KNOWN" else dec["status"]
            if y == "unknown":
                neg_n += 1
                if p in OBJECT_IDS:
                    fa += 1
            else:
                pos_n += 1
                if p == y:
                    tp += 1
                elif p in OBJECT_IDS:
                    wc += 1
        return {
            "overall_accuracy": format_count_pct_ci(tp + (neg_n - fa), pos_n + neg_n),
            "known_recall": format_count_pct_ci(tp, pos_n),
            "far": format_count_pct_ci(fa, neg_n),
            "wrong_class": format_count_pct_ci(wc, pos_n),
        }

    roi_res = {
        "A0_baseline_hybrid": _eval_roi(rec_a0),
        "A1_neg_prototypes": _eval_roi(rec_a1),
        "A3_full_verifier": _eval_roi(rec_a3),
    }

    yolox = YOLOXNanoExtractor(yolox_path)

    def _eval_prop(mode: str, top_k: int) -> dict:
        prop = ObjectProposerVL1b(yolox, mode=mode, top_k=top_k)
        h30 = h50 = total = wc_h50 = wc_tot = 0
        for y, it, img in full_imgs:
            if y == "unknown":
                continue
            gt = tuple(it["roi_bbox_xywh"])
            boxes, _ = prop.propose(img)
            best_iou = 0.0
            for pb in boxes:
                from .common_utils import compute_iou_xywh
                best_iou = max(best_iou, compute_iou_xywh(tuple(pb.bbox_xywh), gt))
            total += 1
            if best_iou >= 0.30:
                h30 += 1
            if best_iou >= 0.50:
                h50 += 1
            if y == "object_01":
                wc_tot += 1
                if best_iou >= 0.50:
                    wc_h50 += 1
        return {
            "recall_iou30": format_count_pct_ci(h30, total),
            "recall_iou50": format_count_pct_ci(h50, total),
            "white_cane_recall_iou50": format_count_pct_ci(wc_h50, wc_tot),
        }

    prop_res = {
        "B0_vl_p1_proposer__k4": _eval_prop("B0_vl_p1_proposer", 4),
        "B_full_v1b__k4": _eval_prop("B_full_v1b", 4),
        "B_full_v1b__k8": _eval_prop("B_full_v1b", 8),
    }

    def _eval_e2e(prop_mode: str, top_k: int, rec: HardNegativeRecognizer, offset: float) -> dict:
        prop = ObjectProposerVL1b(yolox, mode=prop_mode, top_k=top_k)
        pipe = CombinedE2EPipeline(prop, rec, E2EDecisionConfig(e2e_tau_offset=offset))
        tp = fa = pos_n = neg_n = wc = 0
        for y, _, img in full_imgs:
            dec = pipe.predict(img)
            p = dec["label"] if dec["status"] == "KNOWN" else dec["status"]
            if y == "unknown":
                neg_n += 1
                if p in OBJECT_IDS:
                    fa += 1
            else:
                pos_n += 1
                if p == y:
                    tp += 1
                elif p in OBJECT_IDS:
                    wc += 1
        return {
            "overall_accuracy": format_count_pct_ci(tp + (neg_n - fa), pos_n + neg_n),
            "known_recall": format_count_pct_ci(tp, pos_n),
            "far": format_count_pct_ci(fa, neg_n),
            "wrong_class": format_count_pct_ci(wc, pos_n),
        }

    off = float(fr.get("e2e_tau_offset_c4", 0.0))
    e2e_res = {
        "C1_vl_p1_baseline": _eval_e2e("B0_vl_p1_proposer", 4, rec_a0, 0.0),
        "C4_combined_v1b": _eval_e2e("B_full_v1b", 4, rec_a3, off),
    }

    expected_checks = {
        "C1_acc": e2e_res["C1_vl_p1_baseline"]["overall_accuracy"]["ratio"] == "162/214",
        "C1_rec": e2e_res["C1_vl_p1_baseline"]["known_recall"]["ratio"] == "63/100",
        "C1_far": e2e_res["C1_vl_p1_baseline"]["far"]["ratio"] == "15/114",
        "C4_acc": e2e_res["C4_combined_v1b"]["overall_accuracy"]["ratio"] == "142/214",
        "C4_rec": e2e_res["C4_combined_v1b"]["known_recall"]["ratio"] == "66/100",
        "C4_far": e2e_res["C4_combined_v1b"]["far"]["ratio"] == "38/114",
        "A0_rec": roi_res["A0_baseline_hybrid"]["known_recall"]["ratio"] == "67/100",
        "A0_far": roi_res["A0_baseline_hybrid"]["far"]["ratio"] == "8/114",
        "A1_rec": roi_res["A1_neg_prototypes"]["known_recall"]["ratio"] == "33/100",
        "A1_far": roi_res["A1_neg_prototypes"]["far"]["ratio"] == "0/114",
        "A3_rec": roi_res["A3_full_verifier"]["known_recall"]["ratio"] == "74/100",
        "A3_far": roi_res["A3_full_verifier"]["far"]["ratio"] == "20/114",
        "B0_r50": prop_res["B0_vl_p1_proposer__k4"]["recall_iou50"]["ratio"] == "68/100",
        "B_k4_r50": prop_res["B_full_v1b__k4"]["recall_iou50"]["ratio"] == "60/100",
        "B_k8_r50": prop_res["B_full_v1b__k8"]["recall_iou50"]["ratio"] == "74/100",
        "B_k8_cane": prop_res["B_full_v1b__k8"]["white_cane_recall_iou50"]["ratio"] == "2/20",
    }

    payload = {
        "source_commit": "dacd245",
        "vl_p1b_manifest_sha256": compute_sha256(manifest_p),
        "vl_p1b_frozen_thresholds_sha256": compute_sha256(frozen_p),
        "roi_holdout_214": roi_res,
        "proposal_holdout_214": prop_res,
        "e2e_holdout_214": e2e_res,
        "exact_match_checks": expected_checks,
        "all_exact_matches": all(expected_checks.values()),
    }
    out_path = _VL_P1C_ROOT / "reports" / "vl_p1b_frozen_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    res = reproduce_and_freeze_vl_p1b()
    print(json.dumps(res, indent=2))
    if not res["all_exact_matches"]:
        raise SystemExit(1)
