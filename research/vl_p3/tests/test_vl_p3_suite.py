"""20-test comprehensive verification suite for VisionMate VL-P3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.vl_p3.src.cane_verifier_v3 import (
    DeBiasedCaneVerifierV3,
    VERIFIER_FEATURE_NAMES,
    compute_foreground_patch_embedding,
    compute_masked_shaft_crop_embedding,
)
from research.vl_p3.src.common_utils import (
    DINOV2_SHA256,
    VL_P3_DIR,
    WORKTREE_ROOT,
    YOLOX_SHA256,
    compute_mask_metrics,
    compute_sha256,
    decode_mask_rle,
    encode_mask_rle,
    load_image_unicode,
)
from research.vl_p3.src.keypoint_detector import SupervisedKeypointCenterlineDetector
from research.vl_p3.src.shaft_segmenter import SupervisedShaftSegmenter
from research.vl_p3.src.spatial_features import (
    DINOv2SpatialBackbone,
    PatchSemanticHead,
    YOLOXContextExtractor,
    extract_spatial_features,
)

MANIFESTS_DIR = VL_P3_DIR / "manifests"
MODELS_DIR = VL_P3_DIR / "models"


@pytest.fixture(scope="module")
def frozen_cfg() -> dict:
    return json.loads((MANIFESTS_DIR / "frozen_config.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def benchmark_metrics() -> dict:
    return json.loads((MANIFESTS_DIR / "benchmark_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset_manifest() -> dict:
    return json.loads((MANIFESTS_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def annotation_manifest() -> dict:
    return json.loads((MANIFESTS_DIR / "annotation_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shared_models():
    dino = DINOv2SpatialBackbone(MODELS_DIR / "dinov2_small_quantized.onnx")
    yolox = YOLOXContextExtractor(MODELS_DIR / "yolox_nano.onnx")
    patch_head = PatchSemanticHead.from_state_dict(
        json.loads((MODELS_DIR / "patch_semantic_head.json").read_text(encoding="utf-8"))
    )
    kp = SupervisedKeypointCenterlineDetector.load(MODELS_DIR / "keypoint_detector_v3.joblib")
    b1 = SupervisedShaftSegmenter.load(MODELS_DIR / "shaft_segmenter_b1_standalone.joblib")
    b2 = SupervisedShaftSegmenter.load(MODELS_DIR / "shaft_segmenter_b2_kp_guided.joblib")
    ver = DeBiasedCaneVerifierV3.load(MODELS_DIR / "cane_verifier_v3.joblib")
    return dino, yolox, patch_head, kp, b1, b2, ver


def test_01_model_provenance_and_frozen_sha256(frozen_cfg: dict) -> None:
    shas = frozen_cfg["model_sha256"]
    assert compute_sha256(MODELS_DIR / "dinov2_small_quantized.onnx") == DINOV2_SHA256 == shas["dinov2_small_quantized"]
    assert compute_sha256(MODELS_DIR / "yolox_nano.onnx") == YOLOX_SHA256 == shas["yolox_nano"]
    file_map = {
        "patch_semantic_head": MODELS_DIR / "patch_semantic_head.json",
        "keypoint_detector_v3": MODELS_DIR / "keypoint_detector_v3.joblib",
        "shaft_segmenter_b1_standalone": MODELS_DIR / "shaft_segmenter_b1_standalone.joblib",
        "shaft_segmenter_b2_kp_guided": MODELS_DIR / "shaft_segmenter_b2_kp_guided.joblib",
        "cane_verifier_v3": MODELS_DIR / "cane_verifier_v3.joblib",
    }
    for key, path in file_map.items():
        assert path.is_file(), f"Missing model artifact: {path}"
        assert compute_sha256(path) == shas[key]


def test_02_dataset_split_counts(dataset_manifest: dict) -> None:
    counts = dataset_manifest["split_summary"]
    assert len(dataset_manifest["train_v3"]) == counts["train_v3"]["total"] == 183
    assert len(dataset_manifest["val_cal_v3"]) == counts["val_cal_v3"]["total"] == 98
    assert len(dataset_manifest["final_holdout_v3"]) == counts["final_holdout_v3"]["total"] == 245
    assert len(dataset_manifest["ambiguous_cane_v3"]) == counts["ambiguous_cane_v3"]["total"] == 26
    assert sum(1 for r in dataset_manifest["train_v3"] if r["label"] == "object_01") == 34
    assert sum(1 for r in dataset_manifest["val_cal_v3"] if r["label"] == "object_01") == 16
    assert sum(1 for r in dataset_manifest["final_holdout_v3"] if r["label"] == "object_01") == 80
    hh_hold = sum(1 for r in dataset_manifest["final_holdout_v3"] if r.get("person_holding_confuser"))
    assert hh_hold == 125


def test_03_zero_leakage_across_all_splits(dataset_manifest: dict) -> None:
    zl = dataset_manifest["zero_leakage_audit"]
    assert zl["all_clean"] is True
    assert len(zl["checks"]) == 5
    for chk in zl["checks"]:
        assert chk["clean"] is True
        assert chk["sha256_overlap"] == 0
        assert chk["cluster_overlap"] == 0
    dev_items = dataset_manifest["train_v3"] + dataset_manifest["val_cal_v3"]
    hold_items = dataset_manifest["final_holdout_v3"]
    dev_shas = {r["sha256"] for r in dev_items}
    hold_shas = {r["sha256"] for r in hold_items}
    assert len(dev_shas & hold_shas) == 0
    dev_clusters = {r["source_cluster"] for r in dev_items if r.get("source_cluster")}
    hold_clusters = {r["source_cluster"] for r in hold_items if r.get("source_cluster")}
    assert len(dev_clusters & hold_clusters) == 0


def test_04_ground_truth_schema_130_of_130(dataset_manifest: dict, annotation_manifest: dict) -> None:
    ann = annotation_manifest["annotations"]
    pos_items = [
        r for s in ("train_v3", "val_cal_v3", "final_holdout_v3")
        for r in dataset_manifest[s] if r["label"] == "object_01"
    ]
    assert len(pos_items) == 130
    for r in pos_items:
        assert r["id"] in ann, f"Missing annotation for {r['id']}"
        a = ann[r["id"]]
        assert len(a["endpoints_xyxy"]) == 4
        assert len(a["centerline_points_xy"]) == 11
        m = decode_mask_rle(a["mask_rle"])
        assert m.shape == (r["height"], r["width"])
        assert np.any(m > 0)
        assert a["length_px"] > 10.0


def test_05_spatial_feature_grid_36_channels(dataset_manifest: dict, shared_models) -> None:
    dino, yolox, patch_head, *_ = shared_models
    it = dataset_manifest["train_v3"][0]
    img = load_image_unicode(Path(it["abs_path"]))
    bundle = extract_spatial_features(img, dino, yolox, patch_head)
    assert bundle.features_112.shape == (112, 112, 36)
    assert bundle.patches_16.shape == (16, 16, 384)
    assert bundle.cls_emb.shape == (384,)
    assert np.all(np.isfinite(bundle.features_112))


def test_06_keypoint_detector_a1_inference_and_in_bounds(dataset_manifest: dict, shared_models) -> None:
    dino, yolox, patch_head, kp, *_ = shared_models
    it = next(r for r in dataset_manifest["val_cal_v3"] if r["label"] == "object_01")
    img = load_image_unicode(Path(it["abs_path"]))
    h, w = img.shape[:2]
    bundle = extract_spatial_features(img, dino, yolox, patch_head)
    pred = kp.predict(bundle)
    x1, y1, x2, y2 = pred.endpoints_xyxy
    assert 0.0 <= x1 <= w and 0.0 <= x2 <= w
    assert 0.0 <= y1 <= h and 0.0 <= y2 <= h
    assert len(pred.centerline_points_xy) == 11
    assert len(pred.obb_cxcywh_angle) == 5


def test_07_controlled_ablations_recorded() -> None:
    tm = json.loads((MANIFESTS_DIR / "training_metrics.json").read_text(encoding="utf-8"))
    abls = tm["training_ablations"]
    for key in (
        "1_keypoints_only",
        "2_centerline_only",
        "3_keypoints_plus_centerline",
        "4_kp_cl_plus_hard_neg_mining",
        "5_full_aug_and_occlusion",
    ):
        assert key in abls
        assert abls[key]["val_cal_v3"]["mean_endpoint_err_px"] > 0.0
    seg = tm["segmentation_val_cal_v3"]
    assert "B1_standalone" in seg and "B2_kp_guided" in seg


def test_08_shaft_segmenter_b1_b2_and_rle(dataset_manifest: dict, shared_models) -> None:
    dino, yolox, patch_head, kp, b1, b2, _ = shared_models
    it = next(r for r in dataset_manifest["val_cal_v3"] if r["label"] == "object_01")
    img = load_image_unicode(Path(it["abs_path"]))
    bundle = extract_spatial_features(img, dino, yolox, patch_head)
    lp = kp.predict(bundle)
    s1 = b1.predict(bundle)
    s2 = b2.predict(bundle, loc_pred=lp)
    assert s1.mask_orig.shape == img.shape[:2]
    assert s2.mask_orig.shape == img.shape[:2]
    rt = decode_mask_rle(encode_mask_rle(s2.mask_orig))
    assert np.array_equal((rt > 0), (s2.mask_orig > 0))
    mm = compute_mask_metrics(s2.mask_orig, s2.mask_orig > 0)
    assert mm["mask_iou"] == pytest.approx(1.0, abs=1e-4)


def test_09_masked_dinov2_foreground_embeddings(dataset_manifest: dict, shared_models) -> None:
    dino, yolox, patch_head, kp, _, b2, _ = shared_models
    it = next(r for r in dataset_manifest["val_cal_v3"] if r["label"] == "object_01")
    img = load_image_unicode(Path(it["abs_path"]))
    bundle = extract_spatial_features(img, dino, yolox, patch_head)
    lp = kp.predict(bundle)
    sp = b2.predict(bundle, loc_pred=lp)
    pemb = compute_foreground_patch_embedding(bundle, sp, lp)
    cemb, _ = compute_masked_shaft_crop_embedding(img, bundle, sp, lp, dino)
    assert pemb.shape == (384,) and cemb.shape == (384,)
    assert np.linalg.norm(pemb) == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(cemb) == pytest.approx(1.0, abs=1e-3)


def test_10_debiased_verifier_anchor_neutrality(benchmark_metrics: dict, shared_models) -> None:
    *_, ver = shared_models
    base_f = {k: 0.5 for k in VERIFIER_FEATURE_NAMES}
    base_f["gated_hand_person_anchor"] = 0.0
    anchor_f = dict(base_f)
    anchor_f["gated_hand_person_anchor"] = 0.95
    v1 = np.array([base_f[k] for k in VERIFIER_FEATURE_NAMES], dtype=np.float32)
    v2 = np.array([anchor_f[k] for k in VERIFIER_FEATURE_NAMES], dtype=np.float32)
    out1 = ver.score_and_decide(v1, base_f, apply_vetoes=False)
    out2 = ver.score_and_decide(v2, anchor_f, apply_vetoes=False)
    assert abs(out1["raw_prob"] - out2["raw_prob"]) <= 0.10 + 1e-6
    max_hold_delta = max(r["anchor_prob_delta"] for r in benchmark_metrics["c2_per_image_predictions"])
    assert max_hold_delta <= 0.10


def test_11_structural_veto_gates_work(shared_models) -> None:
    *_, ver = shared_models
    good = {
        "margin_crop": 0.25, "pos_crop_top3": 0.80, "margin_patch": 0.25, "pos_patch_top3": 0.80,
        "loc_confidence": 0.90, "loc_kp_confidence": 0.90, "loc_cl_confidence": 0.90,
        "seg_visible_shaft_ratio": 0.90, "seg_white_purity": 0.65, "seg_red_tip": 0.40,
        "seg_width_regularity": 0.85, "seg_edge_straightness": 0.90, "seg_confidence": 0.90,
        "loc_length_ratio": 0.45, "wood_dark_penalty": 0.05, "umbrella_ctx_overlap": 0.0,
        "parallel_dual_pole_score": 0.0, "gated_hand_person_anchor": 0.20,
    }
    cases = [
        ({"loc_length_ratio": 0.04}, "veto_too_short"),
        ({"seg_white_purity": 0.05, "seg_red_tip": 0.0}, "veto_not_white_shaft"),
        ({"wood_dark_penalty": 0.55, "seg_white_purity": 0.15}, "veto_wood_or_dark_confuser"),
        ({"seg_visible_shaft_ratio": 0.10, "loc_confidence": 0.15}, "veto_no_continuous_shaft"),
    ]
    for patch, expected_veto in cases:
        fd = dict(good, **patch)
        vec = np.array([fd[k] for k in VERIFIER_FEATURE_NAMES], dtype=np.float32)
        out = ver.score_and_decide(vec, fd, apply_vetoes=True)
        assert out["decision"] != "KNOWN"
        assert out["veto_reason"] == expected_veto


def test_12_frozen_phase_a_to_b_parity(frozen_cfg: dict, benchmark_metrics: dict) -> None:
    assert frozen_cfg["frozen"] is True
    assert benchmark_metrics["phase"] == "PHASE_B_BLIND_EVALUATION"
    assert benchmark_metrics["frozen_at"] == frozen_cfg["frozen_at"]


def test_13_holdout_c2_far_and_ci95(benchmark_metrics: dict) -> None:
    c2 = benchmark_metrics["final_holdout_v3_presets"]["C2_full_vl_p3_pipeline"]["classification"]
    assert c2["n_total"] == 245 and c2["n_pos"] == 80 and c2["n_neg"] == 165
    assert c2["far"] <= 0.05
    assert c2["known_precision"] >= 0.70
    assert c2["roc_auc"] >= 0.85
    lo, hi = c2["far_ci95"]
    assert 0.0 <= lo <= c2["far"] <= hi <= 1.0


def test_14_holdout_a1_improves_localization_over_a0(benchmark_metrics: dict) -> None:
    a0 = benchmark_metrics["final_holdout_v3_presets"]["A0_v2_lsd_hough_baseline"]
    a1 = benchmark_metrics["final_holdout_v3_presets"]["A1_supervised_kp_centerline"]
    assert a1["localization"]["mean_endpoint_err_px"] < a0["localization"]["mean_endpoint_err_px"]
    assert a1["localization"]["mean_centerline_chamfer_px"] < a0["localization"]["mean_centerline_chamfer_px"]
    assert a1["localization"]["mean_angle_error_deg"] < a0["localization"]["mean_angle_error_deg"]
    assert a1["localization"]["mean_keypoint_both_pdj_10"] > a0["localization"]["mean_keypoint_both_pdj_10"]
    assert a1["occluded_subset_loc"]["mean_endpoint_err_px"] < a0["occluded_subset_loc"]["mean_endpoint_err_px"]


def test_15_handheld_confusers_and_per_group_far(benchmark_metrics: dict) -> None:
    hh = benchmark_metrics["c2_handheld_confuser_subset"]
    assert hh["n_handheld_confusers"] == 125
    assert hh["far"] <= 0.05
    groups = benchmark_metrics["c2_per_confuser_group"]
    assert len(groups) == 7
    assert sum(g["n"] for g in groups.values()) == 165


def test_16_ambiguous_cane_v3_non_false_accept(benchmark_metrics: dict) -> None:
    amb = benchmark_metrics["ambiguous_cane_v3"]
    for preset in (
        "A1_supervised_kp_centerline",
        "B1_supervised_seg_only",
        "B2_kp_guided_supervised_seg",
        "C1_b2_plus_masked_dinov2",
        "C2_full_vl_p3_pipeline",
    ):
        assert amb[preset]["n"] == 26
        assert amb[preset]["non_false_accept_rate"] >= 0.90


def test_17_mcnemar_exact_tests_present(benchmark_metrics: dict) -> None:
    mc = benchmark_metrics["mcnemar_significance_tests"]
    assert len(mc) == 5
    for comp in mc.values():
        assert 0.0 <= comp["positive_recall"]["mcnemar_exact_p"] <= 1.0
        assert 0.0 <= comp["overall_binary_correct"]["mcnemar_exact_p"] <= 1.0


def test_18_full_holdout_latency_and_ram_recorded(benchmark_metrics: dict) -> None:
    sys_res = benchmark_metrics["system_resources"]
    lat = sys_res["c2_full_holdout_latency_ms"]
    assert lat["n_measured"] == 245
    assert lat["p50"] > 0.0 and lat["p95"] >= lat["p50"]
    assert sys_res["rss_peak_mb"] <= 650.0


def test_19_historical_v2_n180_compatibility(benchmark_metrics: dict) -> None:
    h2 = benchmark_metrics["historical_final_holdout_v2_N180"]
    assert h2["n"] == 180
    c2 = h2["c2_vl_p3"]["classification"]
    assert c2["far"] <= 0.05
    assert c2["roc_auc"] >= 0.90


def test_20_historical_vl_p1c_n300_and_contact_sheets(benchmark_metrics: dict) -> None:
    h1c = benchmark_metrics["historical_vl_p1c_N300"]
    assert h1c["n_total"] == 300
    assert h1c["unknown_object_01_far"] <= 0.05
    cs_dir = VL_P3_DIR / "cache" / "contact_sheets"
    for name in ("qual_holdout_tp.jpg", "qual_holdout_fn.jpg", "qual_holdout_fp.jpg", "qual_holdout_tn_handheld.jpg"):
        p = cs_dir / name
        assert p.is_file() and p.stat().st_size > 1000
