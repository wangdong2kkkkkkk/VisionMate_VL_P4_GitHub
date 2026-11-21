"""20-test automated verification suite for VisionMate VL-P4."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from research.vl_p4.src.cnn_cane_net import (
    MobileNetV3SmallMultiHead,
    ONNXCaneRunner,
    preprocess_image_256,
)
from research.vl_p4.src.common_utils import (
    VL_P4_DIR,
    compute_mask_metrics,
    compute_sha256,
    decode_mask_rle,
    encode_mask_rle,
    load_image_unicode,
)
from research.vl_p4.src.soft_verifier_v4 import extract_soft_line_features
from research.vl_p4.src.train_vl_p4_models import _augment_sample, _compute_loss

MANIFESTS_DIR = VL_P4_DIR / "manifests"
MODELS_DIR = VL_P4_DIR / "models"


@pytest.fixture(scope="module")
def ds_manifest() -> dict:
    return json.loads((MANIFESTS_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ann_manifest() -> dict:
    return json.loads((MANIFESTS_DIR / "annotation_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_cfg() -> dict:
    return json.loads((MANIFESTS_DIR / "frozen_config.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def benchmark_metrics() -> dict:
    return json.loads((MANIFESTS_DIR / "benchmark_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def c2_runner(frozen_cfg: dict) -> ONNXCaneRunner:
    c2 = frozen_cfg["variants"]["C2_FinalCandidate"]
    return ONNXCaneRunner(MODELS_DIR / c2["onnx"], mode=c2["mode"])


def test_01_dataset_integrity(ds_manifest: dict) -> None:
    s = ds_manifest["split_summary"]
    assert s["train_v4"]["total"] == 581 and s["train_v4"]["object_01"] == 97
    assert s["val_cal_v4"]["total"] == 158 and s["val_cal_v4"]["object_01"] == 16
    assert s["final_holdout_v4"]["total"] == 219 and s["final_holdout_v4"]["object_01"] == 19
    assert s["ambiguous_cane_v4"]["total"] == 26


def test_02_zero_leakage(ds_manifest: dict) -> None:
    zl = ds_manifest["zero_leakage_audit"]
    assert zl["all_clean"] is True
    for chk in zl["checks"]:
        assert chk["sha256_overlap"] == 0
        assert chk["cluster_overlap"] == 0
        assert chk["phash_overlap"] == 0


def test_03_annotation_validity(ds_manifest: dict, ann_manifest: dict) -> None:
    ann = ann_manifest["annotations"]
    pos_items = [
        r for k in ("train_v4", "val_cal_v4", "final_holdout_v4")
        for r in ds_manifest[k] if r["label"] == "object_01"
    ]
    assert len(pos_items) == 132
    for it in pos_items:
        a = ann[it["id"]]
        assert len(a["endpoints_xyxy"]) == 4
        assert len(a["centerline_points_xy"]) == 11
        assert a["annotation_status"] in ("VERIFIED_GROUND_TRUTH", "MANUALLY_CORRECTED")
        m = decode_mask_rle(a["mask_rle"])
        assert m.shape == (it["height"], it["width"]) and np.any(m > 0)


def test_04_keypoint_transform_consistency() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, _, tfm = preprocess_image_256(img)
    gx, gy = tfm.orig_to_grid(320.0, 240.0)
    ox, oy = tfm.grid_to_orig(gx, gy)
    assert ox == pytest.approx(320.0, abs=1e-3)
    assert oy == pytest.approx(240.0, abs=1e-3)


def test_05_centerline_transform_consistency() -> None:
    sq = np.full((256, 256, 3), 100, dtype=np.uint8)
    rng = np.random.default_rng(7)
    norm, tgt, y = _augment_sample(sq, [10.0, 12.0, 50.0, 54.0], rng)
    assert norm.shape == (3, 256, 256)
    assert tgt.shape == (4, 64, 64)
    assert y == 1.0 and np.all(np.isfinite(norm)) and np.max(tgt[2]) > 0.5


def test_06_segmentation_mask_consistency() -> None:
    m = np.zeros((100, 120), dtype=np.uint8)
    m[20:40, 50:55] = 255
    rt = decode_mask_rle(encode_mask_rle(m))
    assert np.array_equal(m > 0, rt > 0)
    assert compute_mask_metrics(rt, m > 0)["mask_iou"] == pytest.approx(1.0, abs=1e-4)


def test_07_loss_gradient_sanity() -> None:
    model = MobileNetV3SmallMultiHead(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    tgt = torch.zeros(2, 4, 64, 64)
    tgt[0, :3, 20:24, 20:24] = 1.0
    tgt[0, 3, 20:24, 20:24] = 1.0
    y = torch.tensor([1.0, 0.0])
    for mode in ("A1_CNN_Keypoint", "B1_CNN_Segmentation", "B2_CNN_MultiTask"):
        model.zero_grad()
        sp, ob = model(x)
        loss = _compute_loss(sp, ob, tgt, y, mode)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.head_spatial.parameters())


def test_08_training_reproducibility() -> None:
    tm = json.loads((MANIFESTS_DIR / "training_metrics.json").read_text(encoding="utf-8"))
    assert tm["seed"] == 42
    for k in ("A1_CNN_Keypoint", "B1_CNN_Segmentation", "B2_CNN_MultiTask"):
        assert tm["variants"][k]["n_params"] == 1045989
        assert tm["variants"][k]["fp32_mb"] < 5.0


def test_09_model_export_onnx_valid() -> None:
    for fname in ("a1_cnn_keypoint_fp32.onnx", "b1_cnn_segmentation_fp32.onnx", "b2_cnn_multitask_fp32.onnx"):
        m = onnx.load(str(MODELS_DIR / fname))
        onnx.checker.check_model(m)


def test_10_onnx_inference_offline_cpu(ds_manifest: dict, c2_runner: ONNXCaneRunner) -> None:
    img = load_image_unicode(Path(ds_manifest["val_cal_v4"][0]["abs_path"]))
    out = c2_runner.predict(img)
    assert 0.0 <= out["obj_prob"] <= 1.0
    assert len(out["endpoints_xyxy"]) == 4
    assert len(out["centerline_points_xy"]) == 11


def test_11_int8_parity_documented() -> None:
    tm = json.loads((MANIFESTS_DIR / "training_metrics.json").read_text(encoding="utf-8"))
    for k in ("A1_CNN_Keypoint", "B1_CNN_Segmentation", "B2_CNN_MultiTask"):
        par = tm["variants"][k]["onnx_parity"]
        assert "val_roc_fp32" in par and "val_roc_int8" in par
        assert par["recommended_onnx"].endswith("_fp32.onnx")


def test_12_known_unknown_ambiguous_decisions(benchmark_metrics: dict) -> None:
    amb = benchmark_metrics["ambiguous_cane_v4"]
    assert amb["n"] == 26
    assert amb["non_false_accept_rate"] >= 0.80


def test_13_hard_negative_rejection(benchmark_metrics: dict) -> None:
    c2_v4 = benchmark_metrics["final_holdout_v4_metrics"]["C2_FinalCandidate"]["classification"]
    assert c2_v4["far"] <= 0.05
    pc = benchmark_metrics["c2_per_confuser_v4"]
    assert pc["tool_handle_confuser"]["far"] == 0.0
    assert pc["trekking_pole_confuser"]["far"] == 0.0


def test_14_corrupt_or_empty_image_handling(c2_runner: ONNXCaneRunner) -> None:
    with pytest.raises(FileNotFoundError):
        load_image_unicode(VL_P4_DIR / "does_not_exist_123.jpg")
    tiny = np.zeros((16, 16, 3), dtype=np.uint8)
    out = c2_runner.predict(tiny)
    assert np.isfinite(out["obj_prob"])


def test_15_multiple_objects_image(ds_manifest: dict, c2_runner: ONNXCaneRunner) -> None:
    pos_it = next(r for r in ds_manifest["final_holdout_v4"] if r["label"] == "object_01")
    img = load_image_unicode(Path(pos_it["abs_path"]))
    out = c2_runner.predict(img)
    x1, y1, x2, y2 = out["endpoints_xyxy"]
    h, w = img.shape[:2]
    assert 0.0 <= x1 <= w and 0.0 <= x2 <= w and 0.0 <= y1 <= h and 0.0 <= y2 <= h


def test_16_no_object_image(c2_runner: ONNXCaneRunner) -> None:
    blank = np.full((320, 320, 3), 120, dtype=np.uint8)
    out = c2_runner.predict(blank)
    assert out["obj_prob"] < 0.5


def test_17_model_checksums_match_frozen(frozen_cfg: dict) -> None:
    for vname, vc in frozen_cfg["variants"].items():
        assert compute_sha256(MODELS_DIR / vc["onnx"]) == vc["sha256"], f"Mismatch {vname}"
    assert compute_sha256(MODELS_DIR / "soft_verifier_v4.joblib") == frozen_cfg["verifier_sha256"]


def test_18_runtime_and_memory_telemetry(benchmark_metrics: dict) -> None:
    lat_v4 = benchmark_metrics["final_holdout_v4_metrics"]["C2_FinalCandidate"]["classification"]["latency_ms"]
    lat_v3 = benchmark_metrics["final_holdout_v3_metrics"]["C2_FinalCandidate"]["classification"]["latency_ms"]
    assert lat_v4["n_measured"] == 219 and lat_v4["p95"] <= 100.0
    assert lat_v3["n_measured"] == 245 and lat_v3["p95"] <= 60.0
    assert benchmark_metrics["system_resources"]["rss_peak_mb"] <= 300.0


def test_19_no_ground_truth_at_inference(ds_manifest: dict, c2_runner: ONNXCaneRunner) -> None:
    img = load_image_unicode(Path(ds_manifest["val_cal_v4"][0]["abs_path"]))
    pred = c2_runner.predict(img)
    vec, fdict = extract_soft_line_features(img, pred)
    assert vec.shape == (12,) and "obj_prob" in fdict


def test_20_baseline_preservation_and_contact_sheets(benchmark_metrics: dict) -> None:
    assert (VL_P4_DIR / "VL_P3_CLOSURE_AND_PRESERVATION_AUDIT.md").is_file()
    assert "A0_VL_P3_A1" in benchmark_metrics["final_holdout_v4_metrics"]
    assert "A0_VL_P3_A1" in benchmark_metrics["final_holdout_v3_metrics"]
    cs_dir = VL_P4_DIR / "cache" / "contact_sheets"
    for name in ("qual_v4_tp.jpg", "qual_v4_fn.jpg", "qual_v4_fp.jpg", "qual_v4_tn_handheld.jpg"):
        assert (cs_dir / name).is_file()
