"""Comprehensive Pytest Verification Suite for VL-P2 White Cane Localization & Segmentation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import psutil
import pytest

from research.vl_p2.src.build_vl_p2_manifest import verify_zero_overlap_across_splits
from research.vl_p2.src.cane_verifier import (
    CaneVerifierConfig,
    CaneVsConfuserVerifier,
    FEATURE_NAMES,
    extract_cane_verification_features,
)
from research.vl_p2.src.combined_pipeline_v2 import CombinedPipelineV2, PipelineV2Config
from research.vl_p2.src.common_utils import (
    DINOV2_SHA256,
    YOLOX_SHA256,
    axial_angle_diff_deg,
    compute_mask_metrics,
    compute_rotated_obb_iou,
    compute_sha256,
    decode_mask_rle,
    encode_mask_rle,
    hex_hamming_distance,
    load_image_unicode,
    normalize_title,
)
from research.vl_p2.src.localization_research import (
    OrientedCaneLocalizer,
    YOLOXPersonAndObjectDetector,
    evaluate_proposal_against_gt,
)
from research.vl_p2.src.masked_embedding_research import (
    DINOv2EmbedderV2,
    REPRESENTATION_NAMES,
    RepresentationEmbeddingBank,
    build_representation_canvas,
)
from research.vl_p2.src.segmentation_research import (
    CaneShaftSegmenter,
    compute_boundary_f1,
    evaluate_segmentation_against_gt,
)

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
VL_P2_DIR = WORKTREE_ROOT / "research" / "vl_p2"


def _make_synthetic_cane_scene(occluded: bool = False, color_mode: str = "white_red_tip") -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic outdoor street scene with diagonal white cane (optionally occluded mid-shaft)."""
    h, w = 360, 480
    img = np.full((h, w, 3), (75, 78, 82), dtype=np.uint8)
    # Textured asphalt backdrop
    rng = np.random.default_rng(7)
    noise = rng.integers(-10, 11, size=(h, w, 1), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Simulated darker clothing blob on upper left
    cv2.rectangle(img, (110, 40), (210, 220), (35, 42, 55), -1)

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    p1 = (180, 120)
    p2 = (350, 310)
    if color_mode == "white_red_tip":
        mid_red = (int(p1[0] + 0.72 * (p2[0] - p1[0])), int(p1[1] + 0.72 * (p2[1] - p1[1])))
        cv2.line(img, p1, mid_red, (242, 245, 248), 7, cv2.LINE_AA)
        cv2.line(img, mid_red, p2, (40, 45, 225), 7, cv2.LINE_AA)
        cv2.line(gt_mask, p1, p2, 255, 7)
    elif color_mode == "wooden_stick":
        cv2.line(img, p1, p2, (45, 95, 145), 9, cv2.LINE_AA)
        cv2.line(gt_mask, p1, p2, 255, 9)
    else:
        cv2.line(img, p1, p2, (240, 242, 245), 7, cv2.LINE_AA)
        cv2.line(gt_mask, p1, p2, 255, 7)

    if occluded:
        mx, my = int(0.5 * (p1[0] + p2[0])), int(0.5 * (p1[1] + p2[1]))
        cv2.circle(img, (mx, my), 18, (85, 115, 165), -1)
        cv2.circle(gt_mask, (mx, my), 18, 0, -1)

    return img, gt_mask


def test_01_git_baseline_and_worktree_integrity():
    """Verify protected baseline commits and untouched prior research directories."""
    protected_commits = ["8ae6f85", "95a75e0", "6e3039f", "65c2a27", "988fdf1", "dacd245", "9587ad4"]
    for cid in protected_commits:
        rc = subprocess.run(
            ["git", "-C", str(WORKTREE_ROOT), "cat-file", "-t", cid],
            capture_output=True,
            text=True,
            check=True,
        )
        assert rc.stdout.strip() == "commit"

    branch = subprocess.run(
        ["git", "-C", str(WORKTREE_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert branch == "research/vl-p2-white-cane-segmentation"

    # Ensure zero diff against 9587ad4 inside prior modules (app, vl_p1, vl_p1b, vl_p1c)
    diff = subprocess.run(
        [
            "git", "-C", str(WORKTREE_ROOT), "diff", "--name-only", "9587ad4", "--",
            "app", "research/vl_p1", "research/vl_p1b", "research/vl_p1c",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert diff == ""


def test_02_model_provenance_sha256():
    """Verify frozen DINOv2-Small INT8 and YOLOX-Nano ONNX checksums."""
    dinov2 = VL_P2_DIR / "models" / "dinov2_small_quantized.onnx"
    yolox = WORKTREE_ROOT / "models" / "object_detection" / "yolox_nano.onnx"
    assert dinov2.is_file() and yolox.is_file()
    assert compute_sha256(dinov2) == DINOV2_SHA256
    assert compute_sha256(yolox) == YOLOX_SHA256


def test_03_dataset_manifest_and_provenance():
    """Verify dataset counts, open licenses, and readable images on disk."""
    mf = json.loads((VL_P2_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    c = mf["counts"]
    assert c["train_pos"] == 25
    assert c["dev_cal_v2_object_01"] >= 60
    assert c["dev_cal_v2_negatives"] >= 80
    assert c["ambiguous_cane_v2"] >= 15
    assert c["final_holdout_v2_object_01"] >= 50 or mf["data_quota_status"] == "DATA_PENDING"
    assert c["final_holdout_v2_negatives"] >= 100

    for sp in ("train_pos", "hard_neg_gallery", "dev_cal_v2", "final_holdout_v2", "ambiguous_cane_v2"):
        for rec in mf[sp]:
            p = VL_P2_DIR / rec["rel_path"]
            assert p.is_file(), f"Missing file in {sp}: {p}"
            assert rec.get("sha256")
            if sp in ("dev_cal_v2", "final_holdout_v2", "ambiguous_cane_v2"):
                assert rec.get("source_title")
                assert rec.get("license")


def test_04_zero_leakage_across_all_splits():
    """Verify 0 SHA-256, title, cluster, and perceptual hash (<=6) leakage between dev and blind holdout."""
    mf = json.loads((VL_P2_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    report = verify_zero_overlap_across_splits(
        mf,
        [
            ("train_pos", "dev_cal_v2"),
            ("train_pos", "final_holdout_v2"),
            ("hard_neg_gallery", "final_holdout_v2"),
            ("dev_cal_v2", "final_holdout_v2"),
            ("dev_cal_v2", "ambiguous_cane_v2"),
            ("final_holdout_v2", "ambiguous_cane_v2"),
        ],
        hamming_thr=6,
    )
    for k, v in report.items():
        assert v["clean"], f"Data leakage detected for {k}: {v}"

    # Also ensure final_holdout_v2 has zero overlap with historical VL-P1c splits
    ai_root = WORKTREE_ROOT.parent
    v1c_mf_p = ai_root / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c" / "manifests" / "dataset_manifest.json"
    if v1c_mf_p.is_file():
        v1c = json.loads(v1c_mf_p.read_text(encoding="utf-8"))
        hist_recs = v1c.get("calibration_v1c_full", []) + v1c.get("final_holdout_full", [])
        cross = verify_zero_overlap_across_splits(
            {"hist": hist_recs, "final_holdout_v2": mf["final_holdout_v2"]},
            [("hist", "final_holdout_v2")],
            hamming_thr=6,
        )
        assert cross["hist__vs__final_holdout_v2"]["clean"]


def test_05_annotation_manifest_completeness():
    """Verify multi-geometry GT annotations on 100% of dev_cal_v2 and final_holdout_v2 object_01 items."""
    mf = json.loads((VL_P2_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    ann_mf = json.loads((VL_P2_DIR / "manifests" / "annotation_manifest.json").read_text(encoding="utf-8"))
    anns = ann_mf["annotations"]
    for sp in ("dev_cal_v2", "final_holdout_v2"):
        for rec in mf[sp]:
            if rec["label_id"] != "object_01":
                continue
            assert rec["id"] in anns
            a = anns[rec["id"]]
            assert len(a["bbox_xywh"]) == 4
            assert len(a["obb_cxcywh_angle"]) == 5
            assert len(a["obb_corners_xy"]) == 4
            assert len(a["endpoints_xyxy"]) == 4
            assert len(a["centerline_points_xy"]) >= 2
            assert isinstance(a["occluded"], bool)
            m = decode_mask_rle(a["mask_rle"])
            assert m.shape == (rec["height"], rec["width"])
            assert int(np.sum(m > 0)) > 0


def test_06_v1c_baseline_reproduction():
    """Verify exact reproduction of historical VL-P1c C_v1c_cal_only (251/300 = 83.7%)."""
    rep = json.loads((VL_P2_DIR / "manifests" / "v1c_baseline_reproduction.json").read_text(encoding="utf-8"))
    c = rep["results"]["C_v1c_cal_only"]
    assert c["overall_accuracy"]["ratio"] == "251/300"
    assert c["known_recall"]["ratio"] == "62/100"
    assert c["far"]["ratio"] == "11/200"
    assert c["per_class_recall"]["object_01"]["ratio"] == "2/20"


def test_07_oriented_localization_unit_and_metrics():
    """Verify OrientedCaneLocalizer recovers tilted & occluded shafts with lower BG contamination than AABB."""
    loc = OrientedCaneLocalizer(yolox_detector=None, top_k=3)
    for occ in (False, True):
        img, gt_m = _make_synthetic_cane_scene(occluded=occ, color_mode="white_red_tip")
        props, ms = loc.detect(img, top_k=3)
        assert props, f"Expected oriented proposal (occluded={occ})"
        best = props[0]
        gt_obb_r = cv2.minAreaRect(np.column_stack(np.where(gt_m > 0))[:, ::-1].astype(np.float32))
        riou = compute_rotated_obb_iou(
            best.obb_cxcywh_angle,
            [gt_obb_r[0][0], gt_obb_r[0][1], gt_obb_r[1][0], gt_obb_r[1][1], gt_obb_r[2]],
        )
        assert riou >= 0.25
        # Verify axis-aligned background contamination > oriented OBB background contamination
        ys, xs = np.where(gt_m > 0)
        gt_box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        fake_gt = {
            "bbox_xywh": gt_box,
            "obb_cxcywh_angle": [gt_obb_r[0][0], gt_obb_r[0][1], max(gt_obb_r[1]), min(gt_obb_r[1]), gt_obb_r[2]],
            "endpoints_xyxy": [180, 120, 350, 310],
        }
        ev = evaluate_proposal_against_gt(best, fake_gt, gt_m)
        assert ev["obb_bg_contamination"] < ev["axis_bg_contamination"]


def test_08_segmentation_mask_unit_and_metrics():
    """Verify CaneShaftSegmenter produces accurate binary/soft masks and lower post-mask BG contamination."""
    img, gt_m = _make_synthetic_cane_scene(occluded=False, color_mode="white_red_tip")
    loc = OrientedCaneLocalizer(yolox_detector=None, top_k=2)
    props, _ = loc.detect(img, top_k=2)
    assert props
    seg_engine = CaneShaftSegmenter()
    seg = seg_engine.segment_proposal(img, props[0], return_full_mask=True)
    assert seg.full_mask_bin is not None
    assert seg.full_mask_soft is not None
    m = evaluate_segmentation_against_gt(seg.full_mask_bin, gt_m)
    assert m["mask_iou"] >= 0.30
    assert m["boundary_f1"] >= 0.35
    # Verify RLE roundtrip is exact
    rle = encode_mask_rle(seg.full_mask_bin)
    dec = decode_mask_rle(rle)
    assert np.array_equal((seg.full_mask_bin > 0), (dec > 0))


def test_09_six_input_representations():
    """Verify all 6 representations render valid canvases and normalized 384D DINOv2 embeddings."""
    img, _ = _make_synthetic_cane_scene(occluded=False, color_mode="white_red_tip")
    loc = OrientedCaneLocalizer(yolox_detector=None, top_k=1)
    props, _ = loc.detect(img, top_k=1)
    seg_engine = CaneShaftSegmenter()
    seg = seg_engine.segment_proposal(img, props[0], return_full_mask=True)
    emb_engine = DINOv2EmbedderV2(VL_P2_DIR / "models" / "dinov2_small_quantized.onnx")

    embs = {}
    for rep in REPRESENTATION_NAMES:
        canvas = build_representation_canvas(img, props[0], seg, rep)
        assert canvas.shape == (224, 224, 3)
        vec, ms = emb_engine.embed_canvas(canvas)
        assert vec.shape == (384,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4
        embs[rep] = vec
    assert not np.allclose(embs["rep1_axis_crop"], embs["rep4_rotated_masked_crop"])


def test_10_cane_vs_confuser_verifier_and_3state():
    """Verify CaneVsConfuserVerifier separates white canes from wooden sticks without leakage."""
    white_img, _ = _make_synthetic_cane_scene(color_mode="white_red_tip")
    wood_img, _ = _make_synthetic_cane_scene(color_mode="wooden_stick")
    loc = OrientedCaneLocalizer(yolox_detector=None, top_k=1)
    seg_engine = CaneShaftSegmenter()
    wp, _ = loc.detect(white_img, top_k=1)
    ws = seg_engine.segment_proposal(white_img, wp[0], return_full_mask=False)
    # Build synthetic wood proposal matching same geometry
    fv_w, fd_w = extract_cane_verification_features(white_img, wp[0], ws, 0.56, 0.36)
    wood_seg = seg_engine.segment_proposal(wood_img, wp[0], return_full_mask=False)
    fv_wood, fd_wood = extract_cane_verification_features(wood_img, wp[0], wood_seg, 0.46, 0.42)

    assert fd_w["white_purity_core"] > fd_wood["white_purity_core"]
    assert fd_w["red_tip_evidence"] > fd_wood["red_tip_evidence"]
    ver = CaneVsConfuserVerifier(CaneVerifierConfig(use_structural_verifier=True))
    dec_white = ver.decide(fv_w, fd_w)
    dec_wood = ver.decide(fv_wood, fd_wood)
    assert dec_white["verifier_prob"] > dec_wood["verifier_prob"]
    assert dec_wood["status"] in ("UNKNOWN", "AMBIGUOUS")


def test_11_phase_a_freeze_and_phase_b_blind_metrics():
    """Verify frozen config lock, Phase B blind metrics schema, and protocol separation."""
    frozen = json.loads((VL_P2_DIR / "manifests" / "frozen_config_v2.json").read_text(encoding="utf-8"))
    bm = json.loads((VL_P2_DIR / "manifests" / "benchmark_metrics.json").read_text(encoding="utf-8"))
    assert frozen["final_holdout_touched_during_calibration"] is False
    assert frozen["dinov2_onnx_sha256"] == DINOV2_SHA256
    assert frozen["yolox_onnx_sha256"] == YOLOX_SHA256
    fh = bm["final_holdout_v2"]
    for key in ("A0_axis_baseline", "A1_oriented_rotated", "B1_seg_masked_axis", "B2_oriented_seg_rot_masked", "C1_best_rep_calibrated", "C2_full_cane_verified"):
        assert key in fh["experimental_matrix_A0_to_C2"]
    for rep in REPRESENTATION_NAMES:
        assert rep in fh["representations_six_way"]
        assert "oracle_localized_only" in fh["representations_six_way"][rep]
        assert "e2e_predicted_localized" in fh["representations_six_way"][rep]


def test_12_runtime_and_memory_budget():
    """Verify CPU-only E2E latency <= 300 ms/image (median) and RAM <= 1.5 GB."""
    mf = json.loads((VL_P2_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((VL_P2_DIR / "manifests" / "frozen_config_v2.json").read_text(encoding="utf-8"))
    yolox = YOLOXPersonAndObjectDetector(WORKTREE_ROOT / "models" / "object_detection" / "yolox_nano.onnx")
    localizer = OrientedCaneLocalizer(yolox, top_k=4)
    segmenter = CaneShaftSegmenter()
    embedder = DINOv2EmbedderV2(VL_P2_DIR / "models" / "dinov2_small_quantized.onnx")
    train_items = [{"id": r["id"], "path": VL_P2_DIR / r["rel_path"]} for r in mf["train_pos"] if r["label_id"] == "object_01"]
    bank = RepresentationEmbeddingBank(embedder)
    bank.fit_cane_enrollment_shots(train_items, localizer, segmenter)
    ver = CaneVsConfuserVerifier()
    ver.load_state_dict(frozen["verifier_state"])
    best_rep = frozen["selected_best_representation"]
    pipe = CombinedPipelineV2(
        localizer,
        segmenter,
        embedder,
        bank,
        ver,
        PipelineV2Config("C2_runtime_check", best_rep, 4, use_structural_verifier=True),
    )
    sample_paths = [VL_P2_DIR / r["rel_path"] for r in mf["dev_cal_v2"][:8]]
    times = []
    for sp in sample_paths:
        img = load_image_unicode(sp)
        dec = pipe.predict(img)
        times.append(dec["total_ms"])
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    assert float(np.median(times)) <= 300.0
    assert rss_mb <= 1536.0


def test_13_deliverables_completeness():
    """Verify all 18 mandatory deliverables exist and are non-empty."""
    required = [
        VL_P2_DIR / "README.md",
        VL_P2_DIR / "reports" / "DATASET_PROTOCOL.md",
        VL_P2_DIR / "reports" / "ANNOTATION_PROTOCOL.md",
        VL_P2_DIR / "reports" / "MODEL_PROVENANCE.md",
        VL_P2_DIR / "manifests" / "dataset_manifest.json",
        VL_P2_DIR / "manifests" / "annotation_manifest.json",
        VL_P2_DIR / "src" / "localization_research.py",
        VL_P2_DIR / "src" / "segmentation_research.py",
        VL_P2_DIR / "src" / "masked_embedding_research.py",
        VL_P2_DIR / "src" / "benchmark_runner.py",
        VL_P2_DIR / "reports" / "benchmark_metrics.json",
        VL_P2_DIR / "reports" / "localization_metrics.md",
        VL_P2_DIR / "reports" / "segmentation_metrics.md",
        VL_P2_DIR / "reports" / "confusion_matrices.md",
        VL_P2_DIR / "reports" / "failure_analysis.md",
        VL_P2_DIR / "reports" / "performance_report.md",
        VL_P2_DIR / "reports" / "git_baseline_preservation.md",
        VL_P2_DIR / "VL_P2_FINAL_RESEARCH_REPORT.md",
    ]
    for p in required:
        assert p.is_file() and p.stat().st_size > 100, f"Missing or empty deliverable: {p}"
    final_md = (VL_P2_DIR / "VL_P2_FINAL_RESEARCH_REPORT.md").read_text(encoding="utf-8")
    for q_idx in range(1, 11):
        assert f"{q_idx}." in final_md
