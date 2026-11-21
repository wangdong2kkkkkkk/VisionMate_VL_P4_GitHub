"""VL-P1c Complete 20-Group Mandatory Verification Suite (Section X)."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys

import cv2
import numpy as np
import pytest

VL_P1C_DIR = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = VL_P1C_DIR.parents[1]
VL_P1B_DIR = WORKTREE_ROOT / "research" / "vl_p1b"
VL_P1_DIR = WORKTREE_ROOT / "research" / "vl_p1"
MODELS_DIR = VL_P1C_DIR / "models"
YOLOX_PATH = WORKTREE_ROOT / "models" / "object_detection" / "yolox_nano.onnx"
DINO_PATH = MODELS_DIR / "dinov2_small_quantized.onnx"

MANIFEST_PATH = VL_P1C_DIR / "manifests" / "dataset_manifest.json"
FROZEN_CFG_PATH = VL_P1C_DIR / "manifests" / "frozen_config.json"
FROZEN_STATE_PATH = VL_P1C_DIR / "manifests" / "frozen_a3_oof_state.json"
CAL_METRICS_PATH = VL_P1C_DIR / "reports" / "calibration_metrics.json"
BENCHMARK_PATH = VL_P1C_DIR / "reports" / "benchmark_metrics.json"
V1B_FROZEN_BASELINE = VL_P1C_DIR / "reports" / "vl_p1b_frozen_baseline.json"

sys.path.insert(0, str(VL_P1C_DIR))

from src.calibration_research import (  # noqa: E402
    CalibratedRecognizerV1c,
    CalibrationConfigV1c,
    make_group_aware_stratified_folds,
)
from src.combined_pipeline_v1c import CombinedPipelineV1c, MultiProposalControlConfig  # noqa: E402
from src.common_utils import (  # noqa: E402
    OBJECT_IDS,
    compute_ahash,
    compute_dhash,
    compute_iou_xywh,
    compute_sha256,
    hex_hamming_distance,
    list_images,
    load_image_unicode,
)
from src.proposal_research_v1c import (  # noqa: E402
    ObjectProposerV1c,
    ProposalRankerV1c,
    YOLOXNanoExtractor,
    _compute_saliency_map,
)
from src.proposal_types import ProposalBox  # noqa: E402
from src.thin_object_proposer import (  # noqa: E402
    detect_oriented_thin_cane_candidates,
    evaluate_thin_object_metrics,
    extract_thin_aware_roi,
)


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return res.stdout.strip()


# ---------------------------------------------------------------------------
# Group 1: Git Worktree & Baseline Preservation Integrity
# ---------------------------------------------------------------------------
def test_01_git_worktree_and_baseline_integrity() -> None:
    branch = _git(WORKTREE_ROOT, "branch", "--show-current")
    assert branch == "research/vl-p1c-calibration-multiproposal"
    dirty_tracked = _git(WORKTREE_ROOT, "status", "--porcelain")
    for line in dirty_tracked.splitlines():
        if not line.strip():
            continue
        assert "research/vl_p1c" in line, f"Unexpected path modified outside vl_p1c: {line}"

    parent_ws = WORKTREE_ROOT.parent
    for wt_name, expected_prefix in [
        ("VisionMateGlasses", "65c2a27"),
        ("VisionMate_VL_P1_AG", "988fdf1"),
        ("VisionMate_VL_P1b_AG", "dacd245"),
    ]:
        wt = parent_ws / wt_name
        assert wt.is_dir()
        assert _git(wt, "rev-parse", "HEAD").startswith(expected_prefix)
        assert _git(wt, "status", "--short") == ""


# ---------------------------------------------------------------------------
# Group 2: Strict Zero-Overlap Across All Splits & Historical Holdouts
# ---------------------------------------------------------------------------
def test_02_strict_zero_overlap_across_all_splits() -> None:
    assert MANIFEST_PATH.is_file(), f"Missing {MANIFEST_PATH}"
    mf = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert mf["schema_version"] == "vl_p1c_v1"
    assert mf["zero_overlap_verified"] is True

    splits = [
        ("train_pos", mf["train_pos"]),
        ("hard_neg_gallery", mf["hard_neg_gallery"]),
        ("calibration_v1c", mf["calibration_v1c_full"]),
        ("final_holdout_v1c", mf["final_holdout_full"]),
        ("ambiguous_boundary_v1c", mf["ambiguous_boundary_v1c"]),
    ]
    shas: dict[str, set[str]] = {}
    clusters: dict[str, set[str]] = {}
    for sname, rows in splits:
        shas[sname] = {r["sha256"] for r in rows}
        clusters[sname] = {r.get("cluster_id") or r.get("source_cluster") for r in rows if r.get("cluster_id") or r.get("source_cluster")}

    # Historical baselines holdout isolation vs v1c blind holdout & train
    mf_v1 = json.loads((VL_P1_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    mf_v1b = json.loads((VL_P1B_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    hist_v1_test_shas = {r["sha256"] for r in mf_v1["test"]}
    hist_v1b_hold_shas = {r["sha256"] for r in mf_v1b["holdout_full"]}

    for a in range(len(splits)):
        for b in range(a + 1, len(splits)):
            na, nb = splits[a][0], splits[b][0]
            assert not (shas[na] & shas[nb]), f"SHA overlap between {na} and {nb}"
            if clusters[na] and clusters[nb]:
                assert not (clusters[na] & clusters[nb]), f"Cluster overlap between {na} and {nb}"

    assert not (shas["final_holdout_v1c"] & hist_v1_test_shas), "final_holdout overlaps vl_p1 test"
    assert not (shas["final_holdout_v1c"] & hist_v1b_hold_shas), "final_holdout overlaps vl_p1b holdout"


# ---------------------------------------------------------------------------
# Group 3: Cross-Split pHash / dHash Independence
# ---------------------------------------------------------------------------
def test_03_cross_split_phash_independence() -> None:
    assert MANIFEST_PATH.is_file()
    mf = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def _hashes(rows: list[dict], path_key: str) -> list[tuple[str, str, str]]:
        out = []
        for r in rows:
            ah = r.get("ahash")
            dh = r.get("dhash")
            if not ah or not dh:
                img = load_image_unicode(VL_P1C_DIR / r[path_key])
                ah, dh = compute_ahash(img), compute_dhash(img)
            out.append((r["id"], ah, dh))
        return out

    groups = {
        "train_pos": _hashes(mf["train_pos"], "rel_path"),
        "hard_neg": _hashes(mf["hard_neg_gallery"], "full_rel_path"),
        "cal_v1c": _hashes(mf["calibration_v1c_full"], "full_rel_path"),
        "final_holdout": _hashes(mf["final_holdout_full"], "full_rel_path"),
        "ambiguous_v1c": _hashes(mf["ambiguous_boundary_v1c"], "full_rel_path"),
    }
    names = list(groups.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for ida, aha, dha in groups[names[i]]:
                for idb, ahb, dhb in groups[names[j]]:
                    dist = min(hex_hamming_distance(aha, ahb), hex_hamming_distance(dha, dhb))
                    assert dist > 6, f"Near-dup pHash ({dist}) between {ida} ({names[i]}) and {idb} ({names[j]})"


# ---------------------------------------------------------------------------
# Group 4: Image & Annotation Validity Across All Splits
# ---------------------------------------------------------------------------
def test_04_image_and_annotation_validity() -> None:
    assert MANIFEST_PATH.is_file()
    mf = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(mf["train_pos"]) == 25
    assert len(mf["hard_neg_gallery"]) == 50
    assert len(mf["calibration_v1c_full"]) >= 100
    assert len(mf["final_holdout_full"]) == 300
    assert len(mf["ambiguous_boundary_v1c"]) >= 9

    for skey in ("calibration_v1c_full", "final_holdout_full", "ambiguous_boundary_v1c"):
        for r in mf[skey]:
            fp = VL_P1C_DIR / r["full_rel_path"]
            rp = VL_P1C_DIR / r["roi_rel_path"]
            assert fp.is_file() and rp.is_file()
            img = load_image_unicode(fp)
            roi = load_image_unicode(rp)
            assert img.size > 0 and roi.size > 0
            h, w = img.shape[:2]
            if skey != "ambiguous_boundary_v1c":
                assert r["roi_bbox_xywh"] is not None
            if r["roi_bbox_xywh"] is not None:
                bx, by, bw, bh = r["roi_bbox_xywh"]
                assert bw >= 8 and bh >= 8
                assert 0 <= bx and 0 <= by and (bx + bw) <= w and (by + bh) <= h


# ---------------------------------------------------------------------------
# Group 5: Calibration Freeze Integrity & Reproducible Reload
# ---------------------------------------------------------------------------
def test_05_frozen_config_and_reproducible_reload() -> None:
    assert FROZEN_CFG_PATH.is_file() and FROZEN_STATE_PATH.is_file() and CAL_METRICS_PATH.is_file()
    fc = json.loads(FROZEN_CFG_PATH.read_text(encoding="utf-8"))
    assert fc["final_holdout_touched_during_calibration"] is False
    assert fc["frozen_a3_oof_state_sha256"] == compute_sha256(FROZEN_STATE_PATH)
    assert fc["dataset_manifest_sha256"] == compute_sha256(MANIFEST_PATH)
    assert fc["frozen_e2e_k"] in (4, 6)

    rec = CalibratedRecognizerV1c(MODELS_DIR, CalibrationConfigV1c())
    rec.fit_positives(VL_P1C_DIR / "data" / "train_pos")
    rec.fit_hard_negatives(VL_P1C_DIR / "data" / "hard_neg_gallery")
    rec.load_frozen_state(FROZEN_STATE_PATH)

    mf = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    img = load_image_unicode(VL_P1C_DIR / mf["final_holdout_roi"][0]["roi_rel_path"])
    d1 = rec.predict(img)
    d2 = rec.predict(img)
    assert d1["status"] == d2["status"]
    assert d1["pos_score"] == pytest.approx(d2["pos_score"], abs=1e-6)


# ---------------------------------------------------------------------------
# Group 6: 5-Fold Group-Aware CV Stratification & Reproducibility
# ---------------------------------------------------------------------------
def test_06_group_aware_cv_stratification_and_reproducibility() -> None:
    synth = []
    for c_idx in range(30):
        lbl = OBJECT_IDS[c_idx % len(OBJECT_IDS)] if c_idx < 15 else "unknown"
        grp = f"cluster_{c_idx // 2:02d}"
        synth.append({"id": f"s_{c_idx}", "label_id": lbl, "source_cluster": grp})
    f1 = make_group_aware_stratified_folds(synth, n_folds=5)
    f2 = make_group_aware_stratified_folds(synth, n_folds=5)
    assert f1 == f2
    assert len(f1) == 5
    # Verify no shared source_cluster across distinct folds
    fold_clusters = [{synth[i]["source_cluster"] for i in idxs} for idxs in f1]
    for a in range(5):
        for b in range(a + 1, 5):
            assert not (fold_clusters[a] & fold_clusters[b])


# ---------------------------------------------------------------------------
# Group 7: Threshold Floor Enforcement (No Threshold Collapse)
# ---------------------------------------------------------------------------
def test_07_threshold_floor_enforcement() -> None:
    rec = CalibratedRecognizerV1c(MODELS_DIR, CalibrationConfigV1c())
    rec.fit_positives(VL_P1C_DIR / "data" / "train_pos")
    rec.fit_hard_negatives(VL_P1C_DIR / "data" / "hard_neg_gallery")

    rng = np.random.default_rng(7)
    dim = int(rec.pos_prototypes[OBJECT_IDS[0]].shape[0])
    recs, embs, geoms = [], [], []
    for i in range(40):
        lbl = OBJECT_IDS[i % len(OBJECT_IDS)] if i < 20 else "unknown"
        recs.append({"id": f"cal_{i}", "label_id": lbl, "confuser_group": None, "source_cluster": f"g_{i}"})
        if lbl != "unknown":
            v = rec.pos_prototypes[lbl] + 0.05 * rng.normal(size=dim).astype(np.float32)
        else:
            v = rng.normal(size=dim).astype(np.float32)
        v = v / max(1e-8, float(np.linalg.norm(v)))
        embs.append(v.astype(np.float32))
        geoms.append(np.zeros(16, dtype=np.float32))

    cfg = rec.calibrate_mode_on_records(recs, embs, geoms, mode="R5_selective_3state_v1c", delta_floor=0.06)
    expected_floor = max(cfg.tau_floor_abs, round(cfg.tau_global - 0.06, 4))
    for oid in OBJECT_IDS:
        assert cfg.tau_per_class[oid] >= expected_floor, f"{oid} threshold {cfg.tau_per_class[oid]} < {expected_floor}"


# ---------------------------------------------------------------------------
# Group 8: Negative Gallery Verification & Margin Suppression
# ---------------------------------------------------------------------------
def test_08_negative_verification_and_margin_suppression() -> None:
    rec = CalibratedRecognizerV1c(MODELS_DIR, CalibrationConfigV1c(mode="R5_selective_3state_v1c"))
    rec.fit_positives(VL_P1C_DIR / "data" / "train_pos")
    rec.fit_hard_negatives(VL_P1C_DIR / "data" / "hard_neg_gallery")

    # Query aligned to negative prototype of glasses_case_confuser (targeted against object_03)
    neg_proto = rec.neg_prototypes["glasses_case_confuser"]
    ranked = rec.score_features(neg_proto, np.zeros(16, dtype=np.float32))
    assert any(r["neg_sim"] > 0.70 and r["pos_neg_margin"] < 0.0 for r in ranked)
    dec = rec.decide_with_config(ranked, CalibrationConfigV1c(
        mode="R5_selective_3state_v1c",
        tau_per_class={o: 0.45 for o in OBJECT_IDS},
        delta_neg_min={o: 0.02 for o in OBJECT_IDS},
    ))
    assert dec["status"] != "KNOWN"


# ---------------------------------------------------------------------------
# Group 9: Selective 3-State Output (KNOWN / AMBIGUOUS / UNKNOWN)
# ---------------------------------------------------------------------------
def test_09_selective_three_state_classification() -> None:
    rec = CalibratedRecognizerV1c(MODELS_DIR, CalibrationConfigV1c(mode="R5_selective_3state_v1c"))
    cfg = CalibrationConfigV1c(
        mode="R5_selective_3state_v1c",
        tau_per_class={o: 0.50 for o in OBJECT_IDS},
        delta_neg_min={o: 0.02 for o in OBJECT_IDS},
        verifier_min_score={o: 0.0 for o in OBJECT_IDS},
        delta_pos_ambig=0.02,
        tau_ambig_band=0.05,
        neg_ambig_band=0.03,
        ver_ambig_band=0.15,
    )
    rk_known = [
        {"id": "object_03", "pos_score": 0.62, "neg_sim": 0.40, "pos_neg_margin": 0.22, "verifier_score": 0.20, "best_sample": "a"},
        {"id": "object_02", "pos_score": 0.45, "neg_sim": 0.40, "pos_neg_margin": 0.05, "verifier_score": -0.1, "best_sample": "b"},
    ]
    rk_ambig = [
        {"id": "object_03", "pos_score": 0.48, "neg_sim": 0.42, "pos_neg_margin": 0.06, "verifier_score": 0.05, "best_sample": "a"},
        {"id": "object_02", "pos_score": 0.41, "neg_sim": 0.40, "pos_neg_margin": 0.01, "verifier_score": -0.1, "best_sample": "b"},
    ]
    rk_unk = [
        {"id": "object_03", "pos_score": 0.35, "neg_sim": 0.55, "pos_neg_margin": -0.20, "verifier_score": -0.40, "best_sample": "a"},
        {"id": "object_02", "pos_score": 0.30, "neg_sim": 0.45, "pos_neg_margin": -0.15, "verifier_score": -0.50, "best_sample": "b"},
    ]
    assert rec.decide_with_config(rk_known, cfg)["status"] == "KNOWN"
    assert rec.decide_with_config(rk_ambig, cfg)["status"] == "AMBIGUOUS"
    assert rec.decide_with_config(rk_unk, cfg)["status"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Group 10: Multi-Proposal FP Control (Compatibility + Multi-View + FWER)
# ---------------------------------------------------------------------------
def test_10_multi_proposal_fp_control() -> None:
    rec = CalibratedRecognizerV1c(MODELS_DIR, CalibrationConfigV1c(mode="R5_selective_3state_v1c"))
    mp_cfg = MultiProposalControlConfig(enable_control=True, alpha_k=0.02, singleton_penalty_heuristic=0.04, min_thin_cane_evidence=0.50)
    pipe = CombinedPipelineV1c(ObjectProposerV1c(None, "B_v1c_proposer", top_k=4), rec, mp_cfg)

    # 1. Compatibility Gate: thin cane proposal CANNOT emit wallet (object_03)
    p_thin = ProposalBox((10, 10, 20, 120), "oriented_thin_cane", 0.70, source_family="F_thin", thin_evidence_score=0.60)
    dec_wallet = {"status": "KNOWN", "label": "object_03", "top_class": "object_03", "pos_score": 0.65, "margin": 0.20, "neg_sim": 0.3, "pos_neg_margin": 0.35, "verifier_score": 0.4}
    out = pipe.decide_from_evaluated([{"prop": p_thin, "source_group": "thin_cane", "top_class": "object_03", "decision": dec_wallet}], 5.0, 5.0)
    assert out["status"] == "UNKNOWN"

    # 2. Non-redundant multi-view support check: identical dup (IoU > 0.68) does NOT count!
    p_base = ProposalBox((50, 50, 100, 100), "fg_contour", 0.80, source_family="F_primary")
    p_dup = ProposalBox((51, 51, 99, 99), "ms_sal_mid", 0.75, source_family="F_saliency")
    p_diff = ProposalBox((40, 40, 140, 130), "center_72", 0.55, source_family="F_context")
    e_base = {"idx": 0, "prop": p_base, "source_group": "fg_contour", "top_class": "object_02", "decision": {"status": "KNOWN", "pos_score": 0.52}}
    e_dup = {"idx": 1, "prop": p_dup, "source_group": "ms_sal", "top_class": "object_02", "decision": {"status": "KNOWN", "pos_score": 0.50}}
    e_diff = {"idx": 2, "prop": p_diff, "source_group": "context", "top_class": "object_02", "decision": {"status": "KNOWN", "pos_score": 0.46}}
    assert pipe._has_independent_support(e_base, [e_base, e_dup], "object_02") is False
    assert pipe._has_independent_support(e_base, [e_base, e_diff], "object_02") is True


# ---------------------------------------------------------------------------
# Group 11: ProposalRankerV1c Quota & Anti-Thin-Flooding Diversity
# ---------------------------------------------------------------------------
def test_11_proposal_ranker_quota_and_anti_flooding() -> None:
    img = np.full((300, 300, 3), 50, dtype=np.uint8)
    cv2.rectangle(img, (80, 70), (210, 220), (220, 220, 220), -1)
    sal = _compute_saliency_map(img)
    cands = [
        ProposalBox((78, 68, 134, 154), "fg_contour", 0.85, source_family="F_primary"),
        ProposalBox((20, 20, 14, 240), "oriented_thin_cane", 0.90, source_family="F_thin", thin_evidence_score=0.70),
        ProposalBox((40, 20, 14, 240), "person_cane_ray", 0.88, source_family="F_thin", thin_evidence_score=0.68),
        ProposalBox((60, 20, 14, 240), "oriented_thin_cane", 0.86, source_family="F_thin", thin_evidence_score=0.66),
        ProposalBox((40, 40, 210, 210), "center_72", 0.50, source_family="F_context"),
        ProposalBox((0, 0, 300, 300), "full_frame", 0.35, source_family="F_context"),
    ]
    kept = ProposalRankerV1c.score_and_select(img, cands, sal, top_k=4, nms_iou=0.68)
    assert len(kept) == 4
    n_thin = sum(1 for p in kept if p.source_family == "F_thin")
    assert n_thin <= 1, f"Thin proposals flooded K=4 budget: {n_thin}"
    assert any(p.source == "fg_contour" for p in kept)


# ---------------------------------------------------------------------------
# Group 12: Fair Top-K Budget & Full 3x3 Matrix Schema
# ---------------------------------------------------------------------------
def test_12_fair_topk_budget_and_3x3_matrix() -> None:
    img = np.full((240, 320, 3), 100, dtype=np.uint8)
    cv2.rectangle(img, (90, 60), (220, 190), (200, 180, 140), -1)
    yolox = YOLOXNanoExtractor(YOLOX_PATH)
    for k in (4, 6, 8):
        for mode in ("B0_vl_p1_proposer", "B_full_v1b", "B_v1c_proposer"):
            props, _ = ObjectProposerV1c(yolox, mode=mode, top_k=k).propose(img)
            assert 1 <= len(props) <= k

    assert BENCHMARK_PATH.is_file()
    bm = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    for mode in ("B0_vl_p1_proposer", "B_full_v1b", "B_v1c_proposer"):
        for k in (4, 6, 8):
            key = f"{mode}__k{k}"
            assert key in bm["proposal_results"]
            assert "final_holdout_full" in bm["proposal_results"][key]


# ---------------------------------------------------------------------------
# Group 13: Thin-Object Oriented Detector, Soft Mask & 7 Geometry Metrics
# ---------------------------------------------------------------------------
def test_13_thin_object_detector_corridor_and_metrics() -> None:
    img = np.full((360, 480, 3), 75, dtype=np.uint8)
    # Draw diagonal white cane with reflective red tip (simulated handheld scenario)
    cv2.line(img, (110, 60), (370, 310), (242, 244, 245), 7)
    cv2.line(img, (320, 262), (370, 310), (35, 35, 225), 7)
    cands = detect_oriented_thin_cane_candidates(img, yolox_boxes=[], max_candidates=4)
    assert len(cands) >= 1
    assert cands[0].endpoints_xyxy is not None
    roi = extract_thin_aware_roi(img, cands[0])
    assert roi.shape[0] > 0 and roi.shape[1] > 0

    gt_box = (105, 55, 270, 260)
    tm = evaluate_thin_object_metrics(cands, gt_box)
    for k in ("best_iou", "best_containment", "best_bg_contamination", "endpoint_included", "axis_angle_error_deg"):
        assert k in tm
    assert tm["best_containment"] > 0.50


# ---------------------------------------------------------------------------
# Group 14: Zero GT Leakage in Inference & Proposal Signatures / AST
# ---------------------------------------------------------------------------
def test_14_zero_gt_leakage_ast_and_signatures() -> None:
    assert list(inspect.signature(ObjectProposerV1c.propose).parameters.keys()) == ["self", "image"]
    assert list(inspect.signature(CalibratedRecognizerV1c.predict).parameters.keys()) == ["self", "image"]
    assert list(inspect.signature(CombinedPipelineV1c.predict).parameters.keys()) == ["self", "image"]

    for fname in ("proposal_research_v1c.py", "thin_object_proposer.py", "calibration_research.py", "combined_pipeline_v1c.py"):
        src_txt = (VL_P1C_DIR / "src" / fname).read_text(encoding="utf-8")
        tree = ast.parse(src_txt)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "roi_bbox_xywh" not in node.value, f"{fname} references roi_bbox_xywh"
                assert "final_holdout" not in node.value, f"{fname} references final_holdout"


# ---------------------------------------------------------------------------
# Group 15: 100% Offline Network Isolation Verification
# ---------------------------------------------------------------------------
def test_15_offline_execution_blocks_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def _deny_socket(*args, **kwargs):
        raise RuntimeError("Outbound network forbidden during VL-P1c offline inference")

    monkeypatch.setattr(socket, "socket", _deny_socket)
    rec = CalibratedRecognizerV1c(MODELS_DIR, CalibrationConfigV1c())
    rec.fit_positives(VL_P1C_DIR / "data" / "train_pos")
    rec.fit_hard_negatives(VL_P1C_DIR / "data" / "hard_neg_gallery")
    yolox = YOLOXNanoExtractor(YOLOX_PATH)
    pipe = CombinedPipelineV1c(ObjectProposerV1c(yolox, "B_v1c_proposer", top_k=4), rec)
    sample = list_images(VL_P1C_DIR / "data" / "train_pos" / "object_03")[0]
    out = pipe.predict(load_image_unicode(sample))
    assert out["status"] in ("KNOWN", "AMBIGUOUS", "UNKNOWN", "PROPOSAL_MISSED")


# ---------------------------------------------------------------------------
# Group 16: ONNX Model Presence, Readability & SHA-256 Audit
# ---------------------------------------------------------------------------
def test_16_model_checksums_and_availability() -> None:
    assert DINO_PATH.is_file() and DINO_PATH.stat().st_size > 1_000_000
    assert YOLOX_PATH.is_file() and YOLOX_PATH.stat().st_size > 500_000
    if FROZEN_CFG_PATH.is_file():
        fc = json.loads(FROZEN_CFG_PATH.read_text(encoding="utf-8"))
        assert fc["dinov2_onnx_sha256"] == compute_sha256(DINO_PATH)
        assert fc["yolox_onnx_sha256"] == compute_sha256(YOLOX_PATH)


# ---------------------------------------------------------------------------
# Group 17: Benchmark JSON Schema, Statistical Tests & Telemetry Validation
# ---------------------------------------------------------------------------
def test_17_benchmark_metrics_stats_and_telemetry_schema() -> None:
    assert BENCHMARK_PATH.is_file() and CAL_METRICS_PATH.is_file()
    bm = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    cm = json.loads(CAL_METRICS_PATH.read_text(encoding="utf-8"))

    assert len(cm["cv_report"]["sample_size_scaling"]) == 4  # N=50, 100, 150, 200
    for key in ("C1_vl_p1_baseline", "C4_vl_p1b_combined", "C_v1c_cal_only", "C_v1c_prop_only", "C_v1c_full"):
        fh = bm["e2e_results"][key]["final_holdout_full"]
        for mk in ("overall_accuracy", "known_recall", "far"):
            assert 0.0 <= fh[mk]["ci95_low_pct"] <= fh[mk]["pct"] <= fh[mk]["ci95_high_pct"] <= 100.0
        assert len(fh["risk_coverage_curve"]) >= 5

    hw = bm["hardware_telemetry"]
    assert hw["rss_end_mb"] < 2048.0
    assert hw["private_bytes_mb"] > 0.0
    assert len(hw["gallery_scaling"]) == 4
    assert "paired_statistical_tests" in bm


# ---------------------------------------------------------------------------
# Group 18: Historical VL-P1 & VL-P1b Baseline Exact Preservation
# ---------------------------------------------------------------------------
def test_18_historical_baseline_preservation() -> None:
    assert V1B_FROZEN_BASELINE.is_file()
    base = json.loads(V1B_FROZEN_BASELINE.read_text(encoding="utf-8"))
    assert base["source_commit"] == "dacd245"
    assert base["all_exact_matches"] is True


# ---------------------------------------------------------------------------
# Group 19: Corrupt / Empty / Extreme Aspect Ratio Input Safety
# ---------------------------------------------------------------------------
def test_19_corrupt_empty_and_extreme_aspect_safety() -> None:
    yolox = YOLOXNanoExtractor(YOLOX_PATH)
    proposer = ObjectProposerV1c(yolox, "B_v1c_proposer", top_k=4)
    rec = CalibratedRecognizerV1c(MODELS_DIR, CalibrationConfigV1c())
    rec.fit_positives(VL_P1C_DIR / "data" / "train_pos")
    rec.fit_hard_negatives(VL_P1C_DIR / "data" / "hard_neg_gallery")
    pipe = CombinedPipelineV1c(proposer, rec)

    with pytest.raises(ValueError):
        pipe.predict(np.array([], dtype=np.uint8))

    for arr in [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.full((64, 64, 3), 255, dtype=np.uint8),
        np.full((24, 600, 3), 120, dtype=np.uint8),
    ]:
        out = pipe.predict(arr)
        assert out["status"] in ("KNOWN", "AMBIGUOUS", "UNKNOWN", "PROPOSAL_MISSED")


# ---------------------------------------------------------------------------
# Group 20: 5-Repeat Determinism Across Proposer, Recognizer & Pipeline
# ---------------------------------------------------------------------------
def test_20_five_repeat_determinism() -> None:
    yolox = YOLOXNanoExtractor(YOLOX_PATH)
    proposer = ObjectProposerV1c(yolox, "B_v1c_proposer", top_k=4)
    rec = CalibratedRecognizerV1c(MODELS_DIR, CalibrationConfigV1c())
    rec.fit_positives(VL_P1C_DIR / "data" / "train_pos")
    rec.fit_hard_negatives(VL_P1C_DIR / "data" / "hard_neg_gallery")
    if FROZEN_STATE_PATH.is_file():
        rec.load_frozen_state(FROZEN_STATE_PATH)
    pipe = CombinedPipelineV1c(proposer, rec)

    img = load_image_unicode(list_images(VL_P1C_DIR / "data" / "train_pos" / "object_01")[0])
    outs = [pipe.predict(img) for _ in range(5)]
    ref = outs[0]
    for o in outs[1:]:
        assert o["status"] == ref["status"]
        assert o["label"] == ref["label"]
        assert o["selected_box"] == ref["selected_box"]
        assert o["score"] == pytest.approx(ref["score"], abs=1e-6)
