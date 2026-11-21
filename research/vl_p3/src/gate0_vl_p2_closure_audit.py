"""Gate 0: Independent Git, Benchmark Regression, and Ground-Truth Annotation Audit for VL-P2 Closure.

Never modifies any file inside `VisionMate_VL_P2_AG`, `VisionMateGlasses`, or other protected worktrees.
Saves audit artifacts inside `VisionMate_VL_P3_AG/research/vl_p3/manifests/` and `cache/contact_sheets/`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from .common_utils import (
    VL_P3_DIR,
    WORKTREE_ROOT,
    compute_iou_xywh,
    decode_mask_rle,
    format_count_pct_ci,
    load_image_unicode,
    make_contact_sheet,
    obb_from_endpoints,
    sample_centerline_points,
)

AI_ROOT = WORKTREE_ROOT.parent
VL_P2_ROOT = AI_ROOT / "VisionMate_VL_P2_AG"
VL_P2_DIR = VL_P2_ROOT / "research" / "vl_p2"


def _run_git(repo: Path, args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "").strip()
    if proc.stderr and proc.returncode != 0:
        out = (out + "\n" + proc.stderr.strip()).strip()
    return proc.returncode, out


def audit_git_baselines() -> dict:
    """Verify all 8 protected commits, full SHA-1s, ancestry, worktree cleanliness, and tags/remotes."""
    expected = [
        ("English Tables & TTS", "8ae6f85", "hotfix/english-tables-layout-and-tts"),
        ("OD-1 Foundation", "95a75e0", "hotfix/od-1-foundation"),
        ("Static Preview", "6e3039f", "research/vl-p1-manual"),
        ("Webcam FPS Optimization", "65c2a27", "hotfix/webcam-preview-fps-optimization"),
        ("VL-P1", "988fdf1", "research/vl-p1-antigravity"),
        ("VL-P1b", "dacd245", "research/vl-p1b-reliability-proposal"),
        ("VL-P1c", "9587ad4", "research/vl-p1c-calibration-multiproposal"),
        ("VL-P2", "4a10113", "research/vl-p2-white-cane-segmentation"),
    ]
    commits = []
    all_ok = True
    for name, short_sha, branch in expected:
        rc, full_sha = _run_git(VL_P2_ROOT, ["rev-parse", "--verify", f"{short_sha}^{{commit}}"])
        rc_b, b_sha = _run_git(VL_P2_ROOT, ["rev-parse", "--verify", branch])
        matches_branch = (rc == 0 and rc_b == 0 and full_sha == b_sha) if branch != "research/vl-p1-manual" else (rc == 0)
        if branch == "hotfix/english-tables-layout-and-tts":
            matches_branch = (rc == 0)
        if branch == "hotfix/od-1-foundation":
            matches_branch = (rc == 0)
        # Note: hotfix/od-1-foundation/etc. might not have local branch refs if merged/fast-forwarded; verify commit existence & ancestry
        commits.append({
            "component": name,
            "short_sha": short_sha,
            "full_sha1": full_sha if rc == 0 else None,
            "expected_branch": branch,
            "commit_exists": rc == 0,
            "branch_points_to_commit": (full_sha == b_sha) if rc_b == 0 else None,
            "branch_head_sha1": b_sha if rc_b == 0 else None,
        })
        if rc != 0:
            all_ok = False

    ancestry_pairs = [
        ("8ae6f85", "95a75e0"),
        ("95a75e0", "6e3039f"),
        ("6e3039f", "65c2a27"),
        ("6e3039f", "988fdf1"),
        ("988fdf1", "dacd245"),
        ("dacd245", "9587ad4"),
        ("9587ad4", "4a10113"),
        ("4a10113", "HEAD"),
    ]
    ancestry = {}
    for anc, desc in ancestry_pairs:
        repo = WORKTREE_ROOT if desc == "HEAD" else VL_P2_ROOT
        rc, _ = _run_git(repo, ["merge-base", "--is-ancestor", anc, desc])
        ancestry[f"{anc}->{desc}"] = (rc == 0)
        if rc != 0:
            all_ok = False

    worktrees = [
        ("VisionMateGlasses", AI_ROOT / "VisionMateGlasses"),
        ("VisionMate_VL_P1", AI_ROOT / "VisionMate_VL_P1"),
        ("VisionMate_VL_P1_AG", AI_ROOT / "VisionMate_VL_P1_AG"),
        ("VisionMate_VL_P1b_AG", AI_ROOT / "VisionMate_VL_P1b_AG"),
        ("VisionMate_VL_P1c_AG", AI_ROOT / "VisionMate_VL_P1c_AG"),
        ("VisionMate_VL_P2_AG", AI_ROOT / "VisionMate_VL_P2_AG"),
        ("VisionMate_VL_P3_AG", WORKTREE_ROOT),
    ]
    wt_status = []
    for label, path in worktrees:
        rc_h, head = _run_git(path, ["rev-parse", "HEAD"])
        rc_b, bname = _run_git(path, ["branch", "--show-current"])
        rc_s, st = _run_git(path, ["status", "--short"])
        wt_status.append({
            "worktree": label,
            "head_sha1": head if rc_h == 0 else None,
            "branch": bname if rc_b == 0 else None,
            "status_short": st,
            "clean": (st == ""),
        })

    _, tags = _run_git(VL_P2_ROOT, ["tag"])
    _, branches = _run_git(VL_P2_ROOT, ["branch", "-vv"])
    _, p2_diff = _run_git(VL_P2_ROOT, ["diff", "--name-only", "9587ad4", "--", "app", "research/vl_p1", "research/vl_p1b", "research/vl_p1c"])
    _, p3_diff = _run_git(WORKTREE_ROOT, ["diff", "--name-only", "4a10113", "--", "app", "research/vl_p1", "research/vl_p1b", "research/vl_p1c", "research/vl_p2"])

    return {
        "all_protected_commits_present_and_ancestry_valid": all_ok,
        "commits": commits,
        "ancestry_checks": ancestry,
        "worktree_status": wt_status,
        "git_tags": [t for t in tags.splitlines() if t.strip()],
        "branches_vv": branches.splitlines(),
        "p2_out_of_scope_diff_vs_9587ad4": p2_diff,
        "p3_out_of_scope_diff_vs_4a10113": p3_diff,
    }


def audit_vl_p2_post_optimization_regression() -> dict:
    """Run committed 4a10113 VL-P2 C2 pipeline read-only on all 180 final_holdout_v2 images."""
    p2_parent = str(VL_P2_ROOT)
    if p2_parent not in sys.path:
        sys.path.insert(0, p2_parent)

    from research.vl_p2.src.cane_verifier import CaneVsConfuserVerifier
    from research.vl_p2.src.combined_pipeline_v2 import CombinedPipelineV2, PipelineV2Config
    from research.vl_p2.src.localization_research import OrientedCaneLocalizer, YOLOXPersonAndObjectDetector
    from research.vl_p2.src.masked_embedding_research import (
        DINOv2EmbedderV2,
        RepresentationEmbeddingBank,
        build_representation_canvas,
    )
    from research.vl_p2.src.segmentation_research import CaneShaftSegmenter

    mf = json.loads((VL_P2_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    amf = json.loads((VL_P2_DIR / "manifests" / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]
    frozen = json.loads((VL_P2_DIR / "manifests" / "frozen_config_v2.json").read_text(encoding="utf-8"))
    locked_bm = json.loads((VL_P2_DIR / "reports" / "benchmark_metrics.json").read_text(encoding="utf-8"))
    phase_b_c2 = locked_bm["final_holdout_v2"]["experimental_matrix_A0_to_C2"]["C2_full_cane_verified"]

    yolox = YOLOXPersonAndObjectDetector(VL_P2_ROOT / "models" / "object_detection" / "yolox_nano.onnx")
    localizer = OrientedCaneLocalizer(yolox, top_k=6)
    segmenter = CaneShaftSegmenter()
    embedder = DINOv2EmbedderV2(VL_P2_DIR / "models" / "dinov2_small_quantized.onnx")

    train_pos = [{"id": r["id"], "path": VL_P2_DIR / r["rel_path"]} for r in mf["train_pos"] if r["label_id"] == "object_01"]
    best_rep = frozen["selected_best_representation"]
    bank = RepresentationEmbeddingBank(embedder)
    bank.fit_cane_enrollment_shots(train_pos, localizer, segmenter)

    neg_embs = []
    for row in mf.get("hard_neg_gallery", []):
        p = VL_P2_DIR / row["rel_path"]
        if not p.is_file():
            continue
        nimg = load_image_unicode(p)
        nprops, _ = localizer.detect_with_context(nimg, top_k=1)
        if nprops:
            np0 = nprops[0]
            nseg = segmenter.segment_proposal(nimg, np0, return_full_mask=(best_rep == "rep3_masked_crop"))
            ncanvas = build_representation_canvas(nimg, np0, nseg, best_rep)
            ne, _ = embedder.embed_canvas(ncanvas)
        else:
            ne, _ = embedder.embed_raw_roi(nimg)
        neg_embs.append(ne)
    neg_gal_best = np.vstack(neg_embs).astype(np.float32) if neg_embs else np.zeros((0, 384), dtype=np.float32)

    ver = CaneVsConfuserVerifier()
    ver.load_state_dict(frozen["verifier_state"])
    ver.fit_negative_gallery(neg_gal_best)
    pipe = CombinedPipelineV2(
        localizer,
        segmenter,
        embedder,
        bank,
        ver,
        PipelineV2Config(
            "C2_post_opt_audit",
            best_rep,
            int(frozen["selected_top_k"]),
            use_structural_verifier=True,
        ),
    )

    # Extract Phase B positive error IDs for exact positive comparison
    phase_b_pos_errors = {
        err["id"]: err["type"].replace("FN_", "")
        for err in phase_b_c2.get("errors", [])
        if err["type"].startswith("FN_")
    }
    phase_b_fp_subset = {
        err["id"]: err["type"]
        for err in phase_b_c2.get("errors", [])
        if err["type"] == "FP"
    }

    cm = {"object_01": {"object_01": 0, "UNKNOWN": 0, "AMBIGUOUS": 0}, "unknown": {"object_01": 0, "UNKNOWN": 0, "AMBIGUOUS": 0}}
    pos_ok = pos_tot = neg_fp = neg_tot = pos_ambig = neg_ambig = 0
    per_neg_group: dict[str, list[int]] = {}
    per_scenario: dict[str, list[int]] = {"handheld": [0, 0], "occluded": [0, 0], "red_tip": [0, 0], "folded": [0, 0]}
    lat_loc, lat_seg, lat_emb, lat_ver, lat_tot = [], [], [], [], []
    per_image_records = []
    pos_changes = []
    fp_list = []

    for rec in mf["final_holdout_v2"]:
        img = load_image_unicode(VL_P2_DIR / rec["rel_path"])
        dec = pipe.predict(img)
        lat_loc.append(dec["localization_ms"])
        lat_seg.append(dec["segmentation_ms"])
        lat_emb.append(dec["embedding_ms"])
        lat_ver.append(dec["verify_ms"])
        lat_tot.append(dec["total_ms"])

        y = rec["label_id"]
        cg = rec.get("confuser_group") or "general_distractor"
        status = dec["status"]
        pred_lbl = "object_01" if status == "KNOWN" else status
        cm[y][pred_lbl] = cm[y].get(pred_lbl, 0) + 1

        phase_b_status = None
        if y == "object_01":
            phase_b_status = phase_b_pos_errors.get(rec["id"], "KNOWN")
            if phase_b_status != status:
                pos_changes.append({
                    "id": rec["id"],
                    "phase_b_status": phase_b_status,
                    "post_opt_status": status,
                    "post_opt_score": round(float(dec["score"]), 4),
                    "post_opt_reason": dec["reason"],
                })
            pos_tot += 1
            is_ok = 1 if status == "KNOWN" else 0
            pos_ok += is_ok
            if status == "AMBIGUOUS":
                pos_ambig += 1
            ann = amf.get(rec["id"], {})
            tags = ann.get("scenario_tags", [])
            if ann.get("occluded", False):
                per_scenario["occluded"][1] += 1
                per_scenario["occluded"][0] += is_ok
            for tg in ("handheld", "red_tip", "folded"):
                if tg in tags:
                    per_scenario[tg][1] += 1
                    per_scenario[tg][0] += is_ok
        else:
            neg_tot += 1
            per_neg_group.setdefault(cg, [0, 0])
            per_neg_group[cg][1] += 1
            if status == "KNOWN":
                neg_fp += 1
                per_neg_group[cg][0] += 1
                fp_list.append({
                    "id": rec["id"],
                    "group": cg,
                    "post_opt_score": round(float(dec["score"]), 4),
                    "in_locked_top30_fp_list": rec["id"] in phase_b_fp_subset,
                })
            elif status == "AMBIGUOUS":
                neg_ambig += 1

        per_image_records.append({
            "id": rec["id"],
            "gt_label": y,
            "confuser_group": cg,
            "phase_b_status_known_if_pos": phase_b_status,
            "post_opt_status": status,
            "post_opt_score": round(float(dec["score"]), 4),
            "post_opt_margin": round(float(dec["margin"]), 4),
            "total_ms": round(float(dec["total_ms"]), 2),
        })

    total_n = pos_tot + neg_tot
    overall_k = pos_ok + (neg_tot - neg_fp)
    pos_identical = sum(1 for r in per_image_records if r["gt_label"] == "object_01" and r["phase_b_status_known_if_pos"] == r["post_opt_status"])

    return {
        "note": "Regression audit ONLY on VL-P2 final_holdout_v2 (N=180) using committed 4a10113 C2_full_cane_verified. Not a blind test; locked Phase B benchmark_metrics.json remains untouched.",
        "phase_b_locked_reference": {
            "overall_accuracy": phase_b_c2["overall_accuracy"]["formatted"],
            "known_recall_object_01": phase_b_c2["known_recall_object_01"]["formatted"],
            "far": phase_b_c2["far"]["formatted"],
            "confusion_matrix": phase_b_c2["confusion_matrix"],
            "latency": phase_b_c2["latency"],
        },
        "post_optimization_measured": {
            "overall_accuracy": format_count_pct_ci(overall_k, total_n),
            "known_recall_object_01": format_count_pct_ci(pos_ok, pos_tot),
            "far": format_count_pct_ci(neg_fp, neg_tot),
            "unknown_specificity": format_count_pct_ci(neg_tot - neg_fp, neg_tot),
            "ambiguous_counts": {
                "pos_ambig": pos_ambig,
                "neg_ambig": neg_ambig,
                "ambiguous_rate_pct": round(((pos_ambig + neg_ambig) / max(1, total_n)) * 100.0, 2),
            },
            "per_neg_group_far": {g: format_count_pct_ci(v[0], v[1]) for g, v in sorted(per_neg_group.items())},
            "per_scenario_recall": {k: format_count_pct_ci(v[0], v[1]) for k, v in per_scenario.items() if v[1] > 0},
            "confusion_matrix": cm,
            "latency": {
                "loc_p50_ms": round(float(np.percentile(lat_loc, 50)), 2),
                "loc_p95_ms": round(float(np.percentile(lat_loc, 95)), 2),
                "seg_p50_ms": round(float(np.percentile(lat_seg, 50)), 2),
                "emb_p50_ms": round(float(np.percentile(lat_emb, 50)), 2),
                "ver_p50_ms": round(float(np.percentile(lat_ver, 50)), 2),
                "total_e2e_p50_ms": round(float(np.percentile(lat_tot, 50)), 2),
                "total_e2e_p95_ms": round(float(np.percentile(lat_tot, 95)), 2),
            },
        },
        "per_image_comparison": {
            "positives_status_identical_count": pos_identical,
            "positives_total": pos_tot,
            "positives_changed": pos_changes,
            "post_opt_false_positives": fp_list,
            "predictions_changed_behavior_flag": (len(pos_changes) > 0 or neg_fp != phase_b_c2["far"]["k"] or cm != phase_b_c2["confusion_matrix"]),
        },
        "per_image_records": per_image_records,
    }


def _evaluate_line_support_on_raw(img: np.ndarray, endpoints_xyxy: list[float], mask: np.ndarray) -> dict:
    """Measure pixel evidence of white/red shaft alignment along annotated centerline & mask directly on raw image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in endpoints_xyxy]
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 8.0:
        return {"line_len_px": round(length, 2), "cl_support_frac": 0.0, "mask_white_red_purity": 0.0, "mask_bg_bleed": 1.0}

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    l_ch = lab[:, :, 0] * (100.0 / 255.0)
    a_ch = lab[:, :, 1] - 128.0
    b_ch = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_ch, b_ch)
    hue = hsv[:, :, 0]

    # Pixel-level white cane or red reflective tip indicator
    white_px = (l_ch >= 58.0) & (chroma <= 26.0)
    red_px = (a_ch >= 14.0) & (hsv[:, :, 1] >= 70.0) & ((hue <= 12.0) | (hue >= 166.0))
    cane_cue = (white_px | red_px)

    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    nx, ny = -uy, ux
    n_st = 25
    ts = np.linspace(0.0, length, n_st)
    hit_count = 0
    contrast_hits = 0
    for t in ts:
        cx, cy = x1 + ux * t, y1 + uy * t
        best_cue = False
        best_l = -1e9
        side_ls = []
        for off in range(-6, 7):
            px = int(round(cx + nx * off))
            py = int(round(cy + ny * off))
            if 0 <= px < w and 0 <= py < h:
                if cane_cue[py, px]:
                    best_cue = True
                best_l = max(best_l, float(l_ch[py, px]))
        for off in (-12, -9, 9, 12):
            px = int(round(cx + nx * off))
            py = int(round(cy + ny * off))
            if 0 <= px < w and 0 <= py < h:
                side_ls.append(float(l_ch[py, px]))
        if best_cue:
            hit_count += 1
        if side_ls and (best_l - float(np.median(side_ls))) > 3.0:
            contrast_hits += 1

    m_bin = mask > 0
    m_pix = int(np.sum(m_bin))
    if m_pix > 0:
        mask_purity = float(np.mean(cane_cue[m_bin]))
    else:
        mask_purity = 0.0

    # Check if endpoints hit extreme border or degenerate zero length
    ep1_in = (0 <= x1 < w and 0 <= y1 < h)
    ep2_in = (0 <= x2 < w and 0 <= y2 < h)
    return {
        "line_len_px": round(length, 2),
        "line_len_norm_diag": round(length / max(1.0, math.hypot(w, h)), 4),
        "cl_support_frac": round(hit_count / float(n_st), 4),
        "cl_contrast_frac": round(contrast_hits / float(n_st), 4),
        "mask_pixels": m_pix,
        "mask_white_red_purity": round(mask_purity, 4),
        "mask_bg_bleed": round(1.0 - mask_purity, 4),
        "endpoints_in_bounds": bool(ep1_in and ep2_in),
    }


def _draw_side_by_side_audit(img: np.ndarray, ann: dict, tag: str) -> np.ndarray:
    """Render raw image side-by-side with GT OBB/centerline/endpoints/mask overlay for visual QA."""
    mask = decode_mask_rle(ann["mask_rle"])
    over = img.copy()
    if np.any(mask > 0):
        tint = over.copy()
        tint[mask > 0] = (0, 230, 255)
        over = cv2.addWeighted(over, 0.65, tint, 0.35, 0.0)
    pts = np.round(np.asarray(ann["obb_corners_xy"], dtype=np.float32)).astype(np.int32)
    cv2.polylines(over, [pts], isClosed=True, color=(0, 220, 0), thickness=max(2, min(img.shape[:2]) // 180))
    ex1, ey1, ex2, ey2 = [int(round(v)) for v in ann["endpoints_xyxy"]]
    cv2.line(over, (ex1, ey1), (ex2, ey2), (0, 0, 255), max(2, min(img.shape[:2]) // 220))
    cv2.circle(over, (ex1, ey1), max(4, min(img.shape[:2]) // 100), (255, 255, 0), -1)
    cv2.circle(over, (ex2, ey2), max(4, min(img.shape[:2]) // 100), (255, 0, 255), -1)
    h, w = img.shape[:2]
    sc = min(220.0 / max(1, h), 200.0 / max(1, w))
    rw, rh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    left = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    right = cv2.resize(over, (rw, rh), interpolation=cv2.INTER_AREA)
    comb = np.hstack([left, right])
    cv2.putText(comb, tag[:42], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return comb


def audit_vl_p2_annotations() -> dict:
    """Audit all 116 VL-P2 object_01 annotations against original raw images and generate audit sheets."""
    mf = json.loads((VL_P2_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    amf = json.loads((VL_P2_DIR / "manifests" / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]
    overrides = json.loads((VL_P2_DIR / "manifests" / "manual_cane_gt_overrides.json").read_text(encoding="utf-8"))

    # Inspect VL-P2 annotation audit status
    items_dev = [(r, "vl_p2_dev_cal_v2") for r in mf["dev_cal_v2"] if r["label_id"] == "object_01"]
    items_fhold = [(r, "vl_p2_final_holdout_v2") for r in mf["final_holdout_v2"] if r["label_id"] == "object_01"]

    audit_records = []
    sheet_dev, sheet_fhold, sheet_flagged = [], [], []
    tier_counts = {}
    fallback_diagonals = []
    high_bleed = []
    weak_support = []

    for rec, split_name in items_dev + items_fhold:
        iid = rec["id"]
        ann = amf[iid]
        img = load_image_unicode(VL_P2_DIR / rec["rel_path"])
        mask = decode_mask_rle(ann["mask_rle"])
        metrics = _evaluate_line_support_on_raw(img, ann["endpoints_xyxy"], mask)

        in_overrides = iid in overrides
        roi = rec.get("roi_bbox_xywh")
        # Detect generic hint diagonal fallback: endpoints near [hx+0.1*hw, hy+0.08*hh, hx+0.9*hw, hy+0.92*hh]
        is_generic_diag = False
        if not in_overrides and roi is not None:
            hx, hy, hw, hh = [float(v) for v in roi]
            ex1, ey1, ex2, ey2 = [float(v) for v in ann["endpoints_xyxy"]]
            dx1 = abs(ex1 - (hx + 0.1 * hw)) + abs(ey1 - (hy + 0.08 * hh))
            dx2 = abs(ex2 - (hx + 0.9 * hw)) + abs(ey2 - (hy + 0.92 * hh))
            if dx1 < 8.0 and dx2 < 8.0:
                is_generic_diag = True

        # Detect raw image / centerline alignment problems
        needs_repair = False
        reasons = []
        if is_generic_diag:
            needs_repair = True
            reasons.append("hint_bbox_diagonal_fallback")
            fallback_diagonals.append(iid)
        if metrics["cl_support_frac"] < 0.48 and metrics["cl_contrast_frac"] < 0.44:
            needs_repair = True
            reasons.append("weak_centerline_shaft_support")
            weak_support.append(iid)
        if metrics["mask_bg_bleed"] > 0.55:
            needs_repair = True
            reasons.append("mask_excessive_background_bleed")
            high_bleed.append(iid)
        if metrics["line_len_norm_diag"] < 0.08 or metrics["mask_pixels"] < 30:
            needs_repair = True
            reasons.append("short_or_empty_mask")

        if in_overrides and not needs_repair:
            tier = "VERIFIED_MANUAL_GEOMETRY"
        elif in_overrides and needs_repair:
            tier = "MANUAL_OVERRIDE_NEEDS_MASK_OR_LINE_REFINEMENT"
        elif not in_overrides and not needs_repair:
            tier = "PSEUDO_LABEL_HINT_ASSISTED_CLEAN"
        else:
            tier = "PSEUDO_LABEL_NEEDS_REPAIR"

        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        rec_audit = {
            "id": iid,
            "vl_p2_split": split_name,
            "source_origin": "manual_override" if in_overrides else ("hint_bbox_assisted" if roi is not None else "auto"),
            "gt_quality_tier": tier,
            "needs_repair": needs_repair,
            "flag_reasons": reasons,
            "metrics": metrics,
        }
        audit_records.append(rec_audit)

        vis_pair = _draw_side_by_side_audit(img, ann, f"{iid} [{tier[:10]}]")
        if split_name == "vl_p2_dev_cal_v2":
            sheet_dev.append((f"{iid} s={metrics['cl_support_frac']:.2f}", vis_pair))
        else:
            sheet_fhold.append((f"{iid} s={metrics['cl_support_frac']:.2f}", vis_pair))
        if needs_repair:
            sheet_flagged.append((f"{iid} {','.join(reasons)[:24]}", vis_pair))

    sheets_dir = VL_P3_DIR / "cache" / "contact_sheets"
    if sheet_dev:
        make_contact_sheet(sheet_dev[:40], sheets_dir / "gate0_vl_p2_dev_gt_audit_p1.jpg", cols=4, cell_w=380, cell_h=200)
        if len(sheet_dev) > 40:
            make_contact_sheet(sheet_dev[40:], sheets_dir / "gate0_vl_p2_dev_gt_audit_p2.jpg", cols=4, cell_w=380, cell_h=200)
    if sheet_fhold:
        make_contact_sheet(sheet_fhold[:30], sheets_dir / "gate0_vl_p2_fhold_gt_audit_p1.jpg", cols=4, cell_w=380, cell_h=200)
        if len(sheet_fhold) > 30:
            make_contact_sheet(sheet_fhold[30:], sheets_dir / "gate0_vl_p2_fhold_gt_audit_p2.jpg", cols=4, cell_w=380, cell_h=200)
    if sheet_flagged:
        make_contact_sheet(sheet_flagged[:36], sheets_dir / "gate0_vl_p2_flagged_gt_audit.jpg", cols=4, cell_w=380, cell_h=200)

    return {
        "total_vl_p2_annotations": len(audit_records),
        "tier_counts": tier_counts,
        "summary": {
            "total_in_manual_overrides": sum(1 for r in audit_records if r["source_origin"] == "manual_override"),
            "total_hint_bbox_assisted": sum(1 for r in audit_records if r["source_origin"] == "hint_bbox_assisted"),
            "total_clean_without_repair": sum(1 for r in audit_records if not r["needs_repair"]),
            "total_flagged_for_repair": sum(1 for r in audit_records if r["needs_repair"]),
            "generic_diagonal_fallbacks": fallback_diagonals,
            "weak_centerline_support_ids": weak_support,
            "high_mask_bg_bleed_ids": high_bleed,
        },
        "records": audit_records,
    }


def main():
    print("Running Gate 0 Git Audit...")
    git_res = audit_git_baselines()
    print("Running Gate 0 VL-P2 Post-Optimization Regression Audit (all 180 final_holdout_v2 images)...")
    reg_res = audit_vl_p2_post_optimization_regression()
    print("Running Gate 0 VL-P2 Annotation Quality Audit (all 116 object_01 annotations)...")
    ann_res = audit_vl_p2_annotations()

    out_manifest = {
        "git_audit": git_res,
        "vl_p2_post_opt_regression": reg_res,
        "vl_p2_gt_annotation_audit": ann_res,
    }
    out_path = VL_P3_DIR / "manifests" / "vl_p2_post_opt_regression_audit.json"
    out_path.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved Gate 0 audit JSON ->", out_path)
    print("Git OK:", git_res["all_protected_commits_present_and_ancestry_valid"])
    print("Post-opt C2 Recall:", reg_res["post_optimization_measured"]["known_recall_object_01"]["formatted"])
    print("Post-opt C2 FAR:", reg_res["post_optimization_measured"]["far"]["formatted"])
    print("Post-opt Latency P50/P95:", reg_res["post_optimization_measured"]["latency"]["total_e2e_p50_ms"], "/", reg_res["post_optimization_measured"]["latency"]["total_e2e_p95_ms"])
    print("Positives changed:", len(reg_res["per_image_comparison"]["positives_changed"]), reg_res["per_image_comparison"]["positives_changed"])
    print("Annotation tiers:", ann_res["tier_counts"])


if __name__ == "__main__":
    main()
