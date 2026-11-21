"""Step 3c: Curated VL-P1b dataset builder, GT bbox verifier, and split isolation gatekeeper.

Two-stage workflow:
  --stage init     Selects allowlisted, cluster-disjoint splits from _staging_pool*,
                   proposes initial tight GT boxes, and renders high-res green-box
                   verification contact sheets (`gt_verify_*.jpg`) for human/agent inspection.
  --stage finalize Applies any curated `manifests/manual_gt_bboxes.json` box overrides,
                   writes physical split directories, crops verified ROIs, asserts 0 overlaps
                   (SHA-256, title, cluster, aHash/dHash <= 6) across all splits & VL-P1,
                   and freezes `manifests/dataset_manifest.json` + `dataset_manifest.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from .collect_vl_p1b_dataset import _load_vl_p1_signatures, _normalize_title
from .common_utils import (
    LABELS_VI,
    OBJECT_IDS,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    list_images,
    load_image_unicode,
    make_contact_sheet,
    save_image_unicode,
)
from .labeling_protocol import HARD_NEGATIVE_GROUPS, is_allowed_open_license

# Visually audited allowlists per staging source.
# Each tuple is (pool_tag, group_key, allowed_stems_list).
# pool_tag in {"primary", "supp", "extra"}
CLEAN_ALLOWLISTS: dict[str, list[tuple[str, str, list[str]]]] = {
    "pos/object_01": [
        ("primary", "pos/object_01", [
            "cand_004", "cand_006", "cand_012", "cand_013", "cand_015", "cand_016",
            "cand_018", "cand_019", "cand_021", "cand_025", "cand_033", "cand_035",
            "cand_040", "cand_041", "cand_042", "cand_043", "cand_044", "cand_045",
        ]),
        ("supp", "pos/object_01", [
            "supp_001", "supp_007", "supp_009", "supp_010", "supp_011", "supp_012",
            "supp_013", "supp_015", "supp_016", "supp_017", "supp_020", "supp_022",
            "supp_023", "supp_029", "supp_030", "supp_032",
        ]),
    ],
    "pos/object_02": [
        ("primary", "pos/object_02", [
            "cand_009", "cand_025", "cand_030", "cand_032", "cand_039",
        ]),
        ("supp", "pos/object_02", ["supp_005", "supp_018"]),
        ("extra", "pos/object_02_keys", [
            "extra_003", "extra_007", "extra_015", "extra_016", "extra_017", "extra_023",
            "extra_025", "extra_026", "extra_029", "extra_030", "extra_031", "extra_035",
        ]),
        ("extra", "pos/object_02_micro", [
            "micro_002", "micro_008", "micro_009", "micro_012", "micro_019",
        ]),
        ("extra", "pos/object_02_micro2", [
            "micro_003", "micro_006", "micro_007", "micro_010", "micro_012", "micro_013",
        ]),
    ],
    "pos/object_03": [
        ("primary", "pos/object_03", [
            "cand_001", "cand_003", "cand_004", "cand_005", "cand_006", "cand_007",
            "cand_008", "cand_010", "cand_012", "cand_014", "cand_015", "cand_017",
            "cand_021", "cand_023", "cand_027", "cand_028", "cand_029", "cand_032",
            "cand_033", "cand_034", "cand_035", "cand_037", "cand_038", "cand_039",
            "cand_040", "cand_042", "cand_043", "cand_045",
        ]),
        ("extra", "pos/object_03_micro", ["micro_004", "micro_005", "micro_006", "micro_009", "micro_010"]),
    ],
    "pos/object_04": [
        ("primary", "pos/object_04", [
            "cand_001", "cand_003", "cand_004", "cand_005", "cand_006", "cand_007",
            "cand_010", "cand_012", "cand_016", "cand_017", "cand_020", "cand_022",
            "cand_023", "cand_024", "cand_028", "cand_035", "cand_036", "cand_039",
            "cand_040", "cand_041", "cand_045",
        ]),
        ("supp", "pos/object_04", [
            "supp_001", "supp_004", "supp_005", "supp_006", "supp_007", "supp_008",
            "supp_010", "supp_012",
        ]),
    ],
    "pos/object_05": [
        ("primary", "pos/object_05", [
            "cand_001", "cand_002", "cand_003", "cand_005", "cand_007", "cand_009",
            "cand_010", "cand_011", "cand_012", "cand_013", "cand_015", "cand_016",
            "cand_017", "cand_018", "cand_020", "cand_022", "cand_023", "cand_024",
            "cand_026", "cand_027", "cand_028", "cand_029", "cand_030", "cand_031",
            "cand_032",
        ]),
    ],
    "neg/glasses_case_confuser": [
        ("primary", "neg/glasses_case_confuser", [
            "cand_001", "cand_003", "cand_004", "cand_006", "cand_008", "cand_009",
            "cand_014", "cand_015", "cand_016", "cand_017", "cand_018", "cand_019",
            "cand_020", "cand_021", "cand_022", "cand_023", "cand_024", "cand_026",
        ]),
        ("supp", "neg/glasses_case_confuser", ["supp_001", "supp_004", "supp_006", "supp_007"]),
    ],
    "neg/notebook_pouch_confuser": [
        ("primary", "neg/notebook_pouch_confuser", [
            "cand_003", "cand_004", "cand_006", "cand_008", "cand_009", "cand_010",
            "cand_012", "cand_017", "cand_024", "cand_025",
        ]),
        ("supp", "neg/notebook_pouch_confuser", [
            "supp_004", "supp_005", "supp_007", "supp_008", "supp_009", "supp_010",
            "supp_015", "supp_018", "supp_020",
        ]),
    ],
    "neg/walking_stick_confuser": [
        ("primary", "neg/walking_stick_confuser", [
            "cand_001", "cand_002", "cand_003", "cand_009", "cand_010", "cand_014",
            "cand_015", "cand_018", "cand_019", "cand_021", "cand_022", "cand_025",
        ]),
        ("supp", "neg/walking_stick_confuser", [
            "supp_001", "supp_003", "supp_006", "supp_010", "supp_015",
        ]),
    ],
    "neg/long_metal_rod_confuser": [
        ("primary", "neg/long_metal_rod_confuser", [
            "cand_001", "cand_004", "cand_005", "cand_006", "cand_009", "cand_010", "cand_014",
        ]),
        ("supp", "neg/long_metal_rod_confuser", [
            "supp_001", "supp_002", "supp_007", "supp_008", "supp_011",
            "supp_014", "supp_017", "supp_018", "supp_020", "supp_021",
        ]),
    ],
    "neg/lock_key_metal_confuser": [
        ("primary", "neg/lock_key_metal_confuser", [
            "cand_004", "cand_005", "cand_006", "cand_007", "cand_010", "cand_011",
            "cand_014", "cand_017", "cand_018", "cand_021", "cand_022", "cand_023",
            "cand_024", "cand_026",
        ]),
        ("supp", "neg/lock_key_metal_confuser", [
            "supp_001", "supp_002", "supp_004", "supp_005", "supp_007", "supp_008",
            "supp_011", "supp_012", "supp_014", "supp_020", "supp_021", "supp_023", "supp_025",
        ]),
    ],
    "neg/footwear_sock_confuser": [
        ("primary", "neg/footwear_sock_confuser", ["cand_001", "cand_002", "cand_003", "cand_020"]),
        ("supp", "neg/footwear_sock_confuser", ["supp_001", "supp_002", "supp_010", "supp_016"]),
        ("extra", "neg/footwear_sock_extra", [
            "extra_003", "extra_009", "extra_011", "extra_019",
            "extra_020", "extra_022", "extra_023", "extra_024",
        ]),
    ],
    "neg/umbrella_handle_confuser": [
        ("primary", "neg/umbrella_handle_confuser", ["cand_006", "cand_011"]),
        ("extra", "neg/umbrella_handle_extra", [
            "extra_003", "extra_005", "extra_007", "extra_008", "extra_009",
            "extra_011", "extra_019", "extra_021", "extra_022",
        ]),
        ("extra", "neg/umbrella_micro", ["micro_003", "micro_005", "micro_007", "micro_009", "micro_011"]),
    ],
    "neg/general_distractor": [
        ("primary", "neg/general_distractor", [
            "cand_001", "cand_003", "cand_004", "cand_005", "cand_006", "cand_007",
            "cand_008", "cand_010", "cand_011", "cand_012", "cand_017", "cand_018",
            "cand_020", "cand_022", "cand_023", "cand_024", "cand_025", "cand_027",
            "cand_028", "cand_029", "cand_030", "cand_031", "cand_032", "cand_033",
            "cand_034", "cand_035", "cand_039", "cand_040", "cand_043", "cand_044",
            "cand_046", "cand_047", "cand_054", "cand_055",
        ]),
        ("extra", "neg/general_household_extra", [
            "extra_001", "extra_002", "extra_003", "extra_004", "extra_006", "extra_010",
            "extra_011", "extra_012", "extra_013", "extra_015", "extra_018", "extra_021",
            "extra_022", "extra_026", "extra_028", "extra_030", "extra_031", "extra_032",
        ]),
        ("extra", "neg/general_household_extra2", [
            "micro_002", "micro_006", "micro_007", "micro_012",
            "micro_013", "micro_015", "micro_017", "micro_020",
        ]),
    ],
    "amb/ambiguous_boundary": [
        ("primary", "amb/ambiguous_boundary", ["cand_004", "cand_006", "cand_011", "cand_012"]),
        ("primary", "pos/object_02", ["cand_011", "cand_040"]),
        ("primary", "neg/lock_key_metal_confuser", ["cand_002", "cand_008", "cand_012"]),
    ],
}


def _session_cluster_key(norm_title: str) -> str:
    """Group multi-angle shots of the exact same item/table/museum specimen into one cluster."""
    stem = norm_title.rsplit(".", 1)[0]
    # Remove trailing index markers like _01, _(1), -2, 003, _detail
    cleaned = re.sub(r"(\([^)]*\)|[-_ ]+\d+|_detail|_detalj|_\d+[a-z]?)$", "", stem)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if not cleaned or len(cleaned) < 6 or cleaned in {"500px_photo", "i_i", "flickr", "unsplash"}:
        cleaned = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    # Collapse by first 24 chars if long museum archive identifier
    return cleaned[:24] if cleaned else stem[:24]


def _estimate_initial_gt_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Offline foreground saliency heuristic to propose an initial GT box for human visual review."""
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]], axis=0)
    bg_med = np.median(border, axis=0)
    bg_std = np.mean(np.std(border, axis=0))

    # Uniform studio backdrop -> tight foreground mask
    if bg_std < 18.0:
        dist = np.linalg.norm(lab - bg_med, axis=2)
        thr = max(14.0, float(np.percentile(dist, 98)) * 0.16)
        mask = (dist >= thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        ys, xs = np.where(mask > 0)
        if xs.size >= 0.01 * w * h:
            pad_x = max(4, int(0.03 * w))
            pad_y = max(4, int(0.03 * h))
            x1 = max(0, int(xs.min()) - pad_x)
            y1 = max(0, int(ys.min()) - pad_y)
            x2 = min(w, int(xs.max()) + pad_x)
            y2 = min(h, int(ys.max()) + pad_y)
            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                return (x1, y1, x2 - x1, y2 - y1)

    # Complex scene -> central salient foreground region
    blur_f = cv2.GaussianBlur(lab, (5, 5), 0)
    blur_c = cv2.GaussianBlur(lab, (41, 41), 0)
    sal = np.linalg.norm(blur_f - blur_c, axis=2)
    u8 = np.clip((sal / (float(np.percentile(sal, 98)) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    _, otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score, best_box = -1.0, (int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h))
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 0.015 * w * h or area > 0.92 * w * h:
            continue
        cx, cy = (bx + bw / 2.0) / w, (by + bh / 2.0) / h
        c_prior = 1.0 - 0.65 * np.hypot(cx - 0.5, cy - 0.5)
        sc = float(np.mean(sal[by:by + bh, bx:bx + bw])) * c_prior
        if sc > best_score:
            best_score = sc
            px = max(0, bx - int(0.03 * w))
            py = max(0, by - int(0.03 * h))
            pw = min(w - px, bw + 2 * int(0.03 * w))
            ph = min(h - py, bh + 2 * int(0.03 * h))
            best_box = (px, py, pw, ph)
    return best_box


def _load_staging_records(vl_p1b_dir: Path) -> dict[tuple[str, str], dict[str, dict]]:
    sources = {
        "primary": (vl_p1b_dir / "data" / "_staging_pool", "staging_metadata.json"),
        "supp": (vl_p1b_dir / "data" / "_staging_pool_supplements", "supplements_metadata.json"),
        "extra": (vl_p1b_dir / "data" / "_staging_pool_extra", "extra_metadata.json"),
    }
    by_pool_group: dict[tuple[str, str], dict[str, dict]] = {}
    for tag, (root_dir, meta_name) in sources.items():
        mp = root_dir / meta_name
        if not mp.is_file():
            continue
        data = json.loads(mp.read_text(encoding="utf-8"))
        for gkey, recs in data.items():
            stem_map = {}
            for r in recs:
                p = root_dir / r["local_rel"]
                if p.is_file() and is_allowed_open_license(r.get("license")):
                    row = dict(r)
                    row["_abs_path"] = str(p)
                    row["_pool_tag"] = tag
                    stem_map[p.stem] = row
            by_pool_group[(tag, gkey)] = stem_map
    return by_pool_group


def _collect_allowlisted_clusters(by_pool_group: dict, target_key: str) -> list[list[dict]]:
    entries: list[dict] = []
    seen_sha: set[str] = set()
    for pool_tag, gkey, stems in CLEAN_ALLOWLISTS[target_key]:
        stem_map = by_pool_group.get((pool_tag, gkey), {})
        for st in stems:
            rec = stem_map.get(st)
            if not rec or rec["sha256"] in seen_sha:
                continue
            seen_sha.add(rec["sha256"])
            r = dict(rec)
            r["cluster_id"] = _session_cluster_key(r["norm_title"])
            entries.append(r)

    # Deduplicate near-identical perceptual hashes into shared clusters so splits never leak!
    clusters: list[list[dict]] = []
    by_cid: dict[str, list[dict]] = {}
    for r in entries:
        cid = r["cluster_id"]
        merged = False
        for existing_cluster in clusters:
            if any(
                min(hex_hamming_distance(r["ahash"], ex["ahash"]), hex_hamming_distance(r["dhash"], ex["dhash"])) <= 8
                for ex in existing_cluster
            ):
                r["cluster_id"] = existing_cluster[0]["cluster_id"]
                existing_cluster.append(r)
                merged = True
                break
        if not merged:
            if cid in by_cid:
                by_cid[cid].append(r)
            else:
                new_c = [r]
                clusters.append(new_c)
                by_cid[cid] = new_c
    return clusters


def run_init(vl_p1_dir: Path, vl_p1b_dir: Path) -> dict:
    by_pool_group = _load_staging_records(vl_p1b_dir)
    selected: dict[str, list[dict]] = {
        "calibration_v1b": [],
        "holdout": [],
        "hard_neg_gallery": [],
        "ambiguous_boundary": [],
    }
    used_hashes: list[tuple[str, str, str]] = []

    # Step 1: Select positives (5 cal + 20 holdout per class = 25 cal, 100 holdout)
    for oid in OBJECT_IDS:
        gkey = f"pos/{oid}"
        clusters = _collect_allowlisted_clusters(by_pool_group, gkey)
        # Prefer taking at most 1 item per cluster first, then 2nd within holdout ONLY if needed
        distinct_reps = [c[0] for c in clusters]
        if len(distinct_reps) < 25:
            raise RuntimeError(f"{gkey}: only {len(distinct_reps)} distinct clean clusters (< 25 needed)")
        cal_picks = distinct_reps[:5]
        hold_picks = distinct_reps[5:25]
        for idx, rec in enumerate(cal_picks, 1):
            img = load_image_unicode(Path(rec["_abs_path"]))
            rec["split"] = "calibration_v1b"
            rec["label_id"] = oid
            rec["confuser_group"] = None
            rec["id"] = f"cal_{oid}_{idx:02d}"
            rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
            selected["calibration_v1b"].append(rec)
        for idx, rec in enumerate(hold_picks, 1):
            img = load_image_unicode(Path(rec["_abs_path"]))
            rec["split"] = "holdout"
            rec["label_id"] = oid
            rec["confuser_group"] = None
            rec["id"] = f"hold_{oid}_{idx:02d}"
            rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
            selected["holdout"].append(rec)

    # Step 2: Select 7 hard-negative groups across (hard_neg_gallery, calibration_v1b, holdout)
    # Allocate per group deterministically: max diversity across splits, NEVER same cluster in 2 splits!
    total_cal_neg_target = 25  # 3-4 per group across 7 groups
    gallery_per_group = 7      # 7*7 = 49+, capped up to available -> >= 50 total
    for g_idx, gname in enumerate(HARD_NEGATIVE_GROUPS):
        clusters = _collect_allowlisted_clusters(by_pool_group, f"neg/{gname}")
        reps = [c[0] for c in clusters]
        n_avail = len(reps)
        if n_avail < 10:
            raise RuntimeError(f"neg/{gname}: only {n_avail} clean clusters (< 10)")
        n_cal = 4 if g_idx < 4 else 3  # 4+4+4+4+3+3+3 = 25 calibration hard-negatives!
        # Give holdout around 10-12 per group (~75-85 total) and gallery remaining (>=7 per group -> >=50 total)
        rem = n_avail - n_cal
        n_gal = min(max(7, rem // 2), max(5, rem - 7))
        cal_neg = reps[:n_cal]
        gal_neg = reps[n_cal:n_cal + n_gal]
        hold_neg = reps[n_cal + n_gal:]
        # If extra items exist in clusters for hold_neg only (same cluster stays ONLY in holdout), keep 1 rep/cluster
        for i, rec in enumerate(gal_neg, 1):
            rec["split"] = "hard_neg_gallery"
            rec["label_id"] = "unknown"
            rec["confuser_group"] = gname
            rec["id"] = f"hng_{gname[:6]}_{i:02d}"
            rec["roi_bbox_xywh"] = None
            selected["hard_neg_gallery"].append(rec)
        for i, rec in enumerate(cal_neg, 1):
            img = load_image_unicode(Path(rec["_abs_path"]))
            rec["split"] = "calibration_v1b"
            rec["label_id"] = "unknown"
            rec["confuser_group"] = gname
            rec["id"] = f"cal_neg_{gname[:6]}_{i:02d}"
            rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
            selected["calibration_v1b"].append(rec)
        for i, rec in enumerate(hold_neg, 1):
            img = load_image_unicode(Path(rec["_abs_path"]))
            rec["split"] = "holdout"
            rec["label_id"] = "unknown"
            rec["confuser_group"] = gname
            rec["id"] = f"hold_neg_{gname[:6]}_{i:02d}"
            rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
            selected["holdout"].append(rec)

    # Step 3: Add general household distractors to holdout negatives (>= 40 -> total holdout neg >= 120)
    gen_clusters = _collect_allowlisted_clusters(by_pool_group, "neg/general_distractor")
    gen_reps = [c[0] for c in gen_clusters]
    for i, rec in enumerate(gen_reps[:75], 1):
        img = load_image_unicode(Path(rec["_abs_path"]))
        rec["split"] = "holdout"
        rec["label_id"] = "unknown"
        rec["confuser_group"] = "general_distractor"
        rec["id"] = f"hold_gen_{i:02d}"
        rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
        selected["holdout"].append(rec)

    # Step 4: Ambiguous boundary items (6-10 items disjoint from cal/hold/gal)
    used_cluster_ids = {
        r["cluster_id"]
        for lst in (selected["calibration_v1b"], selected["holdout"], selected["hard_neg_gallery"])
        for r in lst
    }
    used_sha_set = {
        r["sha256"]
        for lst in (selected["calibration_v1b"], selected["holdout"], selected["hard_neg_gallery"])
        for r in lst
    }
    amb_clusters = _collect_allowlisted_clusters(by_pool_group, "amb/ambiguous_boundary")
    idx_a = 1
    for cl in amb_clusters:
        rec = cl[0]
        if rec["cluster_id"] in used_cluster_ids or rec["sha256"] in used_sha_set:
            continue
        rec["split"] = "ambiguous_boundary"
        rec["label_id"] = "ambiguous"
        rec["confuser_group"] = "boundary_case"
        rec["id"] = f"amb_{idx_a:02d}"
        rec["roi_bbox_xywh"] = None
        selected["ambiguous_boundary"].append(rec)
        idx_a += 1

    # Save draft selection + render GREEN bbox contact sheets for ALL positives
    draft_path = vl_p1b_dir / "manifests" / "draft_selection.json"
    draft_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    cs_dir = vl_p1b_dir / "manifests" / "contact_sheets_v1b"
    _render_gt_verification_sheets(selected, cs_dir)

    summary = {k: len(v) for k, v in selected.items()}
    summary["holdout_pos"] = sum(1 for r in selected["holdout"] if r["label_id"] != "unknown")
    summary["holdout_neg"] = sum(1 for r in selected["holdout"] if r["label_id"] == "unknown")
    print("[Init Summary]", json.dumps(summary, indent=2))
    return selected


def _render_gt_verification_sheets(selected: dict, cs_dir: Path) -> None:
    cal_pos = [
        (f"{r['id']} | {r['label_id']}", load_image_unicode(Path(r["_abs_path"])), tuple(r["roi_bbox_xywh"]))
        for r in selected["calibration_v1b"]
        if r["label_id"] != "unknown"
    ]
    make_contact_sheet(cal_pos, cs_dir / "gt_verify_cal_pos.jpg", cols=5, cell_w=280, cell_h=220)

    hold_pos = [r for r in selected["holdout"] if r["label_id"] != "unknown"]
    for page in range(4):
        chunk = hold_pos[page * 25:(page + 1) * 25]
        if not chunk:
            continue
        items = [
            (f"{r['id']} | {r['label_id']}", load_image_unicode(Path(r["_abs_path"])), tuple(r["roi_bbox_xywh"]))
            for r in chunk
        ]
        make_contact_sheet(items, cs_dir / f"gt_verify_holdout_pos_{page + 1}.jpg", cols=5, cell_w=280, cell_h=220)

    gal_items = [
        (f"{r['id']} | {r['confuser_group'][:12]}", load_image_unicode(Path(r["_abs_path"])), None)
        for r in selected["hard_neg_gallery"]
    ]
    make_contact_sheet(gal_items, cs_dir / "gt_verify_hard_neg_gallery.jpg", cols=6, cell_w=220, cell_h=175)


def run_finalize(vl_p1_dir: Path, vl_p1b_dir: Path) -> dict:
    draft_path = vl_p1b_dir / "manifests" / "draft_selection.json"
    selected = json.loads(draft_path.read_text(encoding="utf-8"))

    overrides_path = vl_p1b_dir / "manifests" / "manual_gt_bboxes.json"
    overrides: dict[str, list[float]] = {}
    if overrides_path.is_file():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))

    # Apply normalized [x1, y1, x2, y2] manual GT overrides & verify all boxes
    for split_key in ("calibration_v1b", "holdout"):
        for rec in selected[split_key]:
            img = load_image_unicode(Path(rec["_abs_path"]))
            ih, iw = img.shape[:2]
            if rec["id"] in overrides:
                nx1, ny1, nx2, ny2 = overrides[rec["id"]]
                bx = max(0, min(iw - 2, int(round(nx1 * iw))))
                by = max(0, min(ih - 2, int(round(ny1 * ih))))
                bw = max(12, min(iw - bx, int(round((nx2 - nx1) * iw))))
                bh = max(12, min(ih - by, int(round((ny2 - ny1) * ih))))
                rec["roi_bbox_xywh"] = [bx, by, bw, bh]
                rec["bbox_verified_mode"] = "manual_visual_override"
            else:
                rec["bbox_verified_mode"] = "offline_saliency_visual_audited"

    # Re-render GT contact sheets after overrides
    _render_gt_verification_sheets(selected, vl_p1b_dir / "manifests" / "contact_sheets_v1b")

    # Verify strict 0-overlap guarantees against VL-P1 and across ALL VL-P1b splits
    vl_shas, vl_titles, vl_phashes = _load_vl_p1_signatures(vl_p1_dir)
    split_shas: dict[str, set[str]] = {}
    split_clusters: dict[str, set[str]] = {}
    split_phashes: dict[str, list[tuple[str, str, str]]] = {}

    for sp_name, recs in selected.items():
        s_shas, s_clus, s_phs = set(), set(), []
        for r in recs:
            assert is_allowed_open_license(r["license"]), f"Disallowed license in {r['id']}: {r['license']}"
            assert r["sha256"] not in vl_shas, f"SHA collision with VL-P1 on {r['id']}"
            assert r["norm_title"] not in vl_titles, f"Title collision with VL-P1 on {r['id']}"
            for vid, vah, vdh in vl_phashes:
                if min(hex_hamming_distance(r["ahash"], vah), hex_hamming_distance(r["dhash"], vdh)) <= 6:
                    raise RuntimeError(f"pHash <= 6 collision with VL-P1 {vid} on {r['id']}")
            r["cluster_id"] = _session_cluster_key(r["norm_title"])
            s_shas.add(r["sha256"])
            s_clus.add(r["cluster_id"])
            s_phs.append((r["id"], r["ahash"], r["dhash"]))
        split_shas[sp_name] = s_shas
        split_clusters[sp_name] = s_clus
        split_phashes[sp_name] = s_phs

    split_names = list(selected.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            assert not (split_shas[a] & split_shas[b]), f"SHA overlap between {a} and {b}"
            assert not (split_clusters[a] & split_clusters[b]), f"Cluster overlap between {a} and {b}"
            for id_a, ah_a, dh_a in split_phashes[a]:
                for id_b, ah_b, dh_b in split_phashes[b]:
                    if min(hex_hamming_distance(ah_a, ah_b), hex_hamming_distance(dh_a, dh_b)) <= 6:
                        raise RuntimeError(f"pHash <= 6 cross-split collision: {id_a} vs {id_b}")

    # Materialize clean physical dataset directories
    data_dir = vl_p1b_dir / "data"
    for folder in (
        "hard_neg_gallery",
        "calibration_v1b",
        "calibration_v1b_full",
        "holdout_roi",
        "holdout_full",
        "ambiguous_boundary",
    ):
        d = data_dir / folder
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[dict]] = {
        "train_pos": [],
        "hard_neg_gallery": [],
        "calibration_v1b_roi": [],
        "calibration_v1b_full": [],
        "holdout_roi": [],
        "holdout_full": [],
        "ambiguous_boundary": [],
    }

    # Index existing train_pos (copied from VL-P1)
    for oid in OBJECT_IDS:
        for p in list_images(data_dir / "train_pos" / oid):
            img = load_image_unicode(p)
            manifest["train_pos"].append({
                "id": f"train_{oid}_{p.stem}",
                "label_id": oid,
                "label_vi": LABELS_VI[oid],
                "rel_path": p.relative_to(vl_p1b_dir).as_posix(),
                "sha256": compute_sha256(p),
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            })

    # Write hard_neg_gallery
    for r in selected["hard_neg_gallery"]:
        sub = data_dir / "hard_neg_gallery" / r["confuser_group"]
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / f"{r['id']}.jpg"
        shutil.copy2(r["_abs_path"], dst)
        manifest["hard_neg_gallery"].append(_public_record(r, vl_p1b_dir, full_p=dst, roi_p=dst))

    # Write calibration_v1b_full + cropped calibration_v1b
    for r in selected["calibration_v1b"]:
        img = load_image_unicode(Path(r["_abs_path"]))
        cls_dir = r["label_id"] if r["label_id"] != "unknown" else f"unknown/{r['confuser_group']}"
        full_sub = data_dir / "calibration_v1b_full" / cls_dir
        roi_sub = data_dir / "calibration_v1b" / cls_dir
        full_sub.mkdir(parents=True, exist_ok=True)
        roi_sub.mkdir(parents=True, exist_ok=True)
        full_dst = full_sub / f"{r['id']}.jpg"
        roi_dst = roi_sub / f"{r['id']}.jpg"
        shutil.copy2(r["_abs_path"], full_dst)
        bx, by, bw, bh = r["roi_bbox_xywh"]
        crop = img[by:by + bh, bx:bx + bw]
        save_image_unicode(roi_dst, crop if crop.size > 0 else img)
        rec_full = _public_record(r, vl_p1b_dir, full_p=full_dst, roi_p=roi_dst)
        manifest["calibration_v1b_full"].append(rec_full)
        manifest["calibration_v1b_roi"].append(rec_full)

    # Write holdout_full + cropped holdout_roi
    for r in selected["holdout"]:
        img = load_image_unicode(Path(r["_abs_path"]))
        cls_dir = r["label_id"] if r["label_id"] != "unknown" else f"unknown/{r['confuser_group']}"
        full_sub = data_dir / "holdout_full" / cls_dir
        roi_sub = data_dir / "holdout_roi" / cls_dir
        full_sub.mkdir(parents=True, exist_ok=True)
        roi_sub.mkdir(parents=True, exist_ok=True)
        full_dst = full_sub / f"{r['id']}.jpg"
        roi_dst = roi_sub / f"{r['id']}.jpg"
        shutil.copy2(r["_abs_path"], full_dst)
        bx, by, bw, bh = r["roi_bbox_xywh"]
        crop = img[by:by + bh, bx:bx + bw]
        save_image_unicode(roi_dst, crop if crop.size > 0 else img)
        rec_full = _public_record(r, vl_p1b_dir, full_p=full_dst, roi_p=roi_dst)
        manifest["holdout_full"].append(rec_full)
        manifest["holdout_roi"].append(rec_full)

    # Write ambiguous_boundary
    for r in selected["ambiguous_boundary"]:
        dst = data_dir / "ambiguous_boundary" / f"{r['id']}.jpg"
        shutil.copy2(r["_abs_path"], dst)
        manifest["ambiguous_boundary"].append(_public_record(r, vl_p1b_dir, full_p=dst, roi_p=dst))

    out_manifest = {
        "schema_version": "vl_p1b_v1",
        "counts": {k: len(v) for k, v in manifest.items()},
        "holdout_breakdown": {
            "positives_by_class": {
                oid: sum(1 for r in manifest["holdout_full"] if r["label_id"] == oid)
                for oid in OBJECT_IDS
            },
            "negatives_by_group": _count_groups(manifest["holdout_full"]),
        },
        "hard_neg_gallery_by_group": _count_groups(manifest["hard_neg_gallery"]),
        "calibration_v1b_by_group": _count_groups(manifest["calibration_v1b_full"]),
        "zero_overlap_verified": True,
        **manifest,
    }

    mf_txt = json.dumps(out_manifest, indent=2, ensure_ascii=False)
    (vl_p1b_dir / "manifests" / "dataset_manifest.json").write_text(mf_txt, encoding="utf-8")
    (vl_p1b_dir / "dataset_manifest.json").write_text(mf_txt, encoding="utf-8")
    print("[Finalized Manifest Counts]", json.dumps(out_manifest["counts"], indent=2))
    print("[Holdout Breakdown]", json.dumps(out_manifest["holdout_breakdown"], indent=2))
    return out_manifest


def _count_groups(recs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in recs:
        if r["label_id"] == "unknown":
            g = r["confuser_group"]
            out[g] = out.get(g, 0) + 1
    return dict(sorted(out.items()))


def _public_record(r: dict, vl_p1b_dir: Path, full_p: Path, roi_p: Path) -> dict:
    return {
        "id": r["id"],
        "label_id": r["label_id"],
        "label_vi": LABELS_VI.get(r["label_id"], r["label_id"]),
        "confuser_group": r.get("confuser_group"),
        "full_rel_path": full_p.relative_to(vl_p1b_dir).as_posix(),
        "roi_rel_path": roi_p.relative_to(vl_p1b_dir).as_posix(),
        "roi_bbox_xywh": r.get("roi_bbox_xywh"),
        "bbox_verified_mode": r.get("bbox_verified_mode"),
        "sha256": r["sha256"],
        "ahash": r["ahash"],
        "dhash": r["dhash"],
        "cluster_id": r["cluster_id"],
        "title": r["title"],
        "source_url": r["source_url"],
        "license": r["license"],
        "author": r["author"],
        "width": r["width"],
        "height": r["height"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("init", "finalize"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.stage == "init":
        run_init(root / "research" / "vl_p1", root / "research" / "vl_p1b")
    else:
        run_finalize(root / "research" / "vl_p1", root / "research" / "vl_p1b")
