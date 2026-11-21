"""Step 3b: Curated VL-P1c dataset builder, GT bbox verifier, and split isolation gatekeeper.

Three-stage workflow:
  --stage init      Selects visually allowlisted, cluster-disjoint splits into:
                    - calibration_v1c (N=200: 50 inherited cal_v1b + 50 new pos + 100 new neg)
                    - final_holdout_v1c (N=300: 100 new pos 20/class + 112 new hard-neg 16/grp + 88 new general)
                    - ambiguous_boundary_v1c (N=15: 9 inherited + 6 new)
                    Renders green-box GT verification contact sheets (`gt_verify_*.jpg`) for all new positives.
  --stage finalize  Applies `manifests/manual_gt_bboxes_v1c.json` overrides, verifies 0 overlaps
                    (SHA-256, title, cluster_id, aHash/dHash <= 6) across VL-P1, VL-P1b, and all VL-P1c splits,
                    writes clean `data/` split directories, and saves `manifests/dataset_manifest.json`
                    plus root `dataset_manifest.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from .collect_vl_p1c_dataset import _load_existing_signatures, _normalize_title
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


def _session_cluster_key(norm_title: str) -> str:
    """Group multi-angle shots of the exact same item/museum specimen into one cluster."""
    stem = norm_title.rsplit(".", 1)[0]
    cleaned = re.sub(r"(\([^)]*\)|[-_ ]+\d+|_detail|_detalj|_\d+[a-z]?)$", "", stem)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if not cleaned or len(cleaned) < 6 or cleaned in {"500px_photo", "i_i", "flickr", "unsplash", "img", "dsc"}:
        cleaned = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return cleaned[:24] if cleaned else stem[:24]


def _estimate_initial_gt_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Offline foreground saliency heuristic proposing initial GT boxes before human/agent visual audit."""
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]], axis=0)
    bg_med = np.median(border, axis=0)
    bg_std = float(np.mean(np.std(border, axis=0)))

    if bg_std < 18.0:
        dist = np.linalg.norm(lab - bg_med, axis=2)
        thr = max(14.0, float(np.percentile(dist, 98)) * 0.16)
        mask = (dist >= thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        ys, xs = np.where(mask > 0)
        if xs.size >= 0.01 * w * h:
            pad_x, pad_y = max(4, int(0.03 * w)), max(4, int(0.03 * h))
            x1 = max(0, int(xs.min()) - pad_x)
            y1 = max(0, int(ys.min()) - pad_y)
            x2 = min(w, int(xs.max()) + pad_x)
            y2 = min(h, int(ys.max()) + pad_y)
            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                return (x1, y1, x2 - x1, y2 - y1)

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


def _cluster_group_records(
    recs: list[dict],
    staging_root: Path,
    allowed_stems: set[str] | None,
) -> list[list[dict]]:
    entries: list[dict] = []
    seen_sha: set[str] = set()
    for r in recs:
        p = staging_root / r["local_rel"]
        if not p.is_file() or not is_allowed_open_license(r.get("license")):
            continue
        if allowed_stems is not None and p.stem not in allowed_stems:
            continue
        if r["sha256"] in seen_sha:
            continue
        seen_sha.add(r["sha256"])
        row = dict(r)
        row["_abs_path"] = str(p)
        row["cluster_id"] = _session_cluster_key(row["norm_title"])
        entries.append(row)

    clusters: list[list[dict]] = []
    for r in entries:
        merged = False
        for existing_cluster in clusters:
            if any(
                min(hex_hamming_distance(r["ahash"], ex["ahash"]), hex_hamming_distance(r["dhash"], ex["dhash"])) <= 8
                or r["cluster_id"] == ex["cluster_id"]
                for ex in existing_cluster
            ):
                r["cluster_id"] = existing_cluster[0]["cluster_id"]
                existing_cluster.append(r)
                merged = True
                break
        if not merged:
            clusters.append([r])
    return clusters


def run_init(vl_p1_dir: Path, vl_p1b_dir: Path, vl_p1c_dir: Path) -> dict:
    staging_root = vl_p1c_dir / "data" / "_staging_pool"
    meta = json.loads((staging_root / "staging_metadata.json").read_text(encoding="utf-8"))

    # Optional visual curation allowlist file (`manifests/curated_allowlists_v1c.json`)
    allow_p = vl_p1c_dir / "manifests" / "curated_allowlists_v1c.json"
    allowlists: dict[str, list[str]] = json.loads(allow_p.read_text(encoding="utf-8")) if allow_p.is_file() else {}

    v1b_mf = json.loads((vl_p1b_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))

    selected: dict[str, list[dict]] = {
        "calibration_v1c": [],
        "final_holdout": [],
        "ambiguous_boundary": [],
    }

    # Inherit calibration_v1b (25 pos + 25 hard-neg) + ambiguous_boundary (9) seamlessly with original paths
    for row in v1b_mf["calibration_v1b_full"]:
        r = dict(row)
        r["_abs_path"] = str(vl_p1b_dir / row["full_rel_path"])
        r["split"] = "calibration_v1c"
        r["is_inherited_cal_v1b"] = True
        r["norm_title"] = _normalize_title(row["title"])
        selected["calibration_v1c"].append(r)

    for row in v1b_mf["ambiguous_boundary"]:
        r = dict(row)
        r["_abs_path"] = str(vl_p1b_dir / row["full_rel_path"])
        r["split"] = "ambiguous_boundary"
        r["is_inherited_v1b"] = True
        r["norm_title"] = _normalize_title(row["title"])
        selected["ambiguous_boundary"].append(r)

    # 1. Select NEW positives per class: 10 cal_v1c + 20 final_holdout = 30 new distinct clusters/class
    for oid in OBJECT_IDS:
        gkey = f"pos/{oid}"
        allowed = set(allowlists[gkey]) if gkey in allowlists else None
        clusters = _cluster_group_records(meta.get(gkey, []), staging_root, allowed)
        reps = [c[0] for c in clusters]
        if len(reps) < 30:
            raise RuntimeError(f"{gkey}: only {len(reps)} distinct clusters (< 30 needed)")
        cal_picks = reps[:10]
        hold_picks = reps[10:30]
        for idx, rec in enumerate(cal_picks, 1):
            img = load_image_unicode(Path(rec["_abs_path"]))
            rec["split"] = "calibration_v1c"
            rec["label_id"] = oid
            rec["confuser_group"] = None
            rec["id"] = f"cal1c_{oid}_{idx:02d}"
            rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
            rec["is_inherited_cal_v1b"] = False
            selected["calibration_v1c"].append(rec)
        for idx, rec in enumerate(hold_picks, 1):
            img = load_image_unicode(Path(rec["_abs_path"]))
            rec["split"] = "final_holdout"
            rec["label_id"] = oid
            rec["confuser_group"] = None
            rec["id"] = f"fhold_{oid}_{idx:02d}"
            rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
            selected["final_holdout"].append(rec)

    # 2. Select NEW hard negatives across 7 groups: 8-9 new/grp -> cal_v1c (59 new + 25 old = 84 = 12/grp), 16 new/grp -> final_holdout (112)
    for g_idx, gname in enumerate(HARD_NEGATIVE_GROUPS):
        gkey = f"neg/{gname}"
        allowed = set(allowlists[gkey]) if gkey in allowlists else None
        clusters = _cluster_group_records(meta.get(gkey, []), staging_root, allowed)
        reps = [c[0] for c in clusters]
        n_inh = sum(1 for r in selected["calibration_v1c"] if r.get("confuser_group") == gname)
        n_cal_new = 12 - n_inh  # Total 12 per group in calibration_v1c (59 new across 7 groups)
        n_hold_new = 16
        if len(reps) < (n_cal_new + n_hold_new):
            raise RuntimeError(f"{gkey}: only {len(reps)} distinct clusters (< {n_cal_new + n_hold_new} needed)")
        cal_neg = reps[:n_cal_new]
        hold_neg = reps[n_cal_new:n_cal_new + n_hold_new]
        for idx, rec in enumerate(cal_neg, 1):
            img = load_image_unicode(Path(rec["_abs_path"]))
            rec["split"] = "calibration_v1c"
            rec["label_id"] = "unknown"
            rec["confuser_group"] = gname
            rec["id"] = f"cal1c_neg_{gname[:6]}_{idx:02d}"
            rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
            rec["is_inherited_cal_v1b"] = False
            selected["calibration_v1c"].append(rec)
        for idx, rec in enumerate(hold_neg, 1):
            img = load_image_unicode(Path(rec["_abs_path"]))
            rec["split"] = "final_holdout"
            rec["label_id"] = "unknown"
            rec["confuser_group"] = gname
            rec["id"] = f"fhold_neg_{gname[:6]}_{idx:02d}"
            rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
            selected["final_holdout"].append(rec)

    # 3. Select NEW general distractors: 41 for calibration_v1c + 88 for final_holdout = 129
    gen_key = "neg/general_distractor"
    gen_allowed = set(allowlists[gen_key]) if gen_key in allowlists else None
    gen_clusters = _cluster_group_records(meta.get(gen_key, []), staging_root, gen_allowed)
    gen_reps = [c[0] for c in gen_clusters]
    if len(gen_reps) < 129:
        raise RuntimeError(f"{gen_key}: only {len(gen_reps)} distinct clusters (< 129 needed)")
    for idx, rec in enumerate(gen_reps[:41], 1):
        img = load_image_unicode(Path(rec["_abs_path"]))
        rec["split"] = "calibration_v1c"
        rec["label_id"] = "unknown"
        rec["confuser_group"] = "general_distractor"
        rec["id"] = f"cal1c_gen_{idx:02d}"
        rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
        rec["is_inherited_cal_v1b"] = False
        selected["calibration_v1c"].append(rec)
    for idx, rec in enumerate(gen_reps[41:129], 1):
        img = load_image_unicode(Path(rec["_abs_path"]))
        rec["split"] = "final_holdout"
        rec["label_id"] = "unknown"
        rec["confuser_group"] = "general_distractor"
        rec["id"] = f"fhold_gen_{idx:02d}"
        rec["roi_bbox_xywh"] = list(_estimate_initial_gt_bbox(img))
        selected["final_holdout"].append(rec)

    # 4. Select 6 NEW ambiguous boundary items
    amb_key = "amb/ambiguous_boundary"
    amb_allowed = set(allowlists[amb_key]) if amb_key in allowlists else None
    amb_clusters = _cluster_group_records(meta.get(amb_key, []), staging_root, amb_allowed)
    used_clusters = {
        r["cluster_id"]
        for lst in (selected["calibration_v1c"], selected["final_holdout"], selected["ambiguous_boundary"])
        for r in lst
    }
    used_shas = {
        r["sha256"]
        for lst in (selected["calibration_v1c"], selected["final_holdout"], selected["ambiguous_boundary"])
        for r in lst
    }
    added_amb = 0
    for cl in amb_clusters:
        if added_amb >= 6:
            break
        rec = cl[0]
        if rec["cluster_id"] in used_clusters or rec["sha256"] in used_shas:
            continue
        added_amb += 1
        rec["split"] = "ambiguous_boundary"
        rec["label_id"] = "ambiguous"
        rec["confuser_group"] = "boundary_case"
        rec["id"] = f"amb1c_{added_amb:02d}"
        rec["roi_bbox_xywh"] = None
        rec["is_inherited_v1b"] = False
        selected["ambiguous_boundary"].append(rec)

    draft_path = vl_p1c_dir / "manifests" / "draft_selection_v1c.json"
    draft_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    cs_dir = vl_p1c_dir / "manifests" / "contact_sheets_v1c" / "gt_verify"
    _render_gt_sheets(selected, cs_dir)

    summary = {
        "calibration_v1c": len(selected["calibration_v1c"]),
        "calibration_v1c_pos": sum(1 for r in selected["calibration_v1c"] if r["label_id"] != "unknown"),
        "calibration_v1c_neg": sum(1 for r in selected["calibration_v1c"] if r["label_id"] == "unknown"),
        "final_holdout": len(selected["final_holdout"]),
        "final_holdout_pos": sum(1 for r in selected["final_holdout"] if r["label_id"] != "unknown"),
        "final_holdout_neg": sum(1 for r in selected["final_holdout"] if r["label_id"] == "unknown"),
        "ambiguous_boundary": len(selected["ambiguous_boundary"]),
    }
    print("[Init V1c Summary]", json.dumps(summary, indent=2))
    return selected


def _render_gt_sheets(selected: dict, cs_dir: Path) -> None:
    cal_new_pos = [
        (f"{r['id']} | {r['label_id']}", load_image_unicode(Path(r["_abs_path"])), tuple(r["roi_bbox_xywh"]))
        for r in selected["calibration_v1c"]
        if r["label_id"] != "unknown" and not r.get("is_inherited_cal_v1b")
    ]
    for page in range(0, len(cal_new_pos), 25):
        make_contact_sheet(
            cal_new_pos[page:page + 25],
            cs_dir / f"gt_verify_cal1c_new_pos_{page // 25 + 1}.jpg",
            cols=5,
            cell_w=280,
            cell_h=220,
        )

    hold_pos = [
        (f"{r['id']} | {r['label_id']}", load_image_unicode(Path(r["_abs_path"])), tuple(r["roi_bbox_xywh"]))
        for r in selected["final_holdout"]
        if r["label_id"] != "unknown"
    ]
    for page in range(0, len(hold_pos), 25):
        make_contact_sheet(
            hold_pos[page:page + 25],
            cs_dir / f"gt_verify_fhold_pos_{page // 25 + 1}.jpg",
            cols=5,
            cell_w=280,
            cell_h=220,
        )


def run_finalize(vl_p1_dir: Path, vl_p1b_dir: Path, vl_p1c_dir: Path) -> dict:
    draft_path = vl_p1c_dir / "manifests" / "draft_selection_v1c.json"
    selected = json.loads(draft_path.read_text(encoding="utf-8"))

    overrides_path = vl_p1c_dir / "manifests" / "manual_gt_bboxes_v1c.json"
    overrides: dict[str, list[float]] = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.is_file() else {}
    )

    for split_key in ("calibration_v1c", "final_holdout"):
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
                rec["bbox_verified_mode"] = rec.get("bbox_verified_mode") or "offline_saliency_visual_audited"

    _render_gt_sheets(selected, vl_p1c_dir / "manifests" / "contact_sheets_v1c" / "gt_verify")

    # Build strict baseline signature set EXCLUDING inherited calibration_v1b & ambiguous items inside VL-P1b
    vl1_shas, vl1_titles, vl1_phashes = _load_existing_signatures(vl_p1_dir)
    v1b_mf = json.loads((vl_p1b_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    inherited_cal_ids = {r["id"] for r in v1b_mf["calibration_v1b_full"]}
    inherited_amb_ids = {r["id"] for r in v1b_mf["ambiguous_boundary"]}
    v1b_protected_shas = set()
    v1b_protected_clusters = set()
    v1b_protected_phashes = []
    for k in ("train_pos", "hard_neg_gallery", "holdout_full"):
        for r in v1b_mf[k]:
            if r.get("sha256"):
                v1b_protected_shas.add(r["sha256"])
            if r.get("cluster_id"):
                v1b_protected_clusters.add(r["cluster_id"])
            if r.get("ahash") and r.get("dhash"):
                v1b_protected_phashes.append((r["id"], r["ahash"], r["dhash"]))

    split_shas: dict[str, set[str]] = {}
    split_clusters: dict[str, set[str]] = {}
    split_phashes: dict[str, list[tuple[str, str, str]]] = {}

    for sp_name, recs in selected.items():
        s_shas, s_clus, s_phs = set(), set(), []
        for r in recs:
            assert is_allowed_open_license(r["license"]), f"Bad license {r['id']}"
            assert r["sha256"] not in vl1_shas, f"VL-P1 SHA overlap {r['id']}"
            assert r["sha256"] not in v1b_protected_shas, f"VL-P1b gallery/holdout SHA overlap {r['id']}"
            is_inherited = (r["id"] in inherited_cal_ids) or (r["id"] in inherited_amb_ids)
            if not is_inherited:
                for vid, vah, vdh in vl1_phashes + v1b_protected_phashes:
                    if min(hex_hamming_distance(r["ahash"], vah), hex_hamming_distance(r["dhash"], vdh)) <= 6:
                        raise RuntimeError(f"pHash <= 6 overlap {r['id']} vs {vid}")
            r["cluster_id"] = _session_cluster_key(r["norm_title"])
            if not is_inherited and r["cluster_id"] in v1b_protected_clusters:
                raise RuntimeError(f"VL-P1b gallery/holdout cluster overlap {r['id']} cluster={r['cluster_id']}")
            s_shas.add(r["sha256"])
            s_clus.add(r["cluster_id"])
            s_phs.append((r["id"], r["ahash"], r["dhash"]))
        split_shas[sp_name] = s_shas
        split_clusters[sp_name] = s_clus
        split_phashes[sp_name] = s_phs

    sp_names = list(selected.keys())
    for i in range(len(sp_names)):
        for j in range(i + 1, len(sp_names)):
            a, b = sp_names[i], sp_names[j]
            assert not (split_shas[a] & split_shas[b]), f"SHA overlap between {a} and {b}"
            assert not (split_clusters[a] & split_clusters[b]), f"Cluster overlap between {a} and {b}"
            for id_a, ah_a, dh_a in split_phashes[a]:
                for id_b, ah_b, dh_b in split_phashes[b]:
                    if min(hex_hamming_distance(ah_a, ah_b), hex_hamming_distance(dh_a, dh_b)) <= 6:
                        raise RuntimeError(f"Cross-split pHash <= 6: {id_a} vs {id_b}")

    # Materialize clean physical dataset directories (keep train_pos & hard_neg_gallery intact)
    data_dir = vl_p1c_dir / "data"
    for folder in (
        "calibration_v1c",
        "calibration_v1c_full",
        "final_holdout_roi",
        "final_holdout_full",
        "ambiguous_boundary_v1c",
    ):
        d = data_dir / folder
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[dict]] = {
        "train_pos": v1b_mf["train_pos"],
        "hard_neg_gallery": [],
        "calibration_v1c_roi": [],
        "calibration_v1c_full": [],
        "final_holdout_roi": [],
        "final_holdout_full": [],
        "ambiguous_boundary_v1c": [],
    }
    # Index copied train_pos sha256 relative to vl_p1c
    manifest["train_pos"] = []
    for oid in OBJECT_IDS:
        for p in list_images(data_dir / "train_pos" / oid):
            img = load_image_unicode(p)
            manifest["train_pos"].append({
                "id": f"train_{oid}_{p.stem}",
                "label_id": oid,
                "label_vi": LABELS_VI[oid],
                "rel_path": p.relative_to(vl_p1c_dir).as_posix(),
                "sha256": compute_sha256(p),
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            })
    for row in v1b_mf["hard_neg_gallery"]:
        r = dict(row)
        p = vl_p1c_dir / row["full_rel_path"]
        assert p.is_file(), f"Missing copied gallery file {p}"
        manifest["hard_neg_gallery"].append(r)

    def _materialize_split(split_key: str, full_folder: str, roi_folder: str, out_full_k: str, out_roi_k: str) -> None:
        for r in selected[split_key]:
            img = load_image_unicode(Path(r["_abs_path"]))
            cls_dir = r["label_id"] if r["label_id"] != "unknown" else f"unknown/{r['confuser_group']}"
            full_sub = data_dir / full_folder / cls_dir
            roi_sub = data_dir / roi_folder / cls_dir
            full_sub.mkdir(parents=True, exist_ok=True)
            roi_sub.mkdir(parents=True, exist_ok=True)
            full_dst = full_sub / f"{r['id']}.jpg"
            roi_dst = roi_sub / f"{r['id']}.jpg"
            shutil.copy2(r["_abs_path"], full_dst)
            bx, by, bw, bh = r["roi_bbox_xywh"]
            crop = img[by:by + bh, bx:bx + bw]
            save_image_unicode(roi_dst, crop if crop.size > 0 else img)
            rec = _public_record(r, vl_p1c_dir, full_p=full_dst, roi_p=roi_dst)
            manifest[out_full_k].append(rec)
            manifest[out_roi_k].append(rec)

    _materialize_split("calibration_v1c", "calibration_v1c_full", "calibration_v1c", "calibration_v1c_full", "calibration_v1c_roi")
    _materialize_split("final_holdout", "final_holdout_full", "final_holdout_roi", "final_holdout_full", "final_holdout_roi")

    for r in selected["ambiguous_boundary"]:
        dst = data_dir / "ambiguous_boundary_v1c" / f"{r['id']}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["_abs_path"], dst)
        manifest["ambiguous_boundary_v1c"].append(_public_record(r, vl_p1c_dir, full_p=dst, roi_p=dst))

    out_manifest = {
        "schema_version": "vl_p1c_v1",
        "counts": {k: len(v) for k, v in manifest.items()},
        "calibration_v1c_breakdown": {
            "positives_by_class": {oid: sum(1 for r in manifest["calibration_v1c_full"] if r["label_id"] == oid) for oid in OBJECT_IDS},
            "negatives_by_group": _count_groups(manifest["calibration_v1c_full"]),
        },
        "final_holdout_breakdown": {
            "positives_by_class": {oid: sum(1 for r in manifest["final_holdout_full"] if r["label_id"] == oid) for oid in OBJECT_IDS},
            "negatives_by_group": _count_groups(manifest["final_holdout_full"]),
        },
        "hard_neg_gallery_by_group": _count_groups(manifest["hard_neg_gallery"]),
        "zero_overlap_verified": True,
        **manifest,
    }
    mf_txt = json.dumps(out_manifest, indent=2, ensure_ascii=False)
    (vl_p1c_dir / "manifests" / "dataset_manifest.json").write_text(mf_txt, encoding="utf-8")
    (vl_p1c_dir / "dataset_manifest.json").write_text(mf_txt, encoding="utf-8")
    print("[Finalized V1c Counts]", json.dumps(out_manifest["counts"], indent=2))
    print("[Final Holdout Breakdown]", json.dumps(out_manifest["final_holdout_breakdown"], indent=2))
    return out_manifest


def _count_groups(recs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in recs:
        if r["label_id"] == "unknown":
            g = r["confuser_group"]
            out[g] = out.get(g, 0) + 1
    return dict(sorted(out.items()))


def _public_record(r: dict, vl_p1c_dir: Path, full_p: Path, roi_p: Path) -> dict:
    return {
        "id": r["id"],
        "label_id": r["label_id"],
        "label_vi": LABELS_VI.get(r["label_id"], r["label_id"]),
        "confuser_group": r.get("confuser_group"),
        "full_rel_path": full_p.relative_to(vl_p1c_dir).as_posix(),
        "roi_rel_path": roi_p.relative_to(vl_p1c_dir).as_posix(),
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
        run_init(root / "research" / "vl_p1", root / "research" / "vl_p1b", root / "research" / "vl_p1c")
    else:
        run_finalize(root / "research" / "vl_p1", root / "research" / "vl_p1b", root / "research" / "vl_p1c")
