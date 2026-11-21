"""Step 2c: Build Verified Dataset & Multi-Geometry Annotation Manifests for VL-P2.

Constructs:
- `manifests/dataset_manifest.json`:
    * `train_pos` (25 enrollment images across object_01..05)
    * `hard_neg_gallery` (reference confuser gallery: historical 49 + dev slender gallery <= 25)
    * `dev_cal_v2` (>=60 object_01 + >=80 hard negatives, reusing valid prior P1b/P1c images + new dev additions)
    * `final_holdout_v2` (>=50 BRAND-NEW object_01 + >=100 BRAND-NEW negatives, 0 overlap with ANY prior set or dev_cal_v2)
    * `ambiguous_cane_v2` (>=15 ambiguous white sticks/pipes/rods marked LABEL_AMBIGUOUS)
    * `historical_v1c_holdout_obj01` (20 object_01 images from VL-P1c final_holdout for Method A/B regression comparison)
- `manifests/annotation_manifest.json`:
    Complete multi-geometry GT (bbox_xywh, obb_cxcywh_angle, obb_corners_xy, endpoints_xyxy,
    centerline_points_xy, mask_rle, occluded, scenario_tags, label_id, confuser_group).
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import urllib.parse

import cv2
import numpy as np

from .annotation_protocol import (
    derive_gt_geometry_from_endpoints,
    extract_assisted_gt_from_hint_box,
    render_gt_overlay_preview,
)
from .common_utils import (
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    load_image_unicode,
    make_contact_sheet,
    normalize_title,
    save_image_unicode,
)
from .localization_research import OrientedCaneLocalizer, YOLOXPersonAndObjectDetector
from .segmentation_research import CaneShaftSegmenter


def _dedupe_append(
    rec: dict,
    target_list: list[dict],
    seen_shas: set[str],
    seen_titles: set[str],
    seen_phashes: list[tuple[str, str, str]],
    hamming_thr: int = 6,
) -> bool:
    sha = rec["sha256"]
    nt = normalize_title(rec.get("title") or rec.get("source_title") or rec["id"])
    ah = rec["ahash"]
    dh = rec["dhash"]
    if sha in seen_shas or (nt and nt in seen_titles):
        return False
    for _, o_ah, o_dh in seen_phashes:
        if hex_hamming_distance(ah, o_ah) <= hamming_thr or hex_hamming_distance(dh, o_dh) <= hamming_thr:
            return False
    seen_shas.add(sha)
    if nt:
        seen_titles.add(nt)
    seen_phashes.append((rec["id"], ah, dh))
    target_list.append(rec)
    return True


def _copy_to_cache(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not dst_path.is_file():
        shutil.copy2(src_path, dst_path)


def _infer_occluded_and_tags(title: str, desc: str, has_person: bool, obb: list[float], seg_res_img: np.ndarray | None = None) -> tuple[bool, list[str]]:
    blob = f"{title} {desc}".lower()
    tags = []
    if has_person or any(w in blob for w in ("man ", "woman", "person", "walking", "pedestrian", "crossing", "hand")):
        tags.append("handheld")
    if any(w in blob for w in ("fold", "folded", "folding")):
        tags.append("folded")
    if any(w in blob for w in ("red", "tip")):
        tags.append("red_tip")
    if any(w in blob for w in ("street", "city", "sidewalk", "crosswalk", "station", "crowd")):
        tags.append("outdoor_crowded")
    elif any(w in blob for w in ("room", "office", "indoor", "museum", "hall")):
        tags.append("indoor")
    # Held by person or intersecting hand typically exhibits partial shaft occlusion
    occluded = ("handheld" in tags) or ("folded" in tags)
    if not tags:
        tags.append("white_cane_scene")
    return occluded, tags


def verify_zero_overlap_across_splits(
    splits: dict[str, list[dict]],
    pairs_to_check: list[tuple[str, str]],
    hamming_thr: int = 6,
) -> dict:
    """Strictly verify 0 overlap in SHA-256, normalized title, source_cluster, and aHash/dHash <= 6."""
    report = {}
    for s1, s2 in pairs_to_check:
        l1 = splits.get(s1, [])
        l2 = splits.get(s2, [])
        sha1 = {r["sha256"] for r in l1}
        t1 = {normalize_title(r.get("title") or r.get("source_title") or r["id"]) for r in l1}
        c1 = {normalize_title(r.get("source_cluster") or "") for r in l1 if r.get("source_cluster")}
        ph1 = [(r["id"], r["ahash"], r["dhash"]) for r in l1 if r.get("ahash") and r.get("dhash")]

        sha_hits = []
        title_hits = []
        cluster_hits = []
        phash_hits = []
        for r2 in l2:
            if r2["sha256"] in sha1:
                sha_hits.append(r2["id"])
            nt2 = normalize_title(r2.get("title") or r2.get("source_title") or r2["id"])
            if nt2 and nt2 in t1:
                title_hits.append(r2["id"])
            cl2 = normalize_title(r2.get("source_cluster") or "")
            if cl2 and cl2 in c1:
                cluster_hits.append(r2["id"])
            if r2.get("ahash") and r2.get("dhash"):
                for oid, oah, odh in ph1:
                    if (
                        hex_hamming_distance(r2["ahash"], oah) <= hamming_thr
                        or hex_hamming_distance(r2["dhash"], odh) <= hamming_thr
                    ):
                        phash_hits.append((r2["id"], oid))
                        break
        report[f"{s1}__vs__{s2}"] = {
            "sha256_overlaps": len(sha_hits),
            "title_overlaps": len(title_hits),
            "cluster_overlaps": len(cluster_hits),
            "phash_overlaps": len(phash_hits),
            "clean": (len(sha_hits) + len(title_hits) + len(cluster_hits) + len(phash_hits)) == 0,
        }
    return report


def build_vl_p2_manifests(worktree_root: Path) -> tuple[dict, dict]:
    """Assemble dataset_manifest.json and annotation_manifest.json from curated pools + prior dev data."""
    ai_root = worktree_root.parent
    vl_p1_dir = ai_root / "VisionMate_VL_P1_AG" / "research" / "vl_p1"
    vl_p1b_dir = ai_root / "VisionMate_VL_P1b_AG" / "research" / "vl_p1b"
    vl_p1c_dir = ai_root / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c"
    vl_p2_dir = worktree_root / "research" / "vl_p2"
    cache_data = vl_p2_dir / "cache" / "data"
    cache_data.mkdir(parents=True, exist_ok=True)

    curated_path = vl_p2_dir / "manifests" / "curated_allowlist_v2.json"
    if not curated_path.is_file():
        raise FileNotFoundError(f"Missing curated allowlist: {curated_path}")
    curated = json.loads(curated_path.read_text(encoding="utf-8"))

    overrides_path = vl_p2_dir / "manifests" / "manual_cane_gt_overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.is_file() else {}

    mf_v1 = json.loads((vl_p1_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8")) if (vl_p1_dir / "manifests" / "dataset_manifest.json").is_file() else {}
    mf_v1b = json.loads((vl_p1b_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    mf_v1c = json.loads((vl_p1c_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))

    yolox = YOLOXPersonAndObjectDetector(worktree_root / "models" / "object_detection" / "yolox_nano.onnx")
    localizer = OrientedCaneLocalizer(yolox, top_k=5)
    segmenter = CaneShaftSegmenter()

    # Track ALL historical signatures strictly so final_holdout_v2 can NEVER intersect them
    hist_shas: set[str] = set()
    hist_titles: set[str] = set()
    hist_clusters: set[str] = set()
    hist_phashes: list[tuple[str, str, str]] = []

    def _register_url_title(row: dict) -> None:
        su = row.get("source_url", "")
        if "File:" in su:
            fn = urllib.parse.unquote(su.split("File:", 1)[1])
            hist_titles.add(normalize_title("File:" + fn))
            hist_titles.add(normalize_title(fn))
            hist_clusters.add(normalize_title(fn)[:36])

    def _register_prior(base_dir: Path, row: dict, use_full: bool = True) -> dict | None:
        _register_url_title(row)
        rel = row.get("full_rel_path") if (use_full and row.get("full_rel_path")) else (row.get("roi_rel_path") or row.get("rel_path"))
        if not rel:
            return None
        p = base_dir / rel
        if not p.is_file():
            return None
        img = load_image_unicode(p)
        sha = (row.get("full_sha256") if (use_full and row.get("full_rel_path")) else None) or row.get("sha256") or compute_sha256(p)
        ah = (row.get("full_ahash") if (use_full and row.get("full_rel_path")) else None) or row.get("ahash") or compute_ahash(img)
        dh = (row.get("full_dhash") if (use_full and row.get("full_rel_path")) else None) or row.get("dhash") or compute_dhash(img)
        t = row.get("source_title") or row.get("title") or row["id"]
        label_val = row.get("label_id") or row.get("label") or "unknown"
        hist_shas.add(sha)
        hist_titles.add(normalize_title(t))
        hist_titles.add(normalize_title(str(t).replace("File:", "")))
        cl = normalize_title(row.get("source_cluster") or str(t).replace("File:", ""))[:36]
        if cl:
            hist_clusters.add(cl)
        hist_phashes.append((row["id"], ah, dh))
        return {
            "id": row["id"],
            "src_abs": str(p),
            "label_id": label_val,
            "confuser_group": row.get("confuser_group") or ("positive" if label_val != "unknown" else "general_distractor"),
            "sha256": sha,
            "ahash": ah,
            "dhash": dh,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "source_title": t,
            "source_cluster": row.get("source_cluster") or normalize_title(t)[:48],
            "source_url": row.get("source_url", ""),
            "license": row.get("license", ""),
            "author": row.get("author", "Unknown"),
            "roi_bbox_xywh": row.get("roi_bbox_xywh"),
            "has_person": bool(row.get("has_person", False)),
            "origin_phase": row.get("origin_phase", "historical"),
        }

    # 1. train_pos (25) & hard_neg_gallery (49 historical + slender confusers)
    train_pos: list[dict] = []
    for row in mf_v1c["train_pos"]:
        rec = _register_prior(vl_p1c_dir, row, use_full=False)
        if rec:
            dst = cache_data / "train_pos" / rec["label_id"] / f"{rec['id']}.jpg"
            _copy_to_cache(Path(rec["src_abs"]), dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")
            del rec["src_abs"]
            train_pos.append(rec)

    hard_neg_gallery: list[dict] = []
    for row in mf_v1c["hard_neg_gallery"]:
        rec = _register_prior(vl_p1c_dir, row, use_full=False)
        if rec:
            dst = cache_data / "hard_neg_gallery" / f"{rec['id']}.jpg"
            _copy_to_cache(Path(rec["src_abs"]), dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")
            del rec["src_abs"]
            hard_neg_gallery.append(rec)

    # Register all historical splits (VL-P1, VL-P1b, VL-P1c) before building dev_cal_v2 & final_holdout_v2
    for sp in ("train", "calibration", "test"):
        for row in mf_v1.get(sp, []):
            _register_prior(vl_p1_dir, row, use_full=(sp == "test"))
    for row in mf_v1b.get("holdout_full", []):
        _register_prior(vl_p1b_dir, row, use_full=True)
    for sp in ("calibration_v1c_full", "final_holdout_full", "ambiguous_boundary_v1c"):
        for row in mf_v1c.get(sp, []):
            _register_prior(vl_p1c_dir, row, use_full=True)

    # 2. Build dev_cal_v2: reuse 65 historical open-licensed full-image object_01 + >=80 historical + dev hard negatives
    dev_cal_v2: list[dict] = []
    dev_shas: set[str] = set()
    dev_titles: set[str] = set()
    dev_phashes: list[tuple[str, str, str]] = []

    hist_obj01_candidates = []
    for base_d, mf_obj, sp_name, tag in (
        (vl_p1c_dir, mf_v1c, "calibration_v1c_full", "vl_p1c_cal"),
        (vl_p1c_dir, mf_v1c, "final_holdout_full", "vl_p1c_fhold"),
        (vl_p1b_dir, mf_v1b, "calibration_v1b_full", "vl_p1b_cal"),
        (vl_p1b_dir, mf_v1b, "holdout_full", "vl_p1b_hold"),
        (vl_p1_dir, mf_v1, "test", "vl_p1_test"),
    ):
        for row in mf_obj.get(sp_name, []):
            if (row.get("label_id") or row.get("label")) == "object_01":
                lic = str(row.get("license", "")).lower()
                if "unverified" in lic:
                    continue
                r = _register_prior(base_d, row, use_full=True)
                if r:
                    r["origin_phase"] = tag
                    hist_obj01_candidates.append(r)

    for rec in hist_obj01_candidates:
        if _dedupe_append(rec, dev_cal_v2, dev_shas, dev_titles, dev_phashes):
            dst = cache_data / "dev_cal_v2" / "pos" / f"{rec['id']}.jpg"
            _copy_to_cache(Path(rec["src_abs"]), dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")
            del rec["src_abs"]

    # Add curated extra new dev white canes so dev_cal_v2 >= 60 object_01
    for item in curated.get("dev_new_object_01", []):
        p = Path(item["abs_path"])
        if not p.is_file():
            continue
        rec = {
            "id": item["id"],
            "label_id": "object_01",
            "confuser_group": "positive",
            "sha256": item["sha256"],
            "ahash": item["ahash"],
            "dhash": item["dhash"],
            "width": item["width"],
            "height": item["height"],
            "source_title": item["title"],
            "source_cluster": item["source_cluster"],
            "source_url": item["source_url"],
            "license": item["license"],
            "author": item["author"],
            "roi_bbox_xywh": None,
            "has_person": bool(item.get("has_person", True)),
            "origin_phase": "vl_p2_new_dev",
        }
        if _dedupe_append(rec, dev_cal_v2, dev_shas, dev_titles, dev_phashes):
            dst = cache_data / "dev_cal_v2" / "pos" / f"{rec['id']}.jpg"
            _copy_to_cache(p, dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")

    # Historical hard negatives into dev_cal_v2 (prioritize slender confusers + walking sticks + umbrellas)
    hist_neg_candidates = []
    for base_d, mf_obj, sp_name, tag in (
        (vl_p1c_dir, mf_v1c, "calibration_v1c_full", "vl_p1c_cal"),
        (vl_p1c_dir, mf_v1c, "final_holdout_full", "vl_p1c_fhold"),
        (vl_p1b_dir, mf_v1b, "holdout_full", "vl_p1b_hold"),
    ):
        for row in mf_obj.get(sp_name, []):
            if row.get("label_id") == "unknown" and row.get("confuser_group") in (
                "walking_stick_confuser",
                "umbrella_handle_confuser",
                "key_metal_confuser",
                "shoe_distractor",
                "slipper_distractor",
                "wallet_confuser",
                "general_distractor",
            ):
                r = _register_prior(base_d, row, use_full=True)
                if r:
                    r["origin_phase"] = tag
                    hist_neg_candidates.append(r)

    # Sort so cane-relevant hard negatives come first, reaching ~110 dev negatives (>=80 hard-negatives!)
    priority_groups = {"walking_stick_confuser": 0, "umbrella_handle_confuser": 1, "key_metal_confuser": 2}
    hist_neg_candidates.sort(key=lambda r: (priority_groups.get(r["confuser_group"], 5), r["id"]))
    dev_neg_added = 0
    for rec in hist_neg_candidates:
        if dev_neg_added >= 100:
            break
        if _dedupe_append(rec, dev_cal_v2, dev_shas, dev_titles, dev_phashes):
            dst = cache_data / "dev_cal_v2" / "neg" / f"{rec['id']}.jpg"
            _copy_to_cache(Path(rec["src_abs"]), dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")
            del rec["src_abs"]
            dev_neg_added += 1

    # Also include curated new slender dev hard-negatives (trekking poles, metal rods, white poles/lines)
    for item in curated.get("dev_new_negatives", []):
        p = Path(item["abs_path"])
        if not p.is_file():
            continue
        rec = {
            "id": item["id"],
            "label_id": "unknown",
            "confuser_group": item["confuser_group"],
            "sha256": item["sha256"],
            "ahash": item["ahash"],
            "dhash": item["dhash"],
            "width": item["width"],
            "height": item["height"],
            "source_title": item["title"],
            "source_cluster": item["source_cluster"],
            "source_url": item["source_url"],
            "license": item["license"],
            "author": item["author"],
            "roi_bbox_xywh": None,
            "has_person": False,
            "origin_phase": "vl_p2_new_dev",
        }
        if _dedupe_append(rec, dev_cal_v2, dev_shas, dev_titles, dev_phashes):
            dst = cache_data / "dev_cal_v2" / "neg" / f"{rec['id']}.jpg"
            _copy_to_cache(p, dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")

    # 3. Build brand-new blind `final_holdout_v2`: zero overlap against hist + dev!
    all_forbidden_shas = hist_shas | dev_shas
    all_forbidden_titles = hist_titles | dev_titles
    all_forbidden_phashes = list(hist_phashes) + list(dev_phashes)

    final_holdout_v2: list[dict] = []
    for item in curated.get("holdout_new_object_01", []):
        p = Path(item["abs_path"])
        if not p.is_file():
            continue
        rec = {
            "id": item["id"],
            "label_id": "object_01",
            "confuser_group": "positive",
            "sha256": item["sha256"],
            "ahash": item["ahash"],
            "dhash": item["dhash"],
            "width": item["width"],
            "height": item["height"],
            "source_title": item["title"],
            "source_cluster": item["source_cluster"],
            "source_url": item["source_url"],
            "license": item["license"],
            "author": item["author"],
            "roi_bbox_xywh": None,
            "has_person": bool(item.get("has_person", True)),
            "origin_phase": "vl_p2_brand_new_holdout",
        }
        if _dedupe_append(rec, final_holdout_v2, all_forbidden_shas, all_forbidden_titles, all_forbidden_phashes):
            dst = cache_data / "final_holdout_v2" / "pos" / f"{rec['id']}.jpg"
            _copy_to_cache(p, dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")

    for item in curated.get("holdout_new_negatives", []):
        p = Path(item["abs_path"])
        if not p.is_file():
            continue
        rec = {
            "id": item["id"],
            "label_id": "unknown",
            "confuser_group": item["confuser_group"],
            "sha256": item["sha256"],
            "ahash": item["ahash"],
            "dhash": item["dhash"],
            "width": item["width"],
            "height": item["height"],
            "source_title": item["title"],
            "source_cluster": item["source_cluster"],
            "source_url": item["source_url"],
            "license": item["license"],
            "author": item["author"],
            "roi_bbox_xywh": None,
            "has_person": False,
            "origin_phase": "vl_p2_brand_new_holdout",
        }
        if _dedupe_append(rec, final_holdout_v2, all_forbidden_shas, all_forbidden_titles, all_forbidden_phashes):
            dst = cache_data / "final_holdout_v2" / "neg" / f"{rec['id']}.jpg"
            _copy_to_cache(p, dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")

    # 4. Ambiguous cane v2 (>=15 plain white sticks/pipes lacking mobility context)
    ambiguous_cane_v2: list[dict] = []
    for item in curated.get("ambiguous_cane_v2", []):
        p = Path(item["abs_path"])
        if not p.is_file():
            continue
        rec = {
            "id": item["id"],
            "label_id": "ambiguous",
            "confuser_group": "ambiguous_white_rod",
            "sha256": item["sha256"],
            "ahash": item["ahash"],
            "dhash": item["dhash"],
            "width": item["width"],
            "height": item["height"],
            "source_title": item["title"],
            "source_cluster": item["source_cluster"],
            "source_url": item["source_url"],
            "license": item["license"],
            "author": item["author"],
            "roi_bbox_xywh": None,
            "has_person": False,
            "origin_phase": "vl_p2_ambiguous",
        }
        if _dedupe_append(rec, ambiguous_cane_v2, all_forbidden_shas, all_forbidden_titles, all_forbidden_phashes):
            dst = cache_data / "ambiguous_cane_v2" / f"{rec['id']}.jpg"
            _copy_to_cache(p, dst)
            rec["rel_path"] = str(dst.relative_to(vl_p2_dir)).replace("\\", "/")

    # 5. Historical VL-P1c N=300 reference metadata
    historical_v1c_holdout_obj01 = [
        r for r in hist_obj01_candidates if r.get("origin_phase") == "vl_p1c_fhold"
    ]

    # Build GT annotations (`annotation_manifest.json`) for every object_01 in dev_cal_v2 + final_holdout_v2
    annotations: dict[str, dict] = {}
    gt_visuals: dict[str, list[tuple[str, np.ndarray]]] = {"dev_pos": [], "holdout_pos": []}

    for split_key, recs in (("dev_cal_v2", dev_cal_v2), ("final_holdout_v2", final_holdout_v2)):
        for rec in recs:
            if rec["label_id"] != "object_01":
                continue
            img_p = vl_p2_dir / rec["rel_path"]
            img = load_image_unicode(img_p)
            ov = overrides.get(rec["id"])
            if ov and ov.get("endpoints_xyxy"):
                geom = derive_gt_geometry_from_endpoints(
                    img,
                    ov["endpoints_xyxy"],
                    est_thickness_px=ov.get("thickness_px"),
                    segmenter=segmenter,
                )
            else:
                hint = tuple(rec["roi_bbox_xywh"]) if rec.get("roi_bbox_xywh") else None
                geom = extract_assisted_gt_from_hint_box(img, hint, localizer, segmenter)

            occluded, tags = _infer_occluded_and_tags(
                rec.get("source_title", ""), "", rec.get("has_person", False), geom["obb_cxcywh_angle"]
            )
            if ov and "occluded" in ov:
                occluded = bool(ov["occluded"])
            ann_rec = {
                "id": rec["id"],
                "split": split_key,
                "label_id": "object_01",
                "confuser_group": "positive",
                "bbox_xywh": geom["bbox_xywh"],
                "obb_cxcywh_angle": geom["obb_cxcywh_angle"],
                "obb_corners_xy": geom["obb_corners_xy"],
                "endpoints_xyxy": geom["endpoints_xyxy"],
                "centerline_points_xy": geom["centerline_points_xy"],
                "mask_rle": geom["mask_rle"],
                "occluded": occluded,
                "scenario_tags": tags,
                "annotation_source": "manual_override" if (ov and ov.get("endpoints_xyxy")) else "assisted_verified",
            }
            annotations[rec["id"]] = ann_rec
            vis = render_gt_overlay_preview(img, geom)
            bucket = "dev_pos" if split_key == "dev_cal_v2" else "holdout_pos"
            gt_visuals[bucket].append((rec["id"], vis))

    # Write GT overlay contact sheets for mandatory visual inspection
    cs_dir = vl_p2_dir / "cache" / "contact_sheets"
    cs_dir.mkdir(parents=True, exist_ok=True)
    for bname, items in gt_visuals.items():
        for chunk_i in range(0, len(items), 36):
            make_contact_sheet(
                items[chunk_i:chunk_i + 36],
                cs_dir / f"gt_{bname}_{chunk_i // 36 + 1:02d}.png",
                cols=6,
            )

    dev_pos_n = sum(1 for r in dev_cal_v2 if r["label_id"] == "object_01")
    dev_neg_n = sum(1 for r in dev_cal_v2 if r["label_id"] == "unknown")
    hold_pos_n = sum(1 for r in final_holdout_v2 if r["label_id"] == "object_01")
    hold_neg_n = sum(1 for r in final_holdout_v2 if r["label_id"] == "unknown")

    hold_neg_breakdown: dict[str, int] = {}
    for r in final_holdout_v2:
        if r["label_id"] == "unknown":
            g = r["confuser_group"]
            hold_neg_breakdown[g] = hold_neg_breakdown.get(g, 0) + 1

    dev_neg_breakdown: dict[str, int] = {}
    for r in dev_cal_v2:
        if r["label_id"] == "unknown":
            g = r["confuser_group"]
            dev_neg_breakdown[g] = dev_neg_breakdown.get(g, 0) + 1

    splits_dict = {
        "train_pos": train_pos,
        "hard_neg_gallery": hard_neg_gallery,
        "dev_cal_v2": dev_cal_v2,
        "final_holdout_v2": final_holdout_v2,
        "ambiguous_cane_v2": ambiguous_cane_v2,
    }
    leakage = verify_zero_overlap_across_splits(
        splits_dict,
        [
            ("train_pos", "dev_cal_v2"),
            ("train_pos", "final_holdout_v2"),
            ("hard_neg_gallery", "final_holdout_v2"),
            ("dev_cal_v2", "final_holdout_v2"),
            ("dev_cal_v2", "ambiguous_cane_v2"),
            ("final_holdout_v2", "ambiguous_cane_v2"),
        ],
    )

    dataset_manifest = {
        "schema_version": "vl_p2_dataset_manifest_v1",
        "counts": {
            "train_pos": len(train_pos),
            "hard_neg_gallery": len(hard_neg_gallery),
            "dev_cal_v2_total": len(dev_cal_v2),
            "dev_cal_v2_object_01": dev_pos_n,
            "dev_cal_v2_negatives": dev_neg_n,
            "final_holdout_v2_total": len(final_holdout_v2),
            "final_holdout_v2_object_01": hold_pos_n,
            "final_holdout_v2_negatives": hold_neg_n,
            "ambiguous_cane_v2": len(ambiguous_cane_v2),
        },
        "data_quota_status": "READY" if (dev_pos_n >= 60 and dev_neg_n >= 80 and hold_pos_n >= 50 and hold_neg_n >= 100) else "DATA_PENDING",
        "dev_neg_breakdown": dev_neg_breakdown,
        "final_holdout_neg_breakdown": hold_neg_breakdown,
        "zero_leakage_audit": leakage,
        **splits_dict,
    }
    ann_manifest = {
        "schema_version": "vl_p2_annotation_manifest_v1",
        "total_annotated_object_01": len(annotations),
        "annotations": annotations,
    }

    (vl_p2_dir / "manifests" / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2), encoding="utf-8"
    )
    (vl_p2_dir / "manifests" / "annotation_manifest.json").write_text(
        json.dumps(ann_manifest, indent=2), encoding="utf-8"
    )
    return dataset_manifest, ann_manifest


if __name__ == "__main__":
    wt = Path(__file__).resolve().parents[3]
    dm, am = build_vl_p2_manifests(wt)
    print("Dataset counts:", dm["counts"])
    print("Status:", dm["data_quota_status"])
    print("Zero-leakage clean:", all(v["clean"] for v in dm["zero_leakage_audit"].values()))
    print("Total annotated object_01:", am["total_annotated_object_01"])
