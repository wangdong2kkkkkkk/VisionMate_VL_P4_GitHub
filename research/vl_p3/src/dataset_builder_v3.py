"""Dataset & GT Annotation Builder for VL-P3 (`train_v3`, `val_cal_v3`, `final_holdout_v3`, `ambiguous_cane_v3`).

Combines:
  - Visually verified brand-new VL-P3 white canes (`verified_new_pos_ids.json`)
  - Visually verified unused VL-P2 staged white canes (`unused_pos_prefiltered.json`)
  - Audited non-holdout historical `object_01` (`train_pos` + `dev_cal_v2`, excluding ALL `final_holdout_v2` and `vl_p1c/final_holdout_full`)
  - Visually screened brand-new VL-P3 hard negatives (excluding 1 hidden cane + 47 non-photos)
  - Visually verified `ambiguous_cane_v3`
Produces strict group-disjoint splits, initial/repaired rotated OBB + endpoints + 11-pt centerline + mask RLE
annotations, GT overlay contact sheets for mandatory visual inspection, and full zero-leakage verification.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .common_utils import (
    VL_P3_DIR,
    WORKTREE_ROOT,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    decode_mask_rle,
    encode_mask_rle,
    hex_hamming_distance,
    load_image_unicode,
    make_contact_sheet,
    normalize_title,
    obb_from_endpoints,
    sample_centerline_points,
)
from .spatial_features import WORK_RIDGE_SIZE, compute_base_and_ridge_maps, pad_image_to_square_with_transform

VERIFIED_UNUSED_POS_INDICES = [14, 34, 83, 98, 99, 102, 104, 119, 127, 223, 264, 284]
EXCLUDED_V3_NEG_INDICES = {
    0, 3, 4, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24, 28, 29, 31, 37, 38,
    39, 42, 43, 44, 49, 50, 52, 54, 61, 66, 68, 70, 74, 79, 88, 115, 116, 117,
    148, 186, 191, 204, 207, 208, 227, 228, 258, 280,
}
VERIFIED_AMB_INDICES = [
    0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 23, 28, 29, 30,
    31, 32, 33, 34, 35, 37,
]


def _resolve_existing_path(rec: dict, base_repo: Path | None = None) -> Path | None:
    for key in ("abs_path", "path", "src_path"):
        v = rec.get(key)
        if v and Path(v).is_file():
            return Path(v)
    if base_repo and rec.get("rel_path"):
        p = base_repo / rec["rel_path"]
        if p.is_file():
            return p
    return None


def _auto_fit_cane_geometry(img: np.ndarray) -> dict:
    """High-resolution steered-ridge + white/red-tip LSD/Hough line fitter for initial new GT proposals."""
    h, w = img.shape[:2]
    sq, tfm = pad_image_to_square_with_transform(img, grid_size=112)
    _, hi = compute_base_and_ridge_maps(sq, tfm)
    sc4 = float(WORK_RIDGE_SIZE) / float(tfm.side)

    white = hi["white_achrom_448"]
    red = hi["red_tip_448"]
    ridge = hi["steered_ridge_448"]
    wood = hi["wood_cue_448"]
    valid = hi["valid_448"] > 0.5

    # Cane-likeness score field on 448x448
    score_map = np.clip((0.55 * ridge + 0.45 * np.maximum(white, red) - 0.35 * wood), 0.0, 1.0) * valid.astype(np.float32)
    u8 = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
    bw = (score_map >= 0.19).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(bw, rho=1.0, theta=np.pi / 180.0, threshold=18, minLineLength=26, maxLineGap=16)

    cands = []
    if lines is not None:
        segs = [tuple(int(v) for v in row[0]) for row in lines]
        # Score each Hough segment plus bridged collinear segment pairs
        all_pairs = list(segs)
        for i in range(min(18, len(segs))):
            x1, y1, x2, y2 = segs[i]
            L1 = math.hypot(x2 - x1, y2 - y1)
            a1 = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            for j in range(i + 1, min(18, len(segs))):
                u1, v1, u2, v2 = segs[j]
                a2 = math.degrees(math.atan2(v2 - v1, u2 - u1)) % 180.0
                da = abs(a1 - a2) % 180.0
                if min(da, 180.0 - da) > 10.0:
                    continue
                ux, uy = (x2 - x1) / max(1.0, L1), (y2 - y1) / max(1.0, L1)
                mx, my = 0.5 * (u1 + u2), 0.5 * (v1 + v2)
                if abs((mx - x1) * uy - (my - y1) * ux) > 14.0:
                    continue
                pts = [(x1, y1), (x2, y2), (u1, v1), (u2, v2)]
                bd, bp = -1.0, pts[0] + pts[3]
                for a in range(4):
                    for b in range(a + 1, 4):
                        d = math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
                        if d > bd:
                            bd = d
                            bp = pts[a] + pts[b]
                if 28.0 <= bd <= 420.0:
                    all_pairs.append(bp)

        for sx1, sy1, sx2, sy2 in all_pairs:
            L = math.hypot(sx2 - sx1, sy2 - sy1)
            if L < 26.0:
                continue
            ts = np.linspace(0.0, 1.0, 24)
            xs = np.clip(np.round(sx1 + ts * (sx2 - sx1)).astype(int), 0, WORK_RIDGE_SIZE - 1)
            ys = np.clip(np.round(sy1 + ts * (sy2 - sy1)).astype(int), 0, WORK_RIDGE_SIZE - 1)
            if float(np.mean(valid[ys, xs])) < 0.75:
                continue
            wh_cov = float(np.mean(np.maximum(white[ys, xs], red[ys, xs]) >= 0.25))
            sc = float(np.mean(score_map[ys, xs])) * (0.45 + 0.55 * wh_cov) * min(1.4, L / 140.0)
            cands.append((sc, sx1, sy1, sx2, sy2))

    if cands:
        cands.sort(key=lambda t: t[0], reverse=True)
        _, sx1, sy1, sx2, sy2 = cands[0]
        ox1 = float(np.clip(sx1 / sc4 - tfm.pad_left, 0.0, w - 1.0))
        oy1 = float(np.clip(sy1 / sc4 - tfm.pad_top, 0.0, h - 1.0))
        ox2 = float(np.clip(sx2 / sc4 - tfm.pad_left, 0.0, w - 1.0))
        oy2 = float(np.clip(sy2 / sc4 - tfm.pad_top, 0.0, h - 1.0))
    else:
        ox1, oy1, ox2, oy2 = round(w * 0.42, 1), round(h * 0.30, 1), round(w * 0.58, 1), round(h * 0.84, 1)

    if oy1 > oy2:
        ox1, oy1, ox2, oy2 = ox2, oy2, ox1, oy1

    ep = [round(ox1, 2), round(oy1, 2), round(ox2, 2), round(oy2, 2)]
    return _annotation_dict_from_endpoints("", ep, h, w, tier="AUTO_FITTED_NEEDS_REVIEW")


def _annotation_dict_from_endpoints(
    image_id: str,
    ep: list[float],
    h: int,
    w: int,
    thickness_px: float | None = None,
    occluded: bool = False,
    tier: str = "VERIFIED_MANUAL_GEOMETRY",
    note: str = "",
    existing_mask_rle: dict | None = None,
) -> dict:
    x1, y1, x2, y2 = [float(v) for v in ep]
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    ep_clean = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
    L = max(8.0, math.hypot(x2 - x1, y2 - y1))
    thick = float(thickness_px) if thickness_px is not None else max(6.0, round(0.022 * L, 1))
    obb, corners = obb_from_endpoints(x1, y1, x2, y2, thickness=thick)
    cl_pts = sample_centerline_points(ep_clean, 11)

    if existing_mask_rle is not None:
        m_rle = existing_mask_rle
    else:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.line(m, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, max(3, int(round(thick * 0.8))))
        m_rle = encode_mask_rle(m)

    bx1 = max(0, int(math.floor(min(x1, x2) - thick)))
    by1 = max(0, int(math.floor(min(y1, y2) - thick)))
    bx2 = min(w, int(math.ceil(max(x1, x2) + thick)))
    by2 = min(h, int(math.ceil(max(y1, y2) + thick)))
    trunc = bool(min(x1, x2) <= 5 or min(y1, y2) <= 5 or max(x1, x2) >= w - 6 or max(y1, y2) >= h - 6)

    return {
        "image_id": image_id,
        "endpoints_xyxy": ep_clean,
        "centerline_points_xy": cl_pts,
        "obb_cxcywh_angle": obb,
        "obb_corners_xy": corners,
        "bbox_xywh": [bx1, by1, max(1, bx2 - bx1), max(1, by2 - by1)],
        "thickness_px": round(thick, 2),
        "orientation_deg": obb[4],
        "length_px": round(L, 2),
        "visible_ratio": 0.72 if occluded else 1.0,
        "occluded": bool(occluded),
        "truncation": trunc,
        "gt_quality_tier": tier,
        "repair_note": note,
        "mask_rle": m_rle,
    }


def _zero_leakage_audit(splits: dict[str, list[dict]], hist_holdout_v2: list[dict], hist_p1c_fhold: list[dict]) -> dict:
    def _sigs(rows: list[dict]):
        shas = {str(r.get("sha256", "")).lower() for r in rows if r.get("sha256")}
        titles = {normalize_title(str(r.get("norm_title") or r.get("title") or r.get("source_title") or r.get("id"))) for r in rows if (r.get("norm_title") or r.get("title") or r.get("source_title"))}
        clusters = {normalize_title(str(r.get("source_cluster"))) for r in rows if r.get("source_cluster")}
        phashes = [(r["id"], r["ahash"], r["dhash"]) for r in rows if r.get("ahash") and r.get("dhash")]
        return shas, titles, clusters, phashes

    def _pair(name_a: str, rows_a: list[dict], name_b: str, rows_b: list[dict]) -> dict:
        sa, ta, ca, pa = _sigs(rows_a)
        sb, tb, cb, pb = _sigs(rows_b)
        ph_hits = []
        for ida, aa, da in pa:
            for idb, ab, db in pb:
                if hex_hamming_distance(aa, ab) <= 5 and hex_hamming_distance(da, db) <= 5:
                    ph_hits.append((ida, idb))
        return {
            "pair": f"{name_a}__vs__{name_b}",
            "sha256_overlap": len(sa & sb),
            "title_overlap": len(ta & tb),
            "cluster_overlap": len(ca & cb),
            "phash_overlap": len(ph_hits),
            "clean": (len(sa & sb) == 0 and len(ta & tb) == 0 and len(ca & cb) == 0 and len(ph_hits) == 0),
        }

    checks = [
        _pair("train_v3", splits["train_v3"], "val_cal_v3", splits["val_cal_v3"]),
        _pair("train_v3", splits["train_v3"], "final_holdout_v3", splits["final_holdout_v3"]),
        _pair("val_cal_v3", splits["val_cal_v3"], "final_holdout_v3", splits["final_holdout_v3"]),
        _pair("all_v3_splits", splits["train_v3"] + splits["val_cal_v3"] + splits["final_holdout_v3"], "hist_final_holdout_v2", hist_holdout_v2),
        _pair("all_v3_splits", splits["train_v3"] + splits["val_cal_v3"] + splits["final_holdout_v3"], "hist_vl_p1c_fhold", hist_p1c_fhold),
    ]
    return {"checks": checks, "all_clean": all(c["clean"] for c in checks)}


def build_vl_p3_dataset(holdout_pos_target: int = 80, holdout_neg_target: int = 165) -> tuple[Path, Path]:
    manifests_dir = VL_P3_DIR / "manifests"
    staging_v3 = VL_P3_DIR / "cache" / "staging_v3"
    cs_dir = VL_P3_DIR / "cache" / "contact_sheets"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    cs_dir.mkdir(parents=True, exist_ok=True)

    ai_root = WORKTREE_ROOT.parent
    p2_dir = ai_root / "VisionMate_VL_P2_AG" / "research" / "vl_p2"
    p1c_dir = ai_root / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c"

    ds_v2 = json.loads((p2_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    ds_p1c = json.loads((p1c_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    repaired_v2 = json.loads((manifests_dir / "vl_p2_repaired_annotations.json").read_text(encoding="utf-8"))["repaired_annotations"]
    overrides_path = manifests_dir / "manual_cane_gt_overrides_v3.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.is_file() else {}

    # Historical holdout Exclusion sets
    hist_fh_v2 = []
    for it in ds_v2["final_holdout_v2"]:
        r = dict(it)
        r["label"] = r.get("label") or r.get("label_id", "unknown")
        rp = _resolve_existing_path(r, p2_dir)
        if rp:
            im = load_image_unicode(rp)
            r["abs_path"] = str(rp)
            r["ahash"] = compute_ahash(im)
            r["dhash"] = compute_dhash(im)
            if not r.get("source_cluster"):
                r["source_cluster"] = normalize_title(r.get("title") or r.get("source_title") or r["id"])[:30]
        hist_fh_v2.append(r)

    hist_p1c = []
    for it in ds_p1c.get("final_holdout_full", []):
        r = dict(it)
        r["label"] = r.get("label") or r.get("label_id", "unknown")
        rp = _resolve_existing_path(r, p1c_dir)
        if rp:
            im = load_image_unicode(rp)
            r["abs_path"] = str(rp)
            r["ahash"] = compute_ahash(im)
            r["dhash"] = compute_dhash(im)
            if not r.get("source_cluster"):
                r["source_cluster"] = normalize_title(r.get("title") or r.get("source_title") or r["id"])[:30]
        hist_p1c.append(r)

    excl_shas = {str(r.get("sha256", "")).lower() for r in hist_fh_v2 + hist_p1c if r.get("sha256")}
    excl_titles = {normalize_title(str(r.get("norm_title") or r.get("title") or r.get("source_title") or "")) for r in hist_fh_v2 + hist_p1c if (r.get("norm_title") or r.get("title") or r.get("source_title"))}

    # 1. Audited non-holdout historical object_01 (strictly for train_v3 / val_cal_v3)
    hist_dev_pos: list[dict] = []
    for split_k in ("train_pos", "dev_cal_v2"):
        for it in ds_v2[split_k]:
            if (it.get("label") or it.get("label_id")) != "object_01":
                continue
            sha = str(it.get("sha256", "")).lower()
            tit_raw = it.get("title") or it.get("source_title") or it["id"]
            nt = normalize_title(str(it.get("norm_title") or tit_raw))
            if sha in excl_shas or nt in excl_titles:
                continue
            rp = _resolve_existing_path(it, p2_dir)
            if not rp:
                continue
            im = load_image_unicode(rp)
            hist_dev_pos.append({
                "id": it["id"],
                "label": "object_01",
                "abs_path": str(rp),
                "sha256": compute_sha256(rp),
                "ahash": compute_ahash(im),
                "dhash": compute_dhash(im),
                "width": int(im.shape[1]),
                "height": int(im.shape[0]),
                "title": tit_raw,
                "norm_title": nt,
                "source_cluster": normalize_title(str(it.get("source_cluster") or nt[:28])) or it["id"],
                "license": it.get("license", "Open/CC"),
                "author": it.get("author", "VL-P2 Audited"),
                "origin_pool": f"vl_p2_{split_k}",
                "is_brand_new": False,
            })

    # 2. Brand-new & unused unseen object_01 items
    unseen_pos: list[dict] = []
    unused_p = staging_v3 / "unused_pos_prefiltered.json"
    if unused_p.is_file():
        unused_rows = json.loads(unused_p.read_text(encoding="utf-8"))
        for idx in VERIFIED_UNUSED_POS_INDICES:
            if idx < len(unused_rows):
                r = unused_rows[idx]
                rp = _resolve_existing_path(r)
                if not rp:
                    continue
                im = load_image_unicode(rp)
                nt = normalize_title(r.get("title") or r["id"])
                unseen_pos.append({
                    "id": f"v3_pos_{r['id']}",
                    "label": "object_01",
                    "abs_path": str(rp),
                    "sha256": compute_sha256(rp),
                    "ahash": compute_ahash(im),
                    "dhash": compute_dhash(im),
                    "width": int(im.shape[1]),
                    "height": int(im.shape[0]),
                    "title": r.get("title", r["id"]),
                    "norm_title": nt,
                    "source_cluster": normalize_title(r.get("source_cluster") or nt[:28]) or r["id"],
                    "license": r.get("license", "CC"),
                    "author": r.get("author", "Open"),
                    "origin_pool": "v3_unused_p2_screened",
                    "is_brand_new": True,
                })

    ver_ids_path = staging_v3 / "verified_new_pos_ids.json"
    if ver_ids_path.is_file():
        verified_ids = set(json.loads(ver_ids_path.read_text(encoding="utf-8")))
        new_pos_meta = json.loads((staging_v3 / "v3_positives_metadata.json").read_text(encoding="utf-8"))
        for r in new_pos_meta:
            if r["id"] not in verified_ids:
                continue
            rp = _resolve_existing_path(r)
            if not rp:
                continue
            im = load_image_unicode(rp)
            unseen_pos.append({
                "id": r["id"],
                "label": "object_01",
                "abs_path": str(rp),
                "sha256": r["sha256"],
                "ahash": r.get("ahash") or compute_ahash(im),
                "dhash": r.get("dhash") or compute_dhash(im),
                "width": int(im.shape[1]),
                "height": int(im.shape[0]),
                "title": r.get("title", r["id"]),
                "norm_title": r.get("norm_title") or normalize_title(r["id"]),
                "source_cluster": normalize_title(r.get("source_cluster") or r["id"]),
                "license": r.get("license", "CC"),
                "author": r.get("author", "Open"),
                "source_url": r.get("source_url", ""),
                "origin_pool": "v3_fresh_harvest",
                "is_brand_new": True,
            })

    # Deduplicate unseen_pos internally
    dedup_unseen_pos = []
    seen_u_sha = {r["sha256"] for r in hist_dev_pos} | excl_shas
    seen_u_tit = {r["norm_title"] for r in hist_dev_pos} | excl_titles
    seen_u_clu = {r["source_cluster"] for r in hist_dev_pos}
    seen_u_ph = [(r["id"], r["ahash"], r["dhash"]) for r in hist_dev_pos + hist_fh_v2 + hist_p1c if r.get("ahash")]
    for r in unseen_pos:
        if r["sha256"] in seen_u_sha or r["norm_title"] in seen_u_tit or r["source_cluster"] in seen_u_clu:
            continue
        if any(hex_hamming_distance(r["ahash"], a) <= 5 and hex_hamming_distance(r["dhash"], d) <= 5 for _, a, d in seen_u_ph):
            continue
        dedup_unseen_pos.append(r)
        seen_u_sha.add(r["sha256"])
        seen_u_tit.add(r["norm_title"])
        seen_u_ph.append((r["id"], r["ahash"], r["dhash"]))

    # Allocate final_holdout_v3 object_01 strictly from unseen brand-new pool first (>= 80)
    fh_pos: list[dict] = []
    rem_pos: list[dict] = []
    pos_clusters: dict[str, list[dict]] = {}
    for r in dedup_unseen_pos:
        pos_clusters.setdefault(r["source_cluster"], []).append(r)

    # Prefer distinct clusters in holdout first (1 per cluster first, then 2nd only if needed to reach 80)
    sorted_cl = sorted(pos_clusters.items(), key=lambda kv: (len(kv[1]), kv[0]))
    for cl, items in sorted_cl:
        if len(fh_pos) < min(holdout_pos_target, len(dedup_unseen_pos)):
            fh_pos.append(items[0])
            rem_pos.extend(items[1:])
        else:
            rem_pos.extend(items)
    # If fewer clusters than holdout_pos_target, top up from rem_pos carefully
    if len(fh_pos) < holdout_pos_target and rem_pos:
        fh_clusters = {r["source_cluster"] for r in fh_pos}
        extra_out, leftover = [], []
        for r in rem_pos:
            if len(fh_pos) + len(extra_out) < holdout_pos_target and r["source_cluster"] in fh_clusters:
                extra_out.append(r)
            elif len(fh_pos) + len(extra_out) < holdout_pos_target and r["source_cluster"] not in fh_clusters:
                extra_out.append(r)
            else:
                leftover.append(r)
        fh_pos.extend(extra_out)
        rem_pos = leftover

    fh_pos_clusters = {r["source_cluster"] for r in fh_pos}
    dev_unseen_pos = [r for r in rem_pos if r["source_cluster"] not in fh_pos_clusters]
    dev_pos_pool = hist_dev_pos + dev_unseen_pos

    # Split dev_pos_pool group-disjoint into train_v3 (~70%) and val_cal_v3 (~30%)
    dev_pos_by_cl: dict[str, list[dict]] = {}
    for r in dev_pos_pool:
        dev_pos_by_cl.setdefault(r["source_cluster"], []).append(r)
    tr_pos, va_pos = [], []
    target_val_pos = max(16, int(round(0.30 * len(dev_pos_pool))))
    for idx_c, (_, cl_rows) in enumerate(sorted(dev_pos_by_cl.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
        if len(va_pos) < target_val_pos and idx_c % 3 == 1:
            va_pos.extend(cl_rows)
        else:
            tr_pos.extend(cl_rows)

    # 3. Screened brand-new V3 negatives + historical non-holdout negatives
    neg_raw = json.loads((staging_v3 / "v3_negatives_metadata.json").read_text(encoding="utf-8"))
    flat_negs = []
    flat_idx = 0
    for cat, rows in neg_raw.items():
        for r in rows:
            if flat_idx not in EXCLUDED_V3_NEG_INDICES:
                rp = _resolve_existing_path(r)
                if rp:
                    im = load_image_unicode(rp)
                    flat_negs.append({
                        "id": r["id"],
                        "label": "unknown",
                        "confuser_group": r.get("confuser_group", cat),
                        "person_holding_confuser": bool(r.get("person_holding_confuser", False)),
                        "abs_path": str(rp),
                        "sha256": r["sha256"],
                        "ahash": r.get("ahash") or compute_ahash(im),
                        "dhash": r.get("dhash") or compute_dhash(im),
                        "width": int(im.shape[1]),
                        "height": int(im.shape[0]),
                        "title": r.get("title", r["id"]),
                        "norm_title": r.get("norm_title") or normalize_title(r["id"]),
                        "source_cluster": normalize_title(r.get("source_cluster") or r["id"]),
                        "license": r.get("license", "CC"),
                        "author": r.get("author", "Open"),
                        "origin_pool": "v3_fresh_negatives",
                    })
            flat_idx += 1

    # Stratified split of brand-new negatives across confuser groups -> >= 160 in final_holdout_v3
    fh_neg, va_neg, tr_neg = [], [], []
    by_cg: dict[str, list[dict]] = {}
    for r in flat_negs:
        by_cg.setdefault(r["confuser_group"], []).append(r)

    # Stratified split of brand-new negatives across confuser groups -> >= 162 in final_holdout_v3
    fh_neg, va_neg, tr_neg = [], [], []
    by_cg: dict[str, list[dict]] = {}
    for r in flat_negs:
        by_cg.setdefault(r["confuser_group"], []).append(r)

    per_cg_fh = max(18, math.ceil(holdout_neg_target / max(1, len(by_cg))))
    fh_neg_clusters: set[str] = set()
    va_neg_clusters: set[str] = set()
    tr_neg_clusters: set[str] = set()
    va_neg_titles: set[str] = set()
    tr_neg_titles: set[str] = set()

    for cg, rows in sorted(by_cg.items()):
        # Group by source_cluster first so every cluster stays wholly in one split, balanced on person_holding_confuser
        cl_map: dict[str, list[dict]] = {}
        for r in sorted(rows, key=lambda x: x["id"]):
            cl_map.setdefault(r["source_cluster"], []).append(r)
        cl_list = sorted(
            cl_map.items(),
            key=lambda kv: (sum(1 for x in kv[1] if x["person_holding_confuser"]) == 0, kv[0]),
        )
        rows_fh_target = min(per_cg_fh + 1, max(12, int(round(0.65 * len(rows)))))
        cg_fh, cg_rem = [], []
        # Interleave person-holding and non-person-holding clusters evenly
        evens = cl_list[0::2]
        odds = cl_list[1::2]
        for cl_name, cl_items in evens + odds:
            if len(cg_fh) < rows_fh_target and cl_name not in va_neg_clusters and cl_name not in tr_neg_clusters:
                cg_fh.extend(cl_items)
                fh_neg_clusters.add(cl_name)
            elif cl_name not in fh_neg_clusters:
                cg_rem.append((cl_name, cl_items))
        fh_neg.extend(cg_fh)
        # Alternate remainder clusters evenly between va_neg and tr_neg so BOTH get person_holding_confuser
        va_toggle = 0
        for cl_name, cl_items in cg_rem:
            tits = {x["norm_title"] for x in cl_items}
            if cl_name in va_neg_clusters or (tits & va_neg_titles):
                va_neg.extend(cl_items)
                va_neg_clusters.add(cl_name)
                va_neg_titles.update(tits)
            elif cl_name in tr_neg_clusters or (tits & tr_neg_titles):
                tr_neg.extend(cl_items)
                tr_neg_clusters.add(cl_name)
                tr_neg_titles.update(tits)
            elif va_toggle % 2 == 1 and len(va_neg) <= len(tr_neg):
                va_neg.extend(cl_items)
                va_neg_clusters.add(cl_name)
                va_neg_titles.update(tits)
                va_toggle += 1
            else:
                tr_neg.extend(cl_items)
                tr_neg_clusters.add(cl_name)
                tr_neg_titles.update(tits)
                va_toggle += 1

    # Guarantee final_holdout_v3 negatives meet target (>= max(holdout_neg_target, 162)) without cluster overlap
    target_fh_neg = max(holdout_neg_target, 162)
    if len(fh_neg) < target_fh_neg:
        keep_tr = []
        moved_clusters: set[str] = set()
        for r in tr_neg:
            cl = r["source_cluster"]
            if (len(fh_neg) < target_fh_neg or cl in moved_clusters) and cl not in va_neg_clusters and r["norm_title"] not in va_neg_titles:
                fh_neg.append(r)
                fh_neg_clusters.add(cl)
                moved_clusters.add(cl)
            else:
                keep_tr.append(r)
        tr_neg = [r for r in keep_tr if r["source_cluster"] not in fh_neg_clusters]

    # Top up train_v3 and val_cal_v3 negatives cleanly from audited VL-P2 non-holdout negatives
    used_neg_shas = {r["sha256"] for r in fh_neg + va_neg + tr_neg} | excl_shas
    used_neg_tits = {r["norm_title"] for r in fh_neg + va_neg + tr_neg} | excl_titles
    hist_dev_negs = []
    for split_k in ("hard_neg_gallery", "dev_cal_v2"):
        for it in ds_v2[split_k]:
            if (it.get("label") or it.get("label_id")) == "object_01":
                continue
            sha = str(it.get("sha256", "")).lower()
            tit_raw = it.get("title") or it.get("source_title") or it["id"]
            nt = normalize_title(str(it.get("norm_title") or tit_raw))
            if sha in used_neg_shas or nt in used_neg_tits:
                continue
            rp = _resolve_existing_path(it, p2_dir)
            if not rp:
                continue
            im = load_image_unicode(rp)
            cl = normalize_title(str(it.get("source_cluster") or nt[:28])) or it["id"]
            if cl in fh_neg_clusters:
                continue
            used_neg_shas.add(sha)
            used_neg_tits.add(nt)
            hist_dev_negs.append({
                "id": it["id"],
                "label": "unknown",
                "confuser_group": it.get("subtype", it.get("confuser_group", "hist_hard_neg")),
                "person_holding_confuser": bool(it.get("subtype", "") in ("walking_stick", "umbrella", "trekking_pole", "handheld_pole")),
                "abs_path": str(rp),
                "sha256": sha,
                "ahash": compute_ahash(im),
                "dhash": compute_dhash(im),
                "width": int(im.shape[1]),
                "height": int(im.shape[0]),
                "title": tit_raw,
                "norm_title": nt,
                "source_cluster": cl,
                "license": it.get("license", "CC"),
                "author": it.get("author", "VL-P2 Audited"),
                "origin_pool": f"vl_p2_{split_k}",
            })
    for idx_hn, r in enumerate(hist_dev_negs):
        cl = r["source_cluster"]
        nt = r["norm_title"]
        if cl in va_neg_clusters or nt in va_neg_titles:
            va_neg.append(r)
            va_neg_clusters.add(cl)
            va_neg_titles.add(nt)
        elif cl in tr_neg_clusters or nt in tr_neg_titles:
            tr_neg.append(r)
            tr_neg_clusters.add(cl)
            tr_neg_titles.add(nt)
        elif len(va_neg) < 82 and idx_hn % 4 == 0:
            va_neg.append(r)
            va_neg_clusters.add(cl)
            va_neg_titles.add(nt)
        else:
            tr_neg.append(r)
            tr_neg_clusters.add(cl)
            tr_neg_titles.add(nt)

    # 4. Ambiguous pool
    amb_all = json.loads((staging_v3 / "amb_candidates.json").read_text(encoding="utf-8"))
    amb_items = []
    for idx in VERIFIED_AMB_INDICES:
        if idx < len(amb_all):
            r = amb_all[idx]
            rp = Path(r["abs_path"])
            im = load_image_unicode(rp)
            amb_items.append({
                "id": f"amb_v3_{len(amb_items)+1:03d}",
                "label": "ambiguous",
                "abs_path": str(rp),
                "sha256": r["sha256"],
                "ahash": r["ahash"],
                "dhash": r["dhash"],
                "width": int(im.shape[1]),
                "height": int(im.shape[0]),
                "title": rp.name,
                "norm_title": normalize_title(rp.stem),
                "source_cluster": normalize_title(rp.stem),
                "origin_pool": r["source_dir"],
            })

    splits = {
        "train_v3": tr_pos + tr_neg,
        "val_cal_v3": va_pos + va_neg,
        "final_holdout_v3": fh_pos + fh_neg,
        "ambiguous_cane_v3": amb_items,
    }

    # 5. Build complete GT annotations & render visual GT audit contact sheets for ALL new object_01
    annotations: dict[str, dict] = {}
    new_gt_thumbs: list[tuple[str, np.ndarray]] = []
    all_pos = tr_pos + va_pos + fh_pos

    for item in all_pos:
        iid = item["id"]
        h, w = item["height"], item["width"]
        img = load_image_unicode(Path(item["abs_path"]))
        if iid in overrides:
            ov = overrides[iid]
            annotations[iid] = _annotation_dict_from_endpoints(
                iid, ov["endpoints_xyxy"], h, w,
                thickness_px=ov.get("thickness_px"),
                occluded=bool(ov.get("occluded", False)),
                tier="MANUAL_OVERRIDE_V3",
                note=ov.get("repair_note", ""),
            )
        elif iid in repaired_v2:
            ra = repaired_v2[iid]
            annotations[iid] = _annotation_dict_from_endpoints(
                iid, ra["endpoints_xyxy"], h, w,
                thickness_px=ra.get("thickness_px"),
                occluded=bool(ra.get("occluded", False)),
                tier=ra.get("gt_quality_tier", "VERIFIED_MANUAL_GEOMETRY"),
                note=ra.get("repair_note", ""),
                existing_mask_rle=ra.get("mask_rle"),
            )
        else:
            fit = _auto_fit_cane_geometry(img)
            fit["image_id"] = iid
            annotations[iid] = fit

        if iid not in repaired_v2:
            vis = img.copy()
            a = annotations[iid]
            m = decode_mask_rle(a["mask_rle"]) > 0
            vis[m] = cv2.addWeighted(vis[m], 0.55, np.full_like(vis[m], (40, 230, 80)), 0.45, 0)
            ep = [int(round(v)) for v in a["endpoints_xyxy"]]
            cv2.line(vis, (ep[0], ep[1]), (ep[2], ep[3]), (0, 255, 0), 3)
            cv2.circle(vis, (ep[0], ep[1]), 6, (255, 0, 0), -1)
            cv2.circle(vis, (ep[2], ep[3]), 6, (0, 0, 255), -1)
            new_gt_thumbs.append((f"{iid}({w}x{h})", vis))

    for old in cs_dir.glob("v3_new_gt_p*.jpg"):
        old.unlink(missing_ok=True)
    for i in range(0, len(new_gt_thumbs), 30):
        make_contact_sheet(new_gt_thumbs[i:i + 30], cs_dir / f"v3_new_gt_p{i // 30 + 1}.jpg", cols=5, cell_w=260, cell_h=200)

    leakage = _zero_leakage_audit(splits, hist_fh_v2, hist_p1c)

    split_summary = {}
    for sk, rows in splits.items():
        pos_c = sum(1 for r in rows if r["label"] == "object_01")
        neg_c = sum(1 for r in rows if r["label"] == "unknown")
        hh_c = sum(1 for r in rows if r.get("person_holding_confuser"))
        split_summary[sk] = {"total": len(rows), "object_01": pos_c, "neg_or_other": neg_c, "person_holding_confuser": hh_c}

    ds_manifest = {
        "protocol_version": "VL-P3-v1",
        "split_summary": split_summary,
        "zero_leakage_audit": leakage,
        "excluded_indices": {
            "unused_pos_keep": VERIFIED_UNUSED_POS_INDICES,
            "v3_neg_excluded": sorted(EXCLUDED_V3_NEG_INDICES),
            "amb_v3_keep": VERIFIED_AMB_INDICES,
        },
        **splits,
    }
    ann_manifest = {
        "protocol_version": "VL-P3-GT-v1",
        "total_annotated_object_01": len(annotations),
        "new_gt_sheets_count": (len(new_gt_thumbs) + 29) // 30,
        "annotations": annotations,
    }

    ds_path = manifests_dir / "dataset_manifest.json"
    ann_path = manifests_dir / "annotation_manifest.json"
    ds_path.write_text(json.dumps(ds_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    ann_path.write_text(json.dumps(ann_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Dataset built:", json.dumps(split_summary, indent=2))
    print("Zero-leakage clean:", leakage["all_clean"])
    return ds_path, ann_path


if __name__ == "__main__":
    build_vl_p3_dataset()
