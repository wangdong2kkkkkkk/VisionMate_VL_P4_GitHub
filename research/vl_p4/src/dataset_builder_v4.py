"""Build VL-P4 zero-leakage dataset & verified ground-truth annotations."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .common_utils import (
    VL_P4_DIR,
    WORKSPACE_ROOT,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    decode_mask_rle,
    encode_mask_rle,
    hex_hamming_distance,
    load_image_unicode,
    make_percent_grid_sheet,
    normalize_title,
    obb_from_endpoints,
    order_endpoints_top_bottom,
    sample_centerline_points,
)

# Manually verified (x1%, y1%, x2%, y2%, occluded, folded) from visual percent-grid inspection
NEW_VERIFIED_POS_PCT: dict[str, tuple[float, float, float, float, bool, bool]] = {
    "v4_raw_pos_0115": (44, 56, 65, 97, False, False),
    "v4_raw_pos_0116": (38, 63, 62, 87, False, False),
    "v4_raw_pos_0117": (71, 61, 95, 89, False, False),
    "v4_raw_pos_0162": (58, 36, 48, 92, False, False),
    "v4_raw_pos_0176": (71, 40, 81, 67, False, False),
    "v4_raw_pos_0194": (20, 62, 13, 81, False, False),
    "v4_raw_pos_0198": (67, 68, 81, 83, True, False),
    "v4_raw_pos_0200": (28, 43, 35, 95, True, False),
    "v4_raw_pos_0332": (76, 62, 76, 94, True, False),
    "v4_raw_pos_0334": (41, 65, 36, 86, True, False),
    "v4_raw_pos_0339": (45, 58, 46, 89, True, False),
    "v4_raw_pos_0340": (39, 55, 39, 86, False, False),
    "v4_raw_pos_0346": (68, 67, 63, 94, True, False),
    "v4_raw_pos_0350": (82, 34, 72, 74, False, False),
    "v4_raw_pos_0382": (66, 51, 63, 72, True, False),
    "v4_raw_pos_0394": (54, 59, 49, 77, True, False),
    "v4_raw_pos_0395": (45, 56, 51, 82, False, False),
    "v4_raw_pos_0398": (79, 55, 73, 77, True, False),
    "v4_raw_pos_0399": (54, 59, 53, 81, True, False),
    "v4_raw_pos_0400": (26, 66, 37, 89, False, False),
    "v4_raw_pos_0403": (39, 59, 44, 83, True, False),
    "v4_raw_pos_0404": (56, 54, 67, 94, False, False),
    "v4_raw_pos_0431": (18, 53, 18, 86, True, False),
    "v4_raw_pos_0432": (57, 50, 56, 72, True, False),
    "v4_raw_pos_0433": (43, 47, 43, 72, True, False),
    "v4_raw_pos_0434": (56, 55, 52, 77, True, False),
    "v4_raw_pos_0435": (27, 47, 49, 79, False, False),
    "v4_raw_pos_0455": (39, 45, 39, 98, False, False),
    "v4_raw_pos_0456": (32, 48, 32, 99, False, False),
    "v4_raw_pos_0459": (23, 73, 16, 98, True, False),
    "v4_raw_pos_0780": (85, 41, 83, 66, True, False),
    "v4_raw_pos_0790": (42, 48, 42, 94, False, False),
    "v4_raw_pos_0791": (71, 45, 52, 92, False, False),
    "v4_raw_pos_0792": (37, 48, 33, 84, False, False),
    "v4_raw_pos_0793": (58, 49, 52, 75, False, False),
    "v4_raw_pos_0803": (34, 48, 27, 96, False, False),
    "v4_raw_pos_0804": (72, 50, 68, 76, True, False),
    "v4_raw_pos_0805": (20, 52, 32, 90, False, False),
    "v4_raw_pos_0806": (52, 42, 50, 95, False, False),
    "v4_raw_pos_0808": (51, 56, 54, 78, True, False),
    "v4_raw_pos_0809": (30, 60, 29, 81, True, False),
    "v4_raw_pos_0810": (31, 61, 31, 80, True, False),
    "v4_raw_pos_0822": (23, 56, 30, 91, False, False),
    "v4_raw_pos_0835": (74, 52, 69, 81, True, False),
    "v4_raw_pos_0842": (62, 47, 56, 81, False, False),
    "v4_raw_pos_0843": (62, 48, 55, 89, False, False),
    "v4_raw_pos_0844": (49, 46, 51, 81, True, False),
    "v4_raw_pos_0845": (45, 74, 43, 88, True, False),
    "v4_raw_pos_0846": (58, 53, 55, 82, True, False),
    "v4_raw_pos_0849": (44, 52, 43, 82, True, False),
    "v4_raw_pos_0850": (85, 53, 86, 82, True, False),
    "v4_raw_pos_0852": (49, 53, 49, 94, False, False),
    "v4_raw_pos_0853": (42, 56, 42, 83, True, False),
    "v4_raw_pos_0855": (39, 55, 39, 77, True, False),
    "v4_raw_pos_0856": (46, 48, 46, 91, False, False),
    "v4_raw_pos_0857": (53, 56, 53, 94, False, False),
    "v4_raw_pos_0859": (85, 55, 82, 88, False, False),
    "v4_raw_pos_0874": (66, 58, 69, 93, True, False),
    "v4_raw_pos_2113": (39, 47, 46, 65, True, False),
    "v4_raw_pos_2147": (22, 54, 30, 96, False, False),
    "v4_raw_pos_2149": (33, 45, 45, 97, False, False),
    "v4_raw_pos_2160": (48, 56, 56, 94, False, False),
    "v4_raw_pos_2163": (16, 53, 72, 48, False, True),
    "v4_raw_pos_2171": (36, 65, 40, 92, False, False),
    "v4_raw_pos_2172": (39, 45, 19, 92, False, False),
    "v4_raw_pos_2191": (71, 41, 71, 83, False, False),
    "v4_raw_pos_2197": (30, 57, 30, 87, False, False),
    "v4_raw_pos_2198": (29, 50, 32, 74, True, False),
    "v4_raw_pos_2200": (74, 58, 82, 85, False, False),
    "v4_raw_pos_2205": (47, 62, 47, 81, True, False),
    "v4_raw_pos_2210": (39, 77, 34, 98, True, False),
    "v4_raw_pos_2215": (19, 53, 34, 96, False, False),
    "v4_raw_pos_2216": (22, 45, 24, 92, False, False),
    "v4_raw_pos_2246": (68, 54, 71, 86, True, False),
    "v4_raw_pos_2247": (53, 45, 53, 66, True, False),
    "v4_raw_pos_2248": (55, 56, 47, 94, False, False),
    "v4_raw_pos_2249": (62, 48, 76, 90, False, False),
    "v4_raw_pos_3008": (46, 46, 30, 82, False, False),
    "v4_raw_pos_3019": (67, 63, 68, 86, True, False),
    "v4_raw_pos_3022": (78, 63, 73, 87, False, False),
    "v4_raw_pos_3031": (48, 57, 45, 78, True, False),
    "v4_raw_pos_3258": (31, 43, 21, 86, False, False),
}


def _build_annotation_record(item: dict, ep_px: list[float], occluded: bool, folded: bool, status: str) -> dict:
    h, w = int(item["height"]), int(item["width"])
    ep = order_endpoints_top_bottom(ep_px)
    x1, y1, x2, y2 = ep
    length_px = max(8.0, math.hypot(x2 - x1, y2 - y1))
    thick = max(4.0, min(14.0, 0.022 * length_px))
    obb, _ = obb_from_endpoints(x1, y1, x2, y2, thickness=thick)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, max(3, int(round(thick))))
    pad = max(6, int(round(thick)))
    bx1 = max(0, int(math.floor(min(x1, x2) - pad)))
    by1 = max(0, int(math.floor(min(y1, y2) - pad)))
    bx2 = min(w, int(math.ceil(max(x1, x2) + pad)))
    by2 = min(h, int(math.ceil(max(y1, y2) + pad)))
    return {
        "id": item["id"],
        "endpoints_xyxy": ep,
        "centerline_points_xy": sample_centerline_points(ep, 11),
        "bbox_xywh": [bx1, by1, max(4, bx2 - bx1), max(4, by2 - by1)],
        "obb_cxcywh_angle": obb,
        "mask_rle": encode_mask_rle(mask),
        "angle_deg": obb[4],
        "length_px": round(length_px, 2),
        "occluded": bool(occluded),
        "folded": bool(folded),
        "annotation_status": status,
    }


def _group_split_by_cluster(items: list[dict], target_hold: int, target_val: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic cluster-disjoint split into (train, val, holdout)."""
    clusters: dict[str, list[dict]] = {}
    for it in items:
        cl = it.get("source_cluster") or it["id"]
        clusters.setdefault(cl, []).append(it)
    # Sort largest clusters first to pack evenly while keeping clusters intact
    ordered = sorted(clusters.values(), key=lambda g: (len(g), g[0]["id"]))
    hold, val, train = [], [], []
    for grp in ordered:
        if len(hold) < target_hold:
            hold.extend(grp)
        elif len(val) < target_val:
            val.extend(grp)
        else:
            train.extend(grp)
    return train, val, hold


def _check_leakage(a: list[dict], b: list[dict], pair_name: str) -> dict:
    sha_ov = len({r["sha256"] for r in a} & {r["sha256"] for r in b})
    ttl_ov = len({normalize_title(r.get("title", "")) for r in a if r.get("title")} & {normalize_title(r.get("title", "")) for r in b if r.get("title")})
    cl_ov = len({r.get("source_cluster", "") for r in a if r.get("source_cluster")} & {r.get("source_cluster", "") for r in b if r.get("source_cluster")})
    ph_ov = 0
    for ra in a:
        for rb in b:
            if ra.get("ahash") and rb.get("ahash"):
                if hex_hamming_distance(ra["ahash"], rb["ahash"]) <= 4 or hex_hamming_distance(ra["dhash"], rb["dhash"]) <= 4:
                    ph_ov += 1
                    break
    return {
        "pair": pair_name,
        "sha256_overlap": sha_ov,
        "title_overlap": ttl_ov,
        "cluster_overlap": cl_ov,
        "phash_overlap": ph_ov,
        "clean": sha_ov == 0 and ttl_ov == 0 and cl_ov == 0 and ph_ov == 0,
    }


def build_vl_p4_dataset() -> Path:
    manifests_dir = VL_P4_DIR / "manifests"
    cache_dir = VL_P4_DIR / "cache"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    p3_dir = WORKSPACE_ROOT / "VisionMate_VL_P3_AG" / "research" / "vl_p3" / "manifests"
    p3_ds = json.loads((p3_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    p3_ann = json.loads((p3_dir / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]
    raw_v4 = json.loads((cache_dir / "raw_candidates_v4.json").read_text(encoding="utf-8"))

    # 1. Existing VL-P3 dev items (train_v3 + val_cal_v3 ONLY; NEVER touch final_holdout_v3!)
    p3_dev_pos = [dict(r, source_split="vl_p3_dev") for r in (p3_ds["train_v3"] + p3_ds["val_cal_v3"]) if r["label"] == "object_01"]
    p3_dev_neg = [dict(r, source_split="vl_p3_dev") for r in (p3_ds["train_v3"] + p3_ds["val_cal_v3"]) if r["label"] != "object_01"]

    annotations: dict[str, dict] = {}
    for r in p3_dev_pos:
        a = dict(p3_ann[r["id"]])
        a["folded"] = bool(a.get("folded", False))
        a["annotation_status"] = "VERIFIED_GROUND_TRUTH"
        annotations[r["id"]] = a

    # 2. Brand-new visually verified V4 positives
    raw_pos_map = {r["id"]: r for r in raw_v4["pos_records"]}
    new_pos_items: list[dict] = []
    gt_vis_thumbs = []
    for pid, (x1p, y1p, x2p, y2p, occ, fold) in NEW_VERIFIED_POS_PCT.items():
        if pid not in raw_pos_map:
            continue
        it = dict(raw_pos_map[pid], source_split="vl_p4_new")
        w, h = int(it["width"]), int(it["height"])
        ep_px = [x1p * w / 100.0, y1p * h / 100.0, x2p * w / 100.0, y2p * h / 100.0]
        ann_rec = _build_annotation_record(it, ep_px, occ, fold, "MANUALLY_CORRECTED")
        annotations[pid] = ann_rec
        new_pos_items.append(it)
        img = load_image_unicode(Path(it["abs_path"]))
        e = [int(round(v)) for v in ann_rec["endpoints_xyxy"]]
        cv2.line(img, (e[0], e[1]), (e[2], e[3]), (0, 255, 0), 3)
        cv2.circle(img, (e[0], e[1]), 6, (255, 0, 0), -1)
        cv2.circle(img, (e[2], e[3]), 6, (0, 0, 255), -1)
        gt_vis_thumbs.append((pid, img))

    # Render GT verification sheets
    for i in range(0, len(gt_vis_thumbs), 12):
        make_percent_grid_sheet(gt_vis_thumbs[i : i + 12], cache_dir / "screening" / f"gt_verify_v4_{i // 12 + 1:02d}.jpg", cols=3, cell_w=380, cell_h=300)

    # Split new_pos_items strictly by source_cluster:
    # - `international_white_cane_day` (49) + `internation_white_cane_day` (14) -> train_v4 (63 new pos)
    # - Remaining distinct clusters (`vision_australia_international_white`, `ladies_leading_the_way`, singletons) -> final_holdout_v4 (19 new pos)
    new_pos_tr, new_pos_val, new_pos_hold = [], [], []
    for it in new_pos_items:
        cl = it.get("source_cluster", "")
        if cl in ("international_white_cane_day", "internation_white_cane_day"):
            new_pos_tr.append(it)
        else:
            new_pos_hold.append(it)

    # Split brand-new hard negatives (~595) into dev (~395) and blind final_holdout_v4 (~200)
    new_neg_items = [dict(r, source_split="vl_p4_new") for r in raw_v4["neg_records"]]
    new_neg_tr, new_neg_val, new_neg_hold = _group_split_by_cluster(new_neg_items, target_hold=200, target_val=60)

    # Combine VL-P3 dev + V4 new dev (zero overlap with final_holdout_v3 OR final_holdout_v4!)
    p3_tr_ids = {r["id"] for r in p3_ds["train_v3"]}
    p3_va_ids = {r["id"] for r in p3_ds["val_cal_v3"]}
    p3_tr_pos = [r for r in p3_dev_pos if r["id"] in p3_tr_ids]
    p3_va_pos = [r for r in p3_dev_pos if r["id"] in p3_va_ids]
    p3_tr_neg = [r for r in p3_dev_neg if r["id"] in p3_tr_ids]
    p3_va_neg = [r for r in p3_dev_neg if r["id"] in p3_va_ids]

    train_v4 = p3_tr_pos + new_pos_tr + p3_tr_neg + new_neg_tr
    val_cal_v4 = p3_va_pos + new_pos_val + p3_va_neg + new_neg_val
    final_holdout_v4 = new_pos_hold + new_neg_hold
    ambiguous_v4 = list(p3_ds["ambiguous_cane_v3"])

    # Leakage checks
    fh_v3 = p3_ds["final_holdout_v3"]
    checks = [
        _check_leakage(train_v4, val_cal_v4, "train_v4__vs__val_cal_v4"),
        _check_leakage(train_v4, final_holdout_v4, "train_v4__vs__final_holdout_v4"),
        _check_leakage(val_cal_v4, final_holdout_v4, "val_cal_v4__vs__final_holdout_v4"),
        _check_leakage(train_v4 + val_cal_v4, fh_v3, "dev_v4__vs__final_holdout_v3"),
        _check_leakage(final_holdout_v4, fh_v3, "final_holdout_v4__vs__final_holdout_v3"),
    ]
    all_clean = all(c["clean"] for c in checks)

    summary = {
        k: {
            "total": len(v),
            "object_01": sum(1 for r in v if r.get("label") == "object_01"),
            "neg_or_other": sum(1 for r in v if r.get("label") != "object_01"),
            "person_holding_confuser": sum(1 for r in v if r.get("person_holding_confuser")),
        }
        for k, v in [
            ("train_v4", train_v4),
            ("val_cal_v4", val_cal_v4),
            ("final_holdout_v4", final_holdout_v4),
            ("ambiguous_cane_v4", ambiguous_v4),
        ]
    }

    ds_out = manifests_dir / "dataset_manifest.json"
    ds_out.write_text(json.dumps({
        "protocol_version": "VL-P4-v1",
        "split_summary": summary,
        "zero_leakage_audit": {"checks": checks, "all_clean": all_clean},
        "train_v4": train_v4,
        "val_cal_v4": val_cal_v4,
        "final_holdout_v4": final_holdout_v4,
        "ambiguous_cane_v4": ambiguous_v4,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    ann_out = manifests_dir / "annotation_manifest.json"
    ann_out.write_text(json.dumps({
        "count": len(annotations),
        "annotations": annotations,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("VL-P4 split_summary:", json.dumps(summary, indent=2))
    print("Zero-leakage all_clean:", all_clean)
    for c in checks:
        print(" ", c)
    return ds_out


if __name__ == "__main__":
    build_vl_p4_dataset()
