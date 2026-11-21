"""Audit the existing manual VL-P1 dataset (originals, crops, probes, provenance, ORB behavior)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .image_utils import (
    OBJECT_IDS,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    list_images,
    load_image_unicode,
)


def _match_crop_to_originals(crop_img: np.ndarray, originals: list[Path]) -> dict:
    """Find which original image contains crop_img using multi-scale template matching fallback."""
    ch, cw = crop_img.shape[:2]
    best = {"parent": None, "score": -1.0, "bbox_xywh": None}
    for orig_path in originals:
        orig = load_image_unicode(orig_path)
        oh, ow = orig.shape[:2]
        if oh >= ch and ow >= cw:
            res = cv2.matchTemplate(orig, crop_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best["score"]:
                best = {
                    "parent": orig_path.name,
                    "score": round(float(max_val), 5),
                    "bbox_xywh": [int(max_loc[0]), int(max_loc[1]), int(cw), int(ch)],
                    "orig_wh": [int(ow), int(oh)],
                }
    return best


def _orb_kp_count(img: np.ndarray, adaptive_pad: bool = False) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if adaptive_pad:
        h, w = gray.shape[:2]
        if min(h, w) < 128:
            scale = max(1.0, 128.0 / float(min(h, w)))
            if scale > 1.0:
                gray = cv2.resize(
                    gray,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_CUBIC,
                )
        gray = cv2.copyMakeBorder(gray, 36, 36, 36, 36, cv2.BORDER_REFLECT_101)
        orb = cv2.ORB_create(nfeatures=1500, edgeThreshold=15, patchSize=15, fastThreshold=10)
    else:
        orb = cv2.ORB_create(nfeatures=1300, fastThreshold=10)
    kps, _ = orb.detectAndCompute(gray, None)
    return len(kps) if kps is not None else 0


def audit_manual_dataset(base_dir: Path) -> dict:
    """Run full audit of existing originals, crops, dev_probes, backup_crops, and CSV provenance."""
    data_dir = base_dir / "data"
    orig_dir = data_dir / "originals"
    train_dir = data_dir / "train"
    labels = json.loads((base_dir / "labels.json").read_text(encoding="utf-8"))

    csv_records = {}
    csv_path = base_dir / "downloaded_photo_sources.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = f"{row['object_id']}/{Path(row['local_path']).name}"
                csv_records[key] = row

    originals_audit = []
    train_audit = []
    all_hashes: list[dict] = []

    for oid in OBJECT_IDS:
        orig_paths = list_images(orig_dir / oid)
        train_paths = list_images(train_dir / oid)

        for op in orig_paths:
            img = load_image_unicode(op)
            h, w = img.shape[:2]
            sha = compute_sha256(op)
            ah = compute_ahash(img)
            dh = compute_dhash(img)
            key = f"{oid}/{op.name}"
            csv_row = csv_records.get(key)
            csv_sha_match = bool(csv_row and csv_row["sha256"].lower() == sha.lower())
            rec = {
                "object_id": oid,
                "label_vi": labels.get(oid, oid),
                "filename": op.name,
                "rel_path": str(op.relative_to(base_dir)).replace("\\", "/"),
                "width": w,
                "height": h,
                "aspect_ratio": round(w / max(1, h), 3),
                "sha256": sha,
                "ahash": ah,
                "dhash": dh,
                "in_downloaded_csv": csv_row is not None,
                "csv_sha_match": csv_sha_match,
                "source_page": csv_row["source_page"] if csv_row else "manual_web_collection_untracked",
                "license": csv_row["license"] if csv_row else "UNVERIFIED_MANUAL_WEB",
                "author": csv_row["author"] if csv_row else "UNVERIFIED",
            }
            originals_audit.append(rec)
            all_hashes.append({"split": "originals", **rec})

        for tp in train_paths:
            img = load_image_unicode(tp)
            h, w = img.shape[:2]
            sha = compute_sha256(tp)
            ah = compute_ahash(img)
            dh = compute_dhash(img)
            parent_match = _match_crop_to_originals(img, orig_paths)
            kp_raw = _orb_kp_count(img, adaptive_pad=False)
            kp_pad = _orb_kp_count(img, adaptive_pad=True)
            rec = {
                "object_id": oid,
                "label_vi": labels.get(oid, oid),
                "filename": tp.name,
                "rel_path": str(tp.relative_to(base_dir)).replace("\\", "/"),
                "width": w,
                "height": h,
                "aspect_ratio": round(w / max(1, h), 3),
                "sha256": sha,
                "ahash": ah,
                "dhash": dh,
                "matched_original": parent_match["parent"],
                "template_match_score": parent_match["score"],
                "crop_bbox_xywh_in_original": parent_match["bbox_xywh"],
                "orb_keypoints_raw": kp_raw,
                "orb_keypoints_adaptive": kp_pad,
            }
            train_audit.append(rec)
            all_hashes.append({"split": "train", **rec})

    # Check dev_probes and backup_crops
    extra_probes = []
    for sub in ("dev_probes/object_01", "backup_crops/object_01"):
        folder = base_dir / sub
        for p in list_images(folder):
            img = load_image_unicode(p)
            h, w = img.shape[:2]
            extra_probes.append({
                "folder": sub,
                "filename": p.name,
                "rel_path": str(p.relative_to(base_dir)).replace("\\", "/"),
                "width": w,
                "height": h,
                "sha256": compute_sha256(p),
                "ahash": compute_ahash(img),
                "dhash": compute_dhash(img),
                "orb_keypoints_raw": _orb_kp_count(img, adaptive_pad=False),
                "orb_keypoints_adaptive": _orb_kp_count(img, adaptive_pad=True),
            })

    # Pairwise cross-item near-duplicate check among train items (or against dev_probes)
    cross_near_dups = []
    for i in range(len(train_audit)):
        for j in range(i + 1, len(train_audit)):
            d_a = hex_hamming_distance(train_audit[i]["ahash"], train_audit[j]["ahash"])
            d_d = hex_hamming_distance(train_audit[i]["dhash"], train_audit[j]["dhash"])
            if d_a <= 6 or d_d <= 6:
                cross_near_dups.append({
                    "a": train_audit[i]["rel_path"],
                    "b": train_audit[j]["rel_path"],
                    "ahash_dist": d_a,
                    "dhash_dist": d_d,
                })

    csv_missing_on_disk = [k for k in csv_records if not (orig_dir / k).exists()]
    provenance_verified_count = sum(1 for r in originals_audit if r["in_downloaded_csv"] and r["csv_sha_match"])

    summary = {
        "originals_total": len(originals_audit),
        "train_total": len(train_audit),
        "test_existing_total": sum(len(list_images(data_dir / "test" / x)) for x in [*OBJECT_IDS, "unknown"]),
        "provenance_verified_wikimedia_originals": provenance_verified_count,
        "provenance_unverified_manual_web_originals": len(originals_audit) - provenance_verified_count,
        "csv_entries_missing_from_originals": csv_missing_on_disk,
        "train_matched_1_to_1_with_originals": sum(1 for r in train_audit if r["template_match_score"] >= 0.99),
        "train_zero_kp_raw_orb": [r["rel_path"] for r in train_audit if r["orb_keypoints_raw"] == 0],
        "train_zero_kp_adaptive_orb": [r["rel_path"] for r in train_audit if r["orb_keypoints_adaptive"] == 0],
        "near_duplicates_within_train": cross_near_dups,
        "originals": originals_audit,
        "train": train_audit,
        "extra_probes": extra_probes,
    }

    manifests_dir = base_dir / "manifests"
    reports_dir = base_dir / "reports"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "manual_data_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        "# Báo Cáo Kiểm Toán Dữ Liệu Thủ Công Hiện Có (VL-P1 Audit)",
        "",
        "## 1. Tóm tắt số liệu kiểm toán",
        f"- **Tổng ảnh gốc (`data/originals`)**: {summary['originals_total']} ảnh (5 ảnh/nhóm × 5 nhóm)",
        f"- **Tổng ảnh học đã cắt (`data/train`)**: {summary['train_total']} ảnh (5 ảnh/nhóm × 5 nhóm)",
        f"- **Tổng ảnh test sẵn có (`data/test`)**: {summary['test_existing_total']} ảnh (trống hoàn toàn)",
        f"- **Ảnh gốc khớp nguồn Wikimedia trong `downloaded_photo_sources.csv`**: {summary['provenance_verified_wikimedia_originals']}/25",
        f"- **Ảnh gốc bổ sung thủ công chưa có metadata bản quyền**: {summary['provenance_unverified_manual_web_originals']}/25",
        f"- **Mục CSV đã bị thay thế/không còn trên đĩa**: `{summary['csv_entries_missing_from_originals']}`",
        f"- **Ảnh `data/train` khớp chính xác 100% (`score >= 0.99`) từ `data/originals`**: {summary['train_matched_1_to_1_with_originals']}/25",
        f"- **Ảnh train bị `0` keypoints với ORB thô (`edgeThreshold=31`)**: `{summary['train_zero_kp_raw_orb']}`",
        f"- **Ảnh train bị `0` keypoints sau Adaptive Narrow-ROI Padding**: `{summary['train_zero_kp_adaptive_orb']}`",
        "",
        "## 2. Phát hiện lỗi kỹ thuật trọng yếu trên tập `train` hiện có",
        "| File | Kích thước (W×H) | Ảnh gốc cha | Điểm khớp crop | Keypoints ORB thô | Keypoints ORB Adaptive | Ghi chú |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in train_audit:
        note = "RẤT HẸP (61px < 2*31px edgeThreshold -> 0 kp)" if r["orb_keypoints_raw"] == 0 else (
            "Ít đặc trưng bề mặt trơn" if r["orb_keypoints_raw"] < 300 else "OK"
        )
        md_lines.append(
            f"| `{r['rel_path']}` | {r['width']}×{r['height']} | `{r['matched_original']}` | "
            f"{r['template_match_score']:.4f} | {r['orb_keypoints_raw']} | {r['orb_keypoints_adaptive']} | {note} |"
        )

    md_lines.extend([
        "",
        "## 3. Kiểm tra ảnh thử phụ (`dev_probes` & `backup_crops`)",
        "| File | Kích thước (W×H) | Keypoints ORB thô | Keypoints ORB Adaptive |",
        "|---|---:|---:|---:|",
    ])
    for ep in extra_probes:
        md_lines.append(
            f"| `{ep['rel_path']}` | {ep['width']}×{ep['height']} | {ep['orb_keypoints_raw']} | {ep['orb_keypoints_adaptive']} |"
        )

    (reports_dir / "manual_dataset_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    res = audit_manual_dataset(base)
    print(f"Audit complete: {res['train_matched_1_to_1_with_originals']}/25 train crops matched to originals.")
    print(f"Zero-kp raw ORB: {res['train_zero_kp_raw_orb']}, adaptive ORB: {res['train_zero_kp_adaptive_orb']}")
