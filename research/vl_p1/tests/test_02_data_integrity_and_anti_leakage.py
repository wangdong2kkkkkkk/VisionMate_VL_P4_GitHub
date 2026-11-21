"""Tests for exact SHA-256 and perceptual hash anti-leakage across train/cal/test splits."""
from __future__ import annotations

import json
from pathlib import Path

from src.image_utils import (
    OBJECT_IDS,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    list_images,
    load_image_unicode,
)

VL_P1_ROOT = Path(__file__).resolve().parent.parent


def _collect_split_hashes(split_name: str, folder: Path) -> list[dict]:
    entries = []
    for sub in [*OBJECT_IDS, "unknown"]:
        for p in list_images(folder / sub):
            img = load_image_unicode(p)
            entries.append({
                "split": split_name,
                "label": sub,
                "path": str(p.relative_to(VL_P1_ROOT)),
                "sha256": compute_sha256(p),
                "ahash": compute_ahash(img),
                "dhash": compute_dhash(img),
            })
    return entries


def test_zero_sha256_and_perceptual_leakage_across_splits() -> None:
    orig_entries = _collect_split_hashes("originals", VL_P1_ROOT / "data" / "originals")
    train_entries = _collect_split_hashes("train", VL_P1_ROOT / "data" / "train")
    cal_entries = _collect_split_hashes("calibration", VL_P1_ROOT / "data" / "calibration")
    test_roi_entries = _collect_split_hashes("test", VL_P1_ROOT / "data" / "test")
    test_full_entries = _collect_split_hashes("test_full", VL_P1_ROOT / "data" / "test_full")

    train_all = orig_entries + train_entries
    eval_all = cal_entries + test_roi_entries + test_full_entries

    train_shas = {e["sha256"] for e in train_all}
    cal_shas = {e["sha256"] for e in cal_entries}
    test_shas = {e["sha256"] for e in test_roi_entries + test_full_entries}

    assert train_shas.isdisjoint(cal_shas), "SHA-256 leakage between train and calibration"
    assert train_shas.isdisjoint(test_shas), "SHA-256 leakage between train and test"
    assert cal_shas.isdisjoint(test_shas), "SHA-256 leakage between calibration and test"

    for ev in eval_all:
        for tr in train_all:
            d_ah = hex_hamming_distance(ev["ahash"], tr["ahash"])
            d_dh = hex_hamming_distance(ev["dhash"], tr["dhash"])
            # Require combined perceptual distance (both aHash & dHash small indicates structural duplicate)
            assert d_dh > 8 and not (d_ah <= 8 and d_dh <= 12), (
                f"Near-duplicate leakage between {ev['path']} and {tr['path']} (aHash={d_ah}, dHash={d_dh})"
            )


def test_manual_audit_records_all_25_train_parents() -> None:
    audit_path = VL_P1_ROOT / "manifests" / "manual_data_audit.json"
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["train_matched_1_to_1_with_originals"] == 25
    assert "data/train/object_01/03.png" in audit["train_zero_kp_raw_orb"]
    assert audit["train_zero_kp_adaptive_orb"] == []
