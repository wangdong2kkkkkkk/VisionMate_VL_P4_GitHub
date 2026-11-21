"""Unit tests for VL-P1 labels schema, dataset structure, and manifest integrity."""
from __future__ import annotations

import json
from pathlib import Path

from src.image_utils import OBJECT_IDS, list_images

VL_P1_ROOT = Path(__file__).resolve().parent.parent


def test_labels_json_matches_five_vietnamese_categories() -> None:
    labels_path = VL_P1_ROOT / "labels.json"
    assert labels_path.is_file()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert list(labels.keys()) == OBJECT_IDS
    assert labels["object_01"] == "Gậy trắng hỗ trợ người khiếm thị"
    assert labels["object_02"] == "Chùm chìa khóa"
    assert labels["object_03"] == "Ví tiền"
    assert labels["object_04"] == "Đôi dép"
    assert labels["object_05"] == "Đôi giày"


def test_train_and_originals_have_25_samples_each() -> None:
    for split in ("originals", "train"):
        for oid in OBJECT_IDS:
            files = list_images(VL_P1_ROOT / "data" / split / oid)
            assert len(files) == 5, f"{split}/{oid} expected 5, got {len(files)}"


def test_calibration_and_test_splits_populated_and_manifest_valid() -> None:
    manifest_path = VL_P1_ROOT / "manifests" / "dataset_manifest.json"
    assert manifest_path.is_file(), "dataset_manifest.json must exist after split finalization"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for oid in OBJECT_IDS:
        assert len(list_images(VL_P1_ROOT / "data" / "calibration" / oid)) == 3
        assert len(list_images(VL_P1_ROOT / "data" / "test" / oid)) == 10
        assert len(list_images(VL_P1_ROOT / "data" / "test_full" / oid)) == 10
    assert len(list_images(VL_P1_ROOT / "data" / "calibration" / "unknown")) == 10
    assert len(list_images(VL_P1_ROOT / "data" / "test" / "unknown")) >= 20
    assert len(manifest["train"]) == 25
    assert len(manifest["calibration"]) == 25
    assert len(manifest["test"]) >= 70
    for item in [*manifest["calibration"], *manifest["test"]]:
        for field in ("id", "label", "split", "sha256", "ahash", "dhash", "license", "source_url"):
            assert item.get(field), f"Missing {field} in manifest entry {item.get('id')}"
