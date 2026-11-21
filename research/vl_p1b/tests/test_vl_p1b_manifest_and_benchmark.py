"""End-to-end integrity, calibration freeze, and benchmark verification tests for VL-P1b."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.collect_vl_p1b_dataset import _load_vl_p1_signatures
from src.common_utils import OBJECT_IDS, compute_sha256, hex_hamming_distance, load_image_unicode
from src.hard_negative_research import HardNegConfig, HardNegativeRecognizer
from src.labeling_protocol import HARD_NEGATIVE_GROUPS, is_allowed_open_license

VL_P1B_DIR = Path(__file__).resolve().parents[1]
VL_P1_DIR = VL_P1B_DIR.parent / "vl_p1"
MANIFEST_PATH = VL_P1B_DIR / "manifests" / "dataset_manifest.json"
FROZEN_PATH = VL_P1B_DIR / "manifests" / "frozen_thresholds.json"
FROZEN_A3_PATH = VL_P1B_DIR / "manifests" / "frozen_a3_state.json"
BENCHMARK_PATH = VL_P1B_DIR / "reports" / "benchmark_metrics.json"


def _require_artifacts() -> None:
    if not (MANIFEST_PATH.is_file() and FROZEN_PATH.is_file() and BENCHMARK_PATH.is_file()):
        pytest.fail("Required manifest/frozen/benchmark artifacts have not been generated yet.")


def test_manifest_counts_licenses_and_zero_overlaps() -> None:
    _require_artifacts()
    mf = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    counts = mf["counts"]
    assert counts["train_pos"] == 25
    assert counts["hard_neg_gallery"] >= 49
    assert counts["calibration_v1b_full"] >= 50
    assert counts["holdout_full"] >= 200
    assert counts["ambiguous_boundary"] >= 6

    for oid in OBJECT_IDS:
        assert mf["holdout_breakdown"]["positives_by_class"][oid] >= 20
    for grp in HARD_NEGATIVE_GROUPS:
        assert grp in mf["hard_neg_gallery_by_group"]
        assert mf["hard_neg_gallery_by_group"][grp] >= 5

    # License + zero-overlap checks across splits and VL-P1
    vl_shas, _, vl_phashes = _load_vl_p1_signatures(VL_P1_DIR)
    seen_shas: dict[str, set[str]] = {}
    seen_clusters: dict[str, set[str]] = {}
    seen_ph: dict[str, list[tuple[str, str, str]]] = {}

    for split_key in ("hard_neg_gallery", "calibration_v1b_full", "holdout_full", "ambiguous_boundary"):
        s_sha, s_cl, s_ph = set(), set(), []
        for row in mf[split_key]:
            assert is_allowed_open_license(row["license"])
            p = VL_P1B_DIR / row["full_rel_path"]
            assert p.is_file(), f"Missing image file: {p}"
            assert row["sha256"] not in vl_shas
            for vid, vah, vdh in vl_phashes:
                assert min(hex_hamming_distance(row["ahash"], vah), hex_hamming_distance(row["dhash"], vdh)) > 6, (
                    f"VL-P1 overlap {row['id']} vs {vid}"
                )
            s_sha.add(row["sha256"])
            s_cl.add(row["cluster_id"])
            s_ph.append((row["id"], row["ahash"], row["dhash"]))
        seen_shas[split_key] = s_sha
        seen_clusters[split_key] = s_cl
        seen_ph[split_key] = s_ph

    keys = list(seen_shas.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            assert not (seen_shas[a] & seen_shas[b])
            assert not (seen_clusters[a] & seen_clusters[b])
            for ida, aha, dha in seen_ph[a]:
                for idb, ahb, dhb in seen_ph[b]:
                    assert min(hex_hamming_distance(aha, ahb), hex_hamming_distance(dha, dhb)) > 6, (
                        f"Cross-split pHash overlap: {ida} ({a}) vs {idb} ({b})"
                    )


def test_frozen_thresholds_and_reproducible_load() -> None:
    _require_artifacts()
    ft = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    assert ft["holdout_touched_during_calibration"] is False
    assert ft["frozen_a3_state_sha256"] == compute_sha256(FROZEN_A3_PATH)

    rec = HardNegativeRecognizer(VL_P1B_DIR / "models", HardNegConfig(mode="A3_full_verifier"))
    rec.fit_positives(VL_P1B_DIR / "data" / "train_pos")
    rec.fit_hard_negatives(VL_P1B_DIR / "data" / "hard_neg_gallery")
    rec.load_frozen_state(FROZEN_A3_PATH)

    mf = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sample_row = mf["holdout_roi"][0]
    img = load_image_unicode(VL_P1B_DIR / sample_row["roi_rel_path"])
    dec1 = rec.predict(img)
    dec2 = rec.predict(img)
    assert dec1["status"] in ("KNOWN", "AMBIGUOUS", "UNKNOWN")
    assert dec1["status"] == dec2["status"]
    assert dec1["score"] == pytest.approx(dec2["score"], abs=1e-5)


def test_benchmark_metrics_and_budget_schema() -> None:
    _require_artifacts()
    rep = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert rep["frozen_thresholds_sha256"] == compute_sha256(FROZEN_PATH)

    # All required sections present with Wilson CI fields
    for cname in ("C1_vl_p1_baseline", "C2_hard_neg_only", "C3_proposal_only", "C4_combined_v1b"):
        hold = rep["method_c_e2e_ablation"][cname]["holdout_full"]
        for metric_k in ("overall_accuracy", "known_recall", "far"):
            m = hold[metric_k]
            assert 0.0 <= m["ci95_low_pct"] <= m["pct"] <= m["ci95_high_pct"] <= 100.0
            assert "/" in m["ratio"]

    hw = rep["hardware_and_scalability"]
    assert hw["rss_end_mb"] < 2048.0
    assert len(hw["gallery_scaling_n_20_50_100"]) == 3
    db_bytes = rep["calibration"]["calibration_summaries"]["A3_full_verifier"]["db_sizes"]["total_db_bytes"]
    assert db_bytes < 2_000_000
