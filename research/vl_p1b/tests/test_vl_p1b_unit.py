"""Core unit tests for VL-P1b utilities, Method A, Method B, and Method C."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import socket

import cv2
import numpy as np
import pytest

from src.common_utils import (
    OBJECT_IDS,
    compute_ahash,
    compute_containment_and_bg_contamination,
    compute_dhash,
    compute_iou_xywh,
    format_count_pct_ci,
    hex_hamming_distance,
    list_images,
    load_image_unicode,
    wilson_ci_95,
)
from src.combined_pipeline import CombinedE2EPipeline, E2EDecisionConfig
from src.hard_negative_research import (
    HardNegConfig,
    HardNegativeRecognizer,
    extract_roi_geometry_features,
    preprocess_roi_dinov2,
)
from src.labeling_protocol import ALLOWED_OPEN_LICENSES, is_allowed_open_license
from src.proposal_research import (
    ObjectProposerVL1b,
    ProposalBox,
    ProposalRanker,
    YOLOXNanoExtractor,
    _compute_saliency_map,
    propose_thin_and_small_geometry,
)

VL_P1B_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = VL_P1B_DIR / "models"
YOLOX_PATH = VL_P1B_DIR.parent.parent / "models" / "object_detection" / "yolox_nano.onnx"


def test_wilson_ci_known_values() -> None:
    d1 = format_count_pct_ci(0, 25)
    assert d1["pct"] == 0.0
    assert 0.0 <= d1["ci95_low_pct"] <= d1["pct"] <= d1["ci95_high_pct"] < 15.0

    d2 = format_count_pct_ci(20, 20)
    assert d2["pct"] == 100.0
    assert 80.0 < d2["ci95_low_pct"] <= d2["pct"] == d2["ci95_high_pct"]

    d3 = format_count_pct_ci(39, 50)
    assert d3["pct"] == 78.0
    assert 64.0 < d3["ci95_low_pct"] < 65.5
    assert 86.5 < d3["ci95_high_pct"] < 88.0

    assert wilson_ci_95(0, 0) == (0.0, 0.0)


def test_license_filter() -> None:
    for lic in ("CC BY-SA 4.0", "CC-BY-4.0", "CC0", "Public domain", "CC-PD-Mark"):
        assert is_allowed_open_license(lic), f"Expected allowed license: {lic}"
    for bad in ("All rights reserved", "Fair use", "NC-ND 4.0", "Unknown", ""):
        assert not is_allowed_open_license(bad), f"Expected rejected license: {bad}"


def test_phash_duplicate_detection_and_iou_metrics() -> None:
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
    slightly_resized = cv2.resize(base, (96, 96))
    unrelated = np.zeros((120, 120, 3), dtype=np.uint8)
    unrelated[:, :60] = 255

    d_same = min(
        hex_hamming_distance(compute_ahash(base), compute_ahash(slightly_resized)),
        hex_hamming_distance(compute_dhash(base), compute_dhash(slightly_resized)),
    )
    assert d_same <= 6

    d_diff = min(
        hex_hamming_distance(compute_ahash(base), compute_ahash(unrelated)),
        hex_hamming_distance(compute_dhash(base), compute_dhash(unrelated)),
    )
    assert d_diff > 6

    assert compute_iou_xywh((10, 10, 40, 40), (10, 10, 40, 40)) == pytest.approx(1.0)
    assert compute_iou_xywh((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0
    cont, bg = compute_containment_and_bg_contamination((0, 0, 100, 100), (20, 20, 20, 20))
    assert cont == pytest.approx(1.0)
    assert bg == pytest.approx(0.96)


def test_corrupt_and_empty_image_handling(tmp_path: Path) -> None:
    bad_file = tmp_path / "corrupt.jpg"
    bad_file.write_bytes(b"not-a-jpeg-payload")
    with pytest.raises(ValueError):
        load_image_unicode(bad_file)

    with pytest.raises(ValueError):
        preprocess_roi_dinov2(np.zeros((0, 0, 3), dtype=np.uint8))


def test_method_b_thin_and_small_geometry_proposals() -> None:
    # Diagonal bright white cane on dark asphalt + small high-contrast key cluster
    img = np.full((320, 420, 3), 35, dtype=np.uint8)
    cv2.line(img, (60, 40), (320, 280), (245, 245, 245), thickness=6)
    cv2.circle(img, (340, 75), 14, (220, 200, 140), 2)
    cv2.rectangle(img, (332, 85), (348, 118), (200, 200, 210), -1)

    props = propose_thin_and_small_geometry(img)
    assert any(p.source == "geom_thin_line" for p in props)
    assert any(p.source == "geom_small_detail" for p in props)
    cane_gt = (55, 35, 270, 250)
    assert max(compute_iou_xywh(p.bbox_xywh, cane_gt) for p in props) >= 0.45


def test_proposal_ranker_nms_and_tight_preference() -> None:
    img = np.full((240, 240, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (80, 60), (160, 180), (230, 230, 230), -1)
    sal = _compute_saliency_map(img)

    p_tight = ProposalBox((78, 58, 84, 124), "fg_contour", 0.85)
    p_dup = ProposalBox((79, 59, 82, 122), "ms_sal_mid", 0.80)
    p_full = ProposalBox((0, 0, 240, 240), "full_frame", 1.0)
    p_person = ProposalBox((10, 10, 220, 220), "yolox_coco_0", 0.70, coco_class_id=0)

    kept = ProposalRanker.score_and_select(img, [p_full, p_person, p_dup, p_tight], sal, top_k=3, nms_iou=0.72)
    assert len(kept) <= 3
    # Duplicate p_dup should be suppressed by NMS in favor of distinct boxes
    ious_with_tight = [compute_iou_xywh(k.bbox_xywh, p_tight.bbox_xywh) for k in kept[1:]]
    assert all(i < 0.72 for i in ious_with_tight)
    assert kept[0].source == "fg_contour"


def test_zero_gt_bbox_arguments_and_offline_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verify inference APIs accept ONLY image arrays and never ground-truth boxes
    assert list(inspect.signature(ObjectProposerVL1b.propose).parameters.keys()) == ["self", "image"]
    assert list(inspect.signature(HardNegativeRecognizer.predict).parameters.keys()) == ["self", "image"]
    assert list(inspect.signature(CombinedE2EPipeline.predict).parameters.keys()) == ["self", "image"]

    # Block outbound sockets to enforce 100% offline inference
    def _deny_socket(*args, **kwargs):
        raise RuntimeError("Network access forbidden during VL-P1b offline inference")

    monkeypatch.setattr(socket, "socket", _deny_socket)

    rec = HardNegativeRecognizer(MODELS_DIR, HardNegConfig(mode="A3_full_verifier"))
    rec.fit_positives(VL_P1B_DIR / "data" / "train_pos")
    yolox = YOLOXNanoExtractor(YOLOX_PATH)
    proposer = ObjectProposerVL1b(yolox=yolox, mode="B_full_v1b", top_k=4)
    pipe = CombinedE2EPipeline(proposer, rec, E2EDecisionConfig())

    sample_pos = list_images(VL_P1B_DIR / "data" / "train_pos" / "object_03")[0]
    img = load_image_unicode(sample_pos)
    out1 = pipe.predict(img)
    out2 = pipe.predict(img)
    assert out1["status"] in ("KNOWN", "AMBIGUOUS", "UNKNOWN", "PROPOSAL_MISSED")
    assert out1["status"] == out2["status"]
    assert out1["score"] == pytest.approx(out2["score"], abs=1e-5)

    # Empty / no-proposal fallback test
    class _EmptyProposer:
        def propose(self, image):
            return [], 0.1

    missed = CombinedE2EPipeline(_EmptyProposer(), rec).predict(img)
    assert missed["status"] == "PROPOSAL_MISSED"
    assert missed["num_embeddings"] == 0
