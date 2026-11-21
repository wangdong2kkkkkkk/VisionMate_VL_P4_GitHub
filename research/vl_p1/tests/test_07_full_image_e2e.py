"""End-to-end integration tests for Task 2 Full-Image Pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embedding_recognizer import EmbeddingConfig, EmbeddingRecognizer
from src.full_image_pipeline import FullImagePipeline, YOLOXProposer
from src.image_utils import load_image_unicode

VL_P1_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = VL_P1_ROOT / "models"
YOLOX_MODEL = VL_P1_ROOT.parent.parent / "models" / "object_detection" / "yolox_nano.onnx"


def test_full_image_pipeline_yolox_and_hybrid_paths() -> None:
    assert YOLOX_MODEL.is_file(), f"Missing baseline YOLOX model at {YOLOX_MODEL}"
    rec = EmbeddingRecognizer(MODELS_DIR, EmbeddingConfig(backbone="mobilenetv3_small", scoring_mode="hybrid"))
    rec.fit_directory(VL_P1_ROOT / "data" / "train")
    yolox = YOLOXProposer(YOLOX_MODEL)

    # Blank white image triggers PROPOSAL_MISSED under yolox_strict (no objects in COCO)
    blank = np.full((320, 320, 3), 255, dtype=np.uint8)
    pipe_strict = FullImagePipeline(rec, yolox_proposer=yolox, proposer_mode="yolox_strict")
    out_blank = pipe_strict.predict(blank)
    assert out_blank["status"] == "PROPOSAL_MISSED"

    # Hybrid proposer still runs reliably without ground truth boxes on full test scenes
    pipe_hyb = FullImagePipeline(rec, yolox_proposer=yolox, proposer_mode="hybrid")
    sample_full = load_image_unicode(VL_P1_ROOT / "data" / "test_full" / "object_05" / "test_01.jpg")
    out_full = pipe_hyb.predict(sample_full)
    assert out_full["status"] in {"KNOWN", "UNKNOWN", "AMBIGUOUS"}
    assert len(out_full["proposals"]) >= 1
    assert out_full["total_ms"] > 0
