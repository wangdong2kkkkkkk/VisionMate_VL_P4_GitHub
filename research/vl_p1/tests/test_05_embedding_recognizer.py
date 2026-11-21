"""Tests for Method B: Pretrained Image Embedding Recognizer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.embedding_recognizer import EmbeddingConfig, EmbeddingRecognizer
from src.image_utils import load_image_unicode

VL_P1_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = VL_P1_ROOT / "models"


@pytest.mark.parametrize("backbone", ["mobilenetv3_small", "dinov2_small_int8"])
@pytest.mark.parametrize("scoring_mode", ["1nn_max", "prototype", "hybrid"])
def test_embedding_norm_cosine_bounds_and_three_states(backbone: str, scoring_mode: str) -> None:
    rec = EmbeddingRecognizer(MODELS_DIR, EmbeddingConfig(backbone=backbone, scoring_mode=scoring_mode, tau_known=0.55, delta_margin=0.05))
    info = rec.fit_directory(VL_P1_ROOT / "data" / "train")
    assert info["gallery_db_bytes"] > 1024

    sample = load_image_unicode(VL_P1_ROOT / "data" / "train" / "object_02" / "01.png")
    emb = rec.extract_embedding(sample)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4

    ranked, _ = rec.score_embedding(emb)
    for row in ranked:
        assert -1.0001 <= row["max_sim"] <= 1.0001
        assert -1.0001 <= row["proto_sim"] <= 1.0001

    res = rec.predict(sample)
    assert res["status"] == "KNOWN"
    assert res["label"] == "object_02"

    # Synthetic 3-state boundary check via decide()
    assert rec.decide([{"id": "object_01", "score": 0.40}, {"id": "object_02", "score": 0.10}])["status"] == "UNKNOWN"
    assert rec.decide([{"id": "object_01", "score": 0.72}, {"id": "object_02", "score": 0.70}])["status"] == "AMBIGUOUS"
    ok = rec.decide([{"id": "object_01", "score": 0.80}, {"id": "object_02", "score": 0.60}])
    assert ok["status"] == "KNOWN" and ok["label"] == "object_01"


def test_embedding_reproducibility_across_two_runs() -> None:
    img = load_image_unicode(VL_P1_ROOT / "dev_probes" / "object_01" / "probe01.png")
    r1 = EmbeddingRecognizer(MODELS_DIR, EmbeddingConfig(backbone="mobilenetv3_small"))
    r2 = EmbeddingRecognizer(MODELS_DIR, EmbeddingConfig(backbone="mobilenetv3_small"))
    e1 = r1.extract_embedding(img)
    e2 = r2.extract_embedding(img)
    assert np.allclose(e1, e2, atol=1e-6)
