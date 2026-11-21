"""Tests for corrupt/blank/tiny/narrow images, missing model, offline guarantee, and memory bounds."""
from __future__ import annotations

from pathlib import Path
import socket

import numpy as np
import psutil
import pytest

from src.embedding_recognizer import EmbeddingConfig, EmbeddingRecognizer
from src.image_utils import load_image_unicode
from src.orb_recognizer import ORBConfig, ORBRecognizer

VL_P1_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = VL_P1_ROOT / "models"


def test_corrupt_and_missing_image_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ValueError):
        load_image_unicode(missing)

    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not_a_real_jpeg_payload")
    with pytest.raises(ValueError):
        load_image_unicode(corrupt)


def test_missing_model_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EmbeddingRecognizer(tmp_path, EmbeddingConfig(backbone="mobilenetv3_small"))


@pytest.mark.parametrize(
    "img",
    [
        np.full((240, 240, 3), 255, dtype=np.uint8),   # blank white
        np.zeros((240, 240, 3), dtype=np.uint8),       # blank black
        np.full((240, 240, 3), 128, dtype=np.uint8),   # featureless flat gray
        np.full((8, 8, 3), 90, dtype=np.uint8),        # super tiny ROI
        np.full((500, 10, 3), 200, dtype=np.uint8),    # ultra narrow vertical ROI
    ],
)
def test_degenerate_images_handled_gracefully(img: np.ndarray) -> None:
    emb_rec = EmbeddingRecognizer(MODELS_DIR, EmbeddingConfig(backbone="mobilenetv3_small"))
    emb_rec.fit_directory(VL_P1_ROOT / "data" / "train")
    out_e = emb_rec.predict(img)
    assert out_e["status"] in {"KNOWN", "UNKNOWN", "AMBIGUOUS"}
    assert np.isfinite(out_e["score"])

    orb_rec = ORBRecognizer(ORBConfig(adaptive_narrow_roi=True))
    orb_rec.fit_directory(VL_P1_ROOT / "data" / "train")
    out_o = orb_rec.predict(img)
    assert out_o["status"] == "UNKNOWN"


def test_strict_offline_inference_with_network_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def _deny_network(*args, **kwargs):
        raise RuntimeError("NETWORK ACCESS FORBIDDEN IN OFFLINE MODE")

    monkeypatch.setattr(socket, "socket", _deny_network)
    rec = EmbeddingRecognizer(MODELS_DIR, EmbeddingConfig(backbone="mobilenetv3_small"))
    rec.fit_directory(VL_P1_ROOT / "data" / "train")
    probe = load_image_unicode(VL_P1_ROOT / "dev_probes" / "object_01" / "probe01.png")
    out = rec.predict(probe)
    assert out["status"] in {"KNOWN", "UNKNOWN", "AMBIGUOUS"}


def test_memory_stability_over_30_queries() -> None:
    rec = EmbeddingRecognizer(MODELS_DIR, EmbeddingConfig(backbone="mobilenetv3_small"))
    rec.fit_directory(VL_P1_ROOT / "data" / "train")
    probe = load_image_unicode(VL_P1_ROOT / "dev_probes" / "object_01" / "probe01.png")
    proc = psutil.Process()
    rss_start = proc.memory_info().rss / (1024 * 1024)
    for _ in range(30):
        rec.predict(probe)
    rss_end = proc.memory_info().rss / (1024 * 1024)
    assert (rss_end - rss_start) < 60.0, f"Unexpected RAM growth: {rss_end - rss_start:.2f} MB"
