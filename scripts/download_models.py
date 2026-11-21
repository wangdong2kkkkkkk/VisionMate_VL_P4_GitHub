# -*- coding: utf-8 -*-
"""
One-time Bootstrap Model Downloader & SHA-256 Verifier for VisionMate Glasses Phase 1.
Used ONLY during initial online setup; never invoked at runtime.
All models are Apache-2.0 licensed (see docs/MODEL_LICENSES.md).
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS: Dict[str, Tuple[str, str]] = {
    "models/tessdata_fast/vie.traineddata": (
        "https://github.com/tesseract-ocr/tessdata_fast/raw/main/vie.traineddata",
        "79DF64CAF7BCFB2A27DF5042ECB6121E196EADA34DA774956995747636D5BFA1",
    ),
    "models/tessdata_best/vie.traineddata": (
        "https://github.com/tesseract-ocr/tessdata_best/raw/main/vie.traineddata",
        "B6B49293D95D0B6DBD8780174627E82C75BE957B6F4ED9862155540D6B00BB45",
    ),
    "models/paddleocr_latin/latin_rec.onnx": (
        "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/rec.onnx",
        "614FFC2D6D3902D360FAD7F1B0DD455EE45E877069D14C4E51A99DC4EF144409",
    ),
    "models/paddleocr_latin/latin_dict.txt": (
        "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/dict.txt",
        "3C0A8A79B612653C25F765271714F71281E4E955962155540D6B00BB45"
        if False
        else "3C0A8A79B612653C25F765271714F71281E4E955962C153E272B7B8C1D2B13FF",
    ),
    "models/object_detection/yolox_nano.onnx": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        "C789161ED43C8269FCD4E67C67EEEB4E80C622DA2EB296A20BC6007BD18A0B7D",
    ),
    "models/object_detection/yolox_tiny.onnx": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        "427CC366D34E27FF7A03E2899B5E3671425C262EA2291F88BB942BC1CC70B0F7",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def ensure_models() -> None:
    for rel_path, (url, expected_sha256) in MODELS.items():
        target = PROJECT_ROOT / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha256_file(target) == expected_sha256:
            print(f"[OK] {rel_path} verified ({expected_sha256[:12]}...)")
            continue
        print(f"[DOWNLOADING] {rel_path} from {url} ...")
        tmp_path = target.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=30) as resp, tmp_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        actual_sha256 = sha256_file(tmp_path)
        if actual_sha256 != expected_sha256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {rel_path}: expected {expected_sha256}, got {actual_sha256}"
            )
        tmp_path.replace(target)
        print(f"[VERIFIED] {rel_path} ({actual_sha256[:12]}...)")


if __name__ == "__main__":
    ensure_models()
