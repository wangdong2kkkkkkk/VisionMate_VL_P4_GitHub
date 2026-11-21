"""One-time controlled downloader and SHA-256 verifier for open-source ONNX models.

Runtime inference in VL-P1 uses ONLY local files under `models/` and never touches the network.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS_SPEC = [
    {
        "model_id": "mobilenetv3_small_100",
        "repo_id": "onnxmodelzoo/mobilenetv3_small_100_Opset17",
        "filename": "mobilenetv3_small_100_Opset17.onnx",
        "local_name": "mobilenetv3_small_100_Opset17.onnx",
        "license": "Apache-2.0",
        "source_url": "https://huggingface.co/onnxmodelzoo/mobilenetv3_small_100_Opset17",
        "architecture": "MobileNetV3-Small (timm / ONNX Model Zoo Opset 17)",
    },
    {
        "model_id": "efficientnet_b0",
        "repo_id": "onnxmodelzoo/efficientnet_b0_Opset17",
        "filename": "efficientnet_b0_Opset17.onnx",
        "local_name": "efficientnet_b0_Opset17.onnx",
        "license": "Apache-2.0",
        "source_url": "https://huggingface.co/onnxmodelzoo/efficientnet_b0_Opset17",
        "architecture": "EfficientNet-B0 (timm / ONNX Model Zoo Opset 17)",
    },
    {
        "model_id": "dinov2_small_int8",
        "repo_id": "Xenova/dinov2-small",
        "filename": "onnx/model_quantized.onnx",
        "local_name": "dinov2_small_quantized.onnx",
        "license": "Apache-2.0",
        "source_url": "https://huggingface.co/Xenova/dinov2-small",
        "architecture": "DINOv2 ViT-S/14 (INT8 ONNX quantized, 384-d CLS + patch tokens)",
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all_models(base_dir: Path) -> dict:
    models_dir = base_dir / "models"
    manifests_dir = base_dir / "manifests"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for spec in MODELS_SPEC:
        target = models_dir / spec["local_name"]
        if not target.exists():
            cached = Path(
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    filename=spec["filename"],
                    local_dir=str(models_dir),
                )
            )
            if cached.resolve() != target.resolve():
                if target.exists():
                    target.unlink()
                cached.replace(target)
        # Clean empty onnx subfolder if hf_hub_download created models/onnx/
        onnx_subdir = models_dir / "onnx"
        if onnx_subdir.exists() and not any(onnx_subdir.iterdir()):
            onnx_subdir.rmdir()

        size_bytes = target.stat().st_size
        records.append({
            **spec,
            "local_path": str(target.relative_to(base_dir)).replace("\\", "/"),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 3),
            "sha256": _sha256(target),
        })

    manifest = {"models": records}
    (manifests_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    res = download_all_models(base)
    for m in res["models"]:
        print(f"{m['model_id']}: {m['size_mb']} MB | {m['license']} | sha256={m['sha256'][:12]}")
