# VisionMate VL-P4

This repository contains the VisionMate Windows reading application and the separate VL-P4 white-cane detection research pilot. **VL-P4 is research code and is not integrated into the production reading application.**

## What is included

- `src/visionmate/`: Windows desktop UI, camera, OCR, speech, and object-detection application.
- `research/vl_p4/`: VL-P4 data preparation, training, inference, evaluation code, protocols, manifests, and reports.
- `research/vl_p1` through `research/vl_p3/`: earlier research code and metadata used to document the P4 baseline.
- `models/` and `research/vl_p4/models/`: runtime and research model artifacts. See [GitHub publishing notes](GITHUB_READINESS.md) for sizes and license status.
- `tests/`: automated tests. Some regression images and downloaded research datasets are intentionally not included. The retained synthetic fixtures can be regenerated with the scripts documented in their source.

## Requirements and setup

- Windows 11 x64 and Python 3.10 or newer.
- The reading application uses local Tesseract OCR and the Windows Vietnamese OneCore voice. See `docs/MODEL_LICENSES.md` for setup and third-party model details.
- The VL-P4 training/research environment additionally uses PyTorch, torchvision, ONNX, scikit-learn, SciPy, requests, and joblib.

Create a virtual environment and install dependencies from the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

For VL-P4 research and training, install the additional dependencies:

```powershell
python -m pip install -r requirements-vl-p4.txt
```

Start the Windows reading application from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -X utf8 -m visionmate.app
```

See [research/vl_p4/README.md](research/vl_p4/README.md) and `research/vl_p4/TRAINING_PROTOCOL.md` for research workflows. Training and full benchmark reproduction require the original datasets, which are not bundled. Dataset manifests retain source/license metadata but have local machine paths removed.

## Privacy and release notes

No `.env`, virtual environment, Python bytecode, runtime cache, local dataset, or regression-photo directory is included. Do not add private images, videos, API keys, passwords, or local dataset copies. Review `GITHUB_READINESS.md` before making the repository public; in particular, the source files do not declare a distribution license for the custom VL-P4 model weights or for this repository's code.
