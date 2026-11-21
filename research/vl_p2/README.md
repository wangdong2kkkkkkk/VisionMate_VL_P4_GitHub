# VisionMate Glasses — VL-P2: White Cane Localization & Segmentation Research Pilot

## Overview
Independent, offline, CPU-first Windows research pilot solving **white mobility cane (`object_01`) localization, background separation/segmentation, and confuser-resistant recognition** on full uncropped scenes without relying on repetitive threshold-only calibration loops (`VL-P1b`/`VL-P1c`) or online APIs.

- **Worktree:** `VisionMate_VL_P2_AG`
- **Branch:** `research/vl-p2-white-cane-segmentation`
- **Base Commit:** `9587ad4` (VL-P1c) — all prior commits (`8ae6f85`, `95a75e0`, `6e3039f`, `65c2a27`, `988fdf1`, `dacd245`, `9587ad4`) and production `app/` remain completely untouched.
- **Frozen Models:** `dinov2_small_quantized.onnx` (INT8 CPU, Apache-2.0) and `yolox_nano.onnx` (CPU, Apache-2.0).

## Research Pillars Implemented
1. **Method A — Oriented White Cane Localization (`src/localization_research.py`):**
   - Multi-scale morphological white ridge + LSD/Hough line extraction + YOLOX person-hand vicinity recovery + collinear gap bridging across hand/body occlusions + bilateral cross-sectional step verification (`min(center-left, center-right)`) to suppress single-sided architectural borders, steps, and road edges.
   - Predicts oriented bounding boxes (`obb_cxcywh_angle`), 4 corners, centerline endpoints, and axis envelopes strictly without Ground Truth access.
2. **Method B — Thin-Object Segmentation & Masked DINOv2 Embeddings (`src/segmentation_research.py`, `src/masked_embedding_research.py`):**
   - Rectifies the oriented cane corridor horizontally (`cv2.warpAffine`), tracks smooth centerline `y_c(x)` via dynamic programming, estimates local cross-sectional FWHM shaft boundaries, and generates crisp binary + soft Gaussian-tapered masks (`< 2 ms/proposal`).
   - Systematically benchmarks **6 DINOv2 input representations** (`rep1_axis_crop`, `rep2_rotated_crop`, `rep3_masked_crop`, `rep4_rotated_masked_crop`, `rep5_adaptive_pad_crop`, `rep6_minimal_corridor_crop` longitudinal 3-strip packing) with **matched enrollment representation alignment**, keeping Oracle-localized vs E2E-predicted-localized metrics strictly separated.
3. **Method C — White Cane vs Slender Confusers (`src/cane_verifier.py`, `src/combined_pipeline_v2.py`):**
   - Evaluates DINOv2 representation similarity & slender-confuser margin alongside physical/chromatic shaft features (`white_purity_core`, `wood_brown_ratio`, `red_tip_evidence`, `metallic_dark_ratio`, `width_uniformity`, `canopy_flare_penalty`, `hand_person_anchor`), calibrated strictly out-of-fold (`5-fold` group-aware CV on `dev_cal_v2`) into a 3-state selective classifier (`KNOWN` / `AMBIGUOUS` / `UNKNOWN`).

## Directory Structure
```
research/vl_p2/
├── src/
│   ├── common_utils.py              # Unicode I/O, SHA-256, aHash/dHash, rotated OBB IoU, RLE masks, stats
│   ├── reproduce_v1c_baseline.py    # Step 1: exact reproduction of C_v1c_cal_only on historical N=300 holdout
│   ├── collect_vl_p2_dataset.py     # Wikimedia Commons open-license collector (zero-leakage guard)
│   ├── collect_strict_canes.py      # Metadata-screened white-cane category/search scraper
│   ├── collect_diverse_canes.py     # Author/event-capped multilingual cane + white-stick collector
│   ├── annotation_protocol.py       # Multi-geometry GT synthesis & contact-sheet verification
│   ├── build_vl_p2_manifest.py      # Builds dataset_manifest.json + annotation_manifest.json
│   ├── localization_research.py     # Method A: OrientedCaneLocalizer & localization metrics
│   ├── segmentation_research.py     # Method B1: CaneShaftSegmenter & Mask IoU/Dice/Boundary F1
│   ├── masked_embedding_research.py # Method B2: 6 representations + DINOv2-Small INT8 wrapper
│   ├── cane_verifier.py             # Method C: 12D OOF cane-vs-confuser verifier & 3-state logic
│   ├── combined_pipeline_v2.py      # Experimental matrix A0, A1, B1, B2, C1, C2
│   └── benchmark_runner.py          # Strict Phase A (dev freeze) & Phase B (blind holdout) runner
├── manifests/                       # Provenance manifests, GT annotations, frozen_config_v2.json
├── reports/                         # Protocols, localization/segmentation/confusion/perf reports
└── tests/
    └── test_vl_p2_suite.py          # 13-group automated pytest verification suite
```

## Quick Reproduction Commands
```powershell
python -m research.vl_p2.src.reproduce_v1c_baseline
python -m research.vl_p2.src.build_vl_p2_manifest
python -m research.vl_p2.src.benchmark_runner --phase a
python -m research.vl_p2.src.benchmark_runner --phase b
python -m pytest research/vl_p2/tests/test_vl_p2_suite.py -v
```
