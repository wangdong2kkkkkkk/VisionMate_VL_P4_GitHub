# Model & Dependency Provenance — VL-P2

VL-P2 strictly inherits the frozen offline CPU models approved in OD-1 (`95a75e0`) and VL-P1/VL-P1b/VL-P1c (`9587ad4`), without replacing models or introducing proprietary/online dependencies or automatic model downloads at startup.

## 1. Frozen Neural Models

| Model | File Path | Exact SHA-256 | Size | Runtime | License | Role in VL-P2 |
|---|---|---|---|---|---|---|
| **DINOv2-Small INT8** | `research/vl_p2/models/dinov2_small_quantized.onnx` | `3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917` | ~22.1 MB | ONNX Runtime CPU (`CPUExecutionProvider`) | Apache-2.0 (Meta AI / dynamically INT8 quantized) | Extracts 384D L2-normalized visual embeddings on 224×224 rendered cane representations (`rep1`–`rep6`) and supports multi-class `R5` compatibility (`object_02`–`object_05`) |
| **YOLOX-Nano** | `models/object_detection/yolox_nano.onnx` | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` | 3.66 MB | ONNX Runtime CPU (`CPUExecutionProvider`) | Apache-2.0 (Megvii BaseDetection) | Provides pedestrian (`class_id == 0`) context boxes for hand-to-ground white-cane anchoring and multi-class object proposals (`object_02`–`object_05`) |

## 2. Offline Runtime & Algorithm Stack

- **Classical + Structural Vision Modules (CPU-first, Apache/BSD/MIT ecosystem):**
  - OpenCV (`cv2` 4.13.0, Apache-2.0): Multi-scale morphological tophat, LSD (`createLineSegmentDetector`), Probabilistic Hough transform, affine strip rectification, dynamic-programming boundary tracking, bilateral cross-sectional FWHM analysis, exact `rotatedRectangleIntersection`.
  - NumPy (`2.2.6`, BSD-3) & SciPy (`1.15.3`, BSD-3).
  - scikit-learn (`LogisticRegression` + `StandardScaler`, BSD-3): Out-of-fold 5-fold group-aware CV calibration of the 12D `CaneVsConfuserVerifier` strictly on `dev_cal_v2`.
- **Offline Integrity Check:** Startup verifies both ONNX SHA-256 digests against hardcoded constants in `src/common_utils.py`. No network calls are executed during model initialization, localization, segmentation, embedding, or verification.
