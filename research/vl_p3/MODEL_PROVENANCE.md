# MODEL PROVENANCE — VisionMate VL-P3

Tất cả mô hình đều chạy **100% offline trên CPU** (Windows 11, ONNX Runtime + scikit-learn + OpenCV), không phụ thuộc PyTorch lúc suy luận.

| Thành phần | Tệp lưu trữ | Nguồn gốc / Giấy phép | SHA-256 (đã xác minh) |
| --- | --- | --- | --- |
| DINOv2-Small INT8 Backbone | `models/dinov2_small_quantized.onnx` | Meta DINOv2 ViT-S/14 (Apache-2.0), lượng tử hóa INT8 ONNX từ VL-P1 | `3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917` |
| YOLOX-Nano Context Detector | `models/yolox_nano.onnx` | Megvii YOLOX-Nano (Apache-2.0), từ baseline `models/object_detection/yolox_nano.onnx` | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |
| Patch Semantic Head | `models/patch_semantic_head.json` | Huấn luyện trên `train_v3` (PCA 12 chiều + 5 đầu phân lớp tuyến tính trên patch tokens 16×16×384) | `f3f39c2008aa66fc893aa4a3b0bef1561913bcd57ea9828aba9cb9462c19539a` |
| Keypoint & Centerline Detector (`A1`) | `models/keypoint_detector_v3.joblib` | Huấn luyện trên `train_v3` (`HistGradientBoosting`, `sparse_refine_k=160`) | `f19d2ad6d674c3235a6103ec34978c478b39c2678d6ff914bd2dee3846fe4834` |
| Shaft Segmenter Standalone (`B1`) | `models/shaft_segmenter_b1_standalone.joblib` | Huấn luyện trên `train_v3` (`HistGradientBoostingClassifier`, `sparse_refine_k=160`) | `31a849acda618d5d5152b473bb1f2da8cde552d29761a0f27b78f8bd205e3e45` |
| Shaft Segmenter KP-Guided (`B2`) | `models/shaft_segmenter_b2_kp_guided.joblib` | Huấn luyện trên `train_v3` (kèm 4 kênh dẫn hướng từ `A1`, `sparse_refine_k=160`) | `55b5051e041dea304ba902abfa9b2165dd47ac5bf75bb49af02c4b39b0304396` |
| De-Biased Cane Verifier (`C2`) | `models/cane_verifier_v3.joblib` | Huấn luyện OOF trên `train_v3` (LogisticRegression + HistGradientBoosting + DINOv2 banks) | `520bb7d1633f2d2acd68ffa4c51aaf049652ca4fe1ac748395cf3327c7651253` |

Ghi chú: Tệp `models/*.onnx` không commit lên Git (được bảo vệ bởi `.gitignore`), chỉ lưu cục bộ và kiểm tra tính toàn vẹn bằng SHA-256 trong `frozen_config.json` và `test_vl_p3_suite.py`.
