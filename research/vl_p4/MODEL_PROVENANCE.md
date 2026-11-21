# MODEL PROVENANCE — VisionMate VL-P4

| Thành phần | Tệp | Kiến trúc / Nguồn | Tham số / Dung lượng | SHA-256 |
| --- | --- | --- | ---: | --- |
| Backbone khởi tạo | `mobilenet_v3_small-047dcff4.pth` | PyTorch torchvision `MobileNet_V3_Small_Weights.DEFAULT` (BSD-3-Clause, ImageNet-1K) | 9.83 MB | *(cached torch hub)* |
| `A1_CNN_Keypoint` (FP32) | `models/a1_cnn_keypoint_fp32.onnx` | MobileNetV3-Small + FPN64 (3 KP/CL + Obj) | 1,045,989 / 4.002 MB | `13ba4c4cf34605253268cc71bd8c83aa45366bfa3be7901493caa6e18d3b88fe` |
| `B1_CNN_Segmentation` (FP32) | `models/b1_cnn_segmentation_fp32.onnx` | MobileNetV3-Small + FPN64 (Mask + Obj) | 1,045,989 / 4.002 MB | `8cdd91c3c55f1f9abd8b6fa4e15d79b50761b8796a5de349c99551c2c028dca4` |
| `B2_CNN_MultiTask` (FP32, **Selected C2**) | `models/b2_cnn_multitask_fp32.onnx` | MobileNetV3-Small + FPN64 (KP + CL + Mask + Obj) | 1,045,989 / 4.002 MB | `6a5627bc0a4d33dff4f259a2314232f5ac2f222e5e5ea5b34c37146444147270` |
| `B2_CNN_MultiTask` (INT8, *rejected by parity*) | `models/b2_cnn_multitask_int8.onnx` | ONNX Runtime `quantize_dynamic` QUInt8 | 1.171 MB | `0f187f3b8ca6de2352c4f700e5008b0bbf7e82c5136771593a44d383949706b7` |
| `SoftCaneVerifierV4` (`C1`) | `models/soft_verifier_v4.joblib` | StandardScaler + LogisticRegression + HGB (12 features, 0 person-anchor) | — | `f9be750c258d68c6daaf87e57274fed682d50c939d50002cd426a95f5e078918` |

*Lưu ý phục hồi:* Các tệp `.onnx` và `.pt` không commit lên Git (theo `.gitignore`), có thể tái tạo đầy đủ bằng `python -m research.vl_p4.src.train_vl_p4_models` (seed `42`).
