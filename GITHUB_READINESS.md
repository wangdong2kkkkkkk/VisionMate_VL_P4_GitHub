# GitHub publishing notes

This copy was prepared from `VisionMate_VL_P4_AG`. The original project directory was not changed.

## Included model files

| File | Size | SHA-256 |
|---|---:|---|
| `models/object_detection/yolox_nano.onnx` | 3,659,407 bytes (3.66 MB) | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |
| `models/object_detection/yolox_tiny.onnx` | 20,219,662 bytes (20.22 MB) | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` |
| `models/paddleocr_latin/latin_dict.txt` | 2,136 bytes (0.00 MB) | `7629ffd77277fd4093b556769752ee9f7c881754a8544c79a7c438cd9a93d1e6` |
| `models/tessdata_fast/vie.traineddata` | 531,275 bytes (0.53 MB) | `79df64caf7bcfb2a27df5042ecb6121e196eada34da774956995747636d5bfa1` |
| `research/vl_p3/models/cane_verifier_v3.joblib` | 535,850 bytes (0.54 MB) | `520bb7d1633f2d2acd68ffa4c51aaf049652ca4fe1ac748395cf3327c7651253` |
| `research/vl_p3/models/keypoint_detector_v3.joblib` | 434,420 bytes (0.43 MB) | `f19d2ad6d674c3235a6103ec34978c478b39c2678d6ff914bd2dee3846fe4834` |
| `research/vl_p3/models/shaft_segmenter_b1_standalone.joblib` | 149,022 bytes (0.15 MB) | `31a849acda618d5d5152b473bb1f2da8cde552d29761a0f27b78f8bd205e3e45` |
| `research/vl_p3/models/shaft_segmenter_b2_kp_guided.joblib` | 148,884 bytes (0.15 MB) | `55b5051e041dea304ba902abfa9b2165dd47ac5bf75bb49af02c4b39b0304396` |
| `research/vl_p4/models/a1_cnn_keypoint.pt` | 4,325,859 bytes (4.33 MB) | `fa792d1a3d3bca81a72b99a548264b06b06072c7cc3a0e447bbd6a2644b8b3f4` |
| `research/vl_p4/models/a1_cnn_keypoint_fp32.onnx` | 4,196,162 bytes (4.20 MB) | `13ba4c4cf34605253268cc71bd8c83aa45366bfa3be7901493caa6e18d3b88fe` |
| `research/vl_p4/models/a1_cnn_keypoint_int8.onnx` | 1,227,908 bytes (1.23 MB) | `615f4ef1d6b6df81a6564fa7f0aeaf35c281231c675e82fd85d2d2c95500a4d8` |
| `research/vl_p4/models/b1_cnn_segmentation.pt` | 4,325,859 bytes (4.33 MB) | `503a11389ea3614c1b843edbd7fafffe6f9a54cd6573c31f4b9705c22b49ff6a` |
| `research/vl_p4/models/b1_cnn_segmentation_fp32.onnx` | 4,196,162 bytes (4.20 MB) | `8cdd91c3c55f1f9abd8b6fa4e15d79b50761b8796a5de349c99551c2c028dca4` |
| `research/vl_p4/models/b1_cnn_segmentation_int8.onnx` | 1,227,906 bytes (1.23 MB) | `0857fb578cd99f004fd905b7fc1cdad54f09c934695298679749b3c90d8e11ab` |
| `research/vl_p4/models/b2_cnn_multitask.pt` | 4,325,859 bytes (4.33 MB) | `965bd8ce297272e77d2e09fba7e850bbcb452a0f378d72b79bffca9c35db76b1` |
| `research/vl_p4/models/b2_cnn_multitask_fp32.onnx` | 4,196,162 bytes (4.20 MB) | `6a5627bc0a4d33dff4f259a2314232f5ac2f222e5e5ea5b34c37146444147270` |
| `research/vl_p4/models/b2_cnn_multitask_int8.onnx` | 1,227,904 bytes (1.23 MB) | `0f187f3b8ca6de2352c4f700e5008b0bbf7e82c5136771593a44d383949706b7` |
| `research/vl_p4/models/soft_verifier_v4.joblib` | 26,404 bytes (0.03 MB) | `f9be750c258d68c6daaf87e57274fed682d50c939d50002cd426a95f5e078918` |

## License review before public release

- The OCR and YOLOX source/license information is documented in `docs/MODEL_LICENSES.md`; it identifies the Vietnamese Tesseract model and YOLOX Nano/Tiny models as Apache-2.0. The PaddleOCR dictionary is also listed there.
- VL-P3 and VL-P4 architecture, provenance, and hashes are documented in their `MODEL_PROVENANCE.md` files. The source documentation does **not** declare a redistribution license for the custom trained `.onnx`, `.pt`, or `.joblib` outputs. Confirm the intended terms before publishing those weights.
- This repository has no top-level `LICENSE` file. Choose a code license before presenting the code as open source.
- Data collection protocols describe Wikimedia Commons/Openverse sources and per-item licenses. The image datasets and cache/contact sheets were not copied. Keep attribution records for any data you later add.

## Data and privacy exclusions

This copy excludes Git metadata, virtual environments, `.env`/config secrets, caches, `.pyc` files, logs/temp files, the VL-P4 image cache, and regression/object-detection fixture photos. Local absolute paths and local file links in inherited text/JSON/CSV metadata were replaced with non-local placeholders. Only generated synthetic benchmark assets remain under tests. Training and full dataset benchmark reproduction require separately sourced and licensed data.
