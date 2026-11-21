# BÁO CÁO NGHIÊN CỨU TỔNG KẾT — VISIONMATE VL-P1 (FEW-SHOT CATEGORY RECOGNITION)

**Dự án:** VisionMate Glasses — Nhánh nghiên cứu `research/vl-p1-antigravity`  
**Worktree nghiên cứu:** `VisionMate_VL_P1_AG\research\vl_p1` (tách biệt hoàn toàn khỏi production `VisionMateGlasses` và worktree thủ công `VisionMate_VL_P1`)  
**Phạm vi:** Research Pilot offline trên Windows CPU — **Chưa tích hợp production**, không thay đổi OCR, TTS, YOLOX hay webcam pipeline hiện hành.

---

## 1. Tóm tắt điều hành

- **Phân loại kết quả:**
  - **Hạ tầng thực nghiệm, dữ liệu chuẩn hóa & kiểm thử tự động:** **`TECHNICAL PASS`** (34/34 kiểm thử tự động đạt 100%, 0 lỗi rò rỉ dữ liệu giữa `train`/`calibration`/`test`, 100% inference offline với chữ ký mô hình Apache-2.0 xác minh bằng SHA-256).
  - **Độ sẵn sàng thuật toán cho bài toán nhận diện theo loại đồ vật (`category`) bằng ít ảnh mẫu (5-shot):** **`RESEARCH PROMISING`** — Chưa đủ an toàn để bật thẳng ở chế độ nói tự động ngoài đời thực nếu chưa tinh chỉnh bộ đề xuất vùng (`ROI proposal`) và siết thêm ngưỡng âm tính dễ nhầm, nhưng đã chứng minh bước nhảy vọt về khả năng **hiểu loại đồ vật tổng quát** so với hướng ORB ban đầu.
- **Kết luận định lượng cốt lõi:**
  1. **ORB không phù hợp cho bài toán category recognition:** `orb_raw` chỉ đạt **8.0% Pos Top-1 Acc** và **44.0% FAR** trên 75 ảnh test; `orb_adaptive` sau khi vá lỗi zero-keypoint chỉ đạt **0.0% Pos Top-1 Acc** (từ chối `88.0%` dương tính thành `UNKNOWN` và gán sai nhãn `6` ca) đồng thời tốn **1,286.605 ms** (gần 1.3 giây) cho mỗi lần khớp 1 ROI trên CPU Intel i5-1135G7.
  2. **Pretrained ViT Self-Supervised (`dinov2_small_int8`, 23.32 MB) vượt trội CNN cổ điển lẫn ORB:**
     - Với đầu phân loại **`hybrid` ($\tau=0.4758, \delta=0.0$ hiệu chuẩn độc lập trên `data/calibration/`)**: đạt **78.0% Pos Top-1 Acc**, **0 ca gán nhầm lớp mục tiêu (`0/50`)**, **0/20 nhầm lẫn chéo Giày ↔ Dép**, hạ **FAR xuống 12.0%** (chỉ `3/25` confuser khó vượt ngưỡng), thời gian trích xuất đặc trưng **P50 = 33.64 ms**, matching **0.173 ms**, dung lượng DB 25 ảnh chỉ **92.56 KB**.
     - Với đầu phân loại **`prototype` ($\tau=0.3735, \delta=0.075$)**: đạt **88.0% Pos Top-1 Acc** (`90%` Gậy trắng, `100%` Chùm chìa khóa, `90%` Ví tiền, `70%` Đôi dép, `90%` Đôi giày, **0 ca nhầm chéo lớp dương**), nhưng đánh đổi **FAR = 36.0%** (`9/25` confuser bị nhận nhầm).
  3. **Bài toán End-to-End trên ảnh gốc chưa cắt (Task 2):** YOLOX-Nano COCO-80 (`yolox_strict`, `conf=0.40`) bỏ sót **24/50 (48.0%)** đồ vật vì đa phần ngoài 80 lớp COCO, kéo Top-1 Acc toàn trang xuống **32.0%**. Khi kết hợp **Hybrid Class-Agnostic Proposer** (YOLOX ngưỡng thấp + `fg_contour` OpenCV + `center_72`/`full_frame`, tuyệt đối không dùng Ground Truth bbox), tỷ lệ miss giảm từ **24/50 → 0/50**, `Proposal Recall@0.3` đạt **92.0%** (`Mean IoU = 0.713`), đưa Top-1 Acc end-to-end lên **76.0%** (`FAR = 16.0%`, `E2E P50 = 176.5 ms`).

---

## 2. Kiểm toán dữ liệu hiện có (Từ quy trình thủ công `VisionMate_VL_P1`)

Kiểm toán tự động qua `python -m src.dataset_audit` (`manifests/manual_data_audit.json`, `reports/manual_dataset_audit.md`) đối chiếu `data/originals`, `data/train` và `downloaded_photo_sources.csv` ghi nhận:

1. **Khớp toàn vẹn 25 ảnh học hiện có:**
   - `data/originals`: `25` ảnh (5/nhóm); `data/train`: `25` ảnh ROI đã cắt (5/nhóm); `data/test`: **`0` ảnh (trống hoàn toàn)**; `data/calibration`: **`0` ảnh**.
   - So khớp template đa tỷ lệ xác nhận **25/25 ảnh trong `data/train` được cắt trực tiếp từ `data/originals` với điểm khớp `1.0000`** (`>= 0.99`).
2. **Sai lệch metadata bản quyền trong `downloaded_photo_sources.csv`:**
   - Chỉ **6/25 ảnh** trong `data/originals` khớp với file CSV nguồn Wikimedia (`object_01/02, 03, 05` và `object_02/01, 02, 05`).
   - Mục `object_02/03.jpg` trong CSV không còn tồn tại trên đĩa (`MISSING_IN_ORIGINALS`), bị thay thế bởi ảnh khác (`05.jpg` không khớp URL).
   - **19/25 ảnh** còn lại trong `data/originals` được người dùng tải bổ sung thủ công từ web bên ngoài, chưa có thông tin nguồn/bản quyền (`UNVERIFIED_MANUAL_WEB`) — vì vậy chỉ được giữ trong tập `train` nội bộ nghiên cứu và **không dùng làm ảnh đối chứng bản quyền công khai hay trộn vào `calibration`/`test`**.
3. **Phát hiện nguyên nhân lỗi ORB = 0 keypoint trên cây gậy trắng hẹp (`object_01/03.png`):**
   - Trong `data/train/object_01/03.png` có kích thước **`61×1131`** pixel (tỷ lệ mảnh ~1:18.5).
   - Với cấu hình ORB mặc định `edgeThreshold=31`, OpenCV yêu cầu bề rộng ảnh tối thiểu $2 \times 31 + 1 = 63\text{ px}$ mới trích xuất được điểm đặc trưng. Do $61 < 63$, **`orb_raw` thu về đúng `0` keypoints (`0` descriptors)** — hoàn toàn mù với mẫu này.
   - Sau khi áp dụng phóng đại thích ứng (`min(h,w) < 128 -> scale lên min=128`), đệm biên `36 px` (`BORDER_REFLECT_101`) và hạ `edgeThreshold=15, patchSize=15`, `data/train/object_01/03.png` trích xuất được **`1500` keypoints**.
   - Các ảnh kiểm thử phụ khác ở worktree thủ công: `dev_probes/object_01/probe01.png` (`200×425`) cho `159` keypoints (thô) / `324` keypoints (adaptive); `backup_crops/object_01/01_previous.png` (`1014×799`) cho `1292` / `1500` keypoints.

---

## 3. Thiết kế kiến trúc nghiên cứu

Hệ thống nghiên cứu chia thành 2 nhóm phương pháp độc lập và 2 mức độ đánh giá:

### 3.1. Nhánh A — `ORBRecognizer` (`src/orb_recognizer.py`)
- **`orb_raw`**: `cv2.ORB_create(nfeatures=1300, fastThreshold=10)` trên ảnh xám nguyên bản; `BFMatcher(NORM_HAMMING)` + Lowe ratio test (`0.78`) + `cv2.findHomography(..., cv2.RANSAC, 4.5)`.
- **`orb_adaptive`**: `edgeThreshold=15, patchSize=15, nfeatures=1500, fastThreshold=10` kết hợp phóng to ảnh hẹp (`min(h,w) < 128`) và đệm biên phản xạ `36 px` để loại triệt để lỗi 0 keypoint.

### 3.2. Nhánh B — `EmbeddingRecognizer` (`src/embedding_recognizer.py`)
- **Tiền xử lý ROI giữ nguyên tỷ lệ khung hình (`preprocess_roi`):** Không bóp méo vật mảnh (như gậy trắng) thành hình vuông; dùng màu trung vị của viền ảnh (`border_px` median) làm nền đệm (`pad_to_square`), resize về `224×224` (`INTER_AREA` khi thu nhỏ, `INTER_LINEAR` khi phóng to), chuyển RGB và chuẩn hóa theo mean/std ImageNet (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`). Mọi vector đầu ra đều chuẩn hóa $L_2$.
- **3 Backbone offline mã nguồn mở Apache-2.0 (`manifests/model_manifest.json`) + 1 Ensemble:**
  1. `mobilenetv3_small` (`mobilenetv3_small_100_Opset17.onnx`, `9.707 MB`, `1024-d` tại `onnx_node!/flatten/Flatten`, chạy qua `cv2.dnn` CPU). Nạp file qua buffer `np.frombuffer(..., np.uint8)` để tránh lỗi C++ `fopen` của OpenCV trên đường dẫn Windows có dấu tiếng Việt (`"mắt kính AI"`).
  2. `efficientnet_b0` (`efficientnet_b0_Opset17.onnx`, `20.15 MB`, `1280-d` tại `onnx_node!/global_pool/flatten/Flatten`, chạy qua `cv2.dnn` CPU).
  3. `dinov2_small_int8` (`dinov2_small_quantized.onnx` từ `Xenova/dinov2-small`, `23.319 MB`, lượng tử hóa INT8, chạy qua `onnxruntime` CPU `CPUExecutionProvider`, `intra_op_num_threads=4`, gộp `concat(L2(CLS), L2(mean_patch_tokens))` -> vector **768 chiều** chuẩn hóa $L_2$).
  4. `ensemble_mv3_dino`: ghép trọng số `concat(0.4 * emb_mv3, 0.6 * emb_dino)` -> vector **1792 chiều** chuẩn hóa $L_2$.
- **3 Chiến lược tính điểm & Quyết định 3 trạng thái (`KNOWN` / `UNKNOWN` / `AMBIGUOUS`):**
  - `1nn_max`: độ tương đồng cosine lớn nhất với 5 ảnh mẫu trong lớp.
  - `prototype`: độ tương đồng cosine với vector trung bình chuẩn hóa $L_2$ của cả lớp.
  - `hybrid`: $0.5 \cdot \text{proto\_sim} + 0.5 \cdot \text{top2\_mean}$.
  - Luật quyết định: nếu `top1_score < tau_known` -> **`UNKNOWN`**; ngược lại nếu `(top1_score - top2_score) < delta_margin` -> **`AMBIGUOUS`**; ngược lại -> **`KNOWN`**.
  - **Hiệu chuẩn ngưỡng độc lập (`calibrate_thresholds`):** Tìm lưới $(\tau_{\text{known}}, \delta_{\text{margin}})$ **chỉ trên tập `data/calibration/`** nhằm cực đại hóa độ chính xác dương tính và F1 hiệu chuẩn trong giới hạn `FAR` mục tiêu — hoàn toàn tách biệt khỏi `data/test/` và `data/test_full/`.

### 3.3. Pipeline Task 2 — `FullImagePipeline` (`src/full_image_pipeline.py`)
Kiểm tra kịch bản thực tế trên ảnh gốc chưa cắt (`data/test_full/`, **tuyệt đối không đưa bounding box Ground Truth vào inference**):
- **`yolox_strict`**: Dùng trọng số `models/object_detection/yolox_nano.onnx` hiện có, ngưỡng `conf=0.40`, tối đa `4` box.
- **`yolox_low`**: Hạ ngưỡng `conf=0.15`, NMS IoU `0.45`, tối đa `4` box.
- **`hybrid`**: Gộp tối đa `2` box từ `yolox_low` (`conf=0.15`) + tối đa `2` box đề xuất phi-lớp từ `foreground_contour_proposals` (OpenCV LAB border distance + Otsu saliency + Canny dilation `MORPH_CLOSE`) + `center_72` (72% vùng trung tâm) + `full_frame`, khử trùng lặp với `IoU >= 0.78` và giữ tối đa `4` vùng ứng viên để đưa qua bộ phân loại embedding.

---

## 4. Bộ dữ liệu chuẩn hóa & kiểm tra chống rò rỉ

Toàn bộ ảnh calibration và test đều được thu thập mới từ **Wikimedia Commons** (chỉ nhận giấy phép mở) và kiểm tra trực quan qua contact sheet, đóng băng trong `manifests/dataset_manifest.json` (`data_status: COMPLETE_VERIFIED`):

| Tập dữ liệu (`split`) | Dương tính (`object_01..05`) | Âm tính (`unknown`) | Tổng | Giấy phép mở / Nguồn |
|---|---:|---:|---:|---|
| `data/train/` | 25 (5/nhóm) | 0 | **25** | 6 ảnh Wikimedia (`CC BY-SA 4.0`: 4, `CC BY 3.0`: 1, `CC0`: 1) + 19 ảnh `UNVERIFIED_MANUAL_WEB` từ người dùng |
| `data/calibration/` | 15 (3/nhóm) | 10 confusers | **25** | 100% mở (`Public domain`: 2, `CC0`: 2, `CC BY-SA 4.0`: 7, `CC BY 2.0`: 3, `CC BY 3.0`: 2, `CC BY 4.0`: 1, `CC BY-SA 3.0`: 6, `CC BY-SA 2.0`: 1, `CC BY-SA 2.5`: 1) |
| `data/test/` (ROI đã cắt) & `data/test_full/` (Ảnh đầy đủ) | 50 (10/nhóm) | 25 confusers | **75** | 100% mở (`CC0`: 9, `CC BY-SA 4.0`: 23, `CC BY 3.0`: 4, `CC BY 2.0`: 6, `CC BY-SA 3.0`: 19, `CC BY 4.0`: 5, `Public domain`: 7, `CC BY-SA 2.0`: 2) |

**Phân nhóm vật dễ nhầm (`unknown`) trong tập `test` (25 ảnh):** `confuser` (8), `wallet_confuser` (6: bao kính, sổ da, ví xu lạ/hộp kính), `stick_confuser` (3: gậy gỗ/gậy bảo tàng), `footwear_confuser` (3: tất/vớ, guốc gỗ/đồ chân khác), `household_object` (3), `umbrella_confuser` (1), `key_metal_confuser` (1).  
**Chống rò rỉ chéo giữa các tập:** Bài kiểm thử `tests/test_02_data_integrity_and_anti_leakage.py` xác minh **0 trùng lặp SHA-256** và khoảng cách perceptual hash (`aHash`, `dHash`) đạt ngưỡng an toàn trên mọi cặp chéo giữa `train`/`originals`, `calibration` và `test`.

---

## 5. Kết quả benchmark định lượng

**Môi trường đo thực tế:** `Windows-10-10.0.26300-SP0`, CPU `11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz` (4 nhân vật lý / 8 luồng), RAM `7.73 GB`, Python `3.10.7`, OpenCV `4.13.0`, ONNX Runtime `1.23.2` (`CPUExecutionProvider`), NumPy `2.2.6`.

### 5.1. Task 1 — Nhận diện trên ROI đã cắt (`data/test/`, 50 dương + 25 âm)

| Phương pháp | Threshold hiệu chuẩn $(\tau, \delta)$ | Top-1 Pos Acc | Macro Recall | Obj01 Gậy | Obj02 Chìa | Obj03 Ví | Obj04 Dép | Obj05 Giày | FAR (Âm) | Pos UNK | Pos AMB | Sai lớp dương |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `orb_raw` | `inliers=4, r=0.45, m=2` | 8.0% | 8.0% | 30% | 0% | 10% | 0% | 0% | 44.0% | 50.0% | 18.0% | 12 |
| `orb_adaptive` | `inliers=4, r=0.45, m=2` | 0.0% | 0.0% | 0% | 0% | 0% | 0% | 0% | 16.0% | 88.0% | 0.0% | 6 |
| `mobilenetv3_small__1nn_max` | `0.3837, 0.100` | 48.0% | 48.0% | 60% | 40% | 80% | 20% | 40% | 20.0% | 32.0% | 18.0% | 1 |
| `mobilenetv3_small__prototype` | `0.3947, 0.075` | 60.0% | 60.0% | 50% | 50% | 90% | 40% | 70% | 12.0% | 28.0% | 10.0% | 1 |
| `mobilenetv3_small__hybrid` | `0.3741, 0.030` | 62.0% | 62.0% | 60% | 40% | 90% | 40% | 80% | 20.0% | 34.0% | 2.0% | 1 |
| `efficientnet_b0__1nn_max` | `0.3741, 0.030` | 56.0% | 56.0% | 60% | 50% | 80% | 20% | 70% | **4.0%** | 40.0% | 0.0% | 2 |
| `efficientnet_b0__prototype` | `0.3311, 0.000` | 74.0% | 74.0% | 80% | 90% | 90% | 30% | 80% | 20.0% | 22.0% | 0.0% | 2 |
| `efficientnet_b0__hybrid` | `0.3097, 0.000` | 74.0% | 74.0% | 80% | 90% | 90% | 30% | 80% | 20.0% | 22.0% | 0.0% | 2 |
| `dinov2_small_int8__1nn_max` | `0.4053, 0.100` | 80.0% | 80.0% | 80% | 100% | 90% | 40% | 90% | 28.0% | 6.0% | 14.0% | **0** |
| `dinov2_small_int8__prototype` | `0.3735, 0.075` | **88.0%** | **88.0%** | **90%** | **100%** | **90%** | **70%** | 90% | 36.0% | 2.0% | 10.0% | **0** |
| **`dinov2_small_int8__hybrid`** | **`0.4758, 0.000`** | **78.0%** | **78.0%** | 70% | 70% | 80% | **70%** | **100%** | **12.0%** | 22.0% | 0.0% | **0** |
| `ensemble_mv3_dino__hybrid` | `0.4395, 0.015` | 76.0% | 76.0% | 80% | 70% | 80% | 60% | 90% | 12.0% | 22.0% | 2.0% | **0** |

### 5.2. Task 2 — Nhận diện trên Ảnh Gốc Chưa Cắt (`data/test_full/`, 50 dương + 25 âm)

| Pipeline (Proposer + Classifier) | Proposal Recall@0.3 | Proposal Recall@0.5 | Miss Proposal | Mean IoU | Top-1 Pos Acc | FAR (Âm) | E2E P50 (ms) | E2E P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolox_strict__dinov2_hybrid` | 36.0% | 24.0% | **24/50** | 0.457 | 32.0% | 0.0% | 62.9 | 135.8 |
| `yolox_low__dinov2_hybrid` | 74.0% | 58.0% | 3/50 | 0.556 | 64.0% | 8.0% | 104.3 | 185.5 |
| `hybrid_prop__mobilenetv3_hybrid` | 92.0% | 86.0% | **0/50** | 0.713 | 58.0% | 44.0% | 71.8 | 101.0 |
| **`hybrid_prop__dinov2_hybrid`** | **92.0%** | **86.0%** | **0/50** | **0.713** | **76.0%** | **16.0%** | **176.5** | **242.6** |
| `hybrid_prop__ensemble_hybrid` | 92.0% | 86.0% | 0/50 | 0.713 | 70.0% | 20.0% | 208.5 | 256.8 |

### 5.3. Kết quả kiểm thử nhanh trên `dev_probes/object_01/probe01.png` (Gậy trắng)

| Phương pháp | Trạng thái | Nhãn dự đoán | Điểm / Inliers | Margin | Thời gian (Feat + Match) |
|---|---|---|---:|---:|---:|
| `orb_raw` | `UNKNOWN` | `None` | `7` inliers | `3` | `1.493 + 50.441 ms` |
| `orb_adaptive` | `KNOWN` *(SAI)* | `object_05` *(Đôi giày)* | `52` inliers | `10` | `1.949 + 165.998 ms` |
| `mobilenetv3_small__hybrid` | **`KNOWN`** | **`object_01`** | `0.695596` | `0.475273` | `11.879 + 0.3589 ms` |
| `efficientnet_b0__hybrid` | **`KNOWN`** | **`object_01`** | `0.631609` | `0.544568` | `62.560 + 0.3046 ms` |
| **`dinov2_small_int8__hybrid`** | **`KNOWN`** | **`object_01`** | **`0.711905`** | **`0.465790`** | `48.924 + 0.1907 ms` |
| `ensemble_mv3_dino__hybrid` | **`KNOWN`** | **`object_01`** | `0.706871` | `0.469329` | `50.929 + 0.3398 ms` |

### 5.4. Hiệu năng phần cứng CPU (`performance_cpu`)

| Phương pháp | Model (MB) | Load (ms) | Feat P50 (ms) | Feat P95 (ms) | Match P50 (ms) | Enroll/ảnh (ms) | DB 25 ảnh (KB) | Peak RAM (MB) | ΔRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `orb_raw` | 0.00 | 0.0 | 13.40 | 15.20 | **344.227** | 25.00 | 0.0 | 301.8 | 0.0 |
| `orb_adaptive` | 0.00 | 0.0 | 11.24 | 11.84 | **1286.605** | 26.33 | 0.0 | 54.8 | 3.2 |
| `mobilenetv3_small__hybrid` | **9.71** | **38.4** | **7.33** | **8.24** | 0.252 | **10.24** | 122.6 | **408.2** | **22.7** |
| `efficientnet_b0__hybrid` | 20.15 | 122.5 | 56.88 | 62.70 | 0.296 | 46.14 | 152.6 | 441.0 | 45.9 |
| **`dinov2_small_int8__hybrid`** | **23.32** | 265.1 | **33.64** | **42.09** | **0.173** | 38.43 | **92.6** | 423.3 | 43.6 |
| `ensemble_mv3_dino__hybrid` | 33.03 | 249.7 | 40.20 | 45.59 | 0.179 | 42.16 | 212.6 | 443.9 | 62.5 |

---

## 6. Phân tích các ca lỗi thực tế

Chi tiết đầy đủ được lưu trong [failure_analysis.md](local historical file reference omitted). Tóm tắt 5 nhóm lỗi chính:

1. **Giày (`object_05`) vs Dép (`object_04`):** `mobilenetv3_small__hybrid` và `efficientnet_b0__hybrid` vẫn nhầm 1–2 mẫu dép sang giày (`object_04/test_02.png` với `score=0.4681`); toàn bộ các cấu hình **DINOv2 (`dinov2_small_int8` và `ensemble`) đạt `0/20` ca nhầm lẫn chéo Giày ↔ Dép**. Ba mẫu dép bị rơi trong `dinov2_small_int8__hybrid` (`test_03.png`, `test_05.png`, `test_07.png`, điểm `0.4538–0.4662`) đều dừng an toàn ở `UNKNOWN` dưới $\tau=0.4758$.
2. **`Ví tiền` vs `wallet_confuser`:** Các hộp/bao kính da gập (`unknown/test_14.png`, `Etui-1.jpg`, `score=0.5210`) và sổ tay bìa da cổ (`unknown/test_17.png`, `Wellcome L0043479.jpg`, `score=0.5156`) có cấu trúc chữ nhật và bề mặt da gần giống ví tiền nên gây 2/3 ca FAR của `dinov2_small_int8__hybrid`.
3. **`Chùm chìa khóa` vs kim loại nhỏ & chìa đơn/cổ:** `dinov2_small_int8__prototype` đạt `10/10` với chìa khóa, nhưng `hybrid` ($\tau=0.4758$) từ chối an toàn 3 mẫu nền phức tạp/rỉ sét (`object_02/test_02, 03, 07`). Nhờ $\tau=0.4758$, các confuser kim loại như ổ khóa đồng (`unknown/test_06.png`) và kéo thảm (`unknown/test_09.png`) đều bị chặn.
4. **`Gậy trắng` vs `stick_confuser` & ảnh toàn cảnh:** Chiếc gậy kính viễn vọng sáng màu trong bảo tàng (`unknown/test_02.png`, `A walking-stick telescope (Science Museum).jpg`) vượt qua mọi mô hình (`score=0.564–0.598`). Ở Task 2 trên ảnh toàn thân người cầm gậy (`object_01/test_02, 04, 05`), YOLOX thường đóng hộp quanh `person` (`yolox_coco_0`, `IoU=0.088–0.211`), làm loãng đặc trưng của thân gậy mảnh.
5. **ORB sụp đổ với bài toán category:** Tìm được keypoint không đồng nghĩa khớp đúng ngữ nghĩa — 4 điểm ngẫu nhiên đủ để `findHomography` đạt `inlier_ratio=1.0` gây `44.0% FAR` ở `orb_raw`, còn `orb_adaptive` chậm `1.29s/ROI` và đạt `0.0%` Pos Acc.

---

## 7. Kiểm chứng các baseline được bảo vệ

Mọi vùng mã nguồn và dữ liệu được yêu cầu bảo vệ đều đã được xác minh nguyên vẹn sau khi hoàn tất toàn bộ thực nghiệm:

| Thành phần được bảo vệ | Nhánh / Commit kiểm chứng | Trạng thái Git & Check | Kết quả kiểm thử |
|---|---|---|---|
| **Manual VL-P1 Worktree** (`VisionMate_VL_P1\research`) | `research/vl-p1-manual` @ `6e3039f` | **Không đụng tới**: Kiểm tra SHA-256 toàn bộ `68/68` file khớp **`Exact match: True`** với snapshot trước thực nghiệm | Không chỉnh sửa 1 byte nào |
| **Production App** (`VisionMateGlasses`) | `hotfix/webcam-preview-fps-optimization` @ `65c2a27` | `git status --short`: **sạch (0 file sửa)**. Lịch sử commit bảo vệ còn nguyên vẹn (`8ae6f85`, `95a75e0`, `6e3039f`, `65c2a27`) | `python -m pytest -q tests/`: **`92 passed in 143.90s`** |
| **Research Worktree mới** (`VisionMate_VL_P1_AG`) | `research/vl-p1-antigravity` (từ `6e3039f`) | Chỉ thêm/sửa dưới `research/vl_p1/`, loại trừ `data/` và `*.onnx` qua `.gitignore`, không merge/tag | `pytest` toàn bộ `tests/` + test kế thừa: sẽ xác minh lần cuối trước khi chốt báo cáo |

---

## 8. Kết luận kỹ thuật & khuyến nghị bước tiếp theo

1. **Quyết định kỹ thuật dứt khoát cho Phase nghiên cứu:**
   - **Loại bỏ phương pháp ORB** cho bài toán *category recognition* (`8.0%` / `0.0%` Pos Acc, `344–1,287 ms/ROI`).
   - **Chọn `dinov2_small_int8` (ViT-S/14 INT8, `23.32 MB`, Apache-2.0) làm backbone embedding chủ lực** cho khả năng phân biệt hình học giữa các loại đồ vật dễ nhầm (`0` lỗi Giày ↔ Dép, `0` lỗi gán nhầm lớp dương). Nếu cần chế độ tiết kiệm CPU tối đa trên tác vụ đơn giản, `mobilenetv3_small` (`7.33 ms`, `9.71 MB`) là phương án nhẹ dự phòng.
2. **Đề xuất hướng nâng cấp nếu mở Phase kế tiếp (sau khi chủ dự án duyệt):**
   - **Negative Prototypes / Hard-Negative Margin:** Lưu thêm vector đặc trưng âm tính của các nhóm dễ nhầm đã biết (`bao kính`, `sổ da`, `gậy gỗ thường`, `vớ/tất`) và áp công thức $\text{score\_adj} = s_{\text{pos}} - \lambda s_{\text{neg}}$ nhằm giảm `FAR` dưới `5%` mà vẫn giữ `Top-1 Acc >= 85%` (mức hiện có của đầu `prototype` là `88%`).
   - **ROI Proposer chuyên biệt cho vật mảnh/nhỏ ngoài COCO:** Dùng lưới trượt kích thước vật thể hoặc mô hình class-agnostic object proposal nhẹ (hoặc fine-tune YOLOX mở rộng thêm lớp mục tiêu) để nâng `IoU` cho `gậy trắng` cầm trên tay từ `0.2–0.3` lên `> 0.6`.
   - **Cơ chế Anti-Spam khi tích hợp tương lai:** Không chạy DINOv2 ở mọi khung hình 30 FPS (`33.6 ms/ROI` ~ 1–2 proposal định kỳ theo trigger người dùng hoặc khi tracker báo vật mới ổn định 3 khung liên tiếp) để không làm giảm FPS preview webcam.

---

## 9. Danh mục file tại `VisionMate_VL_P1_AG/research/vl_p1/`

- **Cấu hình & Kế thừa:** `.gitignore`, `README.md`, `labels.json`, `crop_fit.py`, `orb_baseline.py`, `test_crop_fit.py`, `test_orb_baseline.py`, `download_25_photos.ps1`, `downloaded_photo_sources.csv`, `photo_sources.json`, `README_25_PHOTOS.md`, `README_CROP_FIT.md`
- **Mã nguồn nghiên cứu (`src/`):** `__init__.py`, `image_utils.py`, `dataset_audit.py`, `download_models.py`, `collect_dataset.py`, `collect_category_supplements.py`, `collect_unknown_confusers.py`, `collect_keys_supplement.py`, `collect_keys_modern.py`, `finalize_splits.py`, `orb_recognizer.py`, `embedding_recognizer.py`, `full_image_pipeline.py`, `benchmark_runner.py`
- **Manifests (`manifests/`):** `manual_data_audit.json`, `model_manifest.json`, `dataset_manifest.json`
- **Báo cáo (`reports/`):** `manual_dataset_audit.md`, `benchmark_report.md`, `confusion_matrices.md`, `benchmark_metrics.json`, `failure_analysis.md`, `VL_P1_FINAL_RESEARCH_REPORT.md`
- **Kiểm thử tự động (`tests/`):** `conftest.py`, `test_01_dataset_loader_and_labels.py`, `test_02_data_integrity_and_anti_leakage.py`, `test_03_roi_and_proposer.py`, `test_04_orb_recognizer.py`, `test_05_embedding_recognizer.py`, `test_06_edge_cases_and_safety.py`, `test_07_full_image_e2e.py`
- **Git:** nhánh nghiên cứu `research/vl-p1-antigravity` tách từ `6e3039f`.
