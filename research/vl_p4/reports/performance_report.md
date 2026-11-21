# Performance & Resource Report — VisionMate VL-P4 (Uncached CPU E2E)

Toàn bộ thời gian dưới đây được đo **từ ảnh gốc trên đĩa (decode + preprocess + ONNX inference + decode/postprocess)**, không dùng feature cache.

## 1. Độ trễ E2E thực tế trên Windows CPU (i5-1135G7)

| Tập kiểm thử | Cấu hình | Mean (ms) | P50 (ms) | P95 (ms) | Load (ms) | Size (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `final_holdout_v4` (N=219) | `A0_VL_P3_A1` *(baseline)* | 652.61 | 547.93 | 1274.76 | — | — |
| `final_holdout_v4` (N=219) | `A1_CNN_Keypoint` (FP32 ONNX) | 45.35 | 39.36 | 80.68 | 147.35 | 4.002 |
| `final_holdout_v4` (N=219) | `C1_CNN_WithVerifier` | 60.61 | 51.03 | 116.24 | 80.71 | 4.002 |
| `final_holdout_v4` (N=219) | **`C2_FinalCandidate` (`B2` FP32 ONNX)** | **39.14** | **36.25** | **59.60** | 84.95 | 4.002 |
| `final_holdout_v3` (N=245) | `A0_VL_P3_A1` *(baseline)* | 399.69 | 375.63 | 576.37 | — | — |
| `final_holdout_v3` (N=245) | **`C2_FinalCandidate` (`B2` FP32 ONNX)** | **24.76** | **24.77** | **30.78** | 121.36 | 4.002 |

## 2. Phân rã độ trễ từng bước của `C2_FinalCandidate`

- Trên `final_holdout_v4` (`mean = 39.14 ms`): `img_decode = 8.94 ms`, `preprocess = 8.94 ms`, `cnn_onnx = 9.84 ms`, `post_verify = 11.32 ms`.
- Trên `final_holdout_v3` (`mean = 24.76 ms`): `img_decode = 6.20 ms`, `preprocess = 5.90 ms`, `cnn_onnx = 6.23 ms`, `post_verify = 6.38 ms`.
- **Tăng tốc so với VL-P3 `A1`:** **15.1×** nhanh hơn trên cả hai tập (`547.93 -> 36.25 ms` và `375.63 -> 24.77 ms`), vượt xa mục tiêu `P95 <= 350–600 ms` (`59.60 ms` và `30.78 ms`).

## 3. Bộ nhớ & CPU (`system_resources`)

- **RSS Start:** `145.46 MB`
- **RSS Peak:** **`150.22 MB`** *(giảm ~3.1× so với `463.32 MB` của VL-P3, rất nhẹ cho máy RAM 8 GB)*
- **Private Bytes:** `930.23 MB` (do nạp cả pipeline VL-P3 `A0` trong cùng tiến trình để đối chứng)
- **CPU Utilization:** `292.1%`
