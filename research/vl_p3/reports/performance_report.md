# Performance & Resource Report — VisionMate VL-P3 (CPU-First Windows 11)

## 1. Độ trễ suy luận thực tế trên `final_holdout_v3` (N=245, Full End-to-End Uncached)

| Preset | Điều kiện đo | Mean (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `C2_full_vl_p3_pipeline` | **Uncached End-to-End (N=245)** | **440.63** | **402.61** | **652.59** | 269.49 | 2275.35 |
| `A0_v2_lsd_hough_baseline` | Uncached localizer + DINOv2 crop (cached `bundle`) | 171.00 | 169.62 | 220.62 | 57.99 | 383.63 |
| `A1_supervised_kp_centerline` | Cached `SpatialFeatureBundle` (N=245) | 40.95 | 37.84 | 57.32 | 26.88 | 294.51 |
| `B1_supervised_seg_only` | Cached `SpatialFeatureBundle` (N=245) | 47.05 | 42.06 | 64.15 | 24.33 | 270.61 |
| `B2_kp_guided_supervised_seg` | Cached `SpatialFeatureBundle` (N=245) | 101.00 | 95.86 | 149.32 | 49.04 | 338.16 |
| `C1_b2_plus_masked_dinov2` | Cached `SpatialFeatureBundle` (N=245) | 156.83 | 151.61 | 222.41 | 65.86 | 266.50 |

*(Trên tập lịch sử `final_holdout_v2`, N=180: `C2` uncached đạt `mean = 433.69 ms`, `p50 = 429.33 ms`, `p95 = 574.19 ms`.)*

## 2. Phân rã nguyên nhân độ trễ của `C2` và đối chiếu mục tiêu

Đo chi tiết từng khâu trên 25 ảnh `val_cal_v3` trước khi giảm `sparse_refine_k`:
- `extract_spatial_features` (`P50 = 181.0 ms`, `P95 = 260.2 ms`):
  - `YOLOX-Nano`: `P50 = 20.9 ms`
  - `DINOv2-Small INT8` (lần 1, toàn ảnh): `P50 = 33.2 ms`
  - `compute_base_and_ridge_maps` (16 lần `cv2.filter2D` 13×13 trên ảnh `448×448` + các phép biến đổi LAB/HSV/Sobel/morph): chiếm ~`127 ms` P50.
- `kp_detector` (`A1`): ~`38–43 ms` P50.
- `seg_b2` (`B2`): ~`42–45 ms` P50.
- Lần chạy `DINOv2` thứ 2 trên masked shaft crop (`C1`/`C2`): ~`33–48 ms` + trích xuất feature verifier.

**Kết luận trung thực về hiệu năng:** Mục tiêu độ trễ `P50 <= 130 ms` và `P95 <= 220 ms` cho luồng end-to-end `C2` **CHƯA ĐẠT** trên CPU thực tế (`402.61 ms` P50 / `652.59 ms` P95). Khâu trích xuất 16 bộ lọc định hướng 13×13 ở độ phân giải `448×448` kết hợp 2 lần chạy DINOv2 là nút thắt cổ chai lớn nhất.

## 3. Tài nguyên bộ nhớ & CPU (`benchmark_metrics.json`)

- **RSS Start:** `138.07 MB`
- **RSS Peak:** `463.32 MB` *(đạt ngưỡng `<= 650 MB`)*
- **Private Bytes:** `1503.11 MB`
- **CPU Utilization (multi-thread ONNX/OpenCV):** `378.5%`
