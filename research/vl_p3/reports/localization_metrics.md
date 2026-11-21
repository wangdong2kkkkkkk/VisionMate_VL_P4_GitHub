# Localization Metrics — VisionMate VL-P3 (`final_holdout_v3`, 80 `object_01`)

## 1. So sánh định vị giữa các cấu hình trên tập mù `final_holdout_v3`

| Chỉ số định vị (`object_01`, N=80) | `A0` (VL-P2 LSD/Hough) | `A1` / `B2` / `C1` / `C2` (VL-P3 Supervised KP+CL) | `B1` (Seg Standalone) | Cải thiện `A1` vs `A0` |
| --- | ---: | ---: | ---: | ---: |
| Mean Endpoint Error (px) | 336.29 | **194.51** | 259.77 | **-141.78 px (-42.2%)** |
| Mean Max Endpoint Error (px) | 456.16 | **254.76** | 366.24 | -201.40 px (-44.2%) |
| Mean Endpoint Err (Norm Diag) | 0.2966 | **0.1709** | 0.2252 | -0.1257 |
| Mean Endpoint Err (Norm Len) | 1.6450 | **0.8963** | 1.1842 | -0.7487 |
| Keypoint PDJ@0.05 | 0.1000 | **0.1812** | 0.0938 | +0.0812 |
| Keypoint PDJ@0.10 | 0.1938 | **0.3937** | 0.2375 | **+0.1999 (+20.0 pp)** |
| Both Endpoints PDJ@0.10 | 0.0875 | **0.2375** | 0.0750 | **+0.1500 (+15.0 pp)** |
| Mean Centerline Chamfer (px) | 204.65 | **143.19** | 121.45 | -61.46 px (-30.0%) |
| Mean Centerline Chamfer (Norm) | 0.1800 | 0.1256 | **0.1053** | -0.0544 |
| Shaft Coverage Fraction | 0.2365 | 0.3250 | **0.3860** | +0.0885 |
| Full Shaft Detected | 0.1750 | **0.2500** | 0.1375 | +0.0750 |
| Endpoint Inclusion | 0.0500 | **0.1750** | 0.0375 | +0.1250 |
| Rotated OBB IoU | 0.0194 | **0.0263** | 0.0248 | +0.0069 |
| Mean Angle Error (°) | 41.17° | **25.66°** | 49.36° | **-15.51°** |

## 2. Phân nhóm bị che khuất (`occluded`, N=27) vs rõ ràng (`clear`, N=53)

| Nhóm ảnh | Preset | Endpoint Err (px) | PDJ@0.10 | Both PDJ@0.10 | Chamfer (px) | Shaft Cov | Angle Err (°) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `occluded` (N=27) | `A0` (VL-P2) | 341.96 | 0.1111 | 0.0370 | 204.79 | 0.1689 | 49.04° |
| `occluded` (N=27) | `A1` / `C2` (VL-P3) | **221.92** | **0.3148** | **0.1481** | **164.49** | **0.2222** | **35.95°** |
| `clear` (N=53) | `A0` (VL-P2) | 333.40 | 0.2358 | 0.1132 | 204.58 | 0.2709 | 37.15° |
| `clear` (N=53) | `A1` / `C2` (VL-P3) | **180.55** | **0.4340** | **0.2830** | **132.34** | **0.3774** | **20.42°** |

## 3. Phân tích định vị

- Bộ định vị có học giám sát `A1` giảm sai số hai đầu mút xuống còn **57.8%** so với baseline LSD/Hough `A0`, tăng tỷ lệ phát hiện đúng cả 2 đầu mút (`Both PDJ@0.10`) từ **8.75% lên 23.75%**, và giảm sai số góc từ **41.17° xuống 25.66°**.
- Trên tập bị che khuất (`N=27`), `A1` vẫn giảm sai số đầu mút `120.04 px` và sai số góc `13.09°` so với `A0`, nhưng mức sai số tuyệt đối (`221.92 px`) vẫn lớn do cây gậy mảnh bị lẫn vào người đi bộ, vạch kẻ đường, hoặc cảnh có nhiều người cầm gậy.
