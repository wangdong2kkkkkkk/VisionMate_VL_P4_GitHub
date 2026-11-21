# Localization Metrics — VisionMate VL-P4

## 1. Tập `final_holdout_v4` (19 `object_01` mới hoàn toàn)

| Mô hình | Endpoint Err (px) | Norm Diag | PDJ@0.05 | PDJ@0.10 | Both PDJ@0.10 | Chamfer (px) | Angle Err (°) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_VL_P3_A1` *(baseline)* | 281.03 | 0.2248 | 0.1053 | 0.2632 | **0.1579** | 245.44 | **20.30°** |
| `A1_CNN_Keypoint` | **230.22** | 0.1858 | **0.2368** | **0.3684** | **0.1579** | **169.18** | 37.98° |
| `B1_CNN_Segmentation` | 276.75 | 0.2189 | 0.0789 | 0.1579 | 0.0000 | 175.51 | 47.08° |
| `B2_CNN_MultiTask` / `C1` / `C2` | 233.31 | **0.1857** | 0.1842 | 0.2895 | 0.0526 | 182.39 | 45.29° |

- **Che khuất (`occluded`, N=8) vs rõ ràng (`clear`, N=11) trên `v4`:** `C2` đạt `223.49 px` (`occluded`) vs `240.45 px` (`clear`), `A0` đạt `285.17 px` vs `278.01 px`.

## 2. Tập `final_holdout_v3` (80 `object_01`, đối chứng trực tiếp với VL-P3 `A1`)

| Mô hình | Endpoint Err (px) | Norm Diag | PDJ@0.05 | PDJ@0.10 | Both PDJ@0.10 | Chamfer (px) | Angle Err (°) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_VL_P3_A1` *(baseline)* | 194.51 | 0.1709 | **0.1812** | **0.3937** | **0.2375** | 144.97 | **25.66°** |
| `A1_CNN_Keypoint` | **188.04** | **0.1641** | 0.1688 | 0.3625 | 0.1500 | 129.06 | 32.81° |
| `B1_CNN_Segmentation` | 211.32 | 0.1846 | 0.1125 | 0.2750 | 0.1000 | **126.32** | 32.86° |
| `B2_CNN_MultiTask` / `C1` / `C2` | 198.53 | 0.1729 | 0.1625 | 0.3750 | 0.1875 | 140.00 | 35.44° |

- **Che khuất (`occluded`, N=27) vs rõ ràng (`clear`, N=53) trên `v3`:** `C2` đạt `212.49 px` (`occluded`, vs `A0` `221.92 px`) và `191.42 px` (`clear`, vs `A0` `180.55 px`).

## 3. Nhận xét trung thực về định vị

- CNN (`A1_CNN_Keypoint`, `C2`) cải thiện **Centerline Chamfer** (`144.97 -> 129.06 / 140.00 px` trên `v3`, `245.44 -> 169.18 / 182.39 px` trên `v4`) và `Endpoint Err` trên `v4` (`281.03 -> 230.22 / 233.31 px`), nhưng **chưa vượt `A0_VL_P3_A1` về `Both PDJ@0.10` (`18.75%` vs `23.75%` trên `v3`) và góc (`35.44°` vs `25.66°`)** vì lưới đầu ra `64×64` ghép đỉnh `top_hm` / `bot_hm` độc lập đôi khi nối chéo giữa hai người khác nhau trong ảnh đông người.
