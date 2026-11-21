# Kết Quả Benchmark Định Lượng — VisionMate VL-P1

## 1. Task 1 — Nhận diện trên ROI đã cắt (`data/test/`, 50 dương + 25 âm)
| Phương pháp | Threshold $(\tau, \delta)$ | Top-1 Pos Acc | Macro Recall | Obj01 Gậy | Obj02 Chìa | Obj03 Ví | Obj04 Dép | Obj05 Giày | FAR (Âm tính) | Pos UNKNOWN | Pos AMBIGUOUS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `orb_raw` | inliers=4, r=0.45 | 8.0% | 8.0% | 30% | 0% | 10% | 0% | 0% | 44.0% | 50.0% | 18.0% |
| `orb_adaptive` | inliers=4, r=0.45 | 0.0% | 0.0% | 0% | 0% | 0% | 0% | 0% | 16.0% | 88.0% | 0.0% |
| `mobilenetv3_small__1nn_max` | 0.384, 0.100 | 48.0% | 48.0% | 60% | 40% | 80% | 20% | 40% | 20.0% | 32.0% | 18.0% |
| `mobilenetv3_small__prototype` | 0.395, 0.075 | 60.0% | 60.0% | 50% | 50% | 90% | 40% | 70% | 12.0% | 28.0% | 10.0% |
| `mobilenetv3_small__hybrid` | 0.374, 0.030 | 62.0% | 62.0% | 60% | 40% | 90% | 40% | 80% | 20.0% | 34.0% | 2.0% |
| `efficientnet_b0__1nn_max` | 0.374, 0.030 | 56.0% | 56.0% | 60% | 50% | 80% | 20% | 70% | 4.0% | 40.0% | 0.0% |
| `efficientnet_b0__prototype` | 0.331, 0.000 | 74.0% | 74.0% | 80% | 90% | 90% | 30% | 80% | 20.0% | 22.0% | 0.0% |
| `efficientnet_b0__hybrid` | 0.310, 0.000 | 74.0% | 74.0% | 80% | 90% | 90% | 30% | 80% | 20.0% | 22.0% | 0.0% |
| `dinov2_small_int8__1nn_max` | 0.405, 0.100 | 80.0% | 80.0% | 80% | 100% | 90% | 40% | 90% | 28.0% | 6.0% | 14.0% |
| `dinov2_small_int8__prototype` | 0.373, 0.075 | 88.0% | 88.0% | 90% | 100% | 90% | 70% | 90% | 36.0% | 2.0% | 10.0% |
| `dinov2_small_int8__hybrid` | 0.476, 0.000 | 78.0% | 78.0% | 70% | 70% | 80% | 70% | 100% | 12.0% | 22.0% | 0.0% |
| `ensemble_mv3_dino__hybrid` | 0.440, 0.015 | 76.0% | 76.0% | 80% | 70% | 80% | 60% | 90% | 12.0% | 22.0% | 2.0% |

## 2. Task 2 — Nhận diện trên Ảnh Gốc Chưa Cắt (`data/test_full/`, 50 dương + 25 âm)
| Pipeline (Proposer + Classifier) | Proposal Recall@0.3 | Proposal Recall@0.5 | Miss Proposal | Mean IoU | Top-1 Pos Acc | Macro Recall | FAR (Âm) | E2E P50 (ms) | E2E P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolox_strict__dinov2_hybrid` | 36.0% | 24.0% | 24/50 | 0.457 | 32.0% | 32.0% | 0.0% | 62.9 | 135.8 |
| `yolox_low__dinov2_hybrid` | 74.0% | 58.0% | 3/50 | 0.556 | 64.0% | 64.0% | 8.0% | 104.3 | 185.5 |
| `hybrid_prop__mobilenetv3_hybrid` | 92.0% | 86.0% | 0/50 | 0.713 | 58.0% | 58.0% | 44.0% | 71.8 | 101.0 |
| `hybrid_prop__dinov2_hybrid` | 92.0% | 86.0% | 0/50 | 0.713 | 76.0% | 76.0% | 16.0% | 176.5 | 242.6 |
| `hybrid_prop__ensemble_hybrid` | 92.0% | 86.0% | 0/50 | 0.713 | 70.0% | 70.0% | 20.0% | 208.5 | 256.8 |

## 3. Hiệu Năng Phần Cứng CPU (Windows 11, Intel i5-1135G7)
| Mô hình / Phương pháp | Model (MB) | Load (ms) | Feat P50 (ms) | Feat P95 (ms) | Match P50 (ms) | Enroll/ảnh (ms) | DB 25 ảnh (KB) | Peak RAM (MB) | ΔRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `orb_raw` | 0.00 | 0.0 | 13.40 | 15.20 | 344.227 | 25.00 | 0.0 | 301.8 | 0.0 |
| `orb_adaptive` | 0.00 | 0.0 | 11.24 | 11.84 | 1286.605 | 26.33 | 0.0 | 54.8 | 3.2 |
| `mobilenetv3_small__hybrid` | 9.71 | 38.4 | 7.33 | 8.24 | 0.252 | 10.24 | 122.6 | 408.2 | 22.7 |
| `efficientnet_b0__hybrid` | 20.15 | 122.5 | 56.88 | 62.70 | 0.296 | 46.14 | 152.6 | 441.0 | 45.9 |
| `dinov2_small_int8__hybrid` | 23.32 | 265.1 | 33.64 | 42.09 | 0.173 | 38.43 | 92.6 | 423.3 | 43.6 |
| `ensemble_mv3_dino__hybrid` | 33.03 | 249.7 | 40.20 | 45.59 | 0.179 | 42.16 | 212.6 | 443.9 | 62.5 |
