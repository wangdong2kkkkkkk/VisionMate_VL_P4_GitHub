# Confusion Matrices & Statistical Comparison — VisionMate VL-P3

## 1. Ma trận nhầm lẫn trên `final_holdout_v3` (N=245: 80 `object_01`, 165 âm tính)

| Preset | TP | Pos-UNC | FN | FP | Neg-UNC | TN | Known Recall (CI95%) | FAR (CI95%) | Precision | F1 | Acc | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_v2_lsd_hough_baseline` | 16 | 55 | 9 | 15 | 69 | 81 | 0.2000 (0.1270–0.3005) | 0.0909 (0.0559–0.1446) | 0.5161 | 0.2883 | 0.3959 | 0.7503 | 0.5518 |
| `A1_supervised_kp_centerline` | **30** | 46 | 4 | 4 | 48 | 113 | **0.3750** (0.2769–0.4845) | 0.0242 (0.0095–0.0607) | **0.8824** | **0.5263** | 0.5837 | **0.9116** | **0.8272** |
| `B1_supervised_seg_only` | 16 | 61 | 3 | 10 | 39 | 116 | 0.2000 (0.1270–0.3005) | 0.0606 (0.0332–0.1080) | 0.6154 | 0.3019 | 0.5388 | 0.8661 | 0.6529 |
| `B2_kp_guided_supervised_seg` | 11 | 62 | 7 | 3 | 52 | 110 | 0.1375 (0.0785–0.2297) | **0.0182** (0.0062–0.0521) | 0.7857 | 0.2340 | 0.4939 | 0.8649 | 0.7101 |
| `C1_b2_plus_masked_dinov2` | 17 | 28 | 35 | 6 | 13 | 146 | 0.2125 (0.1371–0.3142) | 0.0364 (0.0168–0.0771) | 0.7391 | 0.3301 | 0.6653 | 0.8442 | 0.6979 |
| `C2_full_vl_p3_pipeline` | 10 | 36 | 34 | **3** | 8 | 154 | 0.1250 (0.0693–0.2150) | **0.0182** (0.0062–0.0521) | 0.7692 | 0.2151 | **0.6694** | 0.8897 | 0.7636 |

## 2. Phân tích `FAR` theo từng nhóm gây nhiễu (`C2_full_vl_p3_pipeline`)

| Nhóm gây nhiễu (`final_holdout_v3`) | Tổng | FP (`KNOWN`) | `UNCERTAIN` | TN (`UNKNOWN`) | FAR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `general_distractor` | 34 | 1 | 2 | 31 | 0.0294 |
| `metal_rod_pipe_confuser` | 21 | 0 | 2 | 19 | **0.0000** |
| `tool_handle_confuser` | 23 | 0 | 1 | 22 | **0.0000** |
| `trekking_pole_confuser` | 21 | 0 | 1 | 20 | **0.0000** |
| `umbrella_confuser` | 25 | 1 | 0 | 24 | 0.0400 |
| `walking_stick_confuser` | 16 | 1 | 2 | 13 | 0.0625 |
| `white_line_pole_confuser` | 25 | 0 | 0 | 25 | **0.0000** |
| **Subset người cầm vật (`person_holding_confuser`)** | **125** | **2** | **7** | **116** | **0.0160** (CI95: 0.0044–0.0565) |

## 3. Kiểm định McNemar (Exact Two-Sided, `C2` so với các preset khác)

| So sánh | Positive Recall Diff (95% CI) | Recall McNemar p | Overall Binary Correct Diff (95% CI) | Overall McNemar p |
| --- | ---: | ---: | ---: | ---: |
| `C2` vs `A0` | -7.50% (-18.75%, +3.75%) | 0.286279 | +2.45% (-2.45%, +7.35%) | 0.417692 |
| `C2` vs `A1` | **-25.00%** (-35.00%, -16.25%) | **2.0e-06** | **-7.76%** (-11.43%, -4.08%) | **6.6e-05** |
| `C2` vs `B1` | -7.50% (-17.50%, +2.50%) | 0.210114 | +0.41% (-3.67%, +4.49%) | 1.000000 |
| `C2` vs `B2` | -1.25% (-10.00%, +7.50%) | 1.000000 | -0.41% (-3.67%, +2.86%) | 1.000000 |
| `C2` vs `C1` | -8.75% (-18.75%, 0.00%) | 0.118469 | -1.63% (-5.31%, +2.04%) | 0.503445 |

## 4. Tập biên mờ (`ambiguous_cane_v3`, N=26) & Tập lịch sử

- **`ambiguous_cane_v3` (N=26):** `A0` nhận nhầm `4/26` (`non_false_accept_rate = 0.8462`), `A1/B1/B2` đạt `0/26` (`1.0000`), `C1` đạt `1/26` (`0.9615`), `C2` đạt `1/26` (`0.9615`: 1 KNOWN, 1 UNCERTAIN, 24 UNKNOWN).
- **Lịch sử `final_holdout_v2` (N=180: 51 pos, 129 neg):** `C2` đạt `Recall = 0.1569` (8/51), `FAR = 0.0000` (0/129), `Precision = 1.0000`, `Accuracy = 0.7167`, `ROC-AUC = 0.9495`, `PR-AUC = 0.8509` (so với `4a10113` post-opt: `Recall = 0.1961`, `FAR = 0.0465`, `Acc = 0.7389`).
- **Lịch sử `vl_p1c/final_holdout_full` (N=300):** `object_01_recall = 0.2000` (4/20), `object_02_to_05_preserved_rate = 0.6250` (50/80), `unknown_object_01_far = 0.0050` (1/200).
