# Phân Tích Lỗi Trung Thực — VisionMate VL-P1c (`final_holdout`, `N=300`)

> **Nguyên tắc mù đôi:** Mọi phân tích dưới đây được tổng hợp trực tiếp từ `reports/benchmark_metrics.json` sau khi chạy 1 lần duy nhất `--phase b` trên `manifests/frozen_config.json`. Không có bất kỳ chỉnh sửa ngưỡng, `K`, hay penalty nào sau khi xem kết quả holdout.

---

## 1. Phân tích Lỗi Phương pháp A — Recognition & Calibration (`final_holdout_roi`, `N=300`)

| Mode ROI | Overall Acc (95% CI) | Known Recall | FAR | Amb Rate | Wrong Class | Latency p50 / p95 |
|---|---:|---:|---:|---:|---|---:|
| `R0_vl_p1_baseline` | `252/300 (84.0%, [79.43, 87.71])` | `68/100 (68.0%)` | `16/200 (8.0%)` | `0.0%` | `0/100` | `33.65 / 46.34 ms` |
| `R1_global_conservative` | `239/300 (79.7%, [74.75, 83.83])` | `48/100 (48.0%)` | `9/200 (4.5%)` | `0.0%` | `0/100` | `33.72 / 47.60 ms` |
| `R2_per_class_floored` | `250/300 (83.3%, [78.70, 87.12])` | `62/100 (62.0%)` | `12/200 (6.0%)` | `0.0%` | `0/100` | `38.22 / 62.34 ms` |
| `R3_distributional_quantile` | **`254/300 (84.7%, [80.15, 88.30])`** | `65/100 (65.0%)` | `11/200 (5.5%)` | `0.0%` | `0/100` | `35.23 / 51.30 ms` |
| `R4_out_of_fold_verifier` | `253/300 (84.3%, [79.79, 88.01])` | `63/100 (63.0%)` | `10/200 (5.0%)` | `0.0%` | `0/100` | `33.97 / 44.40 ms` |
| **`R5_selective_3state_v1c`** | **`253/300 (84.3%, [79.79, 88.01])`** | `63/100 (63.0%)` | **`10/200 (5.0%)`** | `5.0%` (`9 pos, 6 neg`) | `0/100` | `35.58 / 46.60 ms` |
| `A1_vl_p1b_ref` | `218/300 (72.7%, [67.36, 77.40])` | `21/100 (21.0%)` | **`3/200 (1.5%)`** | `0.0%` | `0/100` | `30.80 / 42.57 ms` |
| `A3_vl_p1b_ref` | `256/300 (85.3%, [80.88, 88.89])` | `73/100 (73.0%)` | `17/200 (8.5%)` | `0.0%` | `0/100` | `35.77 / 48.86 ms` |

### Điểm mạnh đã xác minh và trade-off tồn đọng ở ROI
1. **Cắt giảm có ý nghĩa thống kê các báo giả (`FAR`) so với cả `R0` lẫn `A3_vl_p1b_ref`:**
   - `R5_selective_3state_v1c` đưa FAR từ `8.0%` (`16/200` của `R0`) và `8.5%` (`17/200` của `A3_v1b`) xuống **`5.0% (10/200)`** (`McNemar neg` vs `R0`: `p = 0.03125`; vs `A3_v1b`: `p = 0.039062`).
   - Toàn bộ 88 ảnh nền tự nhiên `general_distractor` đều đạt **`0/88 (0.0%)` FP** trên `R5` (so với `1/88` ở `R0` và `2/88` ở `A3_v1b`), đồng thời `footwear_sock_confuser` và `glasses_case_confuser` cùng đạt **`0/16`**.
   - So với `A3_v1b`, lỗi trên `walking_stick_confuser` giảm mạnh từ **`6/16` xuống còn `2/16`**.
2. **Chi tiết `47` lỗi của `R5` trên `final_holdout_roi` (`28 FN_UNKNOWN + 9 FN_AMBIGUOUS + 10 FP`):**
   - **`9` ca dương tính rơi vào `AMBIGUOUS` (`FN_AMBIGUOUS`):** Phản ánh việc dải bất định giữ an toàn thay vì đoán liều, ví dụ `fhold_object_01_12` (`score=0.4714`), `fhold_object_01_19` (`0.4899`), `fhold_object_02_06` (`0.4584`), `fhold_object_02_19` (`0.4412`).
   - **`28` ca bị từ chối thành `UNKNOWN` (`FN_UNKNOWN`):** Tập trung mạnh ở `object_01` (`8 FN_UNKNOWN + 2 FN_AMBIGUOUS`, chỉ đạt **`10/20`** Recall; ví dụ `fhold_object_01_02` score `0.3244` vs `neg_sim=0.422`, `fhold_object_01_04` `0.4712` vs `neg_sim=0.5201`, `fhold_object_01_05` `0.3171` vs `0.4014`), `object_03` (**`10/20`**, e.g. `fhold_object_03_09` `0.2853`), và `object_04` (**`11/20`**). Với DINOv2-Small đóng băng 5-shot, ngay cả trong ROI cắt chuẩn từ GT, gậy trắng mảnh (`<2%` diện tích pixel hộp trục) vẫn có embedding bị pha bởi nền đường phố/quần áo gần với `walking_stick`/`general`.
   - **`10` ca False Positive dai dẳng trên ROI:** Gồm `3/16` `notebook_pouch_confuser` (`fhold_neg_notebo_11` score `0.6017`, `fhold_neg_notebo_12` `0.5886`, `fhold_neg_notebo_16` `0.546`), `3/16` `lock_key_metal_confuser` (`fhold_neg_lock_k_03` `0.5563`, `_05` `0.6195` bị nhận thành `object_01`, `_15` `0.5308`), `2/16` `walking_stick_confuser` (`fhold_neg_walkin_03` `0.6126`, `_14` `0.6869`), `1/16` `long_metal_rod_confuser` (`fhold_neg_long_m_04` `0.5558`), và `1/16` `umbrella_handle_confuser` (`fhold_neg_umbrel_06` `0.5484`).
   - **`Wrong class` ở ROI = `0/100`** (không có lỗi nhầm chéo lớp nào trên `R0..R5`, kể cả dép vs giày).

---

## 2. Phân tích Lỗi Phương pháp B — Object Proposal & Localization (`final_holdout_full`, `N=100` pos)

| Proposer | Recall@0.3 | Recall@0.5 | Mean IoU | Mean Bg Contam. | `object_01` R@0.5 / R@0.3 | Cane Endpoint Incl. | Latency p50 |
|---|---:|---:|---:|---:|---|---:|---:|
| `yolox_strict__k4` | `26/100 (26.0%)` | `19/100 (19.0%)` | `0.2168` | `0.2093` | `0/20` / `1/20` | `5/20 (25.0%)` | `15.31 ms` |
| `yolox_low__k4` | `60/100 (60.0%)` | `42/100 (42.0%)` | `0.4415` | `0.2158` | `0/20` / `1/20` | `5/20 (25.0%)` | `16.24 ms` |
| `B0_vl_p1_proposer__k4` | `80/100 (80.0%)` | `71/100 (71.0%)` | `0.6600` | `0.2540` | `1/20` / `3/20` | `10/20 (50.0%)` | `20.40 ms` |
| `B_full_v1b__k4` | `77/100 (77.0%)` | `58/100 (58.0%)` | `0.5269` | `0.2480` | `1/20` / `4/20` | `9/20 (45.0%)` | `44.75 ms` |
| `B_full_v1b__k8` | **`86/100 (86.0%)`** | `72/100 (72.0%)` | `0.6435` | `0.2234` | **`2/20` / `8/20`** | `9/20 (45.0%)` | `44.96 ms` |
| **`B_v1c_proposer__k4`** | `81/100 (81.0%)` | **`73/100 (73.0%)`** | `0.6188` | `0.2410` | `1/20` / `4/20` | `10/20 (50.0%)` | `62.54 ms` |
| `B_v1c_proposer__k6` | `83/100 (83.0%)` | `73/100 (73.0%)` | `0.6286` | `0.2338` | `1/20` / `5/20` | `11/20 (55.0%)` | `65.02 ms` |
| `B_v1c_proposer__k8` | `85/100 (85.0%)` | `73/100 (73.0%)` | `0.6588` | `0.2221` | `1/20` / `7/20` | **`13/20 (65.0%)`** | `62.98 ms` |

1. **Thành công sửa lỗi nghẽn khe (`ProposalRankerV1c`) ở `K=4`:**
   - Trong `B_full_v1b__k4`, các hộp `F_thin` chiếm **`54.0%`** slot và đẩy `F_context` xuống còn **`2.5%`**, làm Recall@0.5 tụt từ `71%` (`B0@4`) xuống `58%`.
   - Với bảo vệ đa dạng slot của `ProposalRankerV1c`, ở cùng **`K=4`**, `B_v1c_proposer__k4` cân bằng lại phân bố thành **`F_primary=45.0%, F_context=27.3%, F_thin=24.8%, F_small=3.0%`**, khôi phục Recall@0.5 lên **`73/100 (73.0%)`** (`+15 điểm %` so với `B_full_v1b__k4`, vượt `B0__k4` `71%`). Đặc biệt trên 4 lớp vật thể chuẩn, `object_02=19/20`, `object_03=19/20`, `object_04=16/20`, `object_05=18/20`.
2. **Vì sao định vị gậy trắng (`object_01`) vẫn thất bại ở ngưỡng IoU >= 0.5 (`1/20` ở `K=4..8`):**
   - Đã có cải thiện rõ về bao phủ hình học: `white_cane_r0_3` tăng từ `3/20` (`B0@4`) lên `7/20` (`B_v1c@8`), `endpoint_inclusion_rate` tăng từ `10/20 (50%)` lên **`13/20 (65%)`** (và `5/5 = 100%` trên tập `handheld_person_subset` tại `K=8`).
   - Nhưng gậy trắng là đoạn thẳng chéo cực mảnh trong ảnh đường phố: ngay cả khi hộp `oriented_thin_cane` hoặc `center_72` **bao trọn 100% cây gậy (`containment = 0.98..1.0`)**, hộp chữ nhật trục ngang/dọc bao quanh đoạn chéo vẫn chứa trung bình **`75.73%` (`K=4`) / `70.88%` (`K=8`) nền rác** (`mean_bg_contamination`), đẩy `IoU` xuống còn `0.0982..0.3533` — ví dụ:
     - `fhold_object_01_01`: `containment=1.000`, `best_iou=0.1920` (`oriented_thin_cane`)
     - `fhold_object_01_02`: `containment=1.000`, `best_iou=0.0982` (`center_72`)
     - `fhold_object_01_04`: `containment=1.000`, `best_iou=0.1062` (`center_72`)
     - `fhold_object_01_07`: `containment=1.000`, `best_iou=0.1044` (`oriented_thin_cane`)
     - `fhold_object_01_03`: `containment=0.9797`, `best_iou=0.3533` (`sal_peak_55` ở `K=8`)
   - Hệ quả tất yếu: khi đưa crop axis-aligned `224x224` vào DINOv2-Small (`global token / patch pooling`), hơn 70% tín hiệu đặc trưng trong crop là vỉa hè, quần áo và nền thay vì thân gậy trắng.

---

## 3. Phân tích Lỗi Tổng Hợp End-to-End (`final_holdout_full`, `N=300`)

| Pipeline E2E | Overall Acc (95% CI) | Known Recall | FAR | Amb Rate | `E2E_PROPOSAL_MISSED` | `E2E_VERIFIER_REJECT` | `E2E_AMBIGUOUS` | `E2E_WRONG_CLASS` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C1_vl_p1_baseline` (`B0+A0`) | `243/300 (81.0%, [76.18, 85.04])` | `64/100 (64.0%)` | `21/200 (10.5%)` | `0.0%` | `18` | `18` | `0` | `0` |
| `C4_vl_p1b_combined` (`B_v1b+A3`) | `215/300 (71.7%, [66.32, 76.47])` | **`67/100 (67.0%)`** | `52/200 (26.0%)` | `0.33%` | `16` | `9` | `0` | `8` |
| **`C_v1c_cal_only` (`B0+R5`)** | **`251/300 (83.7%, [79.06, 87.42])`** | `62/100 (62.0%)` | `11/200 (5.5%)` | `6.33%` | `18` | `14` | `5` | `1` |
| `C_v1c_prop_only` (`B_v1c+A0_v1c`) | `237/300 (79.0%, [74.04, 83.23])` | `67/100 (67.0%)` | `30/200 (15.0%)` | `0.0%` | `17` | `16` | `0` | `0` |
| `C_v1c_full__k4` (`B_v1c+R5+MP`) | `242/300 (80.7%, [75.82, 84.74])` | `50/100 (50.0%)` | **`8/200 (4.0%)`** | `12.0%` (`18 pos, 18 neg`) | `18` | `13` | **`18`** | `1` |
| `C_v1c_full__k6` | `242/300 (80.7%, [75.82, 84.74])` | `49/100 (49.0%)` | `7/200 (3.5%)` | `17.0%` (`22 pos, 29 neg`) | `15` | `14` | `22` | `0` |
| `C_v1c_full__k8` | `242/300 (80.7%, [75.82, 84.74])` | `49/100 (49.0%)` | `7/200 (3.5%)` | `19.33%` (`24 pos, 34 neg`) | `13` | `14` | `24` | `0` |

### Ba bài học thực nghiệm cốt lõi từ `C_v1c`
1. **Giảm triệt để bùng nổ FP đa proposal (`C4 26.0% -> C_v1c_full__k4 4.0%`, `C_v1c_prop_only 15.0% -> C_v1c_full__k4 4.0%`):**
   - Trong `C4`, `geom_thin_line` tạo riêng `32` ca FP trên `final_holdout_full` (`52` FP tổng, `12/16` trên `walking_stick_confuser`, `17/88` trên `general_distractor`).
   - Trong `C_v1c_prop_only` (có `B_v1c_proposer` nhưng thiếu kiểm soát đa proposal), `oriented_thin_cane` cũng gây `14/30` ca FP.
   - Khi bật đầy đủ `C_v1c_full__k4`, FAR rơi xuống **`8/200 (4.0%)` (`0/88` FP trên `general_distractor`, `1/16` `lock_key`, `4/16` `notebook_pouch`, `3/16` `walking_stick`)** — chứng minh kiểm soát đa proposal (`alpha_k`, `source_offsets`, `singleton_penalty`) thực sự khóa được tác dụng phụ của proposer mỏng. (Kiểm định McNemar vs `C4`: `neg diff = +22.0%, p = 0.0`; vs `C1`: `neg diff = +6.5%, p = 0.000244`).
2. **Vì sao `C_v1c_full__k4` sụt Recall nặng (`64.0% C1 -> 50.0% C_v1c_full__k4`, `McNemar pos p = 0.001312`):**
   - Có sự **chồng tầng phạt (double-gating)** khi `K=4` kết hợp cùng lúc `B_v1c` + `R5` + `e2e_tau_offset=0.03` + `source_offsets` (`+0.01..+0.045`) + `singleton_penalty` (`+0.008..+0.03`). Ngoài `18` ca hụt proposal (`object_01` tới `2/20` E2E), tới **`18` ca vật thật (`E2E_AMBIGUOUS`)** bị đẩy khỏi `KNOWN` vào vùng `AMBIGUOUS` (như `object_02: 11/20`, `object_03: 11/20`, `object_04: 10/20` so với `C1: 16/20, 16/20, 13/20`) và `13` ca bị `E2E_VERIFIER_REJECT`.
   - Khi tăng `K=4 -> K=6 -> K=8`, `FAR` giảm nhẹ `4.0% -> 3.5%` (không bùng nổ FP nhờ hiệu chỉnh đa proposal!) nhưng `AMBIGUOUS` tăng mạnh `12.0% -> 17.0% -> 19.33%` khiến Recall tiếp tục trượt nhẹ `50% -> 49%`.
3. **Điểm cân bằng thực tế tốt nhất là `C_v1c_cal_only` (`B0 + R5`):**
   - Giữ bộ đề xuất gọn nhẹ `B0@4` (`176.20 ms` p50, `p95 = 218.69 ms`) kết hợp hiệu chuẩn `R5` giúp **`C_v1c_cal_only` đạt `251/300 (83.7%)` Acc** — **cao nhất mọi cấu hình**, vượt `C1` (`81.0%`), giữ Recall gần tương đương (`62.0%` vs `64.0%`), và cắt đôi FAR từ **`10.5%` xuống `5.5%` (`11/200`)** (với `1/88` `general_distractor` và chỉ `5` ca dương rơi vào `AMBIGUOUS`).
4. **Trên tập ranh giới `ambiguous_boundary_v1c` (`N=15`):**
   - `C1` (`10 KNOWN / 5 UNKNOWN`) và `C4` (`12 KNOWN / 3 UNKNOWN`) hoàn toàn không có khả năng báo do dự (`0 AMBIGUOUS`). Ngược lại, `C_v1c_full__k4` phân loại hợp lý thành `5 KNOWN / 5 AMBIGUOUS / 5 UNKNOWN` (`3/5/7` ở ROI là `6/3/6`), chứng minh cơ chế 3 trạng thái hoạt động đúng thiết kế trên các ca lai nhập nhằng.
