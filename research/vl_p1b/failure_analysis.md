# VL-P1b Failure Analysis (Honest Empirical Post-Mortem)

> **Lưu ý nguyên tắc khoa học:** Toàn bộ phân tích dưới đây được thực hiện **sau khi khóa cứng ngưỡng** trên `calibration_v1b` (`manifests/frozen_thresholds.json`, SHA-256 prefix `26e3d034133daa3f`) và **không có bất kỳ vòng tinh chỉnh hậu kiểm (post-hoc tuning)** nào trên tập `holdout` (`N=214`).

## 1. Tổng hợp điểm nghẽn chính trên Holdout (`N=214`) & `vl_p1_test` (`N=75`)

### 1.1 Bài toán A (Hard-Negative & Confuser Recognition — Method A ROI)

| Mode | `holdout_roi` Overall Acc | `holdout_roi` Known Recall | `holdout_roi` FAR | Wrong Class / Shoe-Slipper | ROI Latency p50 |
|---|---:|---:|---:|---|---:|
| `A0_baseline_hybrid` | `173/214 (80.8%, 95% CI [75.04, 85.55])` | `67/100 (67.0%, 95% CI [57.31, 75.44])` | `8/114 (7.0%, 95% CI [3.60, 13.24])` | `2/100` (`2/40`) | `110.25 ms` |
| `A1_neg_prototypes` | `147/214 (68.7%, 95% CI [62.19, 74.53])` | `33/100 (33.0%, 95% CI [24.56, 42.69])` | `0/114 (0.0%, 95% CI [0.00, 3.26])` | `0/100` (`0/40`) | `104.37 ms` |
| `A2_adaptive_thresholds` | `164/214 (76.6%, 95% CI [70.53, 81.80])` | `73/100 (73.0%, 95% CI [63.57, 80.73])` | `23/114 (20.2%, 95% CI [13.84, 28.46])` | `2/100` (`2/40`) | `104.93 ms` |
| `A3_full_verifier` | `168/214 (78.5%, 95% CI [72.52, 83.48])` | `74/100 (74.0%, 95% CI [64.63, 81.60])` | `20/114 (17.5%, 95% CI [11.65, 25.55])` | `2/100` (`2/40`) | `123.12 ms` |

**Hiện tượng trái chiều quan trọng nhất (Calibration-to-Holdout Generalization Gap):**
1. Trên `calibration_v1b` (`N=50`: 25 positive = 5/lớp + 25 hard-negative), `A3_full_verifier` tăng Known Recall từ `15/25 (60.0%)` lên `19/25 (76.0%)` và giảm FAR từ `6/25 (24.0%)` xuống `2/25 (8.0%)`. Tuy nhiên khi đóng băng và chạy trên `holdout_roi` (`N=214`), FAR của `A3_full_verifier` tăng lên **`20/114 (17.5%)`** — cao hơn đáng kể so với `A0_baseline_hybrid` **`8/114 (7.0%)`**, dù `A3` vẫn giữ mức Known Recall cao hơn **`74/100 (74.0%)`** vs **`67/100 (67.0%)`** (nhờ `object_04`: `16/20` vs `11/20`, `object_05`: `19/20` vs `17/20`, `object_01`: `11/20` vs `10/20`).
2. **Nguyên nhân gốc 1 — Overfitting ngưỡng per-class (`tau_per_class`) trên tập calibration nhỏ (`5 pos/lớp`):**
   - Để đạt target calibration, hàm `calibrate_on_split` hạ ngưỡng `tau_per_class` rất thấp: `object_05 = 0.0757`, `object_02 = 0.16`, `object_01 = 0.1704`, `object_03 = 0.1826`, `object_04 = 0.3049` (so với `tau_global = 0.4758` của `A0`). Đồng thời `verifier_min_score` cho `object_02` và `object_05` lần lượt bị hạ xuống `-0.25`.
   - Trong số **`20` False Positives (FP)** của `A3_full_verifier` trên `holdout_roi`, có tới **`13` FP** rơi vào `object_02` (bao gồm `7/12` `lock_key_metal_confuser` như `hold_neg_lock_k_01` score `0.3519`, `hold_neg_lock_k_03` `0.4159`, `hold_neg_lock_k_06` `0.2760`, `hold_neg_lock_k_07` `0.3845`, `hold_neg_lock_k_08` `0.3862`, và `6` mẫu `general_distractor` như `hold_gen_03` score `0.2451`, `hold_gen_06` `0.2507`, `hold_gen_08` `0.2222`, `hold_gen_09` `0.2827`, `hold_gen_15` `0.3256`, `hold_gen_17` `0.2804`) và **`5` FP** rơi vào `object_01` (`hold_neg_walkin_03` `0.3871`, `hold_gen_35` `0.3567`, `hold_gen_47` `0.4232`). Tất cả các mẫu này đều có `score` thấp hơn `tau_global = 0.4758` của `A0`, nhưng lọt qua cổng `tau_per_class['object_02'] = 0.16` hoặc `tau_per_class['object_01'] = 0.1704`.
3. **Ở chiều tích cực có kiểm chứng:**
   - Trừ penalty từ `neg_sim` và kiểm tra biên dương-âm (`A3`) đã **cắt giảm FP trên nhóm `notebook_pouch_confuser`** từ `3/7` (`A0`: `hold_neg_notebo_01`, `_03`, `_07`) xuống còn `1/7` (`A3`: chỉ còn `hold_neg_notebo_01`, chặn được `hold_neg_notebo_07` có `pos_score=0.5944` nhưng `neg_sim=0.6001` và `hold_neg_notebo_03` `0.4907`), và cắt `hold_neg_walkin_06` (`0.4969`).
   - `A1_neg_prototypes` (giữ `tau_global = 0.5437` + trừ `neg_sim`) đạt **`0/114 (0.0%)` FAR** trên `holdout_roi` và **`0/25 (0.0%)` FAR** trên `vl_p1_test`, chứng minh vector âm phản lực có khả năng triệt tiêu hoàn toàn báo giả nếu ngưỡng chung không bị hạ xuống quá mức, nhưng đổi lại khiến Known Recall sụp xuống `33/100 (33.0%)` vì công thức trừ `lambda_neg=0.5` làm tụt điểm của mẫu đúng có nền gần giống confuser (đặc biệt `object_01`: `0/20`).
4. **Confuser dai dẳng trên không gian DINOv2-Small:**
   - **Bao kính (`glasses_case_confuser`):** `0/9` FP trên cả `A0` và `A3` ROI — DINOv2 phân biệt tốt bao kính với ví tiền.
   - **Sổ tay/túi da (`notebook_pouch_confuser`):** `hold_neg_notebo_01` có `pos_score=0.6544` vs `object_03` nên cả `A0` lẫn `A3` đều nhận nhầm.
   - **Gậy leo núi / gậy kim loại (`walking_stick_confuser`, `long_metal_rod_confuser`):** `hold_neg_walkin_05` (`0.5557`) và `hold_neg_long_m_07` (`0.5459`) vượt qua cả `A0` lẫn `A3`.
   - **Ổ khóa & chìa / kim loại (`lock_key_metal_confuser`):** `hold_neg_lock_k_11` (`0.4840`) và `hold_neg_lock_k_12` (`0.5027`) vượt `A0`; `7/12` mẫu vượt `A3` do `tau_per_class['object_02']=0.16`.
   - **Nhầm chéo dép (`object_04`) ↔ giày thể thao (`object_05`):** Cả `A0` và `A3` cùng mắc `2/40` lỗi trên `holdout_roi`: `hold_object_04_10` (dép len cổ cao/bootie -> `object_05`) và `hold_object_04_12` (xăng-đan đi trên tất sọc -> `object_05`).
   - **Missing Recall (`FN_UNKNOWN`):** `object_01` mất nhiều điểm nhất (`9/20` FN trong `A3`, `10/20` FN trong `A0`) vì gậy trắng rất mảnh trong ảnh đường phố, phần lớn diện tích ROI bị nền quần áo/vỉa hè chi phối (`hold_object_01_01`: `pos_score=0.2317`, `hold_object_01_03`: `0.2285`), kế đến là `object_03` (`9/20` FN với các mẫu ví cổ/trưng bày bảo tàng có thẻ thuyết minh hoặc ví vải thêu như `hold_object_03_10`, `03_15`, `03_16`).

---

### 1.2 Bài toán B (Object Proposal Localization — Method B trên `holdout_full`)

| Proposer Mode | `vl_p1_test_full` Recall@0.5 | `holdout_full` Recall@0.3 | `holdout_full` Recall@0.5 | `object_01` Recall@0.5 | Mean Containment | Mean Bg Contam. | p50 Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yolox_strict__k4` | `12/50 (24.0%)` | `36/100 (36.0%)` | `17/100 (17.0%)` | `0/20` | `0.3512` | `0.6150` | `38.38 ms` |
| `yolox_low__k4` | `29/50 (58.0%)` | `64/100 (64.0%)` | `40/100 (40.0%)` | `0/20` | `0.5950` | `0.3405` | `40.09 ms` |
| `B0_vl_p1_proposer__k4` | `43/50 (86.0%)` | `83/100 (83.0%)` | `68/100 (68.0%)` | `0/20` (`7/20` @0.3) | `0.8941` | `0.2622` | `52.71 ms` |
| `B1_multiscale__k4` | `41/50 (82.0%)` | `74/100 (74.0%)` | `64/100 (64.0%)` | `2/20` (`4/20` @0.3) | `0.7301` | `0.2380` | `84.73 ms` |
| `B2_geometry_thin__k4` | `34/50 (68.0%)` | `74/100 (74.0%)` | `61/100 (61.0%)` | `0/20` (`4/20` @0.3) | `0.7283` | `0.2602` | `103.41 ms` |
| `B_full_v1b__k4` | `34/50 (68.0%)` | `74/100 (74.0%)` | `60/100 (60.0%)` | `0/20` (`4/20` @0.3) | `0.6995` | `0.2611` | `98.17 ms` |
| `B_full_v1b__k6` | `44/50 (88.0%)` | `80/100 (80.0%)` | `69/100 (69.0%)` | `2/20` (`6/20` @0.3) | `0.7786` | `0.2579` | `84.34 ms` |
| `B_full_v1b__k8` | **`45/50 (90.0%)`** | **`83/100 (83.0%)`** | **`74/100 (74.0%)`** | `2/20` (`6/20` @0.3) | `0.8417` | `0.2599` | `120.03 ms` |

**Nguyên nhân gốc 2 — Đứt gãy thứ tự xếp hạng (`ProposalRanker`) ở `K=4` so với `K=6..8`:**
1. Với ngân sách lớn hơn (`K=8`), `B_full_v1b__k8` **vượt baseline `B0` trên cả hai tập**: `90.0%` vs `86.0%` Recall@0.5 trên `vl_p1_test_full` (`object_01` tăng từ `5/10` lên `6/10`, `object_04` `9/10 -> 10/10`), và **`74/100 (74.0%)` vs `68/100 (68.0%)`** Recall@0.5 trên `holdout_full` (`object_02` đạt tuyệt đối `20/20` vs `16/20`, `object_03` `17/20` vs `16/20`, `object_01` có `2/20` IoU >= 0.5 vs `0/20`).
2. Ngược lại tại **`K=4` (`B_full_v1b__k4`)**, điểm số rớt xuống `60/100` (dưới `B0` `68/100`). Lý do trực diện nằm trong trường `selected_source_counts`: `geom_thin_line` (3 hộp Hough mảnh sinh bởi `propose_thin_and_small_geometry`) nhận điểm ưu tiên hình học + saliency quá cao trong `ProposalRanker`, dẫn đến **`geom_thin_line` chiếm `130/214` lượt chọn E2E ở `C3` và `125/214` ở `C4`** trong khi `center_72` chỉ còn `2/214` và `fg_contour` chỉ còn `5–7/214`. Hệ quả là các ứng viên bao phủ toàn đồ vật lớn (`center_72`, `full_frame`, `fg_contour`, `yolox_coco`) bị đẩy ra khỏi Top-4 và chỉ xuất hiện lại ở vị trí 5–8 (`K=6..8`).
3. **Thách thức vật lý của `object_01` (Gậy trắng trong scene có người):** Trên `holdout_full`, ngay cả ở `K=8`, `object_01` chỉ đạt `2/20` IoU >= 0.5 (`6/20` IoU >= 0.3) và `mean_bg_contamination = 0.6829`. Đa số hộp dự đoán trúng một đoạn gậy (`containment` `1.0` ở nhiều mẫu như `hold_object_01_04`, `01_08`, `01_13`, `01_16`, `01_17`, `01_20`) nhưng vì gậy trắng là đoạn thẳng mảnh chéo nằm trong hộp trục ngang/dọc (axis-aligned bounding box) có tỷ lệ nền gạch/người chiếm `70–90%`, bất kỳ sai lệch nhẹ nào ở hai đầu mút gậy cũng khiến hộp trục kéo giãn thừa quá 75–90% nền (`bg_contamination` `0.66–0.91`), đẩy IoU xuống `0.15–0.34`. Ngoài ra, `hold_object_05_10` (một đôi giày nhỏ đang cầm trên tay người giữa cửa hàng rộng) chỉ đạt `best_iou = 0.0829` vì YOLOX bắt nhầm cả người cầm giày.

---

### 1.3 Bài toán C (End-to-End Ablation `C1..C4` & Ambiguous Boundary)

| E2E Config | `holdout_full` Overall Acc | `holdout_full` Known Recall | `holdout_full` FAR | `vl_p1_test_full` Acc / Rec / FAR | E2E p50 / p95 |
|---|---:|---:|---:|---|---:|
| `C1_vl_p1_baseline` (`B0 + A0`) | `162/214 (75.7%)` | `63/100 (63.0%)` | `15/114 (13.2%)` | `59/75 (78.7%)` / `38/50 (76.0%)` / `4/25 (16.0%)` | `538.81 / 1050.46 ms` |
| `C2_hard_neg_only` (`B0 + A3`) | `153/214 (71.5%)` | `70/100 (70.0%)` | `31/114 (27.2%)` | `62/75 (82.7%)` / `43/50 (86.0%)` / `6/25 (24.0%)` | `594.90 / 1129.08 ms` |
| `C3_proposal_only` (`B_full_k4 + A0`) | `157/214 (73.4%)` | `60/100 (60.0%)` | `17/114 (14.9%)` | `61/75 (81.3%)` / `39/50 (78.0%)` / `3/25 (12.0%)` | `384.48 / 965.01 ms` |
| `C4_combined_v1b` (`B_full_k4 + A3`) | `142/214 (66.4%)` | `66/100 (66.0%)` | `38/114 (33.3%)` | `59/75 (78.7%)` / `42/50 (84.0%)` / `8/25 (32.0%)` | `332.79 / 534.49 ms` |

1. **Hiệu ứng khuếch đại FP đa crop (Multi-Crop Amplification):** So với đánh giá ROI cô lập, việc chạy bộ nhận dạng trên `K=4` crop/ảnh rồi lấy cực đại làm FAR tăng vọt: `A0 -> C1` tăng FAR từ `7.0%` (`8/114`) lên `13.2%` (`15/114`); `A3 -> C2` từ `17.5%` (`20/114`) lên `27.2%` (`31/114`); và `C4_combined_v1b` vọt lên **`38/114 (33.3%)`** vì (i) `e2e_tau_offset_c4` dừng ở `0.0` khi hiệu chuẩn trên tập `calibration_v1b_full` chỉ có 25 ảnh âm, và (ii) các mảnh `geom_thin_line` vô nghĩa từ ảnh âm (`hold_gen_03`, `hold_gen_05`, `hold_gen_08`, `hold_gen_09`, `hold_gen_18`, `hold_gen_25`, `hold_gen_30`, `hold_gen_48`, `hold_neg_lock_k_01`, `06`, `08`) cắt qua tường/cạnh bàn/khung cửa rồi kích hoạt lớp `object_01` (`tau=0.1704`) hoặc `object_02` (`tau=0.16`).
2. **Độ nhạy trên `ambiguous_boundary` (`n=9`):** Dưới ngưỡng đóng băng `A3`/`C4`, mô hình gán `KNOWN` cho `6/9` mẫu ở ROI (`7/9` ở E2E) và từ chối `UNKNOWN` cho `3/9` (`2/9` E2E), `0/9` rơi vào `AMBIGUOUS`. Nguyên nhân là dải `delta_pos = 0.01` quá hẹp nên mẫu ranh giới (ví dụ giày gỗ/cờ lết, móc khóa gần giống chìa, gậy chống gỗ có tay nắm) hoặc vượt ngưỡng lớp đơn lẻ hoặc rớt xuống `UNKNOWN`.

---

## 2. Hướng khắc phục cụ thể cho vòng nghiên cứu tiếp theo (VL-P1c)

1. **Khử overfitting hiệu chuẩn (Problem A):** Ràng buộc cận dưới cho ngưỡng từng lớp (`tau_per_class[c] >= tau_global - 0.06` với `tau_global >= 0.45`), dùng K-fold cross-validation trên tập calibration tối thiểu `>= 100` mẫu (`>= 10` mẫu dương/lớp, `>= 50` hard-neg) thay vì tìm ngưỡng trên `25 + 25` mẫu.
2. **Cân bằng ngân sách slot trong `ProposalRanker` (Problem B):** Giới hạn `geom_thin_line` tối đa **1 slot** trong `top_k=4`, bắt buộc dành slot cho ít nhất `1` đề xuất vật thể toàn phần (`yolox`/`fg_contour`/`center_72`), hoặc giữ ngân sách `K=6..8` kết hợp tiền lọc nhẹ trước DINOv2 (nhưng cần đo ngân sách trễ `<= 600 ms`).
3. **Hiệu chuẩn đa crop thực sự (`e2e_tau_offset` > 0):** Tách riêng ngưỡng cho proposal phụ (`geom_thin_line`/`person_hand_zone` cần biên phạt `neg_sim` chặt hơn `center_72`/`yolox`) để cắt hiệu ứng chọn mảnh nền tạo FP.
