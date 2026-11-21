# Giao Thức Hiệu Chuẩn & Cô Lập Hai Pha — VisionMate VL-P1c

Tài liệu này mô tả thiết kế hiệu chuẩn khắc phục hiện tượng sụp ngưỡng của VL-P1b (`tau_per_class = 0.0757..0.1826` trên `N=50`), gồm: (1) kiểm soát cận dưới ổn định (`delta_floor`), (2) Cross-Validation 5-fold theo cụm (`cluster_id`), (3) Out-of-Fold (OOF) Ridge verifier, (4) đầu ra 3 trạng thái, (5) bù trừ đa crop E2E, và (6) khóa cứng hai pha (`Phase A` -> `Phase B`).

---

## 1. Quy trình Khóa Cứng Hai Pha (Two-Phase Blind Evaluation)

- **Pha A (`python -u -m src.benchmark_runner_v1c --phase a`):**
  - Thực hiện đăng ký (`enrollment`) trên `train_pos` (`25` ảnh, `1201.02 ms`, `94,802 B`) và `hard_neg_gallery` (`50` ảnh, `1986.11 ms`, `neg_db = 179,174 B`, `verifier = 85,558 B`, tổng DB `359,534 B`).
  - Toàn bộ thao tác tìm ngưỡng (`delta_floor` grid, 5-fold CV, `R0..R5`, E2E `K=4/6/8`, cân chỉnh `source_offsets`, `alpha_k`, `e2e_tau_offset`) **chỉ được phép chạy trên `calibration_v1c` (`N=200`)**.
  - Kiểm tra cổng ổn định (`cv_gate_passed = true`) và xuất ra hai tệp bất biến trước khi đụng `final_holdout`:
    - `manifests/frozen_config.json` (`selected_delta_floor = 0.08`, `frozen_e2e_k = 4`)
    - `manifests/frozen_a3_oof_state.json` (SHA-256: `bf3b7191737d27a7462cb52ce40a3430da2c84c80d9a0e22decc6c8c4583387f`)
  - Trạng thái kiểm định: **`final_holdout_touched_during_calibration = false`**.
- **Pha B (`python -u -m src.benchmark_runner_v1c --phase b`):**
  - Đọc trực tiếp cấu hình từ `manifests/frozen_config.json` và đánh giá mù **1 lần duy nhất** trên `final_holdout_roi` (`300`), `final_holdout_full` (`300`), `ambiguous_boundary_v1c` (`15`), cùng 2 tập hồi quy kế thừa `vl_p1_test` (`75`) và `vl_p1b_holdout` (`214`). Không có bất kỳ chỉnh sửa tham số nào sau Pha B.

---

## 2. Kết quả Group-Aware 5-Fold Cross-Validation trên `calibration_v1c` (`N=200`)

Các fold được chia theo `cluster_id`/nhóm phiên chụp nhằm bảo đảm ảnh cùng người chụp/bộ sản phẩm không nằm đồng thời ở train-fold và val-fold.

### 2.1 Lưới tìm cận dưới `delta_floor` (`tau_c >= tau_global - delta_floor`, `tau_floor_abs = 0.40`)

| `delta_floor` | 5-Fold Known Recall Mean ± Std | 5-Fold FAR Mean ± Std | Kết luận chọn (`Phase A`) |
|---|---:|---:|---|
| `0.04` | `54.67 ± 8.84%` | `7.20 ± 2.99%` | Quá chặt đối với các lớp biên hẹp |
| `0.06` | `58.67 ± 4.99%` | `7.20 ± 2.99%` | Ổn định tốt, kém `0.08` về Recall |
| **`0.08`** *(Đã chọn)* | **`60.00 ± 5.96%`** | **`7.20 ± 2.99%`** | Độ lệch chuẩn ngưỡng giữa các fold thấp (`<= 0.0236`) |

Độ lệch chuẩn ngưỡng `tau_c` qua 5 fold tại `delta_floor = 0.08`:  
`object_01 = 0.0122`, `object_02 = 0.0183`, `object_03 = 0.0103`, `object_04 = 0.0133`, `object_05 = 0.0236`.

### 2.2 So sánh các phương án hiệu chuẩn `R0..R5` (5-Fold OOF Means trên `calibration_v1c`)

| Phương án | Recall Mean ± Std | FAR Mean ± Std | Ambiguous Rate | Vai trò thiết kế |
|---|---:|---:|---:|---|
| `R0` (`vl_p1_baseline`) | `58.67 ± 7.78%` | `12.80 ± 6.40%` | `0.0%` | Ngưỡng toàn cục kế thừa |
| `R1` (`global_conservative`) | `54.67 ± 9.80%` | `10.40 ± 4.08%` | `0.0%` | Ngưỡng toàn cục nâng cao + phản lực âm |
| `R2` (`per_class_floored`) | `61.33 ± 7.78%` | `9.60 ± 4.08%` | `0.5%` | Ngưỡng theo lớp bị chặn sàn `delta_floor=0.08` |
| `R3` (`distributional_quantile`) | `62.66 ± 9.04%` | `12.00 ± 5.66%` | `0.5%` | Chỉnh phân vị dương/âm trên phân bố lớp |
| `R4` (`out_of_fold_verifier`) | `60.00 ± 5.97%` | `7.20 ± 2.99%` | `0.5%` | Kết hợp Ridge OOF verifier (`ridge_alpha=24.0`, `neg_blend_top2_weight=0.55`) |
| **`R5` (`selective_3state_v1c`)** | **`60.00 ± 5.97%`** | **`7.20 ± 2.99%`** | **`7.5%`** | Thêm dải bất định 3 trạng thái (`KNOWN`/`AMBIGUOUS`/`UNKNOWN`) |

### 2.3 Khảo sát độ nhạy cỡ mẫu hiệu chuẩn (`N_cal = 50 .. 200`)

| Cỡ mẫu hiệu chuẩn `N_cal` | Độ lệch `tau_std` | Ngưỡng thấp nhất `min_tau` | Out-of-Sample Recall | Out-of-Sample FAR | Ý nghĩa khoa học |
|---:|---:|---:|---:|---:|---|
| `50` | `0.0303` | `0.4289` | `58.48%` | `9.84%` | Cỡ mẫu nhỏ như VL-P1b gây dao động ngưỡng mạnh (`0.0303`) và FAR cao |
| `100` | `0.0172` | `0.4467` | `50.00%` | `5.16%` | Ngưỡng bắt đầu hội tụ nhưng Recall OOS còn dao động |
| `150` | `0.0156` | `0.4283` | `55.26%` | `9.68%` | Tiếp tục thu hẹp phương sai `tau_std` |
| **`200`** *(Full VL-P1c)* | `0.0` (single fit) | **`0.4639`** | **`56.00%`** | **`4.80%`** | Ngưỡng sàn hội tụ ở `0.4639`, FAR OOS hạ về `4.80%` |

---

## 3. Tham số Đóng Băng Cuối Cùng (`manifests/frozen_config.json`)

### 3.1 Tham số ROI (`R5_selective_3state_v1c`)
- **`tau_per_class`:** `object_01 = 0.5006`, `object_02 = 0.4774`, `object_03 = 0.4652`, `object_04 = 0.4639`, `object_05 = 0.4878` (không còn lớp nào bị sập xuống `0.07..0.18` như VL-P1b).
- **Chặn sàn & verifier:** `tau_floor_abs = 0.40`, `delta_floor = 0.08`, `ridge_alpha = 24.0`, `neg_blend_top2_weight = 0.55`, `delta_neg_min = {object_01..03, 05: 0.0, object_04: 0.03}`, `verifier_min_score = {object_01: 0.15, object_02..05: -0.15}`, `num_threads = 4`.
- **Dải 3 trạng thái `AMBIGUOUS`:** `delta_pos_ambig = 0.01`, `tau_ambig_band = 0.045`, `neg_ambig_band = 0.025`, `ver_ambig_band = 0.14`.

### 3.2 Tham số Đánh Giá Đa Proposal End-to-End (Chọn trên `calibration_v1c_full`)

| Ngân sách Top-`K` trên `calibration_v1c_full` (`N=200`) | E2E Accuracy | E2E Known Recall | E2E FAR | Latency Mean / p50 / p95 | Kết luận Pha A |
|---:|---:|---:|---:|---:|---|
| **`K = 4`** *(Đã đóng băng)* | **`152/200 (76.0%, CI [69.63, 81.39])`** | **`36/75 (48.0%)`** | `9/125 (7.2%)` | `237.17 / 231.80 / 305.24 ms` | **`frozen_e2e_k = 4`** (Acc, Recall và độ trễ tốt nhất) |
| `K = 6` | `148/200 (74.0%)` | `41.3%` | `6.4%` | `— / 311.30 / — ms` | Trễ hơn `~80 ms`, Recall giảm trên `calibration_v1c_full` |
| `K = 8` | `146/200 (73.0%)` | `40.0%` | `7.2%` | `— / 379.96 / — ms` | Chậm nhất (`~380 ms`), không tăng Acc |

- **Kiểm soát bùng nổ FP đa crop:** `alpha_k = 0.02`, `e2e_tau_offset = 0.03`, `singleton_penalty_heuristic = 0.03`, `singleton_penalty_primary = 0.008`, `support_min_pos_score = 0.41` (đòi hỏi đồng thuận không dư thừa với `0.10 <= IoU <= 0.68` hoặc chịu phạt đơn độc), `min_thin_cane_evidence = 0.42`, `use_thin_corridor_mask = true`.
- **Bù trừ ngưỡng theo nguồn proposal (`source_offsets`):** `yolox_non_person = 0.0`, `context = 0.01`, `fg_contour = 0.015`, `thin_cane = 0.02`, `ms_sal = 0.025`, `small_detail = 0.025`, `yolox_person = 0.045`, `thin_legacy = 0.045`.
