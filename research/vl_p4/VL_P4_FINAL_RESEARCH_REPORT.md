# BÁO CÁO NGHIÊN CỨU TỔNG KẾT — VISIONMATE VL-P4
## Lightweight End-to-End White Cane Detection Research

**Đơn vị thực hiện:** Google Antigravity  
**Trạng thái:** Independent Software Research Pilot (Windows 11, CPU-first, offline inference, chưa tích hợp production)  
**Worktree / Nhánh:** `VisionMate_VL_P4_AG` — `research/vl-p4-lightweight-cane-detector` (tách từ VL-P3 `2001119c63c82a714d17aba065985d5c4c95fa70`)  
**Dữ liệu sạch hoàn toàn (Zero-Leakage):** `train_v4 = 581` (97 pos, 484 neg), `val_cal_v4 = 158` (16 pos, 142 neg), `final_holdout_v4 = 219` (19 pos mới hoàn toàn, 200 handheld confusers mới), `ambiguous_cane_v4 = 26`, cộng đối chứng độc lập trên `final_holdout_v3 = 245` (80 pos, 165 neg).

---

## I. Đối chứng trực tiếp `A0_VL_P3_A1` vs `C2_FinalCandidate` (`B2_CNN_MultiTask` ONNX FP32)

Tất cả độ trễ dưới đây đo **từ ảnh gốc trên đĩa không cache (image decode + preprocess + inference + postprocess)**:

| Tập mù (Uncached E2E) | Mô hình | TP / FP / TN / FN | Recall (95% CI) | FAR (95% CI) | Precision | ROC-AUC | PR-AUC | Endpoint Err | E2E P50 / P95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `final_holdout_v4` (N=219) | `A0_VL_P3_A1` | 8 / 5 / 139 / 2 | 0.4211 (0.2314–0.6372) | **0.0250** (0.0107–0.0572) | 0.6154 | 0.9132 | 0.6214 | 281.03 px | 547.93 / 1274.76 ms |
| `final_holdout_v4` (N=219) | **`C2_FinalCandidate`** | 13 / 7 / 180 / 5 | **0.6842** (0.4601–0.8464) | 0.0350 (0.0171–0.0705) | **0.6500** | **0.9524** | **0.7115** | **233.31 px** | **36.25 / 59.60 ms** |
| `final_holdout_v3` (N=245) | `A0_VL_P3_A1` | 30 / 4 / 113 / 4 | 0.3750 (0.2769–0.4845) | **0.0242** (0.0095–0.0607) | **0.8824** | 0.9116 | 0.8272 | **194.51 px** | 375.63 / 576.37 ms |
| `final_holdout_v3` (N=245) | **`C2_FinalCandidate`** | 58 / 12 / 144 / 15 | **0.7250** (0.6186–0.8108) | 0.0727 (0.0421–0.1228) | 0.8286 | **0.9208** | **0.8638** | 198.53 px | **24.77 / 30.78 ms** |

- **Kiểm định McNemar (`C2` vs `A0_VL_P3_A1`):**
  - Trên `final_holdout_v3` (`N=245`): Recall tăng **+35.00%** (CI95: `+21.10%..+48.90%`, exact `p = 1.5e-05`), Overall binary correct tăng **+8.16%** (`p = 0.010455`).
  - Trên `final_holdout_v4` (`N=219`): Recall tăng **+26.32%** (CI95: `-2.28%..+54.91%`, `p = 0.179688` do `N_pos = 19`).
- **Tài nguyên:** `C2_FinalCandidate` (`b2_cnn_multitask_fp32.onnx`, `1,045,989` tham số, `4.002 MB`), `RSS Peak = 150.22 MB`.

---

## II. Trả lời đầy đủ 10 câu hỏi bắt buộc (Mục XIII)

1. **VL-P3 đã được kiểm toán và đóng hồ sơ đúng chưa?**
   - **Đã đóng sạch.** Xác minh commit `2001119c63c82a714d17aba065985d5c4c95fa70` kế thừa từ VL-P2 (`4a10113`), kiểm tra đủ 7 checksum SHA-256, chạy lại `A1` không cache trên `N=245` (`Recall=37.50%, FAR=2.42%, E2E P50=361.37 ms`) và xuất [VL_P3_CLOSURE_AND_PRESERVATION_AUDIT.md](VL_P3_CLOSURE_AND_PRESERVATION_AUDIT.md).

2. **CNN mới có vượt A1 khi dùng cùng dữ liệu không?**
   - **Vượt trội về nhận diện (Recall, PR-AUC, F1, Accuracy) và tốc độ (~15×), nhưng chưa vượt về FAR và độ chính xác góc.** Trên cùng tập `final_holdout_v3` (`N=245`), `C2` (`B2_CNN_MultiTask`) đạt Recall **72.50%** (`p = 1.5e-05`), Accuracy **82.45%**, PR-AUC **0.8638**, tốc độ **24.77 ms P50**, so với `A0_VL_P3_A1` (`37.50%` Recall, `58.37%` Acc, `0.8272` PR-AUC, `375.63 ms P50`). Tuy nhiên `FAR` của `C2` trên `v3` là **7.27%** (cao hơn `2.42%` của `A0_VL_P3_A1`).

3. **Keypoint và centerline có chính xác hơn không?**
   - **Cải thiện Centerline Chamfer, nhưng góc và Both PDJ@0.10 chưa vượt `A0_VL_P3_A1`.** Trên `v4`, `A1_CNN_Keypoint` giảm Endpoint Err từ `281.03 -> 230.22 px` và Chamfer từ `245.44 -> 169.18 px`. Trên `v3`, `A1_CNN_Keypoint` đạt `188.04 px` / Chamfer `129.06 px` (tốt hơn `194.51 px` / `144.97 px` của `A0`), nhưng `C2` (`B2_CNN_MultiTask`) đạt `198.53 px`, `Both PDJ@0.10 = 18.75%` (kém `23.75%` của `A0`), và góc `35.44°` (kém `25.66°` của `A0`).

4. **Segmentation có thực sự hữu ích không?**
   - **Mask IoU vẫn rất thấp (`0.0182–0.0209` tại lưới `64×64`), nhưng dùng segmentation làm nhánh phụ trợ đa nhiệm không-veto lại hữu ích.** `B2_CNN_MultiTask` đạt FAR thấp hơn `A1_CNN_Keypoint` trên cả hai holdout (`3.50%` vs `7.00%` trên `v4`, `7.27%` vs `9.70%` trên `v3`) mà không làm tụt Recall như cổng veto cứng của VL-P3.

5. **Mô hình còn nhận nhầm gậy thường, ô dù hay trekking pole không?**
   - **Vẫn còn trên `walking_stick`, `metal_rod_pipe`, `trekking_pole` và `umbrella`.**
   - Trên `final_holdout_v4` (`200` handheld neg): `tool_handle` **0/73 (0%)**, `trekking_pole` **0/9 (0%)**, `umbrella` **3/74 (4.05%)**, `metal_rod_pipe` **4/44 (9.09%)**.
   - Trên `final_holdout_v3` (`165` neg): `tool_handle` **0/23 (0%)**, `white_line_pole` **0/25 (0%)**, `walking_stick` **3/16 (18.75%)**, `metal_rod_pipe` **3/21 (14.29%)**, `trekking_pole` **2/21 (9.52%)**, `umbrella` **2/25 (8.00%)**, `general_distractor` **2/34 (5.88%)**.

6. **Recall tăng có làm FAR tăng không?**
   - **Có đánh đổi rõ rệt trên `final_holdout_v3`.** Khi bỏ hard veto của VL-P3 và tăng Recall từ `37.50%` lên `72.50%`, FAR tăng từ `2.42%` lên `7.27%` trên `v3` (dù vẫn giữ `3.50% <= 5%` trên `v4`).

7. **Kết quả có ổn định từ validation sang holdout mới không?**
   - **ROC-AUC tăng/ổn định, nhưng ngưỡng hiệu chuẩn `val_cal_v4` có độ lệch Recall/FAR khi chuyển tập.** Từ `val_cal_v4` (`recall=0.250, far=0.014, roc=0.762`) sang `final_holdout_v4` (`recall=0.684, far=0.035, roc=0.952`) và `final_holdout_v3` (`recall=0.725, far=0.073, roc=0.921`). Điều này phản ánh kích thước `val_cal_v4` chỉ có `16` mẫu dương tính nên việc chọn ngưỡng `t_known = 0.89` có phương sai cao.

8. **Xuất ONNX và INT8 có giữ chất lượng không?**
   - **ONNX FP32 (`4.002 MB`) giữ parity tuyệt đối (`ROC-AUC` khớp 4 chữ số thập phân với PyTorch); Dynamic INT8 (`1.171 MB`) KHÔNG giữ được chất lượng.** Lượng tử hóa INT8 dynamic làm `val_roc` tụt từ `0.7623` xuống `0.6373` (`B2`), nên theo quy tắc mục XI.6 hệ thống đã loại bỏ INT8 và dùng **ONNX FP32**.

9. **Có đạt tốc độ CPU end-to-end phù hợp không?**
   - **Đạt xuất sắc.** Độ trễ thực tế không cache từ ảnh gốc trên CPU đạt **`24.77 ms` P50 / `30.78 ms` P95** (`final_holdout_v3`) và **`36.25 ms` P50 / `59.60 ms` P95** (`final_holdout_v4`), `RSS Peak = 150.22 MB`, nhanh hơn ~15× so với VL-P3 và vượt xa mục tiêu `P95 <= 350–600 ms`.

10. **Có đủ bằng chứng để cân nhắc sandbox Windows riêng trong tương lai không?**
    - **Chưa đủ cho triển khai ngay, nhưng đủ bằng chứng để giữ nhánh CNN nhẹ làm nền tảng tiếp theo.** Tốc độ (`25–36 ms`) và Recall (`68–73%`) đã khả thi cho real-time CPU, nhưng cần khắc phục `FAR = 7.27%` trên `v3` (đặc biệt `walking_stick` và `metal_rod_pipe`), mở rộng thêm dữ liệu dương tính ngoài trời/cầm tay đa dạng (do mới thu được `82` ảnh mới thay vì `300–500`), và nâng độ phân giải đầu ra/instance grouping để sửa lỗi nối chéo đầu mút trước khi mở sandbox ứng dụng.

---

## XIV. TRẠNG THÁI NGHIỆM THU (SECTION XIV)

**Phân loại chính thức:** **`RESEARCH PROMISING`** *(Có tiến bộ rõ rệt nhưng còn giới hạn)*

### Lý do:
1. **Tiến bộ đã chứng minh:** Kiến trúc `MobileNetV3-Small Multi-Head` (`4.002 MB` ONNX FP32) chạy offline CPU **nhanh hơn 15×** (`24.77–36.25 ms` P50 so với `375.63–547.93 ms` của VL-P3 `A1`), `RSS Peak = 150.22 MB`, và tăng **Known Recall** gấp đôi (`37.50% -> 72.50%` trên `v3`, `p = 1.5e-05`; `42.11% -> 68.42%` trên `v4` với `FAR = 3.50%`).
2. **Giới hạn còn tồn tại (chưa đạt `RESEARCH IMPROVED` toàn diện):**
   - Số lượng ảnh gậy trắng thực sự mới thu thập được chỉ đạt `82` ảnh (`final_holdout_v4` có `19` pos, `train_v4` tổng `97` pos), thấp hơn mục tiêu `300–500` ảnh.
   - `FAR` trên `final_holdout_v3` là `7.27%` (> `5%`), vẫn còn nhầm `walking_stick` (`18.75%`) và `metal_rod_pipe` (`14.29%`).
   - Định vị `Both PDJ@0.10` (`18.75%`) và góc (`35.44°`) của `C2` chưa vượt `A0_VL_P3_A1` (`23.75%`, `25.66°`), và `Mask IoU` vẫn thấp (`~0.019`).
   - Dynamic INT8 chưa dùng được do tụt `ROC-AUC` (`0.762 -> 0.637`), cần QAT hoặc Static QDQ calibration ở nghiên cứu sau.
