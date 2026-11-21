# VISIONMATE GLASSES — VL-P1c FINAL RESEARCH REPORT
## Calibration & Multi-Proposal Reliability Research (Offline CPU-First DINOv2-Small INT8 + YOLOX-Nano)

**Đơn vị thực hiện:** Google Antigravity  
**Nhánh & worktree nghiên cứu:** `research/vl-p1c-calibration-multiproposal` (`VisionMate_VL_P1c_AG`, base commit `dacd245`)  
**Trạng thái đề xuất phân loại nghiệm thu (Section XII):**  
- **`TECHNICAL PASS`** (pipeline, 20/20 tests, dữ liệu zero-overlap, khóa cứng 2 pha và baseline Git hoàn toàn đạt)  
- **`RESEARCH PROMISING`** đối với nâng cấp **hiệu chuẩn chọn lọc (`C_v1c_cal_only` & `R5`)** giúp cắt một nửa FAR và nâng Accuracy tổng thể trên tập mới  
- **`RESEARCH NOT READY`** đối với tích hợp toàn pipeline (`C_v1c_full__k4` bị sụt Recall xuống `50.0%` và gậy trắng `object_01` vẫn ở mức `2/20` E2E) — **Chưa đưa vào production hay thử nghiệm webcam đầy đủ 5 lớp**

---

## 1. Tóm tắt Điều hành (Executive Summary)

VL-P1c được triển khai nhằm khắc phục trực diện bốn nguyên nhân khiến `VL-P1b` (`dacd245`) thất bại trên tập kiểm thử độc lập, mà vẫn giữ nguyên mô hình **DINOv2-Small INT8 (`dinov2_small_quantized.onnx`)** và **YOLOX-Nano (`yolox_nano.onnx`)** trên CPU (`num_threads=4`, offline 100%):
1. Hiệu chuẩn ngưỡng từng lớp (`tau_per_class`) bị sập trên cỡ mẫu nhỏ `N=50`.
2. Bùng nổ báo giả (FP) khi đánh giá đa proposal/crop (`26.0%` FAR trên `final_holdout_full` của `C4`, `33.3%` trên `vl_p1b_holdout`).
3. `ProposalRanker` của VL-P1b bị `geom_thin_line` chiếm **`54.0%`** khe ở `K=4`, làm mất các hộp chứa toàn đồ vật.
4. Định vị gậy trắng (`object_01`) trong ảnh có người, nền phức tạp.

**Kết quả trung thực từ Pha B — đánh giá mù 1 lần duy nhất trên `final_holdout` (`N=300` ảnh mới hoàn toàn: `100` dương `20/lớp` + `200` âm gồm `112` hard-negative + `88` `general_distractor`) sau khi khóa cứng `frozen_config.json` trên `calibration_v1c` (`N=200`):**

- **Hiệu chuẩn & Giảm FAR thành công rõ rệt (ROI & E2E):**
  - Nhờ `delta_floor=0.08`, group-aware 5-fold CV và OOF Ridge verifier trên `calibration_v1c` (`N=200`), ngưỡng đóng băng hội tụ ổn định tại `0.4639..0.5006` (độ lệch chuẩn xuyên fold `<= 0.0236`).
  - Trên `final_holdout_roi` (`N=300`), **`R5_selective_3state_v1c`** giảm **FAR xuống `10/200 (5.0%, 95% CI [2.74, 8.96])`** so với `R0` **`16/200 (8.0%)`** (`McNemar neg p = 0.03125`) và `A3_vl_p1b_ref` **`17/200 (8.5%)`** (`p = 0.039062`), đạt **`0/88 (0.0%)` FP trên `general_distractor`** và **`0/100` wrong class**, với Overall Accuracy **`253/300 (84.3%)`**.
  - Trên `final_holdout_full` (`N=300`), **`C_v1c_full__k4`** cắt **FAR xuống `8/200 (4.0%, 95% CI [2.04, 7.69])`** (và **`3.5% = 7/200`** ở `K=6, 8`), giảm hơn 6 lần so với `C4_vl_p1b_combined` **`52/200 (26.0%)`** (`p = 0.0`) và giảm hơn một nửa so với `C1_vl_p1_baseline` **`21/200 (10.5%)`** (`p = 0.000244`).
- **Đánh đổi Recall trên `C_v1c_full` & Thành công của cấu hình trung gian `C_v1c_cal_only`:**
  - Việc cộng dồn nhiều lớp phạt trên `C_v1c_full__k4` (`R5` + `e2e_tau_offset=0.03` + `source_offsets` + `singleton_penalty`) đẩy `18/100` vật thật sang trạng thái `AMBIGUOUS`, khiến Known Recall của `C_v1c_full__k4` **sụt xuống `50/100 (50.0%, [40.38, 59.62])`** (kém `C1` `64/100`, `McNemar pos p = 0.001312`) với tổng Accuracy **`242/300 (80.7%)`** (trễ `360.13 ms` p50).
  - Ngược lại, **`C_v1c_cal_only` (`B0@4 + R5`) đạt điểm cân bằng cao nhất: `251/300 (83.7%, [79.06, 87.42])` Overall Accuracy**, Known Recall **`62/100 (62.0%)`**, FAR **`11/200 (5.5%)`**, `p50 = 176.20 ms` (`p95 = 218.69 ms`), vượt `C1` (`81.0%` Acc, `10.5%` FAR).
- **`ProposalRankerV1c` giải quyết tốt nghẽn khe ở `K=4`, nhưng gậy trắng `object_01` vẫn là rào cản kiến trúc:**
  - Ở cùng `K=4`, `B_v1c_proposer__k4` đạt **`73/100 (73.0%)` Recall@0.5** (vượt `B_full_v1b__k4` `58/100` tới `+15 điểm %` và vượt `B0__k4` `71/100`), sửa triệt để lỗi `F_thin` chiếm `54.0%` slot.
  - Tuy `B_v1c_proposer` nâng tỷ lệ bao phủ hai đầu gậy (`endpoint_inclusion_rate`) lên **`13/20 (65.0%)`** ở `K=8` (`5/5` ở ảnh người cầm gậy), hộp trục ngang/dọc (axis-aligned bbox) của gậy chéo vẫn pha tạp **`70.88–75.73%` nền** (`mean_bg_contamination`), làm `object_01` Recall@0.5 dừng ở **`1/20 (5.0%)`** và nhận diện E2E đạt **`2/20 (10.0%)` (`K=4`) / `3/20 (15.0%)` (`K=6, 8`)**.

---

## 2. Xác minh Tập dữ liệu, Chống Rò Rỉ & Khóa Cứng 2 Pha

- **Manifest (`manifests/dataset_manifest.json`, SHA-256 `b6f4c06fa506afaea622a7a87db3bf5ca65b653834f3137d5cf9799e75c3df52`):**
  - `train_pos`: `25` (`5/lớp`); `hard_neg_gallery`: `50` (`7` nhóm confuser); `calibration_v1c`: **`200`** (`75` pos `15/lớp` + `125` neg = `150` mới + `50` kế thừa `cal_v1b`); `final_holdout`: **`300`** (`100` pos `20/lớp` + `200` neg gồm `16/nhóm` × `7` hard-neg + `88` `general_distractor`, **`300/300` mới hoàn toàn**); `ambiguous_boundary_v1c`: **`15`**.
  - Kiểm tra chống rò rỉ 4 tầng (`SHA-256`, `norm_title`, `cluster_id`, `aHash/dHash <= 6`): **`zero_overlap_verified = true`**, `cross_hits = 0` giữa 14 nhóm và với mọi dữ liệu `vl_p1`/`vl_p1b`. Toàn bộ `69` hộp điều chỉnh GT (`manual_gt_bboxes_v1c.json`) được kiểm tra trực quan qua 6 contact sheets `gt_verify_*.jpg`.
- **Khóa Pha A (`manifests/frozen_config.json`, SHA-256 `frozen_a3_oof_state`: `bf3b7191...`):**
  - `cv_gate_passed = true`, `final_holdout_touched_during_calibration = false`, `selected_delta_floor = 0.08`, `frozen_e2e_k = 4` (`152/200 = 76.0%` trên `calibration_v1c_full`, `p50 = 231.80 ms` vs `K=6` `74.0%`, `K=8` `73.0%`).
  - Tham số `R5` đóng băng: `tau_per_class = {object_01: 0.5006, 02: 0.4774, 03: 0.4652, 04: 0.4639, 05: 0.4878}`, `ridge_alpha = 24.0`, `e2e_tau_offset = 0.03`, `alpha_k = 0.02`.

---

## 3. Tổng hợp Kết quả Định lượng (95% Wilson CI)

### 3.1 Phương pháp A — ROI Calibration (`final_holdout_roi`, `N=300` & tập kế thừa)

| Mode ROI | `final_holdout_roi` Acc (`N=300`) | Known Recall | FAR | Amb Rate | `vl_p1_test` Acc/Rec/FAR (`N=75`) | `vl_p1b_holdout` Acc/Rec/FAR (`N=214`) |
|---|---:|---:|---:|---:|---|---|
| `R0_vl_p1_baseline` | `252/300 (84.0%, [79.43, 87.71])` | `68/100 (68.0%)` | `16/200 (8.0%)` | `0.0%` | `81.3% / 78.0% / 12.0%` | `80.8% / 67.0% / 7.0%` |
| `R1_global_conservative` | `239/300 (79.7%, [74.75, 83.83])` | `48/100 (48.0%)` | `9/200 (4.5%)` | `0.0%` | — | — |
| `R2_per_class_floored` | `250/300 (83.3%, [78.70, 87.12])` | `62/100 (62.0%)` | `12/200 (6.0%)` | `0.0%` | — | — |
| `R3_distributional_quantile` | **`254/300 (84.7%, [80.15, 88.30])`** | `65/100 (65.0%)` | `11/200 (5.5%)` | `0.0%` | — | — |
| `R4_out_of_fold_verifier` | `253/300 (84.3%, [79.79, 88.01])` | `63/100 (63.0%)` | `10/200 (5.0%)` | `0.0%` | `80.0% / 72.0% / 4.0%` | `79.9% / 62.0% / 4.4%` |
| **`R5_selective_3state_v1c`** | **`253/300 (84.3%, [79.79, 88.01])`** | `63/100 (63.0%)` | **`10/200 (5.0%)`** | `5.0%` (`9+6`) | `80.0% / 72.0% / 4.0%` | `79.9% / 62.0% / 4.4%` |
| `A1_vl_p1b_ref` | `218/300 (72.7%, [67.36, 77.40])` | `21/100 (21.0%)` | `3/200 (1.5%)` | `0.0%` | `64.0% / 46.0% / 0.0%` | `68.7% / 33.0% / 0.0%` |
| `A3_vl_p1b_ref` | `256/300 (85.3%, [80.88, 88.89])` | `73/100 (73.0%)` | `17/200 (8.5%)` | `0.0%` | `80.0% / 80.0% / 20.0%` | `78.5% / 74.0% / 17.5%` |

*(Trong `calibration_metrics.json`, 5-fold CV `calibration_v1c` cho `R5`: Rec `60.0 ± 5.97%`, FAR `7.2 ± 2.99%`; ở `N_cal = 50 -> 200`, FAR ngoài mẫu giảm từ `9.84%` xuống `4.80%` và `min_tau` ổn định ở `0.4639`.)*

### 3.2 Phương pháp B — Ma trận Công bằng `3×3` Top-`K` (`final_holdout_full`, `N=100` dương, Không GT lúc suy luận)

Chi tiết đầy đủ lưu tại [proposal_ablation.md](proposal_ablation.md). Tóm tắt các điểm chốt trên `final_holdout_full`:
- `yolox_strict__k4`: R@0.3 `26/100 (26.0%)`, R@0.5 `19/100 (19.0%)`, `object_01` R@0.5 `0/20`, `p50 = 15.31 ms`.
- `yolox_low__k4`: R@0.3 `60/100 (60.0%)`, R@0.5 `42/100 (42.0%)`, `object_01` R@0.5 `0/20`, `p50 = 16.24 ms`.
- `B0_vl_p1_proposer` (`K=4 / 6 / 8`): R@0.5 `71% / 75% / 75%`, R@0.3 `80% / 81% / 81%`, `object_01` R@0.5 `1/20`, `endpoint = 10/20 -> 7/20`, `p50 = 20.40..21.21 ms`.
- `B_full_v1b` (`K=4 / 6 / 8`): R@0.5 `58% / 66% / 72%`, R@0.3 `77% / 80% / 86%`, `F_thin` ở `K=4` chiếm `54.0%` slot, `object_01` R@0.5 `1/20 (@4,6) -> 2/20 (@8)`, `p50 = 43.71..44.96 ms`.
- **`B_v1c_proposer` (`K=4 / 6 / 8`):** R@0.5 **`73% / 73% / 73%`**, R@0.3 **`81% / 83% / 85%`**, `mIoU = 0.6188 / 0.6286 / 0.6588`, phân bổ slot ở `K=4` = `F_primary 45.0%, F_context 27.3%, F_thin 24.8%, F_small 3.0%`; gậy trắng R@0.3 `4/20 -> 5/20 -> 7/20`, R@0.5 `1/20`, **`endpoint_inclusion_rate = 10/20 (@4) -> 11/20 (@6) -> 13/20 (65.0%, @8)`** (`5/5` ở `handheld_person_subset`), `mean_bg_contamination = 0.7573 (@4) -> 0.7088 (@8)`, `p50 = 62.54..65.02 ms`.

### 3.3 Thí nghiệm End-to-End (`final_holdout_full`, `N=300`) & Kiểm định McNemar Ghép cặp

| Config E2E | Acc (`N=300`) | Known Recall | FAR | Amb Rate | `object_01` Rec | E2E `IoU>=0.5` | `vl_p1b_holdout` Acc/Rec/FAR | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `C1_vl_p1_baseline` (`B0+A0`) | `243/300 (81.0%, [76.18, 85.04])` | `64/100 (64.0%)` | `21/200 (10.5%)` | `0.0%` | `2/20` | `41/100` | `75.7% / 63.0% / 13.2%` | `158.13 / 197.88 ms` |
| `C4_vl_p1b_combined` (`B_v1b+A3`) | `215/300 (71.7%, [66.32, 76.47])` | **`67/100 (67.0%)`** | `52/200 (26.0%)` | `0.33%` | **`4/20`** | `25/100` | `66.4% / 66.0% / 33.3%` | `204.99 / 251.27 ms` |
| **`C_v1c_cal_only` (`B0+R5`)** | **`251/300 (83.7%, [79.06, 87.42])`** | `62/100 (62.0%)` | `11/200 (5.5%)` | `6.33%` | `2/20` | `37/100` | **`77.1% / 60.0% / 7.9%`** | **`176.20 / 218.69 ms`** |
| `C_v1c_prop_only` (`B_v1c+A0`) | `237/300 (79.0%, [74.04, 83.23])` | `67/100 (67.0%)` | `30/200 (15.0%)` | `0.0%` | `3/20` | **`42/100`** | `73.8% / 60.0% / 14.0%` | `236.16 / 300.22 ms` |
| `C_v1c_full__k4` (`B_v1c+R5+MP`) | `242/300 (80.7%, [75.82, 84.74])` | `50/100 (50.0%)` | **`8/200 (4.0%)`** | `12.0%` | `2/20` | `26/100` | `73.8% / 49.0% / 4.4%` | `360.13 / 492.01 ms` |
| `C_v1c_full__k6` | `242/300 (80.7%, [75.82, 84.74])` | `49/100 (49.0%)` | `7/200 (3.5%)` | `17.0%` | `3/20` | `24/100` | `72.0% / 43.0% / 2.6%` | `313.81 / 403.65 ms` |
| `C_v1c_full__k8` | `242/300 (80.7%, [75.82, 84.74])` | `49/100 (49.0%)` | `7/200 (3.5%)` | `19.33%` | `3/20` | `22/100` | `71.5% / 43.0% / 3.5%` | `372.41 / 469.91 ms` |

- **Kiểm định McNemar trên `final_holdout_full` (`N=300`):**
  - `C_v1c_full__k4` vs `C4_vl_p1b_combined`: Toàn tập `+9.0%` (`p = 0.00109`), Âm tính **`+22.0%` (`p = 0.0`, thắng `45` thua `1`)**, Dương tính **`-17.0%` (`p = 7.6e-5`)**.
  - `C_v1c_full__k4` vs `C1_vl_p1_baseline`: Toàn tập `-0.33%` (`p = 1.0`), Âm tính **`+6.5%` (`p = 0.000244`, thắng `13` thua `0`)**, Dương tính **`-14.0%` (`p = 0.001312`)**.
- **Tập ranh giới `ambiguous_boundary_v1c` (`N=15`):** `C_v1c_full__k4` trả **`5 KNOWN / 5 AMBIGUOUS / 5 UNKNOWN`** (ROI `6/3/6`), trong khi `C1` (`10/0/5`) và `C4` (`12/0/3`) đoán cưỡng bức `0 AMBIGUOUS`.

---

## 4. Trả lời Trực Diện 10 Câu Hỏi Nghiên Cứu Bắt Buộc (Section XI)

1. **Vì sao ngưỡng VL-P1b thất bại và VL-P1c đã khắc phục được đến đâu?**  
   Ngưỡng `A3` của VL-P1b bị hạ xuống `0.0757..0.1826` do fit trực tiếp trên `N=50` mẫu calibration không sàn, cộng với `F_thin` chiếm chỗ và không bù trừ đa crop. VL-P1c khắc phục triệt để vấn đề sập ngưỡng nhờ `calibration_v1c` (`N=200`), 5-fold CV theo `cluster_id` với `delta_floor=0.08` (`tau_per_class = 0.4639..0.5006`), OOF verifier và nguồn bù trừ đa crop, giúp giảm FAR ROI trên `vl_p1b_holdout` từ `17.5%` xuống `4.4%` và trên `final_holdout_roi` xuống `5.0%`. Tuy nhiên VL-P1c lại phát sinh thái cực ngược lại ở E2E đầy đủ: chồng nhiều lớp phạt làm `AMBIGUOUS` tăng lên `12.0%` (`K=4`)..`19.33%` (`K=8`).
2. **FAR có giảm thật trên tập mới không?**  
   **Có, giảm rõ rệt và có ý nghĩa thống kê.** Trên `final_holdout` (`N=300`): ROI FAR giảm từ `8.0%` (`R0`) / `8.5%` (`A3_v1b`) xuống **`5.0%` (`R5`, `p <= 0.039`)** với **`0/88` FP** trên `general_distractor`; E2E FAR giảm từ `10.5%` (`C1`) và `26.0%` (`C4`) xuống **`4.0%` (`C_v1c_full__k4`, `p <= 0.000244`)** và **`5.5%` (`C_v1c_cal_only`)**.
3. **Recall có được bảo toàn không?**  
   **Bảo toàn tương đối ở ROI (`63.0%` `R5` vs `68.0%` `R0`, `p = 0.125`) và ở `C_v1c_cal_only` (`62.0%` vs `64.0%` `C1`), nhưng KHÔNG bảo toàn ở `C_v1c_full__k4` (`50.0%` vs `64.0%` `C1`, `p = 0.001312`).** Chế độ double-gating khiến `18/100` mẫu dương thật bị gán `AMBIGUOUS` thay vì `KNOWN`.
4. **Tăng K có khiến false positive tăng không?**  
   **Không.** Nhờ cơ chế `alpha_k = 0.02`, `source_offsets` và kiểm tra hỗ trợ đa crop, khi tăng `K = 4 -> 6 -> 8`, FAR trên `final_holdout_full` **không tăng mà giảm nhẹ từ `4.0%` (`8/200`) xuống `3.5%` (`7/200`)** (và trên `vl_p1_test_full` giảm từ `4.0%` xuống `0.0%`). Nhưng tăng `K` làm `AMBIGUOUS` tăng `12.0% -> 17.0% -> 19.33%` và `Recall` nhích xuống `50% -> 49%`.
5. **ProposalRanker mới có tốt hơn ở cùng ngân sách K không?**  
   **Có ở `K=4`.** Trên `final_holdout_full`, `B_v1c_proposer__k4` đạt **`73/100 (73.0%)` Recall@0.5** vs `58/100 (58.0%)` của `B_full_v1b__k4` (`+15 điểm %`) và `71/100 (71.0%)` của `B0__k4`, chấm dứt tình trạng `F_thin` chiếm `54.0%` khe (`B_v1c` giữ cân bằng `F_primary=45.0%, F_context=27.3%, F_thin=24.8%, F_small=3.0%`).
6. **Gậy trắng có được khoanh vùng và nhận diện chính xác hơn không?**  
   **Tỷ lệ bao phủ hình học tăng (`endpoint_inclusion_rate` đạt `13/20 = 65.0%` @ `K=8` vs `10/20 = 50.0%` ở `B0`, đạt `5/5` ở ảnh có người cầm), nhưng định vị chặt `IoU>=0.5` (`1/20 = 5.0%`) và nhận diện E2E (`2/20 = 10.0%` @ `K=4`, `3/20 = 15.0%` @ `K=6..8`) VẪN RẤT YẾU.** Lý do căn bản là gậy trắng chéo mảnh nằm trong hộp trục ngang/dọc kéo theo `>70%` nền, làm embedding DINOv2 bị nhiễu nền vỉa hè/người.
7. **Chi phí CPU/RAM có phù hợp laptop i5-1135G7 RAM 8 GB không?**  
   **RAM đạt hoàn toàn** (`rss_loaded = 249.88 MB`, `private_bytes = 555.85 MB`, tổng DB `359,534 B`). Về độ trễ, `C_v1c_cal_only` rất phù hợp (**`176.20 ms` p50, `218.69 ms` p95**), trong khi `C_v1c_full__k4` tốn **`360.13 ms` p50 (`492.01 ms` p95)** — chấp nhận được ở mức chụp theo yêu cầu nhưng chưa đủ nhanh cho stream thời gian thực liên tục.
8. **Pipeline mới có vượt C1 baseline trên cùng tập kiểm thử hay không?**  
   - `C_v1c_full__k4` **KHÔNG vượt `C1`** xét toàn cục (`80.7%` Acc, `50.0%` Rec, `4.0%` FAR, `360.13 ms` vs `C1` `81.0%` Acc, `64.0%` Rec, `10.5%` FAR, `158.13 ms`; `McNemar all p = 1.0`).  
   - Tuy nhiên, nhánh tách biệt **`C_v1c_cal_only` (`B0 + R5`) ĐÃ VƯỢT `C1` về Overall Accuracy (`83.7%` vs `81.0%`) và giảm gần một nửa FAR (`5.5%` vs `10.5%`)** với tốc độ tương đương (`176.20 ms` p50).
9. **Phương án nào có cân bằng chất lượng tốt nhất?**  
   **`C_v1c_cal_only` (`B0@4 + R5`)** là cấu hình cân bằng nhất (`83.7%` Acc, `62.0%` Recall, `5.5%` FAR, `176.20 ms` p50, `Wrong class = 1/100`), vượt trội cho 4 lớp vật thể còn lại (`object_02` 16/20, `03` 14/20, `04` 13/20, `05` 17/20), song vẫn thất bại ở `object_01` (`2/20`).
10. **Có đủ bằng chứng để tiến tới thử nghiệm webcam độc lập không?**  
   **CHƯA ĐỦ cho hệ thống 5 lớp tổng thể (`RESEARCH NOT READY` để mở webcam test với `C_v1c_full` hay `object_01`).** Với bài toán gậy trắng, không nên tiếp tục kéo dài vòng hiệu chuẩn ngưỡng trên kiến trúc crop trục ngang/dọc + DINOv2 global pooling; thay vào đó nên chuyển sang **bộ phát hiện hướng/đoạn thẳng hoặc phân đoạn nhẹ (rotated box / keypoint / fine-tuned small detector cho gậy trắng)** cho `object_01`, trong khi chỉ dùng logic hiệu chuẩn `R5` / `C_v1c_cal_only` làm nền tham khảo cho các vật thể gọn.

---

## 5. Kết quả Kiểm thử Tự động & Bảo toàn Baseline

- **`VisionMate_VL_P1c_AG/research/vl_p1c` (`python -m pytest -v`):** **`20 passed in 32.23s`** (`0 failed`, `0 skipped`).
- **`VisionMate_VL_P1b_AG/research/vl_p1b` (`python -m pytest -v`):** **`10 passed in 12.56s`** (`0 failed`, `0 skipped`, HEAD `dacd245` clean).
- **`VisionMate_VL_P1_AG/research/vl_p1` (`python -m pytest -v`):** **`34 passed in 22.71s`** (`0 failed`, `0 skipped`, HEAD `988fdf1` clean).
- **`VisionMateGlasses` (`65c2a27`):** Clean (`0` sửa đổi production/OCR/TTS/UI/Webcam). Chi tiết tại [reports/git_baseline_preservation.md](reports/git_baseline_preservation.md).
