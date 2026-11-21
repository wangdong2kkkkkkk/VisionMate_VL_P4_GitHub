# VISIONMATE GLASSES — VL-P1b FINAL RESEARCH REPORT
## Recognition Reliability & Object Proposal Research (Offline CPU-First DINOv2-Small INT8 + YOLOX-Nano)

**Đơn vị thực hiện:** Google Antigravity  
**Loại nhiệm vụ:** Independent Research Pilot (`research/vl-p1b-reliability-proposal`, base commit `988fdf1`)  
**Kết luận tổng quát:** **CHƯA ĐẠT NGƯỠNG TÍCH HỢP PRODUCTION — NGHIÊN CỨU MỞ RỘNG CẦN VÒNG HIỆU CHUẨN MỚI (`NOT READY FOR PRODUCTION`)**

---

## 1. Tóm tắt điều hành (Executive Summary)

VL-P1b được triển khai độc lập trên worktree `VisionMate_VL_P1b_AG` (nhánh `research/vl-p1b-reliability-proposal`) nhằm giải quyết hai bài toán còn tồn đọng từ VL-P1 (`988fdf1`):
1. **Bài toán A (False Positive Recognition):** Giảm nhận nhầm các vật gây nhiễu hình thức tương đồng (`glasses_case_confuser`, `notebook_pouch_confuser`, `walking_stick_confuser`, `long_metal_rod_confuser`, `lock_key_metal_confuser`, `footwear_sock_confuser`, `umbrella_handle_confuser`) và nhiễu tổng quát (`general_distractor`) trên nền **DINOv2-Small INT8 (`dinov2_small_quantized.onnx`)** giữ nguyên không huấn luyện lại backbone.
2. **Bài toán B (Object Proposal Localization):** Cải thiện vùng đề xuất trước nhận dạng (đặc biệt với gậy trắng dài/mảnh `object_01`, chìa khóa nhỏ `object_02`, và vật đang cầm trên tay người) mà **tuyệt đối không dùng Ground-Truth bbox lúc suy luận**.

**Kết quả thực nghiệm trung thực trên tập Holdout độc lập (`N=214` = `100` mẫu dương 20/lớp + `114` mẫu âm, đóng băng ngưỡng hoàn toàn từ `calibration_v1b` `N=50` trước khi chạm `holdout`, KHÔNG tinh chỉnh hậu kiểm):**
- **Về Bài toán B (Proposal):** Khi tăng ngân sách crop lên `K=6..8`, `B_full_v1b__k8` đạt **Proposal Recall@0.5 cao nhất trên cả hai tập**: **`74/100 (74.0%, 95% CI [64.63, 81.60])`** trên `holdout_full` (vượt `B0_vl_p1_proposer__k4` **`68/100 = 68.0%`** và `yolox_strict__k4` **`17/100 = 17.0%`**; riêng `object_02` đạt **`20/20`**, `object_01` cải thiện từ `0/20` lên `2/20`) và đạt **`45/50 (90.0%, 95% CI [78.64, 95.65])`** trên `vl_p1_test_full` (so với `43/50 = 86.0%` của `B0`). Tuy nhiên tại `K=4` (`B_full_v1b__k4`: `60/100 = 60.0%`), các hộp mảnh `geom_thin_line` chiếm chỗ trong `ProposalRanker`, đẩy các hộp bao phủ vật thể lớn ra khỏi Top-4.
- **Về Bài toán A (ROI Hard-Negative Verification):** Trên tập `calibration_v1b` (`N=50`), `A3_full_verifier` giảm FAR từ `6/25 (24.0%)` xuống `2/25 (8.0%)` và tăng Recall từ `15/25 (60.0%)` lên `19/25 (76.0%)`. Tuy nhiên trên `holdout_roi` (`N=214`), `A3_full_verifier` dù **tăng Known Recall** từ `67/100 (67.0%)` (`A0`) lên **`74/100 (74.0%, 95% CI [64.63, 81.60])`** và **giảm lỗi nhóm sổ tay/túi da** (`notebook_pouch_confuser`: `3/7 -> 1/7`), lại có **FAR tăng từ `8/114 (7.0%)` (`A0`) lên `20/114 (17.5%, 95% CI [11.65, 25.55])`** do ngưỡng hiệu chuẩn per-class bị hạ quá thấp trên cỡ mẫu `calibration_v1b` nhỏ (`object_02 = 0.16`, `object_01 = 0.1704`, gây `13` FP ở `object_02` và `5` FP ở `object_01`). Ngược lại, `A1_neg_prototypes` (giữ `tau_global=0.5437` + trừ phản lực `neg_sim`) đạt **FAR `0/114 (0.0%, 95% CI [0.00, 3.26])`** trên `holdout_roi` và **`0/25 (0.0%)`** trên `vl_p1_test`, nhưng khiến Recall tụt xuống `33/100 (33.0%)`.
- **Về Pipeline Kết hợp E2E (`C1..C4` ở `K=4`):** Trên `holdout_full`, `C4_combined_v1b` đạt Known Recall `66/100 (66.0%)` (nhỉnh hơn `C1_vl_p1_baseline` `63/100 = 63.0%`), nhưng do kết hợp `geom_thin_line` chiếm ưu thế (`125/214` lựa chọn) với ngưỡng thấp `tau_per_class` từ `A3` và `e2e_tau_offset=0.0`, FAR của `C4_combined_v1b` tăng lên `38/114 (33.3%)` so với `C1_vl_p1_baseline` (`15/114 = 13.2%`), khiến Overall Accuracy giảm từ `162/214 (75.7%)` xuống `142/214 (66.4%)`.
- **Hiệu năng phần cứng & Mở rộng Offline:** RSS khởi đầu `50.55 MB`, khi nạp đủ YOLOX + DINOv2-Small INT8 đạt `113.47 MB`, cuối phiên `86.33 MB`, Private Bytes `712.22 MB` (dưới xa ngưỡng 2 GB); chi phí matching tuyến tính khi mở rộng gallery âm `N=20 / 50 / 100` chỉ tốn **`0.0078 / 0.0101 / 0.0108 ms`** (p50) và tổng RAM ma trận âm `307 KB` tại `N=100`.

---

## 2. Xác minh Tập dữ liệu & Quy trình Đóng băng Ngưỡng

### 2.1 Tập dữ liệu VL-P1b (`manifests/dataset_manifest.json`, SHA-256 prefix `f3976745bc358d73`)
Mọi ảnh mới đều tải trực tiếp từ Wikimedia Commons với giấy phép mở hợp lệ (`CC0`, `Public domain`, `CC BY`, `CC BY-SA`), được gán `roi_bbox_xywh` có kiểm định trực quan (82 hộp hiệu chỉnh thủ công tại `manifests/manual_gt_bboxes.json` + kiểm tra 5 contact sheet `gt_verify_*.jpg`) và vượt qua cổng kiểm tra chống rò rỉ (`zero_overlap_verified: true`: **0** trùng `SHA-256`, **0** trùng `norm_title`, **0** trùng `cluster_id` phiên chụp, **0** va chạm `aHash/dHash <= 6` giữa VL-P1 và giữa mọi split VL-P1b):

- **`train_pos`:** `25` ảnh (`5` ảnh/lớp x `5` lớp, giữ nguyên từ VL-P1).
- **`hard_neg_gallery`:** `50` ảnh across 7 nhóm confuser (`footwear_sock`: 6, `glasses_case`: 9, `lock_key_metal`: 11, `long_metal_rod`: 6, `notebook_pouch`: 6, `umbrella_handle`: 6, `walking_stick`: 6).
- **`calibration_v1b` (ROI & Full):** `50` ảnh (`25` dương = `5/lớp` + `25` hard-neg phân bổ đều 7 nhóm: 3–4/nhóm).
- **`holdout` (ROI & Full):** `214` ảnh (`100` dương = `20/lớp` cho `object_01..05` + `114` âm gồm `58` `general_distractor` + `56` hard-neg: `glasses_case` 9, `lock_key_metal` 12, 5 nhóm còn lại mỗi nhóm 7).
- **`ambiguous_boundary`:** `9` ảnh ranh giới khó.

### 2.2 Ngưỡng đóng băng trên `calibration_v1b` (`manifests/frozen_thresholds.json`, SHA-256 prefix `26e3d034133daa3f`)
- **`holdout_touched_during_calibration`:** `false`.
- Kết quả tự kiểm trên `calibration_v1b`: `A0_baseline_hybrid` (`tau_global=0.4758`) đạt Recall `15/25 (60.0%, 95% CI [40.74, 76.60])` và FAR `6/25 (24.0%, 95% CI [11.50, 43.43])`; `A3_full_verifier` (`verifier_state` SHA-256 `eb6921f638eda4d1fe12a03176b6f6951e87a128ceeefd26ddc77ea0db499560`) đạt Recall `19/25 (76.0%, 95% CI [56.57, 88.50])` và FAR `2/25 (8.0%, 95% CI [2.22, 24.97])`; `e2e_tau_offset_c4 = 0.0`.
- Quy mô database tuần tự hóa `A3`: `pos_db = 94,802 B`, `neg_db = 179,174 B`, `verifier_weights = 85,860 B`, tổng `359,836 B` (`~351 KB`).

---

## 3. Kết quả Định lượng (Benchmark & Ablation với Wilson 95% CI)

### 3.1 Bài toán A — Hard-Negative & Verifier (Standalone ROI)

| Mode | Tập đánh giá | Overall Accuracy | Known Recall | False Accept Rate (FAR) | Wrong Class (`Shoe/Slipper`) | Latency p50 |
|---|---|---:|---:|---:|---|---:|
| `A0_baseline_hybrid` | `vl_p1_test` (`N=75`) | `61/75 (81.3%, [71.07, 88.54])` | `39/50 (78.0%, [64.76, 87.25])` | `3/25 (12.0%, [4.17, 29.96])` | `0/50` (`0/20`) | `160.91 ms` |
| `A1_neg_prototypes` | `vl_p1_test` (`N=75`) | `48/75 (64.0%, [52.70, 73.94])` | `23/50 (46.0%, [32.97, 59.60])` | **`0/25 (0.0%, [0.00, 13.32])`** | `0/50` (`0/20`) | `115.76 ms` |
| `A2_adaptive_thresholds` | `vl_p1_test` (`N=75`) | `61/75 (81.3%, [71.07, 88.54])` | `41/50 (82.0%, [69.20, 90.23])` | `5/25 (20.0%, [8.86, 39.13])` | `1/50` (`1/20`) | `91.31 ms` |
| `A3_full_verifier` | `vl_p1_test` (`N=75`) | `60/75 (80.0%, [69.59, 87.49])` | `40/50 (80.0%, [66.96, 88.76])` | `5/25 (20.0%, [8.86, 39.13])` | `1/50` (`1/20`) | `103.09 ms` |
| **`A0_baseline_hybrid`** | **`holdout_roi` (`N=214`)** | **`173/214 (80.8%, [75.04, 85.55])`** | `67/100 (67.0%, [57.31, 75.44])` | `8/114 (7.0%, [3.60, 13.24])` | `2/100` (`2/40`) | `110.25 ms` |
| `A1_neg_prototypes` | `holdout_roi` (`N=214`) | `147/214 (68.7%, [62.19, 74.53])` | `33/100 (33.0%, [24.56, 42.69])` | **`0/114 (0.0%, [0.00, 3.26])`** | `0/100` (`0/40`) | `104.37 ms` |
| `A2_adaptive_thresholds` | `holdout_roi` (`N=214`) | `164/214 (76.6%, [70.53, 81.80])` | `73/100 (73.0%, [63.57, 80.73])` | `23/114 (20.2%, [13.84, 28.46])` | `2/100` (`2/40`) | `104.93 ms` |
| **`A3_full_verifier`** | **`holdout_roi` (`N=214`)** | `168/214 (78.5%, [72.52, 83.48])` | **`74/100 (74.0%, [64.63, 81.60])`** | `20/114 (17.5%, [11.65, 25.55])` | `2/100` (`2/40`) | `123.12 ms` |

**Chi tiết theo lớp & nhóm confuser trên `holdout_roi` (`A0` vs `A3`):**
- **Recall theo lớp (`A0 -> A3`):** `object_01` `10/20 -> 11/20`; `object_02` `17/20 -> 17/20`; `object_03` `12/20 -> 11/20`; `object_04` `11/20 -> 16/20`; `object_05` `17/20 -> 19/20`.
- **FAR theo nhóm âm (`A0 -> A3`):** `glasses_case` `0/9 -> 0/9`; `notebook_pouch` `3/7 -> 1/7`; `umbrella_handle` `0/7 -> 0/7`; `walking_stick` `2/7 -> 2/7`; `long_metal_rod` `1/7 -> 1/7`; `footwear_sock` `0/7 -> 1/7`; `lock_key_metal` `2/12 -> 7/12`; `general_distractor` `0/58 -> 8/58`.

### 3.2 Bài toán B — Object Proposal Localization (Standalone Full-Image, Không dùng GT lúc suy luận)

| Proposer | Tập đánh giá | Recall@0.3 | Recall@0.5 | Mean Best IoU | Mean Containment | Mean Bg Contam. | p50 Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| `yolox_strict__k4` | `holdout_full` (`N=100` pos) | `36/100 (36.0%)` | `17/100 (17.0%)` | `0.2400` | `0.3512` | `0.6150` | `38.38 ms` |
| `yolox_low__k4` | `holdout_full` (`N=100` pos) | `64/100 (64.0%)` | `40/100 (40.0%)` | `0.4511` | `0.5950` | `0.3405` | `40.09 ms` |
| `B0_vl_p1_proposer__k4` | `holdout_full` (`N=100` pos) | `83/100 (83.0%, [74.45, 89.11])` | `68/100 (68.0%, [58.34, 76.33])` | `0.6700` | `0.8941` | `0.2622` | `52.71 ms` |
| `B1_multiscale__k4` | `holdout_full` (`N=100` pos) | `74/100 (74.0%, [64.63, 81.60])` | `64/100 (64.0%, [54.24, 72.73])` | `0.5793` | `0.7301` | `0.2380` | `84.73 ms` |
| `B2_geometry_thin__k4` | `holdout_full` (`N=100` pos) | `74/100 (74.0%, [64.63, 81.60])` | `61/100 (61.0%, [51.20, 69.98])` | `0.5408` | `0.7283` | `0.2602` | `103.41 ms` |
| `B_full_v1b__k3` | `holdout_full` (`N=100` pos) | `68/100 (68.0%, [58.34, 76.33])` | `51/100 (51.0%, [41.35, 60.58])` | `0.4857` | `0.6451` | `0.2693` | `115.03 ms` |
| `B_full_v1b__k4` | `holdout_full` (`N=100` pos) | `74/100 (74.0%, [64.63, 81.60])` | `60/100 (60.0%, [50.20, 69.06])` | `0.5305` | `0.6995` | `0.2611` | `98.17 ms` |
| `B_full_v1b__k6` | `holdout_full` (`N=100` pos) | `80/100 (80.0%, [71.12, 86.66])` | `69/100 (69.0%, [59.37, 77.22])` | `0.5975` | `0.7786` | `0.2579` | `84.34 ms` |
| **`B_full_v1b__k8`** | **`holdout_full` (`N=100` pos)** | **`83/100 (83.0%, [74.45, 89.11])`** | **`74/100 (74.0%, [64.63, 81.60])`** | `0.6434` | `0.8417` | `0.2599` | `120.03 ms` |
| `B0_vl_p1_proposer__k4` | `vl_p1_test_full` (`N=50` pos) | `46/50 (92.0%, [81.16, 96.85])` | `43/50 (86.0%, [73.81, 93.05])` | `0.7132` | `0.8930` | `0.2057` | `53.35 ms` |
| **`B_full_v1b__k8`** | **`vl_p1_test_full` (`N=50` pos)** | **`48/50 (96.0%, [86.54, 98.90])`** | **`45/50 (90.0%, [78.64, 95.65])`** | `0.7133` | `0.8481` | `0.1728` | `119.44 ms` |

- **Per-category Recall@0.5 trên `holdout_full` (`B0__k4 -> B_full_v1b__k8`):** `object_01` `0/20 -> 2/20`; `object_02` `16/20 -> 20/20`; `object_03` `16/20 -> 17/20`; `object_04` `19/20 -> 19/20`; `object_05` `17/20 -> 16/20`. (Trên `vl_p1_test_full`, `object_01` tăng `5/10 -> 6/10` và `object_04` tăng `9/10 -> 10/10`).

### 3.3 Thí nghiệm đối chứng 4 chiều End-to-End (`Method C` trên `holdout_full` & `vl_p1_test_full`)

| Config | `holdout_full` Acc | `holdout_full` Recall | `holdout_full` FAR | `holdout` Proposal Miss (<0.3) | `holdout` TP w/ IoU>=0.5 | `vl_p1_test` Acc / Rec / FAR | Latency p50 / p95 |
|---|---:|---:|---:|---:|---:|---|---:|
| **`C1_vl_p1_baseline`** (`B0+A0`) | **`162/214 (75.7%, [69.53, 80.96])`** | `63/100 (63.0%, [53.22, 71.82])` | **`15/114 (13.2%, [8.14, 20.58])`** | `17/100 (17.0%)` | `45/63 (71.4%)` | `78.7%` / `76.0%` / `16.0%` | `538.81 / 1050.46 ms` |
| `C2_hard_neg_only` (`B0+A3`) | `153/214 (71.5%, [65.11, 77.12])` | **`70/100 (70.0%, [60.42, 78.11])`** | `31/114 (27.2%, [19.87, 36.00])` | `17/100 (17.0%)` | `51/70 (72.9%)` | **`82.7%`** / **`86.0%`** / `24.0%` | `594.90 / 1129.08 ms` |
| `C3_proposal_only` (`B_full_k4+A0`) | `157/214 (73.4%, [67.07, 78.84])` | `60/100 (60.0%, [50.20, 69.06])` | `17/114 (14.9%, [9.52, 22.59])` | `26/100 (26.0%)` | `37/60 (61.7%)` | `81.3%` / `78.0%` / **`12.0%`** | `384.48 / 965.01 ms` |
| `C4_combined_v1b` (`B_full_k4+A3`) | `142/214 (66.4%, [59.79, 72.35])` | `66/100 (66.0%, [56.28, 74.54])` | `38/114 (33.3%, [25.35, 42.41])` | `26/100 (26.0%)` | `38/66 (57.6%)` | `78.7%` / `84.0%` / `32.0%` | `332.79 / 534.49 ms` |

- **Trên `ambiguous_boundary` (`N=9`):** `C1..C3` trả `6 KNOWN / 0 AMBIGUOUS / 3 UNKNOWN`; `C4` trả `7 KNOWN / 0 AMBIGUOUS / 2 UNKNOWN` (ROI `6 KNOWN / 3 UNKNOWN`).
- Chi tiết ma trận nhầm lẫn lưu tại [reports/confusion_matrices.md](reports/confusion_matrices.md) và phân tích sâu tại [failure_analysis.md](failure_analysis.md).

---

## 4. Trả lời trực diện 3 câu hỏi nghiên cứu

1. **Dùng negative gallery + verifier có giảm False Positive trên confuser mà không làm rớt mạnh Known Recall hay không?**
   - **Trên một phần mục tiêu là Có (`notebook_pouch_confuser` giảm từ `3/7` xuống `1/7` ở `A3`, Recall tăng `67/100 -> 74/100`), nhưng trên toàn cục ở cấu hình đóng băng hiện tại là KHÔNG đạt yêu cầu an toàn:** Cơ chế hiệu chuẩn `calibration_v1b` (`N=50`) hạ `tau_per_class` quá thấp (`0.0757..0.1826`), làm `A3_full_verifier` nâng FAR chung trên holdout từ `7.0%` (`A0`) lên `17.5%`. Nếu giữ ngưỡng cao kết hợp phản lực âm (`A1_neg_prototypes`), FAR giảm về tuyệt đối **`0/114 (0.0%)`** nhưng Known Recall sụp xuống `33.0%`. Điều này cho thấy việc **chuẩn hóa ngưỡng từng lớp trên tập hiệu chuẩn nhỏ** là rào cản chính, không phải thiếu tín hiệu tách biệt ở negative gallery.
2. **Đề xuất mới có tìm đúng gậy trắng (`object_01`), chìa khóa (`object_02`), đồ vật trên tay mà không dùng GT bbox không?**
   - **Có khi ngân sách crop đủ rộng (`K=6..8`), thất bại ở `K=4`:** `B_full_v1b__k8` đạt Recall@0.5 **`74/100 (74.0%)`** trên `holdout_full` (so với `68/100 = 68.0%` của `B0`, `17/100` của YOLOX strict), đưa `object_02` (chìa khóa) lên **`20/20 (100%)`** và đạt **`45/50 (90.0%)`** trên `vl_p1_test_full`. Tuy nhiên với gậy trắng (`object_01`) ở ảnh đường phố tự nhiên, bounding box trục ngang/dọc vẫn có mức nền pha tạp cao (`mean_bg_contamination = 0.6829`, Recall@0.5 `2/20`, Recall@0.3 `6/20`), và ở `K=4` việc thiếu ràng buộc đa dạng slot làm `geom_thin_line` lấn át các hộp toàn vật thể.
3. **Tổng hợp A + B thành `C4` có giữ độ trễ và RAM/CPU phù hợp máy văn phòng để tiến tới bước tiếp theo hay cần chỉnh lại kiến trúc?**
   - **Độ trễ & RAM đạt hoàn toàn mục tiêu phần cứng** (`C4` E2E p50 **`332.79 ms`**, p95 **`534.49 ms`**; RSS **`113.47 MB`** khi nạp cả 2 model, Private Bytes `712.22 MB`; chi phí matching mở rộng `N=100` chỉ `0.0108 ms`), nhưng **kiến trúc ghép nối & hiệu chuẩn (`A + B @ K=4`) CHƯA SẴN SÀNG cho production**: điểm E2E `C4` trên holdout (`66.4%` Acc, `33.3%` FAR) thấp hơn baseline `C1` (`75.7%` Acc, `13.2%` FAR). Cần thực hiện **VL-P1c** (giữ ngân sách `K=6..8` hoặc cân bằng slot trong `ProposalRanker`, chặn cận dưới `tau_per_class`, và hiệu chuẩn trên cỡ mẫu `>=100`) trước khi xem xét tích hợp sản phẩm.

---

## 5. Xác minh tự động & Toàn vẹn Baseline

- `research/vl_p1b/tests/` (`python -m pytest -v tests/`): **10 passed in 34.83s** (unit + manifest zero-overlap + frozen reproducibility + budget/CI schema).
- `VisionMate_VL_P1_AG/research/vl_p1` (`python -m pytest -v`): **34 passed in 54.40s** (0 thay đổi, khớp **`43/43` SHA-256** snapshot).
- Chi tiết xác minh git across 4 worktrees xem tại [reports/git_baseline_preservation.md](reports/git_baseline_preservation.md).
