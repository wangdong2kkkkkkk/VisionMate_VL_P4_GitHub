# BÁO CÁO NGHIÊN CỨU TỔNG KẾT — VISIONMATE VL-P3
## Supervised White Cane Keypoint & Shaft Detection Pilot

**Đơn vị thực hiện:** Google Antigravity  
**Loại nhiệm vụ:** Independent Software Research Pilot (CPU-first, Windows 11 offline, chưa tích hợp production)  
**Worktree / Nhánh:** `VisionMate_VL_P3_AG` — `research/vl-p3-supervised-cane-detection` (tách từ `4a10113`)  
**Dữ liệu sạch hoàn toàn (Zero-Leakage):** `train_v3 = 183` (34 pos), `val_cal_v3 = 98` (16 pos), `final_holdout_v3 = 245` (80 pos, 165 neg gồm 125 handheld confusers), `ambiguous_cane_v3 = 26`. Tất cả 130/130 ảnh `object_01` đã được kiểm tra trực quan và chuẩn hóa GT.

---

## I. Tóm tắt kết quả định lượng trên tập mù `final_holdout_v3` (N=245)

| Preset | TP | FP | TN | FN | Pos-UNC | Neg-UNC | Known Recall | FAR | Precision | ROC-AUC | PR-AUC | Endpoint Err (px) | Latency P50 / P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_v2_lsd_hough_baseline` | 16 | 15 | 81 | 9 | 55 | 69 | 0.2000 | 0.0909 | 0.5161 | 0.7503 | 0.5518 | 336.29 | 169.62 / 220.62 |
| `A1_supervised_kp_centerline` | **30** | 4 | 113 | 4 | 46 | 48 | **0.3750** | 0.0242 | **0.8824** | **0.9116** | **0.8272** | **194.51** | 37.84 / 57.32 *(cached)* |
| `B1_supervised_seg_only` | 16 | 10 | 116 | 3 | 61 | 39 | 0.2000 | 0.0606 | 0.6154 | 0.8661 | 0.6529 | 259.77 | 42.06 / 64.15 *(cached)* |
| `B2_kp_guided_supervised_seg` | 11 | 3 | 110 | 7 | 62 | 52 | 0.1375 | **0.0182** | 0.7857 | 0.8649 | 0.7101 | 194.51 | 95.86 / 149.32 *(cached)* |
| `C1_b2_plus_masked_dinov2` | 17 | 6 | 146 | 35 | 28 | 13 | 0.2125 | 0.0364 | 0.7391 | 0.8442 | 0.6979 | 194.51 | 151.61 / 222.41 *(cached)* |
| `C2_full_vl_p3_pipeline` | 10 | **3** | **154** | 34 | 36 | 8 | 0.1250 | **0.0182** | 0.7692 | 0.8897 | 0.7636 | 194.51 | **402.61 / 652.59** *(uncached)* |

- **Khả năng tách biệt giữa đầu vào sạch và gây nhiễu handheld (`C2`):** `FAR` trên `125` ảnh người cầm vật dài gây nhiễu đạt **0.0160** (2/125, CI95: 0.0044–0.0565), đạt **0.0000** trên `metal_rod_pipe` (0/21), `tool_handle` (0/23), `trekking_pole` (0/21), và `white_line_pole` (0/25).
- **Khử thiên lệch người cầm (`DeBiasedCaneVerifierV3`):** `max anchor_prob_delta` đo trên toàn bộ 245 ảnh holdout là **0.0039** (`<= 0.10`).
- **Tài nguyên RAM:** `RSS Peak = 463.32 MB` (`<= 650 MB`).

---

## II. Trả lời đầy đủ 10 câu hỏi bắt buộc

1. **Bộ định vị có học giám sát (`A1`) có định vị đúng thân gậy và hai đầu mút trên cảnh thực tế hơn LSD/Hough (`A0`) của VL-P2 không?**
   - **Có, cải thiện rõ rệt.** Trên `final_holdout_v3` (80 `object_01`), `A1` giảm sai số trung bình 2 đầu mút từ **336.29 px (`A0`) xuống 194.51 px (`A1`)** (-42.2%), giảm sai số đường tâm Chamfer từ **204.65 px xuống 143.19 px**, giảm sai số góc từ **41.17° xuống 25.66°**, và tăng tỷ lệ phát hiện đúng cả hai đầu mút (`Both PDJ@0.10`) từ **8.75% lên 23.75%**. Tuy nhiên, mức sai số `194.51 px` vẫn còn cao khi ảnh có nhiều người hoặc vạch kẻ đường dài gây phân tâm.

2. **Bộ phân đoạn thân gậy (`B1`, `B2`) có tách được thân gậy khỏi nền, quần áo và người cầm không?**
   - **Chưa đạt.** Trên `final_holdout_v3`, `B1` chỉ đạt `mean_mask_iou = 0.0253` (`dice = 0.0483`, `pixel_recall = 0.3455`, `bg_contamination = 0.9686`) và `B2` đạt `mean_mask_iou = 0.0220` (`dice = 0.0395`, `pixel_recall = 0.0922`, `bg_contamination = 0.9717`). Do thân gậy chỉ rộng ~1–2 px ở lưới `112×112`, bất kỳ độ lệch ngang nhỏ nào của `A1` khiến `tube_gate` của `B2` cắt mất thân gậy thật, còn `B1` không có `tube_gate` thì dính nhiều vạch trắng nền.

3. **Pipeline VL-P3 có thực sự cải thiện Known Recall và giảm FAR trên ảnh tự nhiên có người cầm vật gây nhiễu không?**
   - **Giảm FAR rất mạnh, nhưng `C2` bị tụt mạnh Known Recall so với `A1`.** `C2` giảm FAR xuống **1.82%** (3/165) và **1.60%** (2/125) trên ảnh người cầm vật dài gây nhiễu (so với `A0` là 9.09%), nhưng Known Recall của `C2` chỉ đạt **12.50%** (10/80), thấp hơn cả `A0` (**20.00%**) và kém `A1` (**37.50%**, McNemar exact `p = 2.0e-06`). Ngược lại, riêng **`A1`** đạt **37.50% Recall** tại **2.42% FAR** (`ROC-AUC = 0.9116`, `PR-AUC = 0.8272`).

4. **Những lỗi nào của `A0` đã được giải quyết, và những lỗi mới nào xuất hiện ở VL-P3?**
   - **Lỗi đã giải quyết:** `A0` hay bắt nhầm cạnh kiến trúc/nền tối và thiên lệch theo vùng người (`FAR = 9.09%`, `ambiguous_cane_v3` FP = 4/26, `endpoint_err = 336.29 px`); VL-P3 giảm `FAR` xuống `1.82%–2.42%`, giảm lỗi `ambiguous_cane_v3` xuống `0–1/26`, và giảm `endpoint_err` 42.2%.
   - **Lỗi mới xuất hiện:** Cổng bác bỏ cứng dựa trên mask `B2` trong `C2` tạo ra hiện tượng **bác bỏ oan hàng loạt ảnh dương tính**: `41/80` (`51.25%`) ảnh gậy trắng thật bị veto (`31` bởi `veto_not_white_shaft` và `10` bởi `veto_wood_or_dark_confuser`) vì mask `B2` trượt khỏi thân gậy mảnh sang áo tối hoặc nền.

5. **Kết quả có ổn định giữa `val_cal_v3` (Phase A) và `final_holdout_v3` (Phase B) không?**
   - **FAR ổn định tốt, nhưng Recall và Segmentation IoU suy giảm đáng kể khi sang holdout.**
     - `A1`: `val_cal_v3` Recall `0.375` / FAR `0.049` -> `holdout` Recall `0.375` / FAR `0.024` (**rất ổn định**).
     - `B2`: `val_cal_v3` Recall `0.562` (chạy trước khi giảm k) / `0.438` (k=160) / FAR `0.049` -> `holdout` Recall `0.138` / FAR `0.018` (tụt mạnh).
     - `C2`: `val_cal_v3` Recall `0.312` / FAR `0.024` -> `holdout` Recall `0.125` / FAR `0.018` (tụt 18.7 pp Recall do `B2` mask IoU trên holdout giảm từ `0.0652` xuống `0.0220`).

6. **Khi thân gậy bị che khuất một phần (`occluded`), bộ định vị và phân đoạn có giữ được độ chính xác không?**
   - **Cải thiện tương đối so với `A0`, nhưng chưa đủ tin cậy tuyệt đối.** Trên `27` ảnh bị che khuất của holdout, `A1` giảm sai số đầu mút từ `341.96 px` (`A0`) xuống `221.92 px` và giảm sai số góc từ `49.04°` xuống `35.95°`, `PDJ@0.10` tăng từ `0.1111` lên `0.3148`, nhưng vẫn cao hơn đáng kể so với tập rõ ràng (`180.55 px`, `20.42°`).

7. **Xác nhận ảnh hưởng của người cầm đã được trung hòa chưa (`anchor_prob_delta <= 0.10`)?**
   - **Đã xác nhận trung hòa hoàn toàn.** `DeBiasedCaneVerifierV3` cưỡng chế giới hạn nâng xác suất từ `gated_hand_person_anchor` tối đa `0.10`, kết quả đo thực tế trên toàn bộ 245 ảnh holdout cho `max anchor_prob_delta = 0.0039` (`<= 0.10`).

8. **Độ trễ thực tế và mức dùng RAM trên CPU có đạt ngưỡng khả dụng không?**
   - **RAM đạt ngưỡng, nhưng độ trễ end-to-end của `C2` CHƯA đạt mục tiêu.**
   - `RSS Peak = 463.32 MB` (`<= 650 MB`).
   - Tuy nhiên, độ trễ uncached end-to-end `C2` trên 245 ảnh holdout đạt **`P50 = 402.61 ms`, `P95 = 652.59 ms`** (vượt xa ngưỡng mục tiêu `P50 <= 130 ms` / `P95 <= 220 ms`). Nguyên nhân chính là `compute_base_and_ridge_maps` chạy 16 phép `cv2.filter2D` 13×13 ở `448×448` (~`127 ms` P50) cộng thêm 2 lượt suy luận `DINOv2` (~`33–48 ms`/lượt). Riêng downstream `A1` chỉ tốn `37.84 ms` P50 nếu đã có feature grid.

9. **Dữ liệu `train_v3` hiện tại (`34` gậy trắng) có đủ để mô hình học biểu diễn tổng quát chưa?**
   - **Chưa đủ cho mô hình pixel-level và endpoint-level đa bối cảnh.** Với chỉ **34 ảnh dương tính** trong `train_v3` để học đồng thời heatmap 2 đầu mút, đường tâm và phân đoạn thân gậy mảnh 1–2 px trên nhiều điều kiện ánh sáng/nền phố đông đúc, các bộ cây `HistGradientBoosting` trên lưới `112×112` đã tiệm cận giới hạn biểu diễn (đạt `PR-AUC = 0.8272` cho `A1` nhưng chưa đủ độ chính xác pixel để nuôi mask `B2` IoU cao).

10. **Nếu giữ định hướng thiết bị đeo chi phí thấp chạy CPU, bước tiếp theo tối ưu nhất là gì?**
    - (1) Không dùng cổng veto cứng từ mask `B2` trượt tâm lên kết quả của `A1`; (2) Mở rộng tập ảnh gậy trắng được gán nhãn keypoint/mask lên **>= 300–500 ảnh dương tính thực tế**; (3) Thay `HistGradientBoosting` + 16 bộ lọc `filter2D` 448×448 bằng một **mạng nhẹ end-to-end ONNX INT8** (ví dụ backbone MobileNet/YOLOX-NanoFPN + đầu heatmap độ phân giải cao 1/4 hoặc 1/8) để vừa đạt định vị/phân đoạn chính xác hơn vừa hạ độ trễ CPU về dưới `80–120 ms`.

---

## XIV. ĐÁNH GIÁ & KẾT LUẬN CUỐI CÙNG (SECTION XIV)

**Phân loại chính thức:** **`2. CẦN VL-P4 VỚI DỮ LIỆU / KIẾN TRÚC LỚN HƠN`**

### Lý do kỹ thuật khách quan từ số liệu thực nghiệm:
1. **Không chọn (1) "Tiếp tục tích hợp thử nghiệm"**: Mặc dù `C2` có `FAR = 1.82%` và `A1` có `PR-AUC = 0.8272`, Recall trên holdout thực tế mới đạt `12.50%` (`C2`) / `37.50%` (`A1`), `B2` `mask_iou = 0.0220`, và độ trễ `C2` uncached trên CPU là `402.61 ms` P50 / `652.59 ms` P95 — chưa đủ độ nhạy lẫn tốc độ thời gian thực để đưa vào thử nghiệm trên kính AI.
2. **Không chọn (3) "Không nên đưa vào nhánh sản phẩm hiện tại" theo nghĩa từ bỏ hướng đi**: Hướng học giám sát keypoint + centerline (`A1`) đã chứng minh rõ ràng tính khả thi vượt trội so với heuristic LSD/Hough (`A0`): giảm **42.2%** sai số đầu mút, tăng `Both PDJ@0.10` gần **2.7×**, đạt `ROC-AUC = 0.9116`, `PR-AUC = 0.8272`, `Precision = 88.24%` chỉ với `34` ảnh huấn luyện, và `DeBiasedCaneVerifierV3` đã giải quyết triệt để thiên lệch người cầm (`max anchor_prob_delta = 0.0039`, `handheld FAR = 1.60%`).
3. **Cần VL-P4**: Để vượt qua hai nút thắt cuối cùng (`mask_iou = 0.022` gây veto oan 41/80 ảnh dương tính, và `compute_base_and_ridge_maps` tốn ~127 ms + `402.61 ms` E2E), cần nâng quy mô dữ liệu dương tính gán nhãn từ `34` lên hàng trăm ảnh và chuyển từ GBDT trên lưới thủ công `112×112` sang kiến trúc CNN/ONNX INT8 nhẹ huấn luyện end-to-end.
