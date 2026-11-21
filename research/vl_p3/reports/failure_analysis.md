# Failure Analysis — VisionMate VL-P3

Phân tích trực tiếp dựa trên 4 contact sheets (`cache/contact_sheets/qual_holdout_{tp,fn,fp,tn_handheld}.jpg`) và log dự đoán 245 ảnh của `C2_full_vl_p3_pipeline`:

## 1. Phân bố nguyên nhân từ chối (Veto Counts trên `final_holdout_v3`)

| Lý do bác bỏ (`veto_reason`) | Ảnh dương tính (`object_01`, N=80) | Ảnh âm tính (N=165) | Tác động |
| --- | ---: | ---: | --- |
| `None` (vượt cổng hình học) | 39 | 46 | Chỉ 39/80 ảnh dương tính được verifier đánh giá tự do |
| `veto_not_white_shaft` | **31** | 70 | **Nguyên nhân lớn nhất làm sụt Known Recall của `C2`** |
| `veto_wood_or_dark_confuser` | **10** | 7 | Triệt tiêu thêm 10 ảnh dương tính thực sự |
| `veto_no_continuous_shaft` | 0 | 41 | Chặn rất tốt nền lộn xộn |
| `veto_too_short` | 0 | 1 | Chặn mảnh ngắn |

## 2. Phân tích từng nhóm lỗi từ kiểm tra trực quan

1. **False Negatives do `B2` mask trượt khỏi thân gậy dẫn đến `veto_not_white_shaft` (31/80 dương tính):**
   - Trong `qual_holdout_fn.jpg`, các ảnh như `v3_pos_cane_216`, `v3_pos_v2_ovc_059`, `v3_pos_cane_214`, `v3_pos_cane_085` có `A1` bắt nhầm sang cạnh tủ kính, ghế xe buýt, vạch trắng đường hoặc vỉa hè lát gạch dẫn hướng. Khi `B2` bị khóa trong `tube_gate` quanh đường sai này, `seg_white_purity` rớt dưới `0.26`, kích hoạt `veto_not_white_shaft` (hoặc `veto_wood_or_dark_confuser` như `v3_pos_v2_mcane_001` khi đường tâm đi qua áo tối màu/cờ) và ép điểm `C2` xuống `<= 0.24`.
   - Ngược lại, `A1_supervised_kp_centerline` không dùng cổng cứng của `B2`, nên giữ được **Recall = 0.3750 (30/80)** với **FAR = 0.0242 (4/165)** và **PR-AUC = 0.8272** (cao hơn `C2` một cách có ý nghĩa thống kê: McNemar exact `p = 2.0e-06`).
2. **False Positives còn lại của `C2` (chỉ 3/165 ảnh âm tính — `FAR = 0.0182`):**
   - `v3_neg_umbrella_037`: Đường viền trắng sọc dọc trên ô và mép áo khoác trắng dài.
   - `v3_neg_walking__037`: Vạch an toàn màu trắng/vàng sáng kéo dài trên sân ga tàu điện ngầm.
   - `v3_neg_general__027`: Vạch sơn trắng/đường ray kéo dài trên vỉa hè đô thị.
3. **Điểm mạnh đã chứng minh (`qual_holdout_tp.jpg` & `qual_holdout_tn_handheld.jpg`):**
   - `DeBiasedCaneVerifierV3` đạt `max anchor_prob_delta = 0.0039` (`<= 0.10`), loại bỏ hoàn toàn thiên lệch "thấy người cầm vật dài là báo gậy trắng": đạt **FAR = 0.0160 (2/125)** trên tập người cầm vật gây nhiễu và **FAR = 0.0000** trên `metal_rod_pipe`, `tool_handle`, `trekking_pole`, và `white_line_pole`.
