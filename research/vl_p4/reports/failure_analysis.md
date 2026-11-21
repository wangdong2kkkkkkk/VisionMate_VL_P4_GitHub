# Failure Analysis — VisionMate VL-P4

Phân tích trực tiếp từ 4 contact sheets (`cache/contact_sheets/qual_v4_{tp,fn,fp,tn_handheld}.jpg`) và `benchmark_metrics.json`:

1. **False Positives của `C2_FinalCandidate` (`7/200 = 3.50%` trên `v4`, `12/165 = 7.27%` trên `v3`):**
   - Trong `qual_v4_fp.jpg`, hầu hết FP rơi vào **gậy đi bộ (`walking_stick`: 3/16 = 18.75% trên `v3`)**, **thanh kim loại/gậy selfie (`metal_rod_pipe`: 4/44 = 9.09% trên `v4`, 3/21 = 14.29% trên `v3`)**, và **ô dù cầm tay (`umbrella`: 3/74 = 4.05% trên `v4`, 2/25 = 8.00% trên `v3`)**.
   - Lý do: Độ phân giải đầu vào `256×256` làm mất chi tiết sắc độ trắng/đầu đỏ trên thân thanh mảnh ở khoảng cách xa, nên head phân loại toàn cục (`GAP(c4)`) đôi khi kích hoạt khi thấy người cầm thanh mảnh dọc/chéo giống dáng cầm gậy.
2. **False Negatives còn lại (`5/19` trên `v4`, `15/80` trên `v3`):**
   - Trong `qual_v4_fn.jpg`, các ảnh bị trượt điểm (`w_pos_3008`, `s_cane_085`, `s_cane_215`, `s_cane_002`) thường là ảnh đen trắng cũ (`w_pos_0874`), gậy tháo rời/gập (`s_cane_002`), hoặc nền có nhiều đường thẳng song song khiến ngưỡng hiệu chuẩn bảo thủ `t_known = 0.89` (trên `val_cal_v4`) loại sang `UNCERTAIN`/`UNKNOWN`.
3. **Lỗi ghép chéo đầu mút trong ảnh đông người (`qual_v4_tp.jpg`):**
   - Trong `w_pos_2147`, `w_pos_2197`, `w_pos_2247`, khi ảnh có nhiều người hoặc nhiều gậy, dự đoán `top_hm` và `bot_hm` độc lập ở lưới `64×64` đôi khi nối đầu trên của người này với đầu dưới của gậy kia.
