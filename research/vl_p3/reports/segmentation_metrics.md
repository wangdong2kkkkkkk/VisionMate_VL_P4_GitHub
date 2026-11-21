# Segmentation Metrics — VisionMate VL-P3

## 1. Kết quả phân đoạn thân gậy trên `final_holdout_v3` (`object_01`, N=80)

| Chỉ số phân đoạn | `B1_supervised_seg_only` | `B2_kp_guided_supervised_seg` (`C2` dùng `B2`) |
| --- | ---: | ---: |
| Mean Mask IoU | **0.0253** | 0.0220 |
| Mean Dice F1 | **0.0483** | 0.0395 |
| Mean Pixel Precision | 0.0314 | 0.0283 |
| Mean Pixel Recall | **0.3455** | 0.0922 |
| Mean Boundary F1 | **0.0403** | 0.0401 |
| Mean Background Contamination | **0.9686** | 0.9717 |

*(Trên tập `val_cal_v3` với `sparse_refine_k=560` trong `training_metrics.json`: `B1` đạt `mean_mask_iou = 0.0635`, `pixel_recall = 0.3963`, `bg_contamination = 0.8902`; `B2` đạt `mean_mask_iou = 0.0652`, `pixel_recall = 0.1854`, `bg_contamination = 0.9186`.)*

## 2. Nguyên nhân gốc rễ khiến Mask IoU rất thấp (`0.022–0.025`)

1. **Độ mảnh cực cao của thân gậy:** Trên lưới đặc trưng `112×112`, một cây gậy trắng ngoài đời thực chỉ rộng khoảng **1–2 pixel** (tương đương ~0.2%–0.5% diện tích ảnh). Chỉ cần đường tâm dự đoán lệch ngang 5–15 px ở độ phân giải gốc là IoU giữa hai dải mảnh rơi thẳng về ~0.
2. **`B1_supervised_seg_only` thu nhiều mảnh nền (`bg_contamination = 0.9686`):** Vì `B1` giữ lại tất cả thành phần liên thông có diện tích `>= 14 px` ở tỷ lệ `448×448`, các đường vạch trắng vỉa hè, viền tường hoặc tay áo sáng màu làm nhiễu mạnh precision (`0.0314`), dù pixel recall đạt `0.3455`.
3. **`B2_kp_guided_supervised_seg` bị "nghẽn cổ chai kép" (`pixel_recall = 0.0922`):** `B2` nhân xác suất pixel với `tube_gate` quanh hai đầu mút do `A1` dự đoán. Khi `A1` đặt nhầm 1 đầu mút sang vật thẳng khác (hoặc lệch góc `25.66°`), `tube_gate` triệt tiêu luôn phần thân gậy thật, làm recall pixel tụt từ `0.3455` xuống `0.0922` và kéo theo các cổng bác bỏ `veto_not_white_shaft` ở tầng `C2` kích hoạt nhầm trên ảnh dương tính.
