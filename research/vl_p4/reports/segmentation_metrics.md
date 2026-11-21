# Segmentation Metrics — VisionMate VL-P4

| Tập holdout | Mô hình | Mean Mask IoU | Mean Dice | Pixel Precision | Pixel Recall | BG Contamination |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `final_holdout_v4` (N=19 pos) | `B1_CNN_Segmentation` | 0.0182 | 0.0333 | 0.0225 | 0.2031 | 0.7143 |
| `final_holdout_v4` (N=19 pos) | `B2_CNN_MultiTask` (`C2`) | **0.0190** | **0.0350** | **0.0246** | **0.2564** | 0.7649 |
| `final_holdout_v3` (N=80 pos) | `B1_CNN_Segmentation` | **0.0209** | **0.0388** | 0.0230 | **0.2376** | **0.6520** |
| `final_holdout_v3` (N=80 pos) | `B2_CNN_MultiTask` (`C2`) | 0.0192 | 0.0358 | **0.0294** | 0.2206 | 0.6831 |

**Kết luận trung thực:** Nhánh segmentation ở độ phân giải `64×64` (stride 4) chỉ đạt `Mask IoU ~ 0.019–0.021` (không cải thiện so với VL-P3 `0.022–0.025`). Tuy nhiên, việc **không dùng mask kém chất lượng làm cổng veto cứng** đã giải phóng pipeline khỏi 41/80 ca từ chối oan của VL-P3 `C2`, giúp Recall tăng từ `12.5%` (VL-P3 `C2`) / `37.5%` (VL-P3 `A1`) lên **`72.5%`** trên `final_holdout_v3`.
