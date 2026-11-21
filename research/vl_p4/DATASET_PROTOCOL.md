# DATASET PROTOCOL — VisionMate VL-P4

## 1. Quy mô dữ liệu thực tế thu thập được & tính minh bạch

Mục tiêu lý tưởng của Gate 2 là 300–500 ảnh gậy trắng (`object_01`) và >= 300 ảnh âm tính khó. Trên nguồn ảnh mở (Wikimedia Commons + Openverse CC) sau khi loại bỏ **1,479** ảnh trùng lặp lịch sử từ VL-P1 đến VL-P3 và sàng lọc trực quan loại bỏ trang sách scan/ảnh hội nghị không có gậy:
- Thu thập được **82** ảnh gậy trắng mới đạt chuẩn trực quan (`MANUALLY_CORRECTED`) + **595** ảnh âm tính khó mới (7 nhóm confuser).
- Kết hợp với `50` ảnh gậy trắng đã xác minh từ `train_v3` (`34`) + `val_cal_v3` (`16`) và `231` ảnh âm tính từ `train_v3`/`val_cal_v3` (**tuyệt đối không chạm `final_holdout_v3` `N=245`, `final_holdout_v2` `N=180`, `vl_p1c/final_holdout_full` `N=300`**):
  - Tổng cộng **132** ảnh `object_01` có nhãn xác minh + **826** ảnh âm tính khó.

## 2. Phân chia tập dữ liệu (Cluster-Disjoint, Zero-Leakage)

| Split | Tổng ảnh | `object_01` | Neg / Confuser | `person_holding_confuser` |
| --- | ---: | ---: | ---: | ---: |
| `train_v4` | 581 | 97 | 484 | 223 |
| `val_cal_v4` | 158 | 16 | 142 | 53 |
| `final_holdout_v4` *(100% ảnh mới chưa từng thấy)* | 219 | 19 | 200 | 200 |
| `ambiguous_cane_v4` | 26 | 0 | 26 | 0 |
| `final_holdout_v3` *(đối chứng mù VL-P3, chưa từng dùng train/cal V4)* | 245 | 80 | 165 | 125 |

- **Kiểm tra chống rò rỉ (`zero_leakage_audit`):** Tất cả 5 cặp kiểm tra (`train_v4 vs val_cal_v4`, `train_v4 vs final_holdout_v4`, `val_cal_v4 vs final_holdout_v4`, `dev_v4 vs final_holdout_v3`, `final_holdout_v4 vs final_holdout_v3`) đều đạt `sha256_overlap = 0`, `title_overlap = 0`, `cluster_overlap = 0`, `phash_overlap = 0`, `all_clean: true`.
- Do số ảnh dương tính mới độc lập trong `final_holdout_v4` chỉ đạt `19` (và tổng dev `97 + 16 = 113` < mục tiêu `300–500`), báo cáo trung thực giữ nguyên kết quả thực tế kết hợp đánh giá trên cả `final_holdout_v4` (`N=219`) và `final_holdout_v3` (`N=245`) thay vì trộn ảnh cũ để làm đẹp số lượng.
