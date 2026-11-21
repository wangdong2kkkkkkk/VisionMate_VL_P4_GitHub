# ANNOTATION PROTOCOL — VisionMate VL-P4

## 1. Quy trình gắn nhãn & kiểm định độc lập

1. **Sàng lọc trực quan (Visual Screening):** Tất cả ảnh ứng viên dương tính mới được render lên bảng lưới phần trăm `0..100%` (`cache/screening/pos_screen_*.jpg`) và kiểm tra bằng mắt; các ảnh sách scan, banner sự kiện, hoặc không nhìn thấy gậy trắng đều bị loại bỏ.
2. **Gắn nhãn hình học thủ công:** Với `82` ảnh dương tính mới đạt chuẩn, tọa độ hai đầu mút `(x1%, y1%, x2%, y2%)`, trạng thái `occluded` và `folded` được đọc trực tiếp từ lưới `0..100%`, chuyển đổi sang pixel gốc trong `src/dataset_builder_v4.py`, và kiểm tra lại toàn bộ đường line xanh trên `gt_verify_v4_01..07.jpg` (kèm hiệu chỉnh tinh 13 ảnh).
3. **Không tự sinh nhãn bằng mô hình:** Mô hình CNN VL-P4 không bao giờ được dùng để tạo hoặc sửa Ground Truth.

## 2. Phân loại trạng thái nhãn (`annotation_manifest.json`)

- `VERIFIED_GROUND_TRUTH`: 50 ảnh từ `train_v3` + `val_cal_v3` đã kiểm tra từ VL-P3.
- `MANUALLY_CORRECTED`: 82 ảnh mới thu thập trong VL-P4 được gắn nhãn và tinh chỉnh trực tiếp qua kiểm tra lưới tọa độ.
- Mỗi bản ghi (132/132) chứa đủ: `endpoints_xyxy`, `centerline_points_xy` (11 điểm), `bbox_xywh`, `obb_cxcywh_angle`, `mask_rle`, `angle_deg`, `length_px`, `occluded`, `folded`, `annotation_status`.
