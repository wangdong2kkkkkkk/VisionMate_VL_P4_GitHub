# DATASET PROTOCOL — VisionMate VL-P3

## 1. Phân chia dữ liệu và chống rò rỉ (Zero-Leakage)

Dữ liệu được xây dựng bởi `src/dataset_builder_v3.py` và lưu tại `manifests/dataset_manifest.json` (`protocol_version: VL-P3-v1`):

| Split | Tổng ảnh | `object_01` (Gậy trắng) | Neg / Confuser | `person_holding_confuser` |
| --- | ---: | ---: | ---: | ---: |
| `train_v3` | 183 | 34 | 149 | 22 |
| `val_cal_v3` | 98 | 16 | 82 | 29 |
| `final_holdout_v3` | 245 | 80 | 165 | 125 |
| `ambiguous_cane_v3` | 26 | 0 *(mờ/khó)* | — | — |

### Phân bổ nhóm gây nhiễu trong `final_holdout_v3` (165 ảnh âm tính)
- `general_distractor`: 34
- `metal_rod_pipe_confuser`: 21
- `tool_handle_confuser`: 23
- `trekking_pole_confuser`: 21
- `umbrella_confuser`: 25
- `walking_stick_confuser`: 16
- `white_line_pole_confuser`: 25
- Trong đó có **125** ảnh thuộc nhóm người đang cầm vật gây nhiễu (`person_holding_confuser = True`).

### Kiểm tra chống rò rỉ (`zero_leakage_audit`)
Tất cả 5 cặp kiểm tra đều đạt `clean: true`, `sha256_overlap = 0`, `title_overlap = 0`, `cluster_overlap = 0`, `phash_overlap = 0`:
1. `train_v3__vs__val_cal_v3`
2. `train_v3__vs__final_holdout_v3`
3. `val_cal_v3__vs__final_holdout_v3`
4. `all_v3_splits__vs__hist_final_holdout_v2` (`N=180`)
5. `all_v3_splits__vs__hist_vl_p1c_fhold` (`N=300`)

## 2. Kiểm tra trực quan & Ground Truth gậy trắng (130 / 130)

- Tất cả 130 ảnh `object_01` (`34` trong `train_v3`, `16` trong `val_cal_v3`, `80` trong `final_holdout_v3`) đều đã được kiểm tra trực quan từng ảnh trên lưới tọa độ 100 px (`gt_grid_audit_p01..13.jpg`) và hiệu chỉnh thủ công vào `manifests/manual_cane_gt_overrides_v3.json`.
- 5 ảnh ứng viên holdout không chứa gậy trắng thực sự (`v3_pos_cane_063, 145, 151, 152, 179`) và 1 ảnh chỉ có dây đai chó dẫn đường (`v2_pos_cane_064`) đã được loại bỏ và thay thế bằng 6 ảnh gậy trắng thực tế đã xác minh (`v3_pos_cane_215..220`).
- `manifests/annotation_manifest.json` cung cấp đầy đủ cho 130/130 ảnh: `endpoints_xyxy`, `centerline_points_xy` (11 điểm), `obb_cxcywh_angle`, `mask_rle` (mặt nạ thân gậy đầy đủ kích thước gốc), `angle_deg`, `length_px`, và cờ `occluded`.
