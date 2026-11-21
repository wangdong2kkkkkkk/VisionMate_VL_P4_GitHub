# VisionMate VL-P1 — Nghiên Cứu Nhận Diện Loại Đồ Vật Ít Ảnh Mẫu (Few-Shot Category Recognition)

**Trạng thái:** Research Pilot (thực hiện độc lập bởi Google Antigravity trên nhánh `research/vl-p1-antigravity`)  
**Phạm vi:** Không sửa đổi production app (`VisionMateGlasses`), không merge nhánh, không tạo git tag, không mở Phase 2.

---

## 1. Mục tiêu và 5 nhóm đồ vật mục tiêu (`labels.json`)

Nghiên cứu khả năng nhận diện **category** (cùng loại đồ vật với hình dáng, màu sắc, góc chụp khác nhau, chưa từng xuất hiện trong ảnh mẫu) trên CPU Windows offline, tách biệt với bài toán nhớ một vật cá nhân duy nhất:

- `object_01`: `Gậy trắng hỗ trợ người khiếm thị`
- `object_02`: `Chùm chìa khóa`
- `object_03`: `Ví tiền`
- `object_04`: `Đôi dép`
- `object_05`: `Đôi giày`

---

## 2. Cấu trúc thư mục `research/vl_p1/`

```text
research/vl_p1/
├── labels.json                      # 5 tên danh mục tiếng Việt chuẩn
├── orb_baseline.py                  # Code ORB v0.1 kế thừa từ giai đoạn thủ công
├── crop_fit.py                      # Tiện ích fit/crop kế thừa
├── test_orb_baseline.py             # Test unittest kế thừa (3 test)
├── test_crop_fit.py                 # Test unittest kế thừa (4 test)
├── src/
│   ├── image_utils.py               # Đọc/ghi ảnh Unicode Windows, SHA-256, aHash, dHash, pad_to_square
│   ├── dataset_audit.py             # Kiểm toán originals <-> train crops <-> CSV & phát hiện lỗi ORB 0 kp
│   ├── collect_dataset.py           # Thu thập ảnh Wikimedia Commons có giấy phép mở
│   ├── collect_category_supplements.py
│   ├── collect_unknown_confusers.py
│   ├── collect_keys_supplement.py
│   ├── collect_keys_modern.py
│   ├── finalize_splits.py           # Phân chia calibration/test/test_full và kiểm tra chống rò rỉ
│   ├── download_models.py           # Tải & kiểm tra parity/SHA-256 các mô hình ONNX Apache-2.0
│   ├── orb_recognizer.py            # Method A: ORB raw & ORB adaptive narrow-ROI padding + RANSAC
│   ├── embedding_recognizer.py      # Method B: ONNX embeddings + 1-NN/Prototype/Hybrid + threshold calibration
│   ├── full_image_pipeline.py       # Task 2: YOLOX-Nano vs Hybrid class-agnostic proposer + classifier
│   └── benchmark_runner.py          # Chạy benchmark Task 1, Task 2, dev_probe01 và đo CPU/RAM
├── manifests/
│   ├── manual_data_audit.json       # Kết quả kiểm toán 25 originals / 25 train / CSV
│   ├── model_manifest.json          # Nguồn HuggingFace, Apache-2.0, kích thước, SHA-256 của 3 model
│   └── dataset_manifest.json        # Manifest đầy đủ 125 mẫu train/calibration/test (data_status: COMPLETE_VERIFIED)
├── reports/
│   ├── manual_dataset_audit.md      # Báo cáo kiểm toán dữ liệu thủ công ban đầu
│   ├── benchmark_report.md          # Bảng tổng hợp Task 1, Task 2 & hiệu năng phần cứng
│   ├── confusion_matrices.md        # Ma trận nhầm lẫn & breakdown FAR theo từng nhóm confuser
│   ├── benchmark_metrics.json       # Toàn bộ số liệu JSON và danh sách failure cases
│   ├── failure_analysis.md          # Phân tích chi tiết từng nhóm ca lỗi thực tế
│   └── VL_P1_FINAL_RESEARCH_REPORT.md # Báo cáo nghiên cứu tổng kết chính thức
└── tests/                           # Bộ test pytest tự động (27 test mới + 7 test kế thừa = 34 test)
```

*(Các thư mục dữ liệu nhị phân `data/`, trọng số `models/*.onnx`, ảnh contact sheet và cache HuggingFace được loại trừ qua `.gitignore`.)*

---

## 3. Tóm tắt kết quả chính (Windows 11, Intel i5-1135G7 @ 2.40GHz, 7.73 GB RAM)

| Bài toán / Phương pháp | Pos Top-1 Acc | FAR (Âm tính) | Sai khác Giày/Dép | Điểm nổi bật |
|---|---:|---:|---:|---|
| Method A — `orb_raw` (Task 1 ROI) | 8.0% | 44.0% | 0 / 1 | Phá sản với category đa hình dáng, Match P50 = `344.2 ms` |
| Method A — `orb_adaptive` (Task 1 ROI) | 0.0% | 16.0% | 1 / 0 | Vá được lỗi `0` keypoint (`0 -> 1500`) nhưng sai `dev_probe01`, Match P50 = `1286.6 ms` |
| Method B — `mobilenetv3_small__hybrid` | 62.0% | 20.0% | 0 / 1 | Siêu nhẹ (`9.71 MB`, Feat P50 = `7.33 ms`) |
| Method B — `efficientnet_b0__hybrid` | 74.0% | 20.0% | 0 / 2 | Độ chính xác trung bình, chậm nhất trong nhóm CNN (`56.88 ms`) |
| Method B — `dinov2_small_int8__prototype` | **88.0%** | 36.0% | **0 / 0** | Độ nhạy cao nhất (`100%` chìa khóa, `90%` gậy/ví/giày), chưa siết được hard confusers |
| **Method B — `dinov2_small_int8__hybrid`** | **78.0%** | **12.0%** | **0 / 0** | **Cân bằng tốt nhất**: `0` ca gán nhầm lớp dương, Feat P50 = `33.64 ms`, DB `92.6 KB` |
| Task 2 E2E — `yolox_strict__dinov2_hybrid` | 32.0% | 0.0% | 0 / 0 | YOLOX COCO bỏ sót **24/50** vật ngoài 80 lớp COCO (`Recall@0.3 = 36.0%`) |
| **Task 2 E2E — `hybrid_prop__dinov2_hybrid`** | **76.0%** | **16.0%** | **0 / 0** | Proposer lai giảm miss về **0/50**, `Recall@0.3 = 92.0%`, E2E P50 = `176.5 ms` |

---

## 4. Hướng dẫn tái lập kết quả (Reproduce)

Mọi lệnh chạy tại `VisionMate_VL_P1_AG/research/vl_p1/` (chạy **100% offline** sau khi đã chuẩn bị sẵn `data/` và `models/`):

```powershell
# 1. Kiểm toán dữ liệu gốc & phát hiện lỗi narrow-ROI ORB
python -m src.dataset_audit

# 2. Chạy toàn bộ bộ kiểm thử (27 test mới + 7 test kế thừa = 34 test)
python -m pytest -v tests/ test_crop_fit.py test_orb_baseline.py

# 3. Chạy lại toàn bộ benchmark Task 1, Task 2, dev_probe01 và đo CPU/RAM
python -m src.benchmark_runner
```

Để tải lại mô hình từ HuggingFace hoặc tái tạo dataset mẫu từ Wikimedia Commons (khi có mạng):

```powershell
python -m src.download_models
```
