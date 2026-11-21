# VL-P3 CLOSURE AND PRESERVATION AUDIT (GATE 0)

**Ngày kiểm toán:** 2026-11-16  
**Đơn vị kiểm toán:** Google Antigravity  
**Kết quả Gate 0:** **ĐẠT (PASSED)** — Toàn bộ baseline từ English Tables/TTS đến VL-P3 được bảo vệ nguyên vẹn, tái lập được, không có lỗi phá hủy dữ liệu.

---

## 1. Kiểm toán Git hậu commit

| Thành phần / Nhánh | Worktree | Commit SHA đầy đủ | Trạng thái |
| --- | --- | --- | --- |
| English Tables/TTS | *(branch `hotfix/english-tables-layout-and-tts`)* | `8ae6f85b5ed4e38bf343ca623d943e2fbf99ec9` | Nguyên vẹn |
| OD-1 Foundation | *(ancestor of `6e3039f`)* | `95a75e0` | Nguyên vẹn |
| Static Preview | `VisionMate_VL_P1` (`research/vl-p1-manual`) | `6e3039f97f8065153dc96c4affb4f9371d3bcc7e` | Nguyên vẹn |
| Webcam FPS Optimization | `VisionMateGlasses` (`hotfix/webcam-preview-fps-optimization`) | `65c2a27a21fac65e5e3de78a4feba081dd9e192e` | Nguyên vẹn |
| VL-P1 | `VisionMate_VL_P1_AG` (`research/vl-p1-antigravity`) | `988fdf14a574863f6246017c5ccf8350eef5fdb8` | Sạch, nguyên vẹn |
| VL-P1b | `VisionMate_VL_P1b_AG` (`research/vl-p1b-reliability-proposal`) | `dacd24571ae2c1f6fe51db75e832c0bc5d622215` | Sạch, nguyên vẹn |
| VL-P1c | `VisionMate_VL_P1c_AG` (`research/vl-p1c-calibration-multiproposal`) | `9587ad491b62323f2196bd1749994e7b9d140deb` | Sạch, nguyên vẹn |
| VL-P2 | `VisionMate_VL_P2_AG` (`research/vl-p2-white-cane-segmentation`) | `4a1011308ce8d63cee279354dfcc2ea68d9bb7cd` | Sạch, nguyên vẹn |
| **VL-P3** | `VisionMate_VL_P3_AG` (`research/vl-p3-supervised-cane-detection`) | **`2001119c63c82a714d17aba065985d5c4c95fa70`** | **Sạch, đã commit bàn giao** |

- **Ancestry check (`git merge-base --is-ancestor 4a10113 2001119`):** `exit 0` (VL-P3 kế thừa trực tiếp từ VL-P2).
- **Tags / Merge / Push:** `git tag --list` rỗng; không có merge, tag hoặc push ngoài phạm vi.
- **Worktree sạch:** `git status --short` trên cả `VL_P1_AG`, `VL_P1b_AG`, `VL_P1c_AG`, `VL_P2_AG`, `VL_P3_AG` đều trống.

---

## 2. Kiểm toán dữ liệu, annotation và model VL-P3

- **Tập dữ liệu (`manifests/dataset_manifest.json`):** `train_v3` (183 ảnh, 34 `object_01`), `val_cal_v3` (98 ảnh, 16 `object_01`), `final_holdout_v3` (245 ảnh, 80 `object_01`), `ambiguous_cane_v3` (26 ảnh); kiểm tra rò rỉ 5 cặp `all_clean: true`.
- **Annotation (`manifests/annotation_manifest.json`):** Đầy đủ `130/130` ảnh `object_01` đã được kiểm tra trực quan và khóa trong `manifests/manual_cane_gt_overrides_v3.json`.
- **Checksum SHA-256 toàn bộ 7 thành phần model đã xác minh khớp 100% với `manifests/frozen_config.json`:**
  - `dinov2_small_quantized.onnx`: `3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917`
  - `yolox_nano.onnx`: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
  - `patch_semantic_head.json`: `f3f39c2008aa66fc893aa4a3b0bef1561913bcd57ea9828aba9cb9462c19539a`
  - `keypoint_detector_v3.joblib`: `f19d2ad6d674c3235a6103ec34978c478b39c2678d6ff914bd2dee3846fe4834`
  - `shaft_segmenter_b1_standalone.joblib`: `31a849acda618d5d5152b473bb1f2da8cde552d29761a0f27b78f8bd205e3e45`
  - `shaft_segmenter_b2_kp_guided.joblib`: `55b5051e041dea304ba902abfa9b2165dd47ac5bf75bb49af02c4b39b0304396`
  - `cane_verifier_v3.joblib`: `520bb7d1633f2d2acd68ffa4c51aaf049652ca4fe1ac748395cf3327c7651253`
- **Khôi phục tài nguyên bị `.gitignore` loại khỏi Git:**
  - `models/yolox_nano.onnx`: Sao chép từ `models/object_detection/yolox_nano.onnx` trong repo gốc.
  - `models/dinov2_small_quantized.onnx`: Lưu cục bộ tại `VisionMate_VL_P3_AG/research/vl_p3/models/dinov2_small_quantized.onnx` (và trong `VisionMate_VL_P1_AG/research/vl_p1/models/`).
  - `data/` & `cache/`: Dữ liệu ảnh lưu cục bộ theo `abs_path` trong `dataset_manifest.json` và có thể tái tạo manifest qua `python -m research.vl_p3.src.dataset_builder_v3`.

---

## 3. Xác minh lại baseline `A1` từ ảnh gốc **không cache** (`final_holdout_v3`, N=245)

Chạy kiểm chứng trực tiếp trên `VisionMate_VL_P3_AG` (`2001119c63c82a714d17aba065985d5c4c95fa70`) với `bundle=None` (tính cả tải ảnh, tiền xử lý, `extract_spatial_features`, `kp_detector.predict`, phân loại):

| Chỉ số `A1` (`N=245`) | Đối chứng lịch sử (cached bundle) | Xác minh Gate 0 (**Uncached E2E từ ảnh gốc**) |
| --- | ---: | ---: |
| TP / FP / TN / FN | 30 / 4 / 113 / 4 | **30 / 4 / 113 / 4** *(khớp tuyệt đối)* |
| Known Recall | 37.50% | **37.50%** |
| FAR | 2.42% | **2.42%** |
| Precision | 88.24% | **88.24%** |
| Endpoint Error (px) | 194.51 px | **194.51 px** |
| Both PDJ@0.10 | 23.75% | **23.75%** |
| ROC-AUC / PR-AUC | 0.9116 / 0.8272 | **0.9116 / 0.8272** |
| **Latency P50 / P95 / Mean** | *(37.84 / 57.32 ms chỉ tính downstream)* | **361.37 ms / 586.74 ms / 389.19 ms** |

**Kết luận Gate 0:** Mốc đối chứng E2E thực sự của `A1` là **361.37 ms P50 / 586.74 ms P95** (không dùng con số `37.84 ms` downstream để đại diện cho E2E). Hồ sơ VL-P3 được đóng sạch và làm nền tảng khởi tạo VL-P4.
