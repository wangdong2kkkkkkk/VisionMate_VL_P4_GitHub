# VisionMate — VL-P3: Supervised White Cane Keypoint & Shaft Detection Pilot

**Trạng thái:** Independent Software Research Pilot (CPU-first, Windows offline, chưa tích hợp production).  
**Worktree / Nhánh:** `VisionMate_VL_P3_AG` — `research/vl-p3-supervised-cane-detection` (tách từ `4a10113`).

## Tổng quan kiến trúc

VL-P3 thay thế bộ định vị thủ công LSD/Hough của VL-P2 bằng hệ thống định vị & phân đoạn có học giám sát gọn nhẹ trên CPU:
1. **Multi-scale Spatial Feature Grid (`spatial_features.py`):** Trích xuất lưới đặc trưng `112×112×36` từ 16 kênh trắc quang/định hướng thanh mảnh (`448×448` pooling), `12` kênh PCA và `5` kênh semantic từ `DINOv2-Small INT8` (`16×16×384` patch tokens + `PatchSemanticHead`), cùng `3` kênh ngữ cảnh người/ô từ `YOLOX-Nano`.
2. **Supervised Keypoint & Centerline Detector (`A1` — `keypoint_detector.py`):** Dự đoán đồng thời heatmap 2 đầu mút (`grip`, `tip`), đường tâm (`centerline`), và trường góc kép (`cos2θ`, `sin2θ`) bằng `HistGradientBoosting` hai tầng (dense base + sparse top-`K=160` refinement).
3. **Supervised Shaft Segmenter (`B1`, `B2` — `shaft_segmenter.py`):** Phân đoạn pixel thân gậy ở lưới `112×112` nâng cấp `448×448` rồi map về ảnh gốc, dạng đứng riêng (`B1`) hoặc có dẫn hướng hình học từ `A1` (`B2`).
4. **Background-Suppressed DINOv2 & De-Biased Verifier (`C1`, `C2` — `cane_verifier_v3.py`):** Trích xuất embedding DINOv2 theo patch tiền cảnh và crop đã khử nền dọc thân gậy; bộ xác minh `DeBiasedCaneVerifierV3` giới hạn ảnh hưởng của người cầm (`anchor_prob_delta <= 0.10`) và áp dụng các cổng bác bỏ cấu trúc vật lý (`veto_not_white_shaft`, `veto_wood_or_dark_confuser`, `veto_no_continuous_shaft`, `veto_too_short`, `veto_umbrella_overlap`, `veto_parallel_trekking_poles`).

## Cấu trúc thư mục

- `src/`: Mã nguồn nghiên cứu (`dataset_builder_v3.py`, `spatial_features.py`, `keypoint_detector.py`, `shaft_segmenter.py`, `cane_verifier_v3.py`, `combined_pipeline_v3.py`, `train_vl_p3_models.py`, `benchmark_runner_v3.py`, `common_utils.py`).
- `manifests/`: `dataset_manifest.json`, `annotation_manifest.json`, `manual_cane_gt_overrides_v3.json`, `training_metrics.json`, `frozen_config.json`, `benchmark_metrics.json`.
- `models/`: Trọng số ONNX/Joblib/JSON đã huấn luyện (`models/*.onnx` được gitignore).
- `tests/`: `test_vl_p3_suite.py` (20 kiểm thử tự động).
- `reports/`: Các báo cáo phân tích chuyên sâu và `VL_P3_FINAL_RESEARCH_REPORT.md`.

## Tóm tắt kết quả chính (Blind Holdout `final_holdout_v3`, N=245)

| Preset | Known Recall | FAR | Precision | ROC-AUC | PR-AUC | Endpoint Err (px) | Latency P50 / P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_v2_lsd_hough_baseline` | 0.2000 | 0.0909 | 0.5161 | 0.7503 | 0.5518 | 336.29 | 169.62 / 220.62 |
| `A1_supervised_kp_centerline` | **0.3750** | 0.0242 | **0.8824** | **0.9116** | **0.8272** | **194.51** | 37.84 / 57.32 *(cached)* |
| `B1_supervised_seg_only` | 0.2000 | 0.0606 | 0.6154 | 0.8661 | 0.6529 | 259.77 | 42.06 / 64.15 *(cached)* |
| `B2_kp_guided_supervised_seg` | 0.1375 | 0.0182 | 0.7857 | 0.8649 | 0.7101 | 194.51 | 95.86 / 149.32 *(cached)* |
| `C1_b2_plus_masked_dinov2` | 0.2125 | 0.0364 | 0.7391 | 0.8442 | 0.6979 | 194.51 | 151.61 / 222.41 *(cached)* |
| `C2_full_vl_p3_pipeline` | 0.1250 | **0.0182** | 0.7692 | 0.8897 | 0.7636 | 194.51 | **402.61 / 652.59** *(uncached E2E)* |

Xem phân tích đầy đủ và kết luận hướng đi tiếp theo tại [VL_P3_FINAL_RESEARCH_REPORT.md](VL_P3_FINAL_RESEARCH_REPORT.md).
