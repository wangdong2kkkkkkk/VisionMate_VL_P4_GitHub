# VisionMate Glasses — VL-P1b: Recognition Reliability & Object Proposal Research

> **Trạng thái:** Independent Research Pilot (`research/vl-p1b-reliability-proposal`, base commit `988fdf1`) — **CHƯA TÍCH HỢP PRODUCTION** (`NOT READY FOR PRODUCTION`).  
> Mọi hoạt động nghiên cứu nằm cô lập hoàn toàn trong `research/vl_p1b/`.

## 1. Cấu trúc thư mục

- `src/common_utils.py`, `src/labeling_protocol.py`: Tiện ích offline (CPU-only ONNX, SHA-256, aHash/dHash, Wilson 95% CI, chuẩn nhãn & lọc giấy phép mở).
- `src/reproduce_vl_p1_baseline.py`: Tái lập con số baseline VL-P1 (`988fdf1`) tại `reports/vl_p1_frozen_baseline.json`.
- `src/collect_*.py` & `src/build_vl_p1b_manifest.py`: Thu thập ảnh Wikimedia Commons có giấy phép mở (`CC0`, `PD`, `CC BY`, `CC BY-SA`), xác minh 0 trùng lặp SHA-256 / cụm phiên / pHash <= 6 với VL-P1 và giữa các split, áp dụng `manifests/manual_gt_bboxes.json` đã kiểm định trực quan.
- `src/hard_negative_research.py` (cùng wrapper `hard_negative_research.py`): Bài toán A — `A0_baseline_hybrid`, `A1_neg_prototypes`, `A2_adaptive_thresholds`, `A3_full_verifier` (Ridge + geometry 784-D).
- `src/proposal_research.py` (cùng wrapper `proposal_research.py`): Bài toán B — `yolox_strict`, `yolox_low`, `B0_vl_p1_proposer`, `B1_multiscale`, `B2_geometry_thin`, `B_full_v1b` (`K=3,4,6,8`), tuyệt đối không dùng GT bbox khi suy luận.
- `src/combined_pipeline.py`: Pipeline E2E (`C1..C4`).
- `src/benchmark_runner.py` (cùng wrapper `benchmark_runner.py`): Hiệu chuẩn khóa cứng trên `calibration_v1b` trước khi chạm `holdout`, chạy đánh giá & sinh `reports/benchmark_metrics.json`, `reports/confusion_matrices.md`.
- `tests/`: Unit test & integrity/benchmark verification tests.

## 2. Cách chạy lại kiểm chứng (Offline 100%)

```bash
cd research/vl_p1b
python -m src.benchmark_runner      # Chạy lại hiệu chuẩn + benchmark đơn luồng CPU
python -m pytest -v tests/          # Chạy toàn bộ 10 bài test unit + integrity/schema
```

## 3. Tài liệu & Báo cáo

- Báo cáo tổng kết nghiên cứu: [VL_P1B_FINAL_RESEARCH_REPORT.md](VL_P1B_FINAL_RESEARCH_REPORT.md)
- Phân tích lỗi chi tiết: [failure_analysis.md](failure_analysis.md) & [reports/failure_analysis.md](reports/failure_analysis.md)
- Kiểm toán toàn vẹn baseline & worktree: [reports/git_baseline_preservation.md](reports/git_baseline_preservation.md)
- Giao thức dữ liệu & nguồn mô hình: [DATASET_PROTOCOL.md](DATASET_PROTOCOL.md), [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md)
