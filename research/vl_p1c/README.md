# VisionMate Glasses — VL-P1c: Calibration & Multi-Proposal Reliability Research

> **Trạng thái:** Independent Research Pilot (`research/vl-p1c-calibration-multiproposal`, base commit `dacd245`) — **CHƯA TÍCH HỢP PRODUCTION (`RESEARCH PROMISING` / Selective-Only, KHÔNG DÙNG TRỰC TIẾP TRÊN WHITE CANE E2E)**.  
> Mọi mã nguồn, dữ liệu, cấu hình đóng băng và báo cáo nằm cô lập hoàn toàn trong `research/vl_p1c/`. Production (`app/`), `research/vl_p1/` (`988fdf1`) và `research/vl_p1b/` (`dacd245`) không bị chỉnh sửa.

---

## 1. Cấu trúc thư mục

- `src/common_utils.py`, `src/labeling_protocol.py`: Tiện ích CPU-only ONNX (`num_threads=4`), Wilson 95% CI, McNemar exact paired test, SHA-256, `aHash`/`dHash`, chuẩn phân lớp & lọc giấy phép Wikimedia Commons.
- `src/reproduce_vl_p1b_baseline.py`: Tái lập đầy đủ 16/16 chỉ số đóng băng của VL-P1b (`dacd245`) tại `reports/vl_p1b_frozen_baseline.json` (`all_exact_matches=true`).
- `src/collect_vl_p1c_dataset.py`, `src/collect_vl_p1c_supplements*.py`, `src/build_vl_p1c_manifest.py`: Thu thập ảnh mới mở rộng, kiểm tra chống rò rỉ 4 tầng (`SHA-256`, tiêu đề, `cluster_id`, `aHash/dHash <= 6`), áp dụng `manifests/manual_gt_bboxes_v1c.json` (69 mục), và đóng gói `manifests/dataset_manifest.json` (`zero_overlap_verified: true`).
- `src/calibration_research.py`: Nghiên cứu hiệu chuẩn `R0..R5` (group-aware 5-fold CV, `delta_floor` grid `0.04/0.06/0.08`, out-of-fold Ridge verifier, safety floor, 3 trạng thái `KNOWN`/`AMBIGUOUS`/`UNKNOWN`, và sample-size scaling `N=50..200`).
- `src/thin_object_proposer.py`: Mô-đun đề xuất chuyên biệt cho vật mảnh dài có định hướng (`oriented_thin_cane`, `person_cane_ray`, thin corridor mask).
- `src/proposal_research_v1c.py`, `src/proposal_types.py`: Bộ xếp hạng cân bằng khe nguồn `ProposalRankerV1c` (`F_primary`, `F_context`, `F_small`, `F_thin`), thí nghiệm công bằng `K=4, 6, 8` (`B0`, `B_full_v1b`, `B_v1c_proposer`) kèm bộ chỉ số hình học riêng cho gậy trắng.
- `src/combined_pipeline_v1c.py`: Pipeline E2E (`C1`, `C4`, `C_v1c_cal_only`, `C_v1c_prop_only`, `C_v1c_full__k4/k6/k8`) tích hợp kiểm soát FP đa crop (`alpha_k`, `source_offsets`, non-redundant support `0.10 <= IoU <= 0.68`).
- `src/benchmark_runner_v1c.py`: Thực thi nghiêm ngặt 2 pha mù đôi (`--phase a` đóng băng trên `calibration_v1c` `N=200`, `--phase b` chấm mù 1 lần duy nhất trên `final_holdout` `N=300` + `vl_p1_test` + `vl_p1b_holdout` + `ambiguous_boundary_v1c`).
- `tests/test_vl_p1c_suite.py`: Bộ 20 nhóm kiểm thử bắt buộc (độc lập dữ liệu, tái lập hiệu chuẩn, sàn ngưỡng, kiểm soát đa proposal, đa dạng slot, ngân sách `K` công bằng, không dùng GT khi suy luận, checksum ONNX, phần cứng, hồi quy).

---

## 2. Nguồn gốc & Checksum Mô hình (Offline 100%, CPU-First)

| Thành phần | Đường dẫn | SHA-256 | Vai trò trong VL-P1c |
|---|---|---|---|
| **DINOv2-Small INT8** | `models/dinov2_small_quantized.onnx` (kế thừa `VL-P1`/`VL-P1b`, Apache-2.0) | `3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917` | Trích xuất embedding 384-D trên ROI & top-K proposal crops |
| **YOLOX-Nano** | `models/yolox_nano.onnx` (kế thừa production VisionMate `65c2a27`, Apache-2.0) | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` | Sinh đề xuất đối tượng tổng quát & neo người (`person_hand_zone`, `person_cane_ray`) |
| **Dataset Manifest** | `manifests/dataset_manifest.json` | `b6f4c06fa506afaea622a7a87db3bf5ca65b653834f3137d5cf9799e75c3df52` | Tập `calibration_v1c` (`N=200`), `final_holdout` (`N=300`), `ambiguous_boundary_v1c` (`N=15`) |
| **OOF Verifier State** | `manifests/frozen_a3_oof_state.json` | `bf3b7191737d27a7462cb52ce40a3430da2c84c80d9a0e22decc6c8c4583387f` | Ridge verifier 784-D đóng băng từ Phase A (`ridge_alpha=24.0`, `cv_gate_passed=true`) |

---

## 3. Cách tái lập kiểm chứng (Offline 100%)

```powershell
cd research/vl_p1c

# 1. Xác minh manifest và cổng chống rò rỉ
python -m src.build_vl_p1c_manifest --stage finalize

# 2. Tái lập baseline VL-P1b (dacd245)
python -m src.reproduce_vl_p1b_baseline

# 3. Pha A: Chạy hiệu chuẩn 5-fold CV trên calibration_v1c (N=200) và đóng băng cấu hình
python -u -m src.benchmark_runner_v1c --phase a

# 4. Pha B: Đánh giá mù 1 lần duy nhất trên final_holdout (N=300) và các tập kế thừa
python -u -m src.benchmark_runner_v1c --phase b

# 5. Chạy toàn bộ 20 nhóm test kiểm chứng VL-P1c
python -m pytest -v
```

---

## 4. Tài liệu & Báo cáo bàn giao

- **Báo cáo tổng kết nghiên cứu:** [VL_P1C_FINAL_RESEARCH_REPORT.md](VL_P1C_FINAL_RESEARCH_REPORT.md) (bản sao: [reports/VL_P1C_FINAL_RESEARCH_REPORT.md](reports/VL_P1C_FINAL_RESEARCH_REPORT.md))
- **Giao thức dữ liệu & chống rò rỉ:** [DATASET_PROTOCOL.md](DATASET_PROTOCOL.md) ([reports/dataset_protocol.md](reports/dataset_protocol.md))
- **Giao thức hiệu chuẩn 5-fold CV & OOF:** [calibration_protocol.md](calibration_protocol.md) ([reports/calibration_protocol.md](reports/calibration_protocol.md))
- **Số liệu đóng băng & kết quả thô:** [frozen_config.json](frozen_config.json), [calibration_metrics.json](calibration_metrics.json), [benchmark_metrics.json](benchmark_metrics.json)
- **Ma trận nhầm lẫn & ablation proposal:** [confusion_matrices.md](confusion_matrices.md), [proposal_ablation.md](proposal_ablation.md)
- **Phân tích lỗi chi tiết:** [failure_analysis.md](failure_analysis.md) ([reports/failure_analysis.md](reports/failure_analysis.md))
- **Báo cáo hiệu năng CPU/RAM:** [performance_report.md](performance_report.md) ([reports/performance_report.md](reports/performance_report.md))
- **Kiểm toán bảo toàn baseline Git & worktrees:** [git_baseline_preservation.md](git_baseline_preservation.md) ([reports/git_baseline_preservation.md](reports/git_baseline_preservation.md))
