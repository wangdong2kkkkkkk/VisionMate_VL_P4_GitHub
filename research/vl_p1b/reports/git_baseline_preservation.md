# Git & Baseline Integrity Preservation Audit (`VL-P1b`)

**Ngày kiểm toán:** 2026-11-14  
**Nhánh làm việc nghiên cứu VL-P1b:** `research/vl-p1b-reliability-proposal` (worktree `VisionMate_VL_P1b_AG`)  
**Commit nền tảng (base commit):** `988fdf1`

## 1. Trạng thái các Worktree & Baseline được bảo vệ

| Worktree | Nhánh | Commit HEAD | Trạng thái Git (`git status --short`) | Toàn vẹn dữ liệu / code |
|---|---|---|---|---|
| `VisionMateGlasses` | `hotfix/webcam-preview-fps-optimization` | `65c2a27` | Clean (`''`, 0 thay đổi) | Production UI, OCR, TTS, `yolox_nano.onnx`, baseline `8ae6f85`/`95a75e0`/`6e3039f`/`65c2a27` **nguyên vẹn 100%** |
| `VisionMate_VL_P1` | `research/vl-p1-manual` | `6e3039f` | Chỉ có `?? research/` (68 file nghiên cứu thủ công) | **Không đụng chạm** (0 file tracked thay đổi) |
| `VisionMate_VL_P1_AG` | `research/vl-p1-antigravity` | `988fdf1` | Clean (`''`, 0 thay đổi) | Khớp **`43/43` SHA-256** với snapshot trước phiên làm việc (`scratch/vl_p1_ag_snapshot.json`) |
| `VisionMate_VL_P1b_AG` | `research/vl-p1b-reliability-proposal` | `988fdf1` | Chỉ thêm mới `?? research/vl_p1b/` | Toàn bộ code, tài liệu, test, manifest, báo cáo chỉ nằm trong `research/vl_p1b/` |

## 2. Xác minh kiểm thử baseline VL-P1 & chống rò rỉ

- **Suite kiểm thử gốc `VisionMate_VL_P1_AG/research/vl_p1` (`python -m pytest -v`):** **`34 passed in 54.40s`** (0 failed, 0 skipped — giữ nguyên toàn vẹn nghiệm thu `988fdf1`).
- **Tái lập con số gốc VL-P1 (`reports/vl_p1_frozen_baseline.json`):** `ROI 39/50 = 78.0%`, `ROI FAR 3/25 = 12.0%`; `E2E 38/50 = 76.0%`, `E2E FAR 4/25 = 16.0%`, `Proposal Recall@0.5 43/50 = 86.0%` — khớp hoàn toàn với kết quả `A0_baseline_hybrid`, `B0_vl_p1_proposer__k4`, và `C1_vl_p1_baseline` trên `vl_p1_test` trong `reports/benchmark_metrics.json`.
- **Không hợp nhất nhánh / không tạo git tag / không mở Phase 2:** Chưa merge vào `main` hay `hotfix/*`, không sửa code production `app/`, và không commit file dữ liệu (`data/`), trọng số (`*.onnx`), `draft_selection.json`, hay `contact_sheets_v1b/` lên git (đã chặn bởi `research/vl_p1b/.gitignore`).
