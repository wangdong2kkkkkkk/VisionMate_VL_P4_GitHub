# Kiểm Toán Toàn Vẹn Git & Bảo Toàn Baseline — VisionMate VL-P1c

**Ngày kiểm toán:** 2026-11-15  
**Nhánh nghiên cứu VL-P1c:** `research/vl-p1c-calibration-multiproposal` (worktree `VisionMate_VL_P1c_AG`)  
**Commit nền tảng (base commit):** `dacd24571ae2c1f6fe51db75e832c0bc5d622215` (cây lịch sử kế thừa `65c2a27 -> 988fdf1 -> dacd245`)

---

## 1. Trạng thái Toàn bộ Worktree & Baseline Được Bảo Vệ

| Worktree | Nhánh | Commit HEAD | Trạng thái Git (`git status --short`) | Toàn vẹn code / dữ liệu |
|---|---|---|---|---|
| `VisionMateGlasses` | `hotfix/webcam-preview-fps-optimization` | `65c2a27a21fac65e5e3de78a4feba081dd9e192e` | Clean (`''`, `0` thay đổi) | Production UI, OCR, OneCore TTS, `yolox_nano.onnx`, mốc `8ae6f85`/`95a75e0`/`6e3039f`/`65c2a27` **nguyên vẹn 100%** |
| `VisionMate_VL_P1` | `research/vl-p1-manual` | `6e3039f97f8065153dc96c4affb4f9371d3bcc7e` | `?? research/` (nguyên trạng ban đầu) | **Không đụng chạm** (`0` file tracked thay đổi) |
| `VisionMate_VL_P1_AG` | `research/vl-p1-antigravity` | `988fdf14a574863f6246017c5ccf8350eef5fdb8` | Clean (`''`, `0` thay đổi) | **Nguyên vẹn 100%** (`0` file sửa đổi) |
| `VisionMate_VL_P1b_AG` | `research/vl-p1b-reliability-proposal` | `dacd24571ae2c1f6fe51db75e832c0bc5d622215` | Clean (`''`, `0` thay đổi) | **Nguyên vẹn 100%** (`0` file sửa đổi) |
| **`VisionMate_VL_P1c_AG`** | **`research/vl-p1c-calibration-multiproposal`** | Khởi tạo từ `dacd245` | Mọi thay đổi cô lập hoàn toàn trong `research/vl_p1c/` | Chỉ chứa code, manifest, báo cáo và tests VL-P1c; loại trừ `data/`, `*.onnx`, `contact_sheets*`, `staging*`, `draft_selection_v1c.json` |

---

## 2. Xác minh Kiểm thử Tự động Qua Cả 3 Thế hệ Nghiên cứu (Chạy thực tế ngày 2026-11-15)

| Bộ kiểm thử | Lệnh chạy thực tế | Kết quả | Ghi chú |
|---|---|---|---|
| **VL-P1c (Mới — 20 nhóm bắt buộc)** | `python -m pytest -v` (`VisionMate_VL_P1c_AG/research/vl_p1c`) | **`20 passed in 32.23s`** (`0 failed`, `0 skipped`) | Toàn bộ 20/20 bài kiểm tra (`test_01`..`test_20`) đạt |
| **VL-P1b (Kế thừa — `dacd245`)** | `python -m pytest -v` (`VisionMate_VL_P1b_AG/research/vl_p1b`) | **`10 passed in 12.56s`** (`0 failed`, `0 skipped`) | Không hồi quy |
| **VL-P1 (Kế thừa — `988fdf1`)** | `python -m pytest -v` (`VisionMate_VL_P1_AG/research/vl_p1`) | **`34 passed in 22.71s`** (`0 failed`, `0 skipped`) | Không hồi quy |

---

## 3. Tái lập Số liệu Đóng Băng VL-P1b & Chính sách Cô lập Git

- **Tái lập `reports/vl_p1b_frozen_baseline.json` (`source_commit = dacd245`):** **`all_exact_matches = true` (`16/16` kiểm tra khớp chính xác)** (`A0 = 173/214 = 80.8%`, `A1 = 147/214 = 68.7%`, `A3 = 168/214 = 78.5%`, `C1 = 162/214 = 75.7%`, `C4 = 142/214 = 66.4%`).
- **Quy tắc dừng & không can thiệp nhánh chính:** Không hợp nhất vào `main`/`master` hay `hotfix/*`, không tạo Git tag, không chỉnh sửa `app/` hay `yolox_nano.onnx` mặc định của VisionMate, không mở Phase 2 phần cứng hay VL-P1d.
