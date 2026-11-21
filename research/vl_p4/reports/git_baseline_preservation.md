# Git & Baseline Preservation Audit — VisionMate VL-P4

## 1. Trạng thái Git trên toàn bộ 6 worktree

| Worktree | Nhánh | Commit HEAD | Trạng thái |
| --- | --- | --- | --- |
| `VisionMate_VL_P1_AG` | `research/vl-p1-antigravity` | `988fdf1` | Sạch, không thay đổi |
| `VisionMate_VL_P1b_AG` | `research/vl-p1b-reliability-proposal` | `dacd245` | Sạch, không thay đổi |
| `VisionMate_VL_P1c_AG` | `research/vl-p1c-calibration-multiproposal` | `9587ad4` | Sạch, không thay đổi |
| `VisionMate_VL_P2_AG` | `research/vl-p2-white-cane-segmentation` | `4a10113` | Sạch, không thay đổi |
| `VisionMate_VL_P3_AG` | `research/vl-p3-supervised-cane-detection` | `2001119` | Sạch, không thay đổi |
| `VisionMate_VL_P4_AG` | `research/vl-p4-lightweight-cane-detector` | `2001119` *(trước commit V4)* | Chỉ có `research/vl_p4/` |

- Không chạm vào `app/`, production OCR/TTS/webcam, `models/object_detection/yolox_nano.onnx`, hay các nhánh lịch sử; không merge, tag hoặc push.

## 2. Kết quả kiểm thử tự động & hồi quy (tổng 144 test thực chạy)

| Suite | Kết quả | Thời gian | Ghi chú |
| --- | --- | ---: | --- |
| `VL-P4` (`test_vl_p4_suite.py`) | **20 passed** | 52.87s | Toàn bộ 20 test mới đạt |
| `VL-P3` | **20 passed** | 25.83s | Nguyên vẹn |
| `VL-P2` | **12 passed, 1 failed** | 32.26s | `test_12_runtime_and_memory_budget` (`473.525 ms > 300.0 ms`), không sửa baseline cũ |
| `VL-P1c` | **20 passed** | 98.50s | Nguyên vẹn |
| `VL-P1b` | **10 passed** | 43.58s | Nguyên vẹn |
| `VL-P1` | **61 passed** | 64.73s | Nguyên vẹn |
