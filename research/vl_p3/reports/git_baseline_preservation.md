# Git & Baseline Preservation Audit — VisionMate VL-P3

## 1. Trạng thái các worktree trước khi commit VL-P3

| Worktree | Nhánh | HEAD Commit | Trạng thái sửa đổi ngoài phạm vi |
| --- | --- | --- | --- |
| `VisionMate_VL_P1_AG` | `research/vl-p1-antigravity` | `988fdf1` | Sạch (không thay đổi) |
| `VisionMate_VL_P1b_AG` | `research/vl-p1b-reliability-proposal` | `dacd245` | Sạch (không thay đổi) |
| `VisionMate_VL_P1c_AG` | `research/vl-p1c-calibration-multiproposal` | `9587ad4` | Sạch (không thay đổi) |
| `VisionMate_VL_P2_AG` | `research/vl-p2-white-cane-segmentation` | `4a10113` | Sạch (không thay đổi) |
| `VisionMate_VL_P3_AG` | `research/vl-p3-supervised-cane-detection` | `4a10113` *(trước commit)* | Chỉ có `research/vl_p3/` |

## 2. Kiểm thử hồi quy Read-Only trên các baseline trước

| Suite | Lệnh kiểm tra | Kết quả | Thời gian |
| --- | --- | --- | ---: |
| `VL-P3` | `python -m pytest research/vl_p3/tests/test_vl_p3_suite.py -v` | **20 passed** | 17.50s |
| `VL-P2` | `python -m pytest research/vl_p2/tests -q` | **13 passed** | 18.02s |
| `VL-P1c` | `python -m pytest research/vl_p1c/tests -q` | **20 passed** | 50.60s |
| `VL-P1b` | `python -m pytest tests -q` *(từ `research/vl_p1b`)* | **10 passed** | 17.73s |
| `VL-P1` | `python -m pytest research/vl_p1/tests tests/unit -q` | **61 passed** | 38.47s |

- Tổng cộng **124 / 124 test** trên toàn bộ 5 worktree đều đạt, không có sửa đổi nào chạm vào `app/`, production code hay các nhánh nghiên cứu trước.
- Tệp `models/*.onnx` được giữ ngoài Git thông qua `.gitignore` (chỉ lưu hash SHA-256 trong `MODEL_PROVENANCE.md` và `frozen_config.json`).
