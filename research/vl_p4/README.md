# VisionMate — VL-P4: Lightweight End-to-End White Cane Detection Research

**Trạng thái:** Independent Software Research Pilot (CPU-first, Windows 11 offline, chưa tích hợp production).  
**Worktree / Nhánh:** `VisionMate_VL_P4_AG` — `research/vl-p4-lightweight-cane-detector` (tách từ VL-P3 `2001119c63c82a714d17aba065985d5c4c95fa70`).

## Tóm tắt kiến trúc & kết quả chính

VL-P4 thay thế pipeline GBDT + 16 bộ lọc thủ công `448×448` + 2 lượt `DINOv2` của VL-P3 bằng một mạng **MobileNetV3-Small Multi-Head FPN (`1,045,989` tham số, `4.002 MB` ONNX FP32)** xử lý trực tiếp ảnh `256×256` đầu vào một lượt duy nhất trên CPU.

### Đối chứng trực tiếp `A0_VL_P3_A1` vs `C2_FinalCandidate` (`B2_CNN_MultiTask` ONNX FP32, Uncached E2E)

| Tập kiểm thử mù (Uncached E2E) | Mô hình | Known Recall | FAR | Precision | ROC-AUC | PR-AUC | Endpoint Err | E2E P50 / P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `final_holdout_v4` (`N=219`: 19 pos, 200 handheld neg) | `A0_VL_P3_A1` | 0.4211 (8/19) | 0.0250 (5/200) | 0.6154 | 0.9132 | 0.6214 | 281.03 px | 547.93 / 1274.76 ms |
| `final_holdout_v4` (`N=219`: 19 pos, 200 handheld neg) | **`C2_FinalCandidate`** | **0.6842** (13/19) | **0.0350** (7/200) | **0.6500** | **0.9524** | **0.7115** | **233.31 px** | **36.25 / 59.60 ms** |
| `final_holdout_v3` (`N=245`: 80 pos, 165 neg) | `A0_VL_P3_A1` | 0.3750 (30/80) | **0.0242** (4/165) | **0.8824** | 0.9116 | 0.8272 | **194.51 px** | 375.63 / 576.37 ms |
| `final_holdout_v3` (`N=245`: 80 pos, 165 neg) | **`C2_FinalCandidate`** | **0.7250** (58/80) | 0.0727 (12/165) | 0.8286 | **0.9208** | **0.8638** | 198.53 px | **24.77 / 30.78 ms** |

Xem báo cáo phân tích đầy đủ và phân loại Section XIV tại [VL_P4_FINAL_RESEARCH_REPORT.md](VL_P4_FINAL_RESEARCH_REPORT.md).
