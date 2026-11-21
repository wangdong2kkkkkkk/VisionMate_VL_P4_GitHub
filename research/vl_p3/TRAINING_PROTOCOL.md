# TRAINING PROTOCOL — VisionMate VL-P3

## 1. Quy trình huấn luyện hai giai đoạn (Zero-Leakage)

1. **Huấn luyện bộ trích xuất & định vị/phân đoạn (`src/train_vl_p3_models.py`):**
   - Chỉ sử dụng `train_v3` (`N=183`: `34` dương tính, `149` âm tính).
   - Tổng thời gian huấn luyện (`total_train_sec`): **303.54 giây**.
   - Học `PatchSemanticHead` (PCA 12 chiều + 5 đầu ngữ nghĩa trên patch tokens DINOv2 `16×16×384`).
   - Thực hiện 5 thí nghiệm ablation có kiểm soát cho `A1` và đánh giá trên `val_cal_v3` (`16` ảnh dương tính):

| Ablation (`train_vl_p3_models.py`) | Mẫu pixel | Mean Endpoint Err (px) | Norm Diag | PDJ@0.10 | Both PDJ@0.10 | Chamfer (px) | Shaft Cov | Angle Err (°) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1_keypoints_only` | 63,403 | 203.53 | 0.1585 | 0.5000 | 0.3750 | 143.95 | 0.4025 | 21.68 |
| `2_centerline_only` | 63,403 | 227.93 | 0.1817 | 0.4375 | 0.2500 | 145.08 | 0.3600 | 18.99 |
| `3_keypoints_plus_centerline` | 63,403 | 192.40 | 0.1545 | 0.5312 | 0.3750 | 128.59 | 0.4225 | 23.71 |
| `4_kp_cl_plus_hard_neg_mining` | 63,430 | **161.58** | 0.1325 | 0.4688 | 0.3750 | **83.30** | 0.4500 | 21.68 |
| `5_full_aug_and_occlusion` *(final)* | 113,001 | 171.81 | **0.1252** | **0.5938** | **0.3750** | 106.04 | **0.4625** | 24.84 |

   - Huấn luyện `B1_standalone` và `B2_kp_guided` (`mean_mask_iou` trên `val_cal_v3` trước khi tinh chỉnh `sparse_refine_k`: `B1 = 0.0635`, `B2 = 0.0652`).

2. **Tối ưu suy luận không mất mát (`sparse_refine_k=160`, `np.partition`):**
   - Thay `np.median` 8 kênh bằng `np.partition(..., (3, 4))` (giữ nguyên giá trị toán học).
   - Giảm `sparse_refine_k` khi suy luận từ `480`/`560` xuống `160` mà không cần huấn luyện lại cây HGB.

3. **Hiệu chuẩn & Đóng băng Phase A (`benchmark_runner_v3.py --phase a`):**
   - Huấn luyện OOF `DeBiasedCaneVerifierV3` trên `train_v3` (`N=183`: `oof_pos_mean = 0.9311`, `oof_neg_mean = 0.0180`).
   - Hiệu chuẩn ngưỡng `t_known`, `t_uncertain` cho tất cả 6 preset trên `val_cal_v3` (`N=98`) với mục tiêu `FAR <= 5%`, sau đó khóa cứng vào `manifests/frozen_config.json` (`frozen_at: 2026-11-15T23:10:18`).
   - Ngưỡng `C2` đã khóa: `verifier_t_known = 0.41`, `verifier_t_uncertain = 0.23`, `apply_structural_vetoes = true`.

4. **Đánh giá mù một lần duy nhất Phase B (`benchmark_runner_v3.py --phase b`):**
   - Chạy duy nhất 1 lần sau khi khóa `frozen_config.json` trên `final_holdout_v3` (`N=245`), `ambiguous_cane_v3` (`N=26`), `final_holdout_v2` (`N=180`), và `vl_p1c/final_holdout_full` (`N=300`).
