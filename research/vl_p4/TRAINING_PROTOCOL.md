# TRAINING PROTOCOL — VisionMate VL-P4

- **Môi trường:** Windows 11, Python 3.10.7, `torch 2.14.1+cpu`, `torchvision 0.29.2+cpu`, `onnx 1.23.2`, `onnxruntime 1.23.2`, seed cố định `42`.
- **Tiền xử lý:** Letterbox vuông giữ tỷ lệ ảnh gốc -> `256×256` RGB chuẩn hóa ImageNet; đầu ra lưới không gian `64×64` (stride 4).
- **Lịch trình huấn luyện (`src/train_vl_p4_models.py`):**
  - **Class-balanced sampling:** Mỗi epoch lấy mẫu cân bằng (`160–192` dương tính + `160–240` âm tính khó từ `train_v4`).
  - **Augmentation nhất quán hình học:** Lật ngang (`p=0.5`), xoay `±15°`, scale `0.86–1.14`, dịch `±6%`, random occlusion cutout (`p=0.25`), jitter độ sáng/tương phản; tự động biến đổi đồng bộ endpoints, centerline và mask.
  - **Stage 1 (Warmup):** Đóng băng `features`, huấn luyện FPN + head (`lr=1.5e-3..1.8e-3`).
  - **Stage 2 (Fine-tune):** Mở băng toàn bộ (`lr_backbone=1.8e-4..2.2e-4`, `lr_head=5.5e-4..6.5e-4`, `AdamW` + `CosineAnnealingLR`, chọn checkpoint tốt nhất trên `val_cal_v4`).
- **Kết quả xuất ONNX & kiểm tra parity FP32 vs INT8 (`val_cal_v4`):**
  - `A1_CNN_Keypoint`: `val_roc_fp32 = 0.7386` -> `val_roc_int8 = 0.6629` (`max_prob_diff = 0.8947`, `int8_viable = false`).
  - `B1_CNN_Segmentation`: `val_roc_fp32 = 0.7575` -> `val_roc_int8 = 0.6470` (`max_prob_diff = 0.8333`, `int8_viable = false`).
  - `B2_CNN_MultiTask`: `val_roc_fp32 = 0.7623` -> `val_roc_int8 = 0.6373` (`max_prob_diff = 0.8743`, `int8_viable = false`).
  - Vì dynamic INT8 làm sụt mạnh `ROC-AUC` (`-0.076` đến `-0.125`), toàn bộ đánh giá chính thức sử dụng **ONNX FP32 (`4.002 MB`)** với parity khớp 100% PyTorch.
- **Hiệu chuẩn & Đóng băng Phase A (`val_cal_v4`, N=158):**
  - `A1_CNN_Keypoint`: `t_known = 0.73`, `t_uncertain = 0.55` (`recall=0.250, far=0.049, roc=0.738`).
  - `B1_CNN_Segmentation`: `t_known = 0.76`, `t_uncertain = 0.58` (`recall=0.312, far=0.042, roc=0.757`).
  - `B2_CNN_MultiTask`: `t_known = 0.89`, `t_uncertain = 0.71` (`recall=0.250, far=0.014, roc=0.762`).
  - `C1_CNN_WithVerifier`: `t_known = 0.51`, `t_uncertain = 0.33` (`recall=0.250, far=0.028, roc=0.803`).
  - **Chọn `C2_FinalCandidate = B2_CNN_MultiTask`** trên `val_cal_v4` (FAR thấp nhất `0.014`, ưu tiên kết hợp multi-task).
