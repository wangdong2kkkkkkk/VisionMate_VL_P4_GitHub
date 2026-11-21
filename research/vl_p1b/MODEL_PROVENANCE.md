# Nguồn Gốc, Giấy Phép & Mã Kiểm Chứng Mô Hình — VisionMate VL-P1b

Mọi mô hình sử dụng trong VL-P1b đều chạy **100% offline trên CPU (`CPUExecutionProvider` / OpenCV DNN CPU)**, không gọi mạng trong quá trình inference, không thêm checkpoint nặng và không thay thế `yolox_nano.onnx` mặc định của VisionMate OD-1.

| Mã mô hình | Vai trò trong VL-P1b | Tệp cục bộ | Kích thước | SHA-256 | Nguồn gốc | Giấy phép |
|---|---|---|---:|---|---|---|
| `dinov2_small_int8` *(Chủ lực)* | Trích xuất embedding 768-d (`L2(CLS) \|\| L2(mean_patch)`), phương pháp A & C | `research/vl_p1b/models/dinov2_small_quantized.onnx` | 24,451,943 bytes (`23.319 MB`) | `3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917` | `https://huggingface.co/Xenova/dinov2-small` (`onnx/model_quantized.onnx`) | Apache-2.0 |
| `yolox_nano` *(Proposer baseline)* | Bộ phát hiện 80 lớp COCO cho `yolox_strict`, `yolox_low`, và thành phần trong proposer lai | `models/object_detection/yolox_nano.onnx` | 3,659,407 bytes (`3.490 MB`) | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` | Kế thừa nguyên trạng từ baseline OD-1 (`95a75e0`) | Apache-2.0 (YOLOX OD-1 baseline) |
| `mobilenetv3_small_100` *(Đối chứng VL-P1)* | Bản sao phục vụ đối chứng tái lập khi cần | `research/vl_p1b/models/mobilenetv3_small_100_Opset17.onnx` | 10,178,712 bytes (`9.707 MB`) | `6bfccfd39c49295c353ee96da83eaa47b1a4f2fbbfc300f39779ca8dee9c73b7` | `https://huggingface.co/onnxmodelzoo/mobilenetv3_small_100_Opset17` | Apache-2.0 |
| `efficientnet_b0` *(Đối chứng VL-P1)* | Bản sao phục vụ đối chứng tái lập khi cần | `research/vl_p1b/models/efficientnet_b0_Opset17.onnx` | 21,128,851 bytes (`20.150 MB`) | `e76596a2b9e27c7c734c38550859105b43fec926f13447a84dad175eb994068a` | `https://huggingface.co/onnxmodelzoo/efficientnet_b0_Opset17` | Apache-2.0 |

## Ghi chú kiến trúc & không gian đặc trưng
- **`dinov2_small_int8`**: Đầu vào RGB `224×224` (đệm nền trung vị biên giữ nguyên tỷ lệ khung hình, chuẩn hóa ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`). Tensor đầu ra `last_hidden_state` có shape `(1, 257, 384)`. Token `CLS` (`hidden[0, 0, :]`) và trung bình 256 patch tokens (`hidden[0, 1:, :].mean(axis=0)`) được chuẩn hóa $L_2$ riêng lẻ rồi nối ghép và chuẩn hóa $L_2$ lại thành vector 768 chiều.
- **Không fine-tune backbone**: Toàn bộ trọng số DINOv2-Small INT8 được đóng băng hoàn toàn; cải tiến VL-P1b chỉ tác động lên lớp *Negative Prototypes*, *Adaptive Confidence Threshold*, *Lightweight Candidate Verification* (siêu phẳng tuyến tính $L_2$ + 16 đặc trưng hình học/quang học của ROI) và *Multi-Scale / Geometry Object Proposal*.
