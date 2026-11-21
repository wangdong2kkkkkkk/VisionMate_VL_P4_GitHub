# Giao Thức Dữ Liệu & Quy Tắc Gắn Nhãn — VisionMate VL-P1b

Tài liệu này quy định ranh giới phân loại 5 lớp đồ vật, 7 nhóm âm tính dễ nhầm (`hard-negative`), các ca ranh giới (`LABEL_AMBIGUOUS`), quy chuẩn thu thập từ Wikimedia Commons và cơ chế cô lập chống rò rỉ dữ liệu tuyệt đối.

---

## 1. Quy tắc gắn nhãn ranh giới (Không bẻ định nghĩa lớp)

| Lớp mục tiêu | Thuộc về lớp (`POSITIVE`) | Đưa vào `unknown` (Confuser) | Đánh dấu `LABEL_AMBIGUOUS` (báo cáo riêng) |
|---|---|---|---|
| `object_01` — **Gậy trắng hỗ trợ người khiếm thị** | Gậy dò đường / white cane / long cane / red-tipped cane (thẳng hoặc gập, đứng riêng hoặc người mù đang cầm). | Gậy chống thời trang, gậy gỗ, gậy trekking, gậy kính viễn vọng, nạng, ô dù, thanh kim loại. | Gậy chống sơn trắng trơn nhưng thiếu bối cảnh/đầu típ đặc trưng của white cane. |
| `object_02` — **Chùm chìa khóa** | $\ge 2$ chìa trên khoen, hoặc 1 chìa gắn móc khóa/fob rõ là chùm chìa cá nhân. | Ổ khóa đơn, kéo, thìa, dụng cụ kim loại, khóa cửa cố định trên tường. | Chìa khóa đơn trần (ví dụ chìa cổ, 1 chìa rời không khoen/không fob). |
| `object_03` — **Ví tiền** | Ví gập đôi/gập ba (`bi-fold`/`tri-fold`), ví dài khóa kéo, ví đựng xu (`coin purse`) cầm tay. | Bao kính/hộp kính da, sổ tay bìa da, túi đeo chéo/cặp, hộp đựng cứng. | Túi zip nhỏ đa năng không phân biệt rõ ví tiền hay túi đựng cáp/đồ trang điểm. |
| `object_04` — **Đôi dép** | Dép xỏ ngón (`flip-flops`), dép quai ngang (`slides`), dép đi trong nhà (`slippers`) hở kết cấu dép. | Vớ/tất, guốc mộc truyền thống không rõ kết cấu dép hiện đại. | Clog/sục bít mũi hở gót lai giữa giày và dép; dép rọ bít mũi kín. |
| `object_05` — **Đôi giày** | Sneakers, giày thể thao, giày tây/da, bốt ôm trọn bàn chân và có đế giày. | Vớ/tất, bao bọc giày mỏng. | Sandal quai hậu bít gần hết mu bàn chân (lai giày-sandal). |

---

## 2. Phân tách các tập dữ liệu & Chống rò rỉ

| Tập dữ liệu | Thư mục trong `research/vl_p1b/data/` | Vai trò trong thực nghiệm | Quy tắc độc lập |
|---|---|---|---|
| **Train Positives** | `train_pos/` (`25` ảnh: `5/nhóm`) | Bộ 5-shot học mẫu dương tính (kế thừa `vl_p1/data/train`). | **Không dùng** để chọn ngưỡng hay đánh giá holdout. |
| **Hard-Negative Gallery** | `hard_neg_gallery/` (`50` ảnh across 7 nhóm confuser) | Kho mẫu âm tính khó để tạo *Negative Prototypes* và huấn luyện `CandidateVerifier`. | Tuyệt đối không chứa bất kỳ ảnh nào thuộc `vl_p1/data/test`, `calibration_v1b`, hay `holdout_v1b`. |
| **Calibration Set** | `calibration_v1b/` (ROI) & `calibration_v1b_full/` (`50` ảnh: `25` pos + `25` hard-neg) | Hiệu chuẩn mọi ngưỡng $\tau_c$, $\delta_{\text{pos}}$, $\delta_{\text{neg}}$ và chọn cấu hình tốt nhất trước khi đóng băng. | Chỉ hiệu chuẩn trên tập này rồi ghi vào `manifests/frozen_thresholds.json`. |
| **Independent Holdout** | `holdout_roi/` & `holdout_full/` (`220` ảnh: `100` pos `20/nhóm` + `120` neg) | Đánh giá tổng quát hóa **1 lần duy nhất** trên ảnh hoàn toàn mới chưa từng chạm tới. | Kiểm tra chéo 0 trùng `SHA-256`, `source_url`, `aHash`/`dHash` ($d > 6$) với tất cả các tập còn lại và toàn bộ 125 ảnh VL-P1. |
| **Ambiguous Boundary** | `ambiguous_boundary/` | Các ca biên `LABEL_AMBIGUOUS` nhằm kiểm tra xem hệ thống có tự tin đoán bừa hay biết do dự/từ chối. | Báo cáo tách biệt, không trộn vào Acc/FAR chính thức. |
| **VL-P1 Historical Regression** | `../vl_p1/data/` (`75` ảnh test: `50` pos + `25` neg) | Đối chiếu hồi quy với kết quả VL-P1 (`78.0% / 12.0%` và `76.0% / 16.0%`). | Không dùng để chứng minh tổng quát hóa mới. |

---

## 3. Chính sách bản quyền dữ liệu

- 19/25 ảnh trong `train_pos/` kế thừa từ nguồn thủ công VL-P1 (`UNVERIFIED_MANUAL_WEB`) được **giữ nội bộ nghiên cứu và không mặc nhiên phân phối công khai** (bị loại khỏi Git qua `.gitignore`).
- Mọi ảnh thu thập mới cho `hard_neg_gallery`, `calibration_v1b`, `holdout_v1b` và `ambiguous_boundary` chỉ chấp nhận ảnh thật từ **Wikimedia Commons** có giấy phép mở (`CC0`, `Public domain`, `CC BY`, `CC BY-SA`), ghi nhận đầy đủ: `id`, `split`, `label`, `source_url`, `license`, `author`, `sha256`, `ahash`, `dhash`, `roi_bbox_xywh`, `session_key` (mã file Wikimedia/người chụp chống trùng cùng phiên/mẫu sản phẩm).
