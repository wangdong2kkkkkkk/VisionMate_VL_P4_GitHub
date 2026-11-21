# Giao Thức Dữ Liệu, Phân Tách & Chống Rò Rỉ — VisionMate VL-P1c

Tài liệu này mô tả chuẩn gắn nhãn 5 lớp đồ vật mục tiêu, 7 nhóm âm tính khó (*hard-negative*), lớp ranh giới (`AMBIGUOUS` / `LABEL_AMBIGUOUS`), phân tách tập hiệu chuẩn `calibration_v1c` (`N=200`) và tập giữ lại cuối cùng `final_holdout` (`N=300`), cùng cơ chế kiểm tra chống rò rỉ đa tầng (`SHA-256`, `cluster_id`, perceptual `aHash`/`dHash`).

---

## 1. Quy chuẩn gắn nhãn ranh giới lớp (Giữ nguyên nhất quán từ VL-P1b)

| Lớp mục tiêu | Thuộc về lớp (`POSITIVE`) | Đưa vào `unknown` (Confuser) | Đánh dấu `AMBIGUOUS` (`ambiguous_boundary_v1c`) |
|---|---|---|---|
| `object_01` — **Gậy trắng khiếm thị** | Gậy dò đường (`white cane` / `long cane` / `red-tipped cane`, thẳng hoặc gập, nằm riêng hoặc có người khiếm thị cầm). | Gậy chống thời trang, gậy gỗ, gậy trekking, gậy kính viễn vọng, nạng, ô dù, thanh kim loại. | Gậy sơn trắng trơn thiếu bối cảnh/đầu típ đặc trưng của white cane. |
| `object_02` — **Chùm chìa khóa** | $\ge 2$ chìa trên khoen hoặc 1 chìa gắn móc khóa/fob rõ ràng là chùm chìa cá nhân. | Ổ khóa đơn, kéo, thìa, dụng cụ kim loại, khóa cửa gắn cố định trên tường. | Chìa khóa đơn trần (không khoen/không fob, ví dụ chìa cổ). |
| `object_03` — **Ví tiền** | Ví gập đôi/gập ba (`bi-fold`/`tri-fold`), ví dài khóa kéo, ví xu (`coin purse`) cầm tay. | Bao kính/hộp kính da, sổ tay bìa da, túi đeo chéo/cặp, hộp đựng cứng. | Túi zip nhỏ đa năng không phân biệt rõ ví tiền hay túi đựng cáp/đồ trang điểm. |
| `object_04` — **Đôi dép** | Dép xỏ ngón (`flip-flops`), dép quai ngang (`slides`), dép đi trong nhà (`slippers`) hở kết cấu dép. | Vớ/tất, guốc mộc truyền thống không rõ kết cấu dép hiện đại. | Clog/sục bít mũi hở gót lai giày-dép; dép rọ bít mũi kín. |
| `object_05` — **Đôi giày** | Sneakers, giày thể thao, giày tây/da, bốt ôm trọn bàn chân và có đế giày. | Vớ/tất, bao bọc giày mỏng. | Sandal quai hậu bít gần hết mu bàn chân (lai giày-sandal). |

---

## 2. Cơ cấu phân tách tập dữ liệu (`manifests/dataset_manifest.json`)

**SHA-256 manifest:** `b6f4c06fa506afaea622a7a87db3bf5ca65b653834f3137d5cf9799e75c3df52`  
**Trạng thái kiểm tra rò rỉ:** `zero_overlap_verified = true`

| Tập dữ liệu | Số lượng | Thành phần chi tiết | Vai trò thực nghiệm & Ràng buộc cô lập |
|---|---:|---|---|
| **`train_pos`** | `25` | `5` ảnh/lớp × `5` lớp (`object_01..05`, kế thừa từ `vl_p1/data/train`). | Tập support 5-shot cố định. Không dùng để dò ngưỡng hay đánh giá. |
| **`hard_neg_gallery`** | `50` | 7 nhóm confuser: `footwear_sock_confuser` (`6`), `glasses_case_confuser` (`9`), `lock_key_metal_confuser` (`11`), `long_metal_rod_confuser` (`6`), `notebook_pouch_confuser` (`6`), `umbrella_handle_confuser` (`6`), `walking_stick_confuser` (`6`). | Tạo Negative Prototypes + huấn luyện `RidgeClassifier` (`OOF` trong `calibration_v1c` và trạng thái đóng băng cho holdout). `0` ảnh trùng với mọi split test/cal/holdout. |
| **`calibration_v1c` (ROI & Full)** | `200` | `75` dương (`15/lớp`) + `125` âm (`12` ảnh × `7` nhóm hard-neg = `84` + `41` `general_distractor`). Gồm `150` ảnh thu thập mới (`cal1c_*`) + `50` ảnh kế thừa `calibration_v1b`. | Chỉ dùng trong **Phase A** (`--phase a`) để chạy group-aware 5-fold CV, kiểm tra `delta_floor` grid, chọn `K=4` và khóa cứng `frozen_config.json`. |
| **`final_holdout` (ROI & Full)** | `300` | **`100` dương (`20/lớp`) + `200` âm (`16` ảnh × `7` nhóm hard-neg = `112` + `88` `general_distractor`)**. **`300/300` ảnh thu thập mới (`fhold_*`), không kế thừa.** | Đánh giá mù **1 lần duy nhất** trong **Phase B** (`--phase b`) sau khi `frozen_config.json` đã khóa. |
| **`ambiguous_boundary_v1c`** | `15` | `9` ảnh kế thừa (`amb_01..09`) + `6` ảnh mới (`amb1c_01..06`). Đánh giá ở cả ROI lẫn full-image (không gán `roi_bbox_xywh` bắt buộc vì đánh giá hành vi ranh giới toàn cảnh/crop). | Kiểm chứng phân bố `KNOWN / AMBIGUOUS / UNKNOWN`; không trộn vào Accuracy/FAR chính thức. |
| **Tập hồi quy kế thừa (`vl_p1_test`, `vl_p1b_holdout`)** | `75` / `214` | `vl_p1_test`: `50` pos + `25` neg; `vl_p1b_holdout`: `100` pos + `114` neg (khớp 100% tập VL-P1b). | Đo hồi quy xuyên thế hệ (`vl_p1 -> vl_p1b -> vl_p1c`) trên cùng `frozen_config.json`. |

---

## 3. Giấy phép ảnh thu thập mới & Kiểm định Bounding Box

### 3.1 Phân bố giấy phép mở trên Wikimedia Commons
Mọi ảnh thu thập mới đều lấy từ Wikimedia Commons với giấy phép mở hợp lệ (không có ảnh `UNVERIFIED_MANUAL_WEB` trong các tập mới; 25 ảnh kế thừa `train_pos` giữ nguyên nội bộ, không commit `data/` lên Git):

- **`hard_neg_gallery` (`50`):** `CC BY-SA 4.0` (`15`), `CC BY 2.0` (`9`), `Public domain` (`8`), `CC BY-SA 2.0` (`5`), `CC0` (`4`), `CC BY-SA 3.0` (`4`), `CC BY 3.0` (`3`), `CC BY 4.0` (`2`).
- **`calibration_v1c` (`200`):** `CC BY-SA 4.0` (`80`), `CC BY-SA 3.0` (`40`), `CC BY 2.0` (`19`), `CC BY 3.0` (`15`), `Public domain` (`14`), `CC BY-SA 2.0` (`12`), `CC0` (`10`), `CC BY 4.0` (`6`), plus 4 bản địa phương `de` (`CC BY-SA 3.0 de` 1, `CC BY 2.0 de` 1, `CC BY 3.0 de` 1, `CC BY-SA 2.0 de` 1).
- **`final_holdout` (`300`):** `CC BY-SA 4.0` (`104`), `CC BY-SA 3.0` (`47`), `CC BY 2.0` (`35`), `Public domain` (`26`) + `Public Domain` (`1`), `CC BY 4.0` (`24`), `CC0` (`20`), `CC BY-SA 2.0` (`17`), `CC BY 3.0` (`14`), `CC BY-SA 2.5` (`3`), `CC BY-SA 2.0 de` (`2`), `CC BY-SA 2.0 kr` (`2`), các bản quốc gia còn lại (`nl`, `it`, `pl`, `br`, `1.0`) mỗi loại `1`.
- **`ambiguous_boundary_v1c` (`15`):** `CC BY-SA 3.0` (`4`) + `CC BY-SA 3.0 at` (`1`), `CC BY-SA 4.0` (`4`), `Public domain` (`2`), `CC0` (`1`), `CC BY 3.0` (`1`), `CC BY-SA 2.5` (`1`), `CC BY 2.0` (`1`).

### 3.2 Quy trình kiểm định chống rò rỉ 4 tầng & GT Bounding Box
1. **Lọc trùng 4 tầng trước `--stage init` và xác minh lại ở `--stage finalize`:**
   - So khớp tuyệt đối `SHA-256`, tiêu đề chuẩn hóa (`norm_title`), mã phiên/tác giả (`cluster_id`), và khoảng cách Hamming `aHash`/`dHash` $\le 6$ bit giữa mọi 14 nhóm cho phép (`curated_allowlists_v1c.json`, `cross_hits = 0`) cũng như chéo với toàn bộ dữ liệu `vl_p1` và `vl_p1b`.
2. **Hiệu chỉnh & kiểm định trực quan Bounding Box:**
   - Kế thừa `68` hộp đã khớp SHA-256 từ `manual_gt_bboxes` và bổ sung/tinh chỉnh thêm `fhold_object_01_20` cùng `cal1c_object_01_06` (tổng `69` mục trong `manifests/manual_gt_bboxes_v1c.json`).
   - Kiểm tra trực quan toàn bộ 6 bảng đối chiếu `gt_verify_cal1c_new_pos_1..2.jpg` và `gt_verify_fhold_pos_1..4.jpg` trước khi chạy bất kỳ mô hình nào.
3. **Chính sách tách biệt thư mục trên đĩa:**
   - `calibration_v1c/` (ROI) và `calibration_v1c_full/` (full-image), `final_holdout_roi/` và `final_holdout_full/` cùng chia sẻ đúng danh sách ảnh gốc của từng split nhưng cô lập hoàn toàn khỏi quá trình tìm tham số và đều được loại trừ khỏi Git qua `.gitignore`.
