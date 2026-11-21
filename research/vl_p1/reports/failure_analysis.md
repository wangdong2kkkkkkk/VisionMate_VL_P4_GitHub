# Phân Tích Ca Lỗi Thực Tế — VisionMate VL-P1

Tài liệu này phân tích chi tiết các ca thất bại, nhầm lẫn và ranh giới phân tách dựa trên kết quả chạy thực tế từ `reports/benchmark_metrics.json`, `reports/confusion_matrices.md`, `reports/benchmark_report.md` và `manifests/dataset_manifest.json`.

---

## 1. Cặp khó giữa hai nhóm mục tiêu: `Đôi giày` (`object_05`) vs `Đôi dép` (`object_04`)

Đây là cặp thử thách trực tiếp bài toán phân loại theo **category** trong điều kiện ngoại hình, màu sắc và góc chụp đa dạng:

| Phương pháp (Task 1) | `object_04` (Đôi dép) đúng | `object_05` (Đôi giày) đúng | Giày -> Dép | Dép -> Giày | Nhận xét hành vi |
|---|---:|---:|---:|---:|---|
| `orb_raw` | 0/10 | 0/10 | 0 | 1 | Gần như không khớp chéo cá thể; `object_04/test_10.png` (`Blue slippers in home.jpg`) bị nhận nhầm sang `object_05`. |
| `orb_adaptive` | 0/10 | 0/10 | 1 | 0 | `88%` dương tính bị trả `UNKNOWN`; 1 ca giày (`object_05`) bị nhận nhầm sang `object_04`. |
| `mobilenetv3_small__hybrid` | 4/10 | 8/10 | 0 | 1 | `object_04/test_02.png` (`White flip flops with Brazilian flag.jpg`) bị nhận nhầm thành `object_05` (`score=0.4681, margin=0.1257`); `object_04/test_07.png` rơi vào `AMBIGUOUS` (`score=0.3936, margin=0.0161`); 4 ca khác dưới ngưỡng $\tau=0.3741$. |
| `efficientnet_b0__hybrid` | 3/10 | 8/10 | 0 | 2 | Tăng recall chung nhưng 2 mẫu `object_04` vẫn bị kéo sang `object_05`, 5 mẫu rơi vào `UNKNOWN`. |
| `dinov2_small_int8__prototype` | 7/10 | 9/10 | **0** | **0** | **Không có bất kỳ ca nhầm lẫn chéo nào (`0/20`)** giữa Giày và Dép; 3 ca dép còn lại (`test_03.png`, `test_05.png`, `test_07.png`) và 1 ca giày (`test_10.png`, `Eliofeet Black-Green Kalenji.jpg`) chỉ dừng ở `AMBIGUOUS` do margin < $\delta=0.075$. |
| `dinov2_small_int8__hybrid` | 7/10 | **10/10** | **0** | **0** | Không nhầm chéo; 3 ca dép mang trên chân / chụp gần nền phức tạp rơi nhẹ xuống dưới $\tau=0.4758$ nên từ chối an toàn là `UNKNOWN`: `object_04/test_03.png` (`Person wearing black flip flops.jpg`, `score=0.4538`), `object_04/test_05.png` (`Flip flops on feet.jpg`, `score=0.4611`), `object_04/test_07.png` (`My Flip Flops.jpg`, `score=0.4662`). |

**Nguyên nhân gốc:** Các CNN huấn luyện ImageNet (`MobileNetV3-Small`, `EfficientNet-B0`) thiên về kết cấu bề mặt và hình bóng giày dép nói chung, dễ bị "dép trắng kẹp quai mảnh trên nền sáng" hoặc "dép đang mang trên chân người" kéo về phía lớp `giày`. Ngược lại, **DINOv2-Small INT8** (ViT-S/14 self-supervised) giữ ranh giới hình học giữa cấu trúc dép hở mũi/quai kẹp và giày kín tốt hơn rõ rệt (`0` lỗi chéo trên cả tập calibration 6 ảnh và tập test 20 ảnh của hai lớp này).

---

## 2. `Ví tiền` (`object_03`) vs nhóm `wallet_confuser` (bao kính, sổ tay bìa da, túi/hộp đựng)

Nhóm âm tính `wallet_confuser` gồm 6 ảnh trong tập test có hình dáng chữ nhật, chất liệu da/vải gập rất giống ví tiền:

- Các ca gây dương tính giả (`False Accept`) dai dẳng nhất trên Task 1:
  - `unknown/test_14.png` (`File:Etui-1.jpg` — bao đựng kính bằng da gập): bị `dinov2_small_int8__hybrid` nhận nhầm là `object_03` với `score=0.5210, margin=0.1657`, và `ensemble_mv3_dino__hybrid` với `score=0.4608, margin=0.1349`.
  - `unknown/test_17.png` (`File:A notebook allegedly covered in human skin Wellcome L0043479.jpg` — sổ tay đóng bìa da cổ): bị `mobilenetv3_small__hybrid` (`score=0.4610`), `dinov2_small_int8__hybrid` (`score=0.5156`), và `ensemble_mv3_dino__hybrid` (`score=0.4942`) nhận nhầm thành `object_03`.
  - `unknown/test_15.png` (`File:Burberry glasses case.JPG`) và `unknown/test_13.png` (`File:Chrome Hearts Freedom Sunglasses.JPG` có hộp kính da bên cạnh): gây FAR trên `mobilenetv3_small__hybrid` (`test_15.png`, `score=0.3794`) và `dinov2_small_int8__prototype` (`test_13.png`, `score=0.4696`; `test_14.png`, `score=0.5312`; `test_17.png`, `score=0.5438`).
- Ngược lại, trên các mẫu dương tính `object_03` có kiểu dáng lạ so với 5 ảnh train (ví gập hiện đại):
  - `object_03/test_03.png` (`File:Purse, coin (AM 1967.159-1).jpg` — túi xu cổ trang trí) có `score=0.3954` ở `dinov2_small_int8__hybrid` -> bị từ chối thành `UNKNOWN`.
  - `object_03/test_04.png` (`File:Crocodile wallets.jpg` — nhiều ví da cá sấu xếp chồng) có `score=0.4541` -> dưới $\tau=0.4758$ -> `UNKNOWN`.

**Bài học kỹ thuật:** Khi không có mô hình phát hiện vật thể phân tách "chi tiết ngăn thẻ/khóa kéo" ở độ phân giải cao và không bổ sung `negative prototypes` cho các đồ vật hình hộp chữ nhật (hộp kính, sổ tay), không gian đặc trưng global 224×224 gần như không thể tách tuyệt đối một bao kính da gập (`Etui-1.jpg`) khỏi một chiếc ví da trơn.

---

## 3. `Chùm chìa khóa` (`object_02`) vs vật kim loại nhỏ và chìa cổ/chìa đơn

- Trên Task 1, DINOv2 phân loại khá mạnh nhóm này (`dinov2_small_int8__1nn_max` và `dinov2_small_int8__prototype` đạt **10/10 = 100%** recall), trong khi CNN thuần (`mobilenetv3_small__hybrid`) chỉ đạt **4/10 = 40%**.
- Khi siết ngưỡng lên $\tau=0.4758$ ở `dinov2_small_int8__hybrid` để hạ FAR tổng xuống `12.0%`, 3 mẫu dương tính khó rơi vào `UNKNOWN` (không có mẫu nào bị gán nhầm lớp khác):
  - `object_02/test_02.png` (`File:Cheraman Masjid, Kodungaloor...09.jpg` — khóa & chìa cổ chụp gắn kiến trúc gỗ, `score=0.4027`).
  - `object_02/test_03.png` (`File:Bring It (5067801381).jpg` — móc khóa/thẻ trang trí dày hơn chìa kim loại, `score=0.4525`).
  - `object_02/test_07.png` (`File:Rusty keys Schlüssel Keys on key rings Schlüsselbund 01.JPG` — chùm chìa rỉ sét trên nền nhiễu, `score=0.4659`).
- Ở phía âm tính (`unknown`): với ngưỡng thấp của `dinov2_small_int8__prototype` ($\tau=0.3735$), `unknown/test_06.png` (`File:Brass St. Louis Padlock From the 1904 World's Fair.jpg`, `score=0.4476`) và `unknown/test_09.png` (`File:Carpet tools, scissors (Azerbaijan).JPG`, `score=0.4009`) bị nhận nhầm là chùm chìa khóa (`object_02`). Khi dùng `dinov2_small_int8__hybrid` ($\tau=0.4758$), cả 2 mẫu này cùng `key_metal_confuser` đều bị từ chối chính xác (`FAR=0.0` cho `key_metal_confuser`).
- Ở Task 2 (ảnh gốc chưa cắt): `yolox_strict` bỏ sót **6/10** chùm chìa khóa, `yolox_low` vẫn bỏ sót **2/10** (`PROPOSAL_MISSED`: `object_02/test_02.jpg`, `test_07.jpg`), còn `hybrid_prop__dinov2_hybrid` kéo về **0/10 missed proposal** (chỉ 2 ca còn `UNKNOWN` sau crop: `test_02.jpg`, `test_07.jpg`).

---

## 4. `Gậy trắng hỗ trợ người khiếm thị` (`object_01`) vs `stick_confuser` & ảnh toàn cảnh người cầm gậy

Nhóm `object_01` phơi bày rõ hai hiện tượng:
1. **Dương tính bị từ chối khi vật mảnh/dài nằm chéo cùng người cầm (`in-the-wild`):**
   - `object_01/test_02.png` (`File:Blinden, slechtzienden, stations, blindenstokken...932-3839.jpg`): ảnh tư liệu đen trắng cổ, người đứng cầm gậy mảnh; mọi mô hình đều cho điểm thấp (`dinov2_small_int8__hybrid` `score=0.3082`; ngay cả `prototype` cũng trả `UNKNOWN` với `score=0.3049`).
   - `object_01/test_05.png` (người cầm gậy trước tòa nhà, `dinov2_small_int8__hybrid` `score=0.4575`) và `object_01/test_10.png` (`AR6A2941.jpg`, `score=0.3802`).
   - Trong Task 2 (`hybrid_prop__dinov2_hybrid`), vùng đề xuất tốt nhất cho các ảnh người cầm gậy nhiều khi chỉ đạt `IoU = 0.21–0.31` (vì YOLOX chỉ phát hiện hộp bao quanh cả `person` = `yolox_coco_0`, hoặc `center_72` chỉ lấy phần giữa, làm thân gậy mảnh bị cắt đầu/đuôi hoặc lẫn nền người). Kết quả: `object_01` có 6/10 ca rơi về `UNKNOWN` ở Task 2 (`test_02, 03, 04, 05, 09, 10`).
2. **Âm tính `stick_confuser` rất giống hình học:**
   - `unknown/test_02.png` (`File:A walking-stick telescope (Science Museum).jpg` — gậy chống có màu sáng/trắng tương phản cao trong bảo tàng): đánh lừa **tất cả** mọi model embedder với điểm rất cao (`mobilenetv3_small__hybrid` `score=0.5799`, `dinov2_small_int8__prototype` `score=0.5980`, `dinov2_small_int8__hybrid` `score=0.5683`, `ensemble_mv3_dino__hybrid` `score=0.5643`). Thực chất đây cũng là một cây gậy chống dài có tay cầm, chỉ khác ngữ nghĩa ("gậy kính viễn vọng bảo tàng" thay vì "gậy dò đường người khiếm thị").
   - Với `dinov2_small_int8__hybrid` ($\tau=0.4758$), 2 confuser gậy gỗ khác là `unknown/test_03.png` (`Kawokh Bukkok Tampak depan.jpg`) và `unknown/test_04.png` (`Alligator canes 2...jpg`) cũng như chiếc ô (`umbrella_confuser`) đều đã bị từ chối thành công (`FAR stick_confuser = 1/3`, `umbrella_confuser = 0/1`).

---

## 5. Sự sụp đổ của ORB (`orb_raw` & `orb_adaptive`) và bài học từ `dev_probe_01`

1. **Giả định sai bản chất bài toán (Homography phẳng cùng cá thể vs Nhận diện theo loại):**
   - `orb_raw` chỉ đạt **8.0% Top-1 Pos Acc** (Obj01: 30%, Obj02: 0%, Obj03: 10%, Obj04: 0%, Obj05: 0%) kèm **44.0% FAR** (11/25 ảnh âm tính bị nhận nhầm thành đối tượng biết!) và **12 ca** gán sai hẳn nhãn dương tính (`wrong_pos=12`).
   - Lý do kỹ thuật quan sát được trên `dev_probe01` (`score=7`, `object_05/01.png` có `good=4, inliers=4, ratio=1.0`): ma trận Homography 2D cần tối thiểu **4 cặp điểm**, nên bất kỳ nhiễu nền nào lọt qua Lowe ratio test với chỉ `4–7` điểm đều dễ đạt `ratio = 1.0` một cách giả tạo.
2. **Hiện tượng trên `orb_adaptive`:**
   - Sửa lỗi zero-keypoint trên ảnh rất hẹp (`data/train/object_01/03.png` `61×1131` tăng từ `0` lên `1500` keypoints) là **cần thiết để code không câm hoàn toàn** ở bước trích xuất đặc trưng, nhưng **không thể cứu phương pháp matching hình học ORB** cho bài toán *category recognition*:
   - Trên `dev_probe01` (`probe01.png`), `orb_adaptive` tìm ra `324` keypoints, khớp nhầm mạnh nhất vào **giày `object_05/03.png` (`52 inliers`, `margin=10` so với `42 inliers` của `object_04`)**, trong khi toàn bộ 4 mô hình embedding đều nhận đúng **`object_01` với margin cosine `0.466–0.545`**.
   - Trên tập 75 ảnh test, `orb_adaptive` trả `UNKNOWN` tới **88.0% (44/50)** số ảnh dương tính và gán sai nhãn cho **6/50** ảnh còn lại (`Top-1 Pos Acc = 0.0%`), đồng thời tốn tới **`1,286.605 ms` (1.29 giây)** cho mỗi lần khớp 1 ROI với 25 ảnh gallery trên CPU Intel i5-1135G7.
