# Báo Cáo Kiểm Toán Dữ Liệu Thủ Công Hiện Có (VL-P1 Audit)

## 1. Tóm tắt số liệu kiểm toán
- **Tổng ảnh gốc (`data/originals`)**: 25 ảnh (5 ảnh/nhóm × 5 nhóm)
- **Tổng ảnh học đã cắt (`data/train`)**: 25 ảnh (5 ảnh/nhóm × 5 nhóm)
- **Tổng ảnh test sẵn có (`data/test`)**: 0 ảnh (trống hoàn toàn)
- **Ảnh gốc khớp nguồn Wikimedia trong `downloaded_photo_sources.csv`**: 6/25
- **Ảnh gốc bổ sung thủ công chưa có metadata bản quyền**: 19/25
- **Mục CSV đã bị thay thế/không còn trên đĩa**: `['object_02/03.jpg']`
- **Ảnh `data/train` khớp chính xác 100% (`score >= 0.99`) từ `data/originals`**: 25/25
- **Ảnh train bị `0` keypoints với ORB thô (`edgeThreshold=31`)**: `['data/train/object_01/03.png']`
- **Ảnh train bị `0` keypoints sau Adaptive Narrow-ROI Padding**: `[]`

## 2. Phát hiện lỗi kỹ thuật trọng yếu trên tập `train` hiện có
| File | Kích thước (W×H) | Ảnh gốc cha | Điểm khớp crop | Keypoints ORB thô | Keypoints ORB Adaptive | Ghi chú |
|---|---:|---|---:|---:|---:|---|
| `data/train/object_01/01.png` | 1039×949 | `02.jpg` | 1.0000 | 1297 | 1500 | OK |
| `data/train/object_01/02.png` | 247×1461 | `03.jpg` | 1.0000 | 1012 | 1390 | OK |
| `data/train/object_01/03.png` | 61×1131 | `05.jpg` | 1.0000 | 0 | 1500 | RẤT HẸP (61px < 2*31px edgeThreshold -> 0 kp) |
| `data/train/object_01/04.png` | 418×411 | `images (3).jpeg` | 1.0000 | 295 | 753 | Ít đặc trưng bề mặt trơn |
| `data/train/object_01/05.png` | 1048×1121 | `white-cane-scaled.webp` | 1.0000 | 1027 | 1500 | OK |
| `data/train/object_02/01.png` | 1123×1030 | `01.jpg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_02/02.png` | 852×652 | `02.jpg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_02/03.png` | 1231×848 | `05.jpg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_02/04.png` | 623×271 | `images.jpeg` | 1.0000 | 1208 | 1500 | OK |
| `data/train/object_02/05.png` | 2238×2167 | `pngtree-bunch-of-keys-with-keychain-png-image_15109528.png` | 1.0000 | 1313 | 1504 | OK |
| `data/train/object_03/01.png` | 533×395 | `images.jpeg` | 1.0000 | 1199 | 1498 | OK |
| `data/train/object_03/02.png` | 630×322 | `V9.jpg` | 1.0000 | 1094 | 1393 | OK |
| `data/train/object_03/03.png` | 589×527 | `vi_kep_tien-handmade_4fed56e5ca2447c69ada79aed225431e_grande.jpg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_03/04.png` | 675×555 | `vi_nam_da_that_dung_tien_vd141_2.jpg` | 1.0000 | 1239 | 1501 | OK |
| `data/train/object_03/05.png` | 426×353 | `vi-epsom.jpg` | 1.0000 | 1116 | 1468 | OK |
| `data/train/object_04/01.png` | 572×639 | `2_6b187103febc43eeb99fe9f74c1363e1_1024x1024.webp` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_04/02.png` | 671×754 | `4_a6e238a0986b499a94916eba374cea6e.webp` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_04/03.png` | 1161×1184 | `Dep-le-be-trai-G63-xanh-nhat-2.jpg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_04/04.png` | 782×949 | `vn-11134207-81ztc-mlp1p5oo2rk461.jpeg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_04/05.png` | 596×429 | `W0OJHhS99KMT90fMoTokR9K3LKQyYW8uSdTchzoN.jpg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_05/01.png` | 521×280 | `88ea5549d99ad9bdb9a16b6ed3158f75.jpg_720x720q80.jpg` | 1.0000 | 1105 | 1500 | OK |
| `data/train/object_05/02.png` | 648×424 | `giay-bong-ro-cho-nguoi-moi-choi-16_fecfb4795b4849dba7ff208cc27a035f.webp` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_05/03.png` | 852×518 | `GIAY-DA-NAM-GI3DEAU008D0 (9).jpg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_05/04.png` | 729×532 | `giay-the-thao-nu-ag0155-13.jpg` | 1.0000 | 1300 | 1500 | OK |
| `data/train/object_05/05.png` | 266×352 | `images.jpeg` | 1.0000 | 1003 | 1346 | OK |

## 3. Kiểm tra ảnh thử phụ (`dev_probes` & `backup_crops`)
| File | Kích thước (W×H) | Keypoints ORB thô | Keypoints ORB Adaptive |
|---|---:|---:|---:|
| `dev_probes/object_01/probe01.png` | 200×425 | 159 | 324 |
| `backup_crops/object_01/01_previous.png` | 1014×799 | 1292 | 1500 |
