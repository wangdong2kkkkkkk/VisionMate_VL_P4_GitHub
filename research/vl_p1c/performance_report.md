# Báo Cáo Hiệu Năng Phần Cứng, Độ Trễ & Khả Năng Mở Rộng CPU — VisionMate VL-P1c

Mọi phép đo được thực hiện **100% offline trên CPU (`num_threads=4`, không GPU)** trong tiến trình độc lập không có tác vụ nền cạnh tranh (`--phase b`), hướng tới máy tham chiếu Windows 11, Intel i5-1135G7, RAM 8 GB.

---

## 1. Tài nguyên Bộ nhớ & CPU (`hardware_telemetry` trong `reports/benchmark_metrics.json`)

| Chỉ số phần cứng | Giá trị đo thực tế | Ngưỡng mục tiêu tham khảo | Đánh giá |
|---|---:|---:|---|
| **RSS khởi đầu (`rss_start_mb`)** | `51.25 MB` | — | Nhẹ |
| **RSS khi nạp đủ YOLOX-Nano + DINOv2-Small INT8 (`rss_loaded_mb`)** | **`249.88 MB`** | `<= 2,048 MB` | Đạt rộng rãi |
| **RSS cuối phiên (`rss_end_mb`)** | `121.87 MB` | — | Không rò rỉ bộ nhớ đáng kể |
| **Private Bytes tối đa (`private_bytes_mb`)** | **`555.85 MB`** | `< 1,000 MB` | Đạt |
| **CPU sử dụng trung bình (`cpu_percent_avg`)** | `127.8%` (trên `num_threads=4`) | Phù hợp CPU đa nhân i5 | Đạt |

---

## 2. Chi phí Đăng ký Gallery & Khả năng Mở rộng Tuyến tính Offline

- **Đăng ký (`Enrollment`) 1 lần:**
  - `25` ảnh dương (`train_pos`): **`1,201.02 ms`** -> `pos_db = 94,802 B` (`~92.6 KB`).
  - `50` ảnh âm (`hard_neg_gallery`) + huấn luyện Ridge 784-D: **`1,986.11 ms`** -> `neg_db = 179,174 B`, `verifier = 85,558 B` (tổng DB tuần tự hóa = **`359,534 B` ~ `351.1 KB`**).

| Số ảnh Negative Gallery (`N_neg`) | Match-only Latency (ms) | Kích thước Vector DB (Bytes) | Nhận xét mở rộng |
|---:|---:|---:|---|
| `20` | `0.00439 ms` | `153,600 B` (`150.0 KB`) | Chi phí matching vector không đáng kể |
| `50` *(Hiện dùng)* | `0.00576 ms` | `245,760 B` (`240.0 KB`) | Dưới `0.01 ms` |
| `100` | `0.00785 ms` | `399,360 B` (`390.0 KB`) | Chiếm `< 1/10,000` tổng độ trễ khung hình |
| `150` | **`0.01173 ms`** | **`552,960 B` (`540.0 KB`)** | Cho phép mở rộng gallery âm hàng trăm mẫu |

---

## 3. Độ trễ Suy luận Từng Tầng trên `final_holdout` (`N=300`)

### 3.1 Tầng Verifier trên ROI cô lập & Tầng Proposal đơn lẻ

| Thành phần / Cấu hình | Latency p50 | Latency p95 | Ghi chú |
|---|---:|---:|---|
| **ROI Baseline `R0` / `R5_selective_3state_v1c`** | **`33.65 ms` / `35.58 ms`** | `46.34 ms` / **`46.60 ms`** | Dưới `50 ms`, cộng thêm OOF verifier/3-state chỉ tốn `+1.93 ms` p50 |
| **YOLOX-only (`yolox_strict__k4` / `yolox_low__k4`)** | `15.31 ms` / `16.24 ms` | `21.32 ms` / `22.39 ms` | Nhanh nhưng bỏ sót toàn bộ `object_01` (`0/20` R@0.5) |
| **Baseline Proposal `B0_vl_p1_proposer__k4`** | **`20.40 ms`** | `29.45 ms` | Rất nhanh (`avg_k = 3.5`) |
| **VL-P1b Proposal `B_full_v1b__k4` / `__k8`** | `44.75 ms` / `44.96 ms` | `61.92 ms` / `63.36 ms` | Tăng `+24 ms` do Hough/geometry cũ |
| **VL-P1c Proposal `B_v1c_proposer__k4` / `__k6` / `__k8`** | **`62.54 ms` / `65.02 ms` / `62.98 ms`** | `78.65 / 83.96 / 78.37 ms` | Tăng so với `B0` do trích `oriented_thin_cane` + thin corridor mask |

### 3.2 Toàn Pipeline End-to-End (`final_holdout_full`, `N=300`)

| Cấu hình E2E | `avg_num_embeddings` | Proposal p50 | Verifier p50 | Total p50 | Total p95 | So với mục tiêu `< 250 ms` |
|---|---:|---:|---:|---:|---:|---|
| `C1_vl_p1_baseline` (`B0@4 + A0`) | `3.43` | `29.68 ms` | `129.16 ms` | **`158.13 ms`** | **`197.88 ms`** | Đạt |
| **`C_v1c_cal_only` (`B0@4 + R5`)** | **`3.43`** | `30.11 ms` | `147.02 ms` | **`176.20 ms`** | **`218.69 ms`** | **Đạt** (kể cả p95 `< 220 ms`) |
| `C4_vl_p1b_combined` (`B_v1b@4 + A3`) | `4.00` | `57.09 ms` | `146.43 ms` | `204.99 ms` | `251.27 ms` | Đạt p50 |
| `C_v1c_prop_only` (`B_v1c@4 + A0_v1c`) | `4.00` | `77.25 ms` | `159.69 ms` | `236.16 ms` | `300.22 ms` | Sát ngưỡng 250 ms ở p50 |
| `C_v1c_full__k4` (`B_v1c@4 + R5 + MP`) | `4.00` | `109.37 ms` | `251.69 ms` | **`360.13 ms`** | **`492.01 ms`** | **Vượt ngưỡng 250 ms** (thin mask overhead + verifier) |
| `C_v1c_full__k6` (`B_v1c@6`) | `5.98` | `79.73 ms` | `231.94 ms` | `313.81 ms` | `403.65 ms` | Chạy `~6` crop DINOv2 (`> 300 ms`) |
| `C_v1c_full__k8` (`B_v1c@8`) | `7.76` | `79.53 ms` | `292.72 ms` | `372.41 ms` | `469.91 ms` | Tốn gần `300 ms` cho DINOv2 batch |

### Đánh giá tính khả thi phần cứng cho laptop i5-1135G7 / RAM 8 GB
- **Dung lượng & RAM:** Hoàn toàn phù hợp (`249.88 MB` RSS khi nạp đủ model, `555.85 MB` Private Bytes, `351 KB` database).
- **Độ trễ E2E:** `C_v1c_cal_only` (`176.20 ms` p50, `218.69 ms` p95 — ~`5.7 FPS` nếu chạy liên tục hoặc `<0.22s` chế độ bấm đọc theo yêu cầu) đạt hoàn toàn tiêu chí CPU-first. Trái lại, `C_v1c_full__k4` (`360.13 ms` p50, `492.01 ms` p95) nặng hơn do tạo mặt nạ khe mỏng và đánh giá thêm nhiều đặc trưng/crop; nếu muốn dùng pipeline tương tự khi bấm chụp trên laptop văn phòng vẫn chấp nhận được cho chế độ *on-demand* (`~0.36s–0.49s`), nhưng chưa phù hợp chạy liên tục từng khung hình webcam.
