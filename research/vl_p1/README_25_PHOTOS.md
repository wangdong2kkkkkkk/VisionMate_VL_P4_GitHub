# VisionMate VL-P1 — 25 ảnh ứng viên từ Wikimedia Commons

Đây là **bộ danh sách nguồn ảnh + script tải về**, KHÔNG phải ZIP chứa sẵn các tệp ảnh.
Nguồn đã được chọn từ Wikimedia Commons cho 5 lớp: gậy trắng, chùm chìa khóa, ví tiền, đôi dép, đôi giày.
Mỗi lớp có 5 tệp ứng viên. Một số ảnh chụp vật trong bối cảnh thật hoặc nhiều vật trong cùng khung hình.
Bạn cần **xem ảnh, kiểm tra nhãn rồi khoanh ROI** trước khi sử dụng.

## Chạy trên Windows

1. Giải nén ZIP vào thư mục `VisionMate_VL_P1` sao cho script ở `research\vl_p1\download_25_photos.ps1`.
2. Mở PowerShell trong thư mục `VisionMate_VL_P1`.
3. Chạy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\research\vl_p1\download_25_photos.ps1
```

Script sẽ tải ảnh JPEG 1280px (hoặc kích thước gốc nếu nhỏ hơn) vào:
`research\vl_p1\data\originals\object_01` … `object_05`.

Mỗi tệp có số thứ tự `01.jpg` ... `05.jpg`. Nguồn, tác giả, giấy phép và SHA-256 được ghi trong `research\vl_p1\downloaded_photo_sources.csv`.
Nếu máy không kết nối mạng hoặc Wikimedia từ chối yêu cầu, mở `photo_sources.json` để nhấn vào trang ảnh và tải thủ công.

## Quy tắc VL-P1

- **Đây mới là 25 ứng viên cho bộ học**, không phải 25 ảnh đã được gán nhãn và xác nhận.
- Không trộn ảnh học vào tập kiểm thử. Cần thu thập 50 ảnh kiểm thử khác biệt, tốt nhất lấy từ nguồn/sản phẩm chưa xuất hiện ở bộ học; thêm ít nhất 20 ảnh âm tính.
- Khi ảnh có nhiều vật, chỉ crop đúng vật mục tiêu; không gán nhãn cho toàn bộ cảnh.
- Với `object_01` phải là gậy trắng hỗ trợ người khiếm thị (white mobility cane), không phải gậy chống thông thường.
- Với `object_04`/`object_05` xác định ranh giới dép và giày cho nhất quán (giày hở mũi hoặc giày lười cần được định nghĩa trước khi thử).
- `data\originals` là dữ liệu thô. Chỉ đưa ROI đã kiểm tra sang `data\train` theo hướng dẫn ORB Starter.
- Giấy phép phụ thuộc từng ảnh, nhiều ảnh CC BY/CC BY-SA cần ghi công tác giả, link nguồn, thông tin chỉnh sửa. Kiểm tra riêng giấy phép trước khi chia sẻ/tích hợp sản phẩm.
- Các tệp ảnh tải từ Internet có thể chứa metadata cá nhân; không tải hoặc phát tán thông tin nhạy cảm không cần thiết.
- Không thay đổi code VisionMate chính hoặc Git baseline.

### Lưu ý môi trường
Ở môi trường tạo gói này, các ảnh gốc không tải xuống được, nên **không có byte ảnh trong ZIP**. Script sẽ tải tại máy Windows của bạn và cần kết nối Internet một lần, sau đó chạy ORB offline.
