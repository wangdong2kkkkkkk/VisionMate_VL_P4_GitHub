# AN TOÀN, QUYỀN RIÊNG TƯ & TIẾP CẬN (SAFETY & PRIVACY)

**Phiên bản:** 0.1.0 (Phase 1)  
**Ngày cập nhật:** 09/11/2026  

---

## 1. Giới hạn Sử dụng An toàn (Safety Boundaries)

> [!CAUTION]
> **VisionMate Glasses Phase 1 là công cụ hỗ trợ đọc chữ in trong nhà, KHÔNG phải thiết bị y tế và KHÔNG phải hệ thống dẫn đường.**
> - Không sử dụng phần mềm để băng qua đường, phát hiện phương tiện giao thông, né tránh vật cản hoặc di chuyển độc lập ngoài trời.
> - Không thay thế gậy trắng, chó dẫn đường, người hướng dẫn di chuyển hoặc chuyên gia phục hồi chức năng.
> - Không tự ý dùng kết quả đọc nhãn thuốc hoặc hóa chất làm chỉ định y khoa nếu chưa kiểm chứng với người hỗ trợ hoặc nguồn tin cậy.

---

## 2. Nguyên tắc Fail-Closed (Thất bại An toàn)

1. **Không bịa đặt nội dung khi không có chữ (`NO_TEXT`):**
   - Khi ảnh không có văn bản rõ ràng, hệ thống phản hồi đúng câu chuẩn:
     *“Mình chưa nhìn thấy chữ rõ để đọc.”*
   - Tuyệt đối không tự sinh mô tả cảnh vật hay đoán mò nội dung.
2. **Từ chối ảnh chất lượng kém (`BAD_IMAGE`):**
   - Ảnh mờ: *“Ảnh đang mờ, hãy giữ camera ổn định và đưa nội dung vào gần hơn.”*
   - Ảnh quá tối: *“Ảnh đang quá tối, hãy bật thêm đèn hoặc hướng nội dung ra nơi đủ sáng.”*
   - Ảnh chói sáng/lóa: *“Ảnh bị chói sáng, hãy nghiêng nhẹ nhãn hoặc đổi góc để giảm phản xạ ánh sáng.”*
3. **Từ chối khi độ tin cậy nhận dạng thấp hoặc ảnh tài liệu chữ quá nhỏ (`UNCERTAIN`) — Tách bạch Chất lượng Ảnh, Độ tin cậy OCR và Chính sách Phát âm (Phase 1.2):**
   - Hệ thống tách rõ 3 thuộc tính độc lập trong `PolicyDecision` và `SessionJobResult`:
     1. `image_quality_status`: `"OK"` | `"LOW_RESOLUTION"` | mã lỗi `QualityIssue`.
     2. `ocr_confidence_status`: `"HIGH"` | `"MEDIUM"` | `"LOW"` | `"NONE"`.
     3. `speech_policy`: `"ALLOW_FULL_SPEECH"` | `"SPEAK_WARNING_ONLY"` | `"SPEAK_NO_TEXT"` | `"SPEAK_ERROR"`.
   - Khi ảnh tài liệu có mật độ dòng dày và phân phối chiều cao ký tự thực sự dưới ngưỡng Nyquist cho dấu tiếng Việt (`median < 8.5px` hoặc `median < 9.5px và p75 < 12.0px`, ví dụ ảnh chụp trang A4 ở độ phân giải thấp `379x527` khiến chữ chú thích chỉ cao `5–7px`) kèm tỷ lệ dòng bị biến dạng cao, hệ thống chuyển sang trạng thái `UNCERTAIN` (`speech_policy = "SPEAK_WARNING_ONLY"`) và phát cảnh báo âm thanh rõ ràng:
     *“Ảnh có chữ quá nhỏ hoặc không đủ rõ. Hãy đưa camera lại gần hoặc dùng ảnh độ phân giải cao hơn”*
   - Ngược lại, với các ảnh văn bản rõ nét trên màn hình hoặc tài liệu chuẩn (như ảnh `996x587` có chữ trắng/cam trên nền tối hoặc chữ đen trên nền sáng với chiều cao chữ thường `x-height ~ 11–12px` và chiều cao chữ hoa/dấu `p75 ~ 15px`), hệ thống chuẩn hóa phân cực (`light_on_dark` / `dark_on_light`) và đánh giá đúng là `image_quality_status = "OK"`, đọc đầy đủ `15/15` dòng (`READABLE`).
   - Hệ thống tuyệt đối **không** gắn nhãn văn bản là `"đã xác minh"` khi chưa đủ bằng chứng kiểm chứng, và không đọc to các dòng chữ vỡ nét/vô nghĩa.
4. **Bảo toàn nguyên văn chữ số, ngày tháng và đơn vị (`FR-04`, `A05`):**
   - Hệ thống **không** dùng LLM hay bộ sửa lỗi tự động để đoán hoặc viết lại chữ số, ngày sản xuất (`NSX`), hạn sử dụng (`HSD`), hàm lượng (`mg`, `ml`) hay giá tiền.
   - Khi phát hiện từ khóa nhạy cảm (thuốc, liều lượng, hạn dùng, hóa chất) hoặc chuỗi số liệu, hệ thống tự động đính kèm cảnh báo âm thanh và hình ảnh:
     *“Lưu ý: Nội dung có chứa con số, hạn dùng hoặc thông tin nhạy cảm, vui lòng kiểm tra lại với người hỗ trợ hoặc nguồn tin cậy.”*
5. **Trung thực về trạng thái giọng đọc (`TTS_BLOCKED`):**
   - Nếu máy thiếu giọng đọc tiếng Việt offline hợp lệ, hệ thống báo rõ `TTS tiếng Việt chưa sẵn sàng (TTS_BLOCKED)`, không giả giọng tiếng Anh đọc chữ Việt và không tự động gọi API đám mây.

---

## 3. Quyền Riêng tư Mặc định (Privacy by Default — FR-08, A11)

- **Chỉ xử lý trên RAM (RAM-only Runtime):**
  - Khung hình chụp từ webcam, văn bản nhận dạng được (`OCRResult`) và luồng âm thanh tổng hợp (`SAPI.SpMemoryStream`) đều nằm hoàn toàn trong bộ nhớ RAM và được giải phóng ngay sau khi hoàn tất lượt đọc.
  - Mặc định ứng dụng **không** tạo bất kỳ file ảnh, video, âm thanh `.wav` hay văn bản `.txt/.json` nào trên ổ cứng.
- **Không kết nối mạng ở Runtime (100% Offline Runtime):**
  - Sau bước cài đặt ban đầu, ứng dụng không thực hiện bất kỳ kết nối HTTP/TCP nào ra Internet. Đã được kiểm chứng tự động bằng bài kiểm thử chặn toàn bộ `socket.socket.connect` và `urllib.request.urlopen`.
- **Chế độ Debug Opt-in có Cảnh báo (`FR-11`):**
  - Chế độ ghi log debug chỉ dành cho nhà phát triển, mặc định luôn tắt (`False`) mỗi khi khởi động ứng dụng.
  - Khi người phát triển tích chọn trên giao diện, ứng dụng hiển thị hộp thoại cảnh báo quyền riêng tư bắt buộc xác nhận trước khi bật, và cung cấp nút **“Xóa log debug”** để xóa sạch thư mục `debug_captures/` bất cứ lúc nào.

---

## 4. Thiết kế Tiếp cận (Accessibility — Section 6.2, A10)

- **Điều khiển 100% bằng bàn phím (không cần chuột):**
  - `Phím Cách (Space)`: Chụp một khung hình và đọc chữ (tự động xóa kết quả cũ).
  - `Phím R`: Đọc lại kết quả vừa nhận dạng.
  - `Phím Esc`: Dừng đọc hoặc hủy tác vụ ngay lập tức.
  - `Tổ hợp Ctrl + L`: Xóa toàn bộ kết quả OCR hiện tại, dừng giọng đọc ngay lập tức, hủy tác vụ đang chạy và đặt lại trạng thái phát lại (`R`) mà vẫn giữ nguyên khung hình camera trực tiếp.
  - `Phím O`: Mở hộp thoại chọn ảnh thử nghiệm (Dev/Test mode).
  - `Phím F1`: Nghe hướng dẫn sử dụng các phím tắt bằng tiếng Việt.
  - `Phím Tab / Shift+Tab`: Di chuyển tuần tự qua các điều khiển với viền tiêu điểm vàng tương phản cao (`3px solid #FFD600`).
- **Hỗ trợ trình đọc màn hình:** Toàn bộ nút bấm, thanh trượt, danh sách chọn và vùng văn bản đều được gán thuộc tính `accessibleName` và `accessibleDescription` bằng tiếng Việt.
