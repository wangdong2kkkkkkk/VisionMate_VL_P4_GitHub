# VisionMate Glasses — Phase 1.1 Real-Document OCR & UI Reset Hotfix

## Bối cảnh và kết quả người dùng đã quan sát

Phase 1 ở `phase1/offline-vietnamese-reading`, commit `8383af3`, được báo cáo 21/21 pytest PASS nhưng người dùng thực tế thử ảnh `administrative_form_379x527.png` và thu được OCR sai nặng, đảo thứ tự dòng/cột và sai dấu; xem `observed_ocr.txt`. Giao diện không thể xóa kết quả cũ trước khi thử webcam. Trạng thái nghiệm thu thực tế: CHƯA ĐẠT; không tuyên bố toàn Phase 1 hoặc cải thiện 1.1 PASS cho đến khi người dùng thử.

## Mục tiêu

1. Sửa lỗi quản lý kết quả OCR: nút 'Xóa kết quả' rõ ràng, accessibleName, phím tắt trống chưa xung đột (đề xuất Ctrl+L). Xóa ngay text và nội dung đang đọc, dừng TTS, hủy/đánh dấu bất hợp lệ OCR đang chạy, reset kết quả để R không phát lại nội dung cũ. Không tắt camera preview. Sau khi xóa, Space và O tạo phiên thử mới. Các job mới phải THAY THẾ kết quả cũ mặc định, không vô ý nối vào nội dung trước.
2. Chống race: generation/session id cho camera, mở ảnh, OCR async, TTS; clearing/cancel/new capture/close phải ngăn mọi callback cũ cập nhật UI hoặc phát lại. Test ít nhất 20 lần clear trong lúc OCR/TTS, 20 lần thay nguồn ảnh/camera và đóng ứng dụng.
3. Điều tra OCR thực tế: ảnh 379x527 chứa chữ quá nhỏ, bố cục hành chính hai cột ở đầu trang, đường kẻ chấm và chú thích cuối trang. Bắt đầu bằng logging kiểm soát (mặc định không lưu ảnh/text), phân tích layout; so sánh Tesseract fast/best, PSM 3/4/6/11 phù hợp theo loại vùng, preprocessing grayscale/contrast, denoise/deskew và scale 2x/3x khi có lợi. Không coi scale làm khôi phục chi tiết đã mất. Nên có detect vùng/ROI, tách cột header và reading order, tránh đưa đường chấm/gạch ngang vào lời đọc.
4. Chất lượng đáng tin: nếu chất lượng nguồn không đủ hoặc sự bất định cao, phát thông báo tiếng Việt 'Ảnh có chữ quá nhỏ hoặc không đủ rõ. Hãy đưa camera lại gần hoặc dùng ảnh độ phân giải cao hơn', không đọc ra chuỗi vô nghĩa như kết quả chính thức. Không cố đoán nội dung mẫu pháp lý theo template; không dùng LLM để tự điền nội dung vô căn cứ. Các trường số, ngày tháng, tên, nội dung pháp lý phải giữ nguyên chữ nhìn thấy; không tự chỉnh thành câu có vẻ hợp lý. Nếu confidence không đủ độ tin cậy, dùng nhiều tín hiệu (quality, OCR confidences, disagreement, field heuristics) và phân loại uncertain thay vì suy diễn.
5. Có thể bổ sung chọn vùng đọc (ROI) hoặc chế độ Đọc tài liệu để tăng khả năng dùng, nhưng chỉ khi giữ đơn giản, offline, dễ tiếp cận. Không mở Phase 2 và không nâng cấp ngoài phạm vi.

## QA bắt buộc

- Thêm file regression của người dùng này vào test assets mà không chỉnh sửa; giữ SHA-256. Các chuỗi trong `expected_fragments.txt` là mốc tham khảo KHÔNG phải transcript hoàn chỉnh. Không tuning cứng theo riêng ảnh này.
- Trên ảnh gốc, hoặc nhận đúng các heading thực sự nhìn thấy trong ROI đủ rõ, hoặc thông báo chất lượng thấp một cách trung thực. Không xuất bản một đoạn chữ rác dài với giọng đọc tự tin. Không thể bảo đảm OCR đầy đủ từ 379x527.
- Thêm tập tài liệu THẬT có nhãn thủ công độc lập, gồm ít nhất 20 tài liệu ở độ phân giải đủ đọc (văn bản một cột và form nhiều cột), 10 ảnh chất lượng kém, một tập nhãn sản phẩm/bao bì cũ. Chấm CER/WER trên tập được gắn nhãn, thứ tự đọc nhiều cột, tỷ lệ từ chối đúng và sai. Phân biệt synthetic và real-world. Không tăng điểm nhờ template hardcode.
- Benchmark OCR/TTS end-to-end, RAM, FPS preview; đo p50/p95 và trường hợp mất mạng. Duy trì các test A01-A12 và 21/21 gốc, báo rõ regressions nếu có.
- Kiểm tra Ctrl+L/nút xóa bằng keyboard, screen-reader/accessibility, đổi ảnh mở O sang webcam Space, R sau clear, ngắt TTS bằng Esc, không rò rỉ kết quả giữa session.

## Bảo vệ baseline và bàn giao

- Tạo branch/sandbox mới kế thừa commit 8383af3; không ghi đè baseline, không merge tự động, không sửa dự án khác.
- Một nhiệm vụ lớn tự khảo sát → sửa → test → đo → báo cáo. Không cần xin phép ở mỗi bước trong phạm vi này. Không tạo hoặc tuyên bố test PASS nếu thiếu bằng chứng thực tế.
- Bàn giao source, launcher riêng nếu cần, báo cáo trước-sau (đặc biệt ảnh này), ảnh chụp UI, pytest và benchmark, Git branch/commit/status, đường dẫn bản chạy, checklist người dùng thao tác 5 phút. Kết luận tối đa là TECHNICAL PASS / CONDITIONAL chờ người dùng xác nhận.
