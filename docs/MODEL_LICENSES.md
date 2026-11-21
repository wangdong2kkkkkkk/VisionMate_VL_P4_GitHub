# GIẤY PHÉP MÔ HÌNH, MÃ NGUỒN & PHỤ THUỘC (MODEL & PACKAGE LICENSES)

**Phiên bản:** 0.1.0 (Phase 1)  
**Ngày kiểm tra:** 09/11/2026  

---

## 1. Danh mục Mô hình & Công cụ Nhận dạng / Giọng nói Cục bộ

| Thành phần | Phiên bản / File | Nguồn tải chính thức | SHA-256 Hash | Giấy phép (License) | Vai trò trong Phase 1 |
|---|---|---|---|---|---|
| **Tesseract OCR Binary (Windows x64)** | `v5.4.0.20240606` (`tesseract.exe`) | `https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe` | Được xác minh qua WinGet manifest (`UB-Mannheim.TesseractOCR`) | **Apache License 2.0** | Engine OCR CPU chính thức |
| **Tesseract Vietnamese Fast Model** | `models/tessdata_fast/vie.traineddata` (`531,275` bytes) | `https://github.com/tesseract-ocr/tessdata_fast/raw/main/vie.traineddata` | `79DF64CAF7BCFB2A27DF5042ECB6121E196EADA34DA774956995747636D5BFA1` | **Apache License 2.0** | Model OCR tiếng Việt mặc định (`tesseract_fast_vie`) |
| **Tesseract Vietnamese Best Model** | `models/tessdata_best/vie.traineddata` (`12,435,550` bytes) | `https://github.com/tesseract-ocr/tessdata_best/raw/main/vie.traineddata` | `B6B49293D95D0B6DBD8780174627E82C75BE957B6F4ED9862155540D6B00BB45` | **Apache License 2.0** | Model OCR tiếng Việt độ chính xác cao tùy chọn (`tesseract_best_vie`) |
| **Tesseract English Fast Model** | `models/tessdata_fast/eng.traineddata` (`4,113,088` bytes) | `https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata` | `7D4322BD2A7749724879683FC3912CB542F19906C83BCC1A52132556427170B2` | **Apache License 2.0** | Model bổ trợ |
| **Tesseract English Best Model** | `models/tessdata_best/eng.traineddata` (`15,400,601` bytes) | `https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata` | `8280AED0782FE27257A68EA10FE7EF324CA0F8D85BD2FD145D1C2B560BCB66BA` | **Apache License 2.0** | Model bổ trợ |
| **PaddleOCR Latin/vi ONNX Recognition Model** | `models/paddleocr_latin/latin_rec.onnx` (`7,862,832` bytes) | `https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/rec.onnx` (chuyển đổi từ PaddleOCR PP-OCRv3/v5) | `614FFC2D6D3902D360FAD7F1B0DD455EE45E877069D14C4E51A99DC4EF144409` | **Apache License 2.0** | Dùng để đối chiếu benchmark thực nghiệm (`FR-12`) |
| **PaddleOCR Latin Dictionary** | `models/paddleocr_latin/latin_dict.txt` (`1,634` bytes) | `https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/dict.txt` | `3C0A8A79B612653C25F765271714F71281E4E955962C153E272B7B8C1D2B13FF` | **Apache License 2.0** | Từ điển ký tự cho model đối chiếu PaddleOCR Latin |
| **YOLOX-Nano Object Detection Model** | `models/object_detection/yolox_nano.onnx` (`3,659,407` bytes) | `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx` (Megvii YOLOX) | `C789161ED43C8269FCD4E67C67EEEB4E80C622DA2EB296A20BC6007BD18A0B7D` | **Apache License 2.0** | Model nhận diện đồ vật CPU chính thức (OD-1 Foundation, 80 COCO classes) |
| **YOLOX-Tiny Object Detection Model** | `models/object_detection/yolox_tiny.onnx` (`20,219,662` bytes) | `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx` (Megvii YOLOX) | `427CC366D34E27FF7A03E2899B5E3671425C262EA2291F88BB942BC1CC70B0F7` | **Apache License 2.0** | Model đối chiếu / dự phòng độ chính xác cao hơn (OD-1 Benchmark) |
| **Windows OneCore Vietnamese TTS Voice** | `Microsoft An - Vietnamese (Vietnam)` (`MSTTS_V110_viVN_An`) | Cài sẵn trong hệ điều hành Windows 11 (`C:\WINDOWS\Speech_OneCore\Engines\TTS\vi-VN\`) | Thành phần bản quyền hệ điều hành Windows 11 | **Microsoft Windows OS License** | Giọng đọc tiếng Việt offline chính thức (`vi-VN`) |

---

## 2. Thư viện Python Chính (`requirements.lock`)

| Thư viện | Phiên bản Pinned | Giấy phép | Mục đích sử dụng |
|---|---|---|---|
| `opencv-python` | `4.13.0.90` | Apache-2.0 | Mở camera UVC, đọc ảnh, tính toán chất lượng ảnh (Laplacian, CLAHE) |
| `numpy` | `2.2.6` | BSD-3-Clause | Xử lý mảng ảnh và âm thanh PCM trên RAM |
| `Pillow` | `12.1.0` | MIT-CMU | Xử lý ảnh và tạo bộ dữ liệu kiểm thử tổng hợp (synthetic fixtures) |
| `PySide6` | `6.12.0` | LGPL-3.0-only | Giao diện desktop Windows hỗ trợ tiếp cận (Accessibility & Shortcuts) |
| `pytesseract` | `0.3.13` | Apache-2.0 | Giao tiếp với binary `tesseract.exe` cục bộ |
| `pywin32` | `304` | PSF-2.0 | Kết nối COM `SAPI.SpVoice` / `SAPI.SpMemoryStream` cho giọng đọc Windows OneCore |
| `PyAudio` | `0.2.12` | MIT | Phát luồng âm thanh PCM theo từng khối 46ms hỗ trợ ngắt tức thì |
| `onnxruntime` | `1.23.2` | MIT | Thực thi mô hình ONNX CPU cho đối chiếu PaddleOCR |
| `rapidocr-onnxruntime` | `1.4.4` | Apache-2.0 | Pipeline chạy mô hình PaddleOCR ONNX trên CPU |
| `psutil` | `7.2.2` | BSD-3-Clause | Kiểm toán phần cứng và đo bộ nhớ RAM thực tế (Peak Working Set) |
| `pytest` | `9.2.0` | MIT | Chạy bộ kiểm thử tự động Unit & Integration |

---

## 3. Bản quyền Bộ Dữ liệu Kiểm thử (`tests/fixtures/`)

Toàn bộ 70 ảnh kiểm thử trong `tests/fixtures/dataset/` và `tests/fixtures/manifest.json` được tạo tự động bằng script `scripts/generate_benchmark_dataset.py` sử dụng phông chữ hệ thống trên máy cục bộ (`synthetic: true`, `consent_safe: true`). Bộ dữ liệu không chứa hình ảnh cá nhân, không chứa tài liệu riêng tư của bên thứ ba và được phát hành nội bộ theo chuẩn **CC0-1.0**.
