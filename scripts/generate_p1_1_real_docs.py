# -*- coding: utf-8 -*-
"""
Generates the Phase 1.1 Real-Style Vietnamese Document & Packaging Evaluation Suite (`real_docs_p1_1`)
and `manifest_p1_1.json`.

Dataset composition (40 realistic images + the untouched original regression sample in `regression_p1_1/`):
1. `readable_doc_single` (10 images, ~1080x1480):
   Single-column official Vietnamese notices, letters, clinic instructions, utility bills, and academic notices
   with paper texture, subtle camera perspective/lighting, horizontal separator rules, and dotted lines.
2. `readable_doc_multicol` (10 images, ~1120x1540):
   2-column Vietnamese administrative documents (Nghị quyết, Quyết định, Công văn, Thông báo, Biên bản,
   Giấy giới thiệu, Tờ trình, Hợp đồng, Phiếu tiếp nhận, Giấy xác nhận) featuring:
   - Top 2-column header (Left: Agency & Số/Ký hiệu; Right: Quốc hiệu & Tiêu ngữ + Địa danh/Ngày)
   - Horizontal separator rules (`____`) and fill-in-the-blank dotted leaders (`..........`)
   - Centered title & body lines
   - Bottom 2-column footer (Left: `Nơi nhận:` list; Right: Authority & Signer block)
   Includes exact ground truth in expected reading order (Left Column -> Right Column -> Body -> Left Footer -> Right Footer).
3. `low_quality_unreadable_doc` (10 images):
   Low-resolution thumbnail forms (`350x490` to `390x540` with 5-7px footnotes), severe motion blur,
   heavy underexposure, specular glare, and extreme defocus. Expected status: `UNCERTAIN` or `BAD_IMAGE`.
4. `product_packaging_receipt` (10 images, ~920x680):
   Multi-line Vietnamese medicine boxes, food nutrition labels, pharmacy prescriptions, and supermarket receipts
   with dates, dosages, prices, batch codes, and cardboard/paper textures.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DOCS_DIR = PROJECT_ROOT / "tests" / "fixtures" / "real_docs_p1_1"
MANIFEST_P1_1_PATH = REAL_DOCS_DIR / "manifest_p1_1.json"
REGRESSION_IMG_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "regression_p1_1" / "administrative_form_379x527.png"
)


def _font(family: str, size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    candidates: List[str] = []
    if family == "serif":
        if bold and italic:
            candidates = [r"C:\Windows\Fonts\timesbi.ttf", r"C:\Windows\Fonts\timesbd.ttf"]
        elif bold:
            candidates = [r"C:\Windows\Fonts\timesbd.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
        elif italic:
            candidates = [r"C:\Windows\Fonts\timesi.ttf", r"C:\Windows\Fonts\times.ttf"]
        else:
            candidates = [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\arial.ttf"]
    else:
        if bold:
            candidates = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]
        elif italic:
            candidates = [r"C:\Windows\Fonts\ariali.ttf", r"C:\Windows\Fonts\segoeuii.ttf"]
        else:
            candidates = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]

    for c in candidates:
        if Path(c).is_file():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()  # type: ignore[return-value]


def save_image_unicode(path: Path, bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if ok:
        buf.tofile(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _apply_paper_camera_effect(img: Image.Image, seed: int, vignette: float = 0.06) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Subtle illumination gradient simulating indoor desk lamp / camera capture
    gx = np.linspace(1.0 - vignette, 1.0 + vignette * 0.5, w, dtype=np.float32)
    gy = np.linspace(1.0, 1.0 - vignette * 0.7, h, dtype=np.float32)
    illum = gy[:, None] * gx[None, :]
    arr = arr * illum[:, :, None]

    # Mild sensor grain
    noise = rng.normal(0.0, 1.8, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


SINGLE_COL_SPECS: List[Dict[str, Any]] = [
    {
        "id": "doc_single_01",
        "title": "THÔNG BÁO LỊCH TIÊM CHỦNG ĐỊNH KỲ",
        "lines": [
            "TRẠM Y TẾ PHƯỜNG BẾN NGHÉ",
            "THÔNG BÁO LỊCH TIÊM CHỦNG ĐỊNH KỲ",
            "Thời gian buổi sáng: Từ 07 giờ 30 đến 11 giờ 00 ngày 15/11/2026",
            "Địa điểm: Số 18 đường Nguyễn Thị Minh Khai, Quận 1",
            "Khi đi phụ huynh mang theo sổ tiêm chủng của trẻ em.",
        ],
    },
    {
        "id": "doc_single_02",
        "title": "GIẤY HẸN TÁI KHÁM NHÃN KHOA",
        "lines": [
            "BỆNH VIỆN MẮT TRUNG ƯƠNG",
            "GIẤY HẸN TÁI KHÁM NHÃN KHOA",
            "Họ và tên bệnh nhân: Nguyễn Văn Minh",
            "Ngày hẹn khám lại: 24/11/2026 tại Phòng khám số 04",
            "Lưu ý: Nhịn ăn sáng trước khi làm xét nghiệm máu.",
        ],
    },
    {
        "id": "doc_single_03",
        "title": "THÔNG BÁO TẠM NGỪNG CẤP NƯỚC",
        "lines": [
            "CÔNG TY CẤP NƯỚC SÀI GÒN",
            "THÔNG BÁO TẠM NGỪNG CẤP NƯỚC",
            "Khu vực áp dụng: Phường Võ Thị Sáu, Quận 3",
            "Thời gian: Từ 22 giờ 00 ngày 18/11/2026 đến 04 giờ 00 ngày 19/11/2026",
            "Lý do: Bảo trì đường ống dẫn nước sạch định kỳ.",
        ],
    },
    {
        "id": "doc_single_04",
        "title": "HƯỚNG DẪN NỘP HỒ SƠ HÀNH CHÍNH",
        "lines": [
            "BỘ PHẬN TIẾP NHẬN VÀ TRẢ KẾT QUẢ",
            "HƯỚNG DẪN NỘP HỒ SƠ HÀNH CHÍNH",
            "Bước 1: Lấy số thứ tự tự động tại quầy hướng dẫn",
            "Bước 2: Xuất trình căn cước công dân bản chính",
            "Bước 3: Nhận phiếu hẹn trả kết quả tại quầy số 02.",
        ],
    },
    {
        "id": "doc_single_05",
        "title": "THÔNG BÁO THU TIỀN ĐIỆN THÁNG 11",
        "lines": [
            "ĐIỆN LỰC THÀNH PHỐ HÀ NỘI",
            "THÔNG BÁO THU TIỀN ĐIỆN THÁNG 11",
            "Mã khách hàng: PE010024891",
            "Chỉ số công tơ tiêu thụ: 245 kWh",
            "Tổng số tiền thanh toán: 612500 đồng.",
        ],
    },
    {
        "id": "doc_single_06",
        "title": "NỘI QUY PHÒNG ĐỌC THƯ VIỆN",
        "lines": [
            "THƯ VIỆN KHOA HỌC TỔNG HỢP",
            "NỘI QUY PHÒNG ĐỌC THƯ VIỆN",
            "1. Giữ trật tự và tắt chuông điện thoại di động",
            "2. Không mang đồ ăn thức uống vào phòng đọc sách",
            "3. Trả sách đúng quầy quy định trước 17 giờ 00.",
        ],
    },
    {
        "id": "doc_single_07",
        "title": "PHIẾU XÁC NHẬN ĐĂNG KÝ HỌC PHẦN",
        "lines": [
            "PHÒNG ĐÀO TẠO ĐẠI HỌC",
            "PHIẾU XÁC NHẬN ĐĂNG KÝ HỌC PHẦN",
            "Học kỳ 1 Năm học 2026 - 2027",
            "Môn học: Nhập môn Trí tuệ Nhân tạo - Mã lớp: IT4060",
            "Phòng học: Giảng đường B1 - Thứ Hai tiết 1 đến tiết 3.",
        ],
    },
    {
        "id": "doc_single_08",
        "title": "THÔNG BÁO HỌP TỔ DÂN PHỐ",
        "lines": [
            "BAN ĐIỀU HÀNH KHU PHỐ 5",
            "THÔNG BÁO HỌP TỔ DÂN PHỐ",
            "Kính mời đại diện các hộ gia đình tham dự cuộc họp",
            "Thời gian: 19 giờ 00 tối Thứ Bảy ngày 21/11/2026",
            "Địa điểm: Nhà sinh hoạt cộng đồng khu phố 5.",
        ],
    },
    {
        "id": "doc_single_09",
        "title": "HƯỚNG DẪN SỬ DỤNG THIẾT BỊ ĐỌC CHỮ",
        "lines": [
            "TRUNG TÂM HỖ TRỢ NGƯỜI KHIẾM THỊ",
            "HƯỚNG DẪN SỬ DỤNG THIẾT BỊ ĐỌC CHỮ",
            "Đặt tài liệu cách camera khoảng 25 đến 35 cm",
            "Nhấn phím Cách để chụp ảnh và nghe đọc tiếng Việt",
            "Nhấn tổ hợp Ctrl cộng L để xóa kết quả cũ.",
        ],
    },
    {
        "id": "doc_single_10",
        "title": "GIẤY MỜI NHẬN TRỢ CẤP XÃ HỘI",
        "lines": [
            "ỦY BAN NHÂN DÂN PHƯỜNG TÂN ĐỊNH",
            "GIẤY MỜI NHẬN TRỢ CẤP XÃ HỘI",
            "Thời gian chi trả: Từ ngày 05/11/2026 đến ngày 10/11/2026",
            "Địa điểm: Hội trường Ủy ban nhân dân phường",
            "Đề nghị mang theo sổ lĩnh trợ cấp hàng tháng.",
        ],
    },
]


MULTICOL_ADMIN_SPECS: List[Dict[str, Any]] = [
    {
        "id": "doc_multicol_01",
        "left_header": ["ỦY BAN NHÂN DÂN", "QUẬN HOÀN KIẾM", "Số: 12/NQ-UBND"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Hà Nội, ngày 09 tháng 11 năm 2026"],
        "body": [
            "NGHỊ QUYẾT",
            "Về việc phê duyệt kế hoạch hỗ trợ người cao tuổi",
            "QUYẾT NGHỊ:",
            "Điều 1. Thông qua danh sách hỗ trợ khám sức khỏe định kỳ.",
        ],
        "left_footer": ["Nơi nhận:", "- Như Điều 1;", "- Lưu: VT."],
        "right_footer": ["TM. ỦY BAN NHÂN DÂN", "CHỦ TỊCH"],
    },
    {
        "id": "doc_multicol_02",
        "left_header": ["SỞ Y TẾ HÀ NỘI", "BỆNH VIỆN ĐA KHOA", "Số: 45/QĐ-BVĐK"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Hà Nội, ngày 10 tháng 11 năm 2026"],
        "body": [
            "QUYẾT ĐỊNH",
            "Về việc ban hành quy trình tiếp đón bệnh nhân",
            "QUYẾT ĐỊNH:",
            "Điều 1. Áp dụng quy trình ưu tiên cho người khuyết tật.",
        ],
        "left_footer": ["Nơi nhận:", "- Các khoa phòng;", "- Lưu: VT."],
        "right_footer": ["GIÁM ĐỐC BỆNH VIỆN", "PGS.TS. Trần Quốc Bảo"],
    },
    {
        "id": "doc_multicol_03",
        "left_header": ["SỞ GIÁO DỤC VÀ ĐÀO TẠO", "TRƯỜNG THPT CHU VĂN AN", "Số: 88/TB-CVA"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Hà Nội, ngày 12 tháng 11 năm 2026"],
        "body": [
            "THÔNG BÁO",
            "Về lịch kiểm tra học kỳ 1 năm học 2026 - 2027",
            "Nhà trường thông báo lịch kiểm tra tập trung cho các khối lớp.",
        ],
        "left_footer": ["Nơi nhận:", "- Giáo viên chủ nhiệm;", "- Lưu: VP."],
        "right_footer": ["HIỆU TRƯỞNG", "Nguyễn Thị Lan Hương"],
    },
    {
        "id": "doc_multicol_04",
        "left_header": ["TỔNG CÔNG TY ĐIỆN LỰC", "CÔNG TY ĐIỆN LỰC ĐỐNG ĐA", "Số: 104/CV-PCĐĐ"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Hà Nội, ngày 14 tháng 11 năm 2026"],
        "body": [
            "Kính gửi: Quý khách hàng sử dụng điện trên địa bàn",
            "Công ty Điện lực thông báo lịch kiểm tra an toàn lưới điện mùa khô.",
        ],
        "left_footer": ["Nơi nhận:", "- Như trên;", "- Lưu: VT, KD."],
        "right_footer": ["KT. GIÁM ĐỐC", "PHÓ GIÁM ĐỐC"],
    },
    {
        "id": "doc_multicol_05",
        "left_header": ["ỦY BAN NHÂN DÂN", "PHƯỜNG LÁNG HẠ", "Số: 29/GXN-UBND"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Láng Hạ, ngày 15 tháng 11 năm 2026"],
        "body": [
            "GIẤY XÁC NHẬN",
            "Ủy ban nhân dân phường xác nhận ông Lê Hoàng Nam hiện cư trú tại địa phương.",
        ],
        "left_footer": ["Nơi nhận:", "- Người đề nghị;", "- Lưu: Tư pháp."],
        "right_footer": ["TM. ỦY BAN NHÂN DÂN", "KT. CHỦ TỊCH"],
    },
    {
        "id": "doc_multicol_06",
        "left_header": ["HỘI NGƯỜI MÙ VIỆT NAM", "THÀNH HỘI HÀ NỘI", "Số: 16/GGT-HNM"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Hà Nội, ngày 16 tháng 11 năm 2026"],
        "body": [
            "GIẤY GIỚI THIỆU",
            "Trân trọng giới thiệu đồng chí Phạm Minh Tuấn đến liên hệ công tác.",
        ],
        "left_footer": ["Nơi nhận:", "- Như trên;", "- Lưu: VP."],
        "right_footer": ["TM. BAN THƯỜNG VỤ", "CHỦ TỊCH"],
    },
    {
        "id": "doc_multicol_07",
        "left_header": ["BỘ LAO ĐỘNG THƯƠNG BINH", "CỤC BẢO TRỢ XÃ HỘI", "Số: 63/TTr-BTXH"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Hà Nội, ngày 17 tháng 11 năm 2026"],
        "body": [
            "TỜ TRÌNH",
            "Về việc cấp phát thiết bị hỗ trợ tiếp cận thông tin cho người khiếm thị",
        ],
        "left_footer": ["Nơi nhận:", "- Lãnh đạo Bộ;", "- Lưu: VT."],
        "right_footer": ["CỤC TRƯỞNG", "Đặng Văn Thành"],
    },
    {
        "id": "doc_multicol_08",
        "left_header": ["TRUNG TÂM DỊCH VỤ VIỆC LÀM", "PHÒNG TƯ VẤN ĐÀO TẠO", "Số: 72/BB-TTDV"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Hà Nội, ngày 18 tháng 11 năm 2026"],
        "body": [
            "BIÊN BẢN BÀN GIAO TÀI LIỆU",
            "Hai bên tiến hành bàn giao 50 bộ tài liệu hướng dẫn kỹ năng tin học.",
        ],
        "left_footer": ["ĐẠI DIỆN BÊN GIAO", "Nguyễn Văn Hòa"],
        "right_footer": ["ĐẠI DIỆN BÊN NHẬN", "Vũ Thị Mai"],
    },
    {
        "id": "doc_multicol_09",
        "left_header": ["BẢO HIỂM XÃ HỘI VIỆT NAM", "BHXH QUẬN CẦU GIẤY", "Số: 215/PTN-BHXH"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Cầu Giấy, ngày 19 tháng 11 năm 2026"],
        "body": [
            "PHIẾU TIẾP NHẬN HỒ SƠ",
            "Đã tiếp nhận hồ sơ cấp lại thẻ bảo hiểm y tế của bà Hoàng Thị Cúc.",
        ],
        "left_footer": ["NGƯỜI NỘP HỒ SƠ", "Hoàng Thị Cúc"],
        "right_footer": ["CÁN BỘ TIẾP NHẬN", "Lê Thanh Tùng"],
    },
    {
        "id": "doc_multicol_10",
        "left_header": ["TRƯỜNG ĐẠI HỌC BÁCH KHOA", "VIỆN CÔNG NGHỆ THÔNG TIN", "Số: 91/QĐ-CNTT"],
        "right_header": ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "Hà Nội, ngày 20 tháng 11 năm 2026"],
        "body": [
            "QUYẾT ĐỊNH",
            "Về việc giao đề tài nghiên cứu ứng dụng đọc văn bản ngoại tuyến",
            "QUYẾT ĐỊNH:",
            "Điều 1. Giao nhiệm vụ phát triển phần mềm đọc tài liệu tiếng Việt.",
        ],
        "left_footer": ["Nơi nhận:", "- Nhóm nghiên cứu;", "- Lưu: Văn thư."],
        "right_footer": ["VIỆN TRƯỞNG", "GS.TS. Phạm Quang Minh"],
    },
]


PACKAGING_SPECS: List[Dict[str, Any]] = [
    {
        "id": "pkg_01",
        "bg": (244, 248, 252),
        "accent": (20, 90, 160),
        "lines": [
            "VIÊN NÉN PARACETAMOL 500MG",
            "Chỉ định: Giảm đau, hạ sốt nhanh",
            "Liều dùng: Uống 1 viên/lần, cách nhau 6 giờ",
            "NSX: 10/02/2026 - HSD: 10/02/2029",
        ],
    },
    {
        "id": "pkg_02",
        "bg": (250, 246, 236),
        "accent": (150, 75, 20),
        "lines": [
            "SỮA TƯƠI TIỆT TRÙNG ÍT ĐƯỜNG",
            "Thể tích thực: 180 ml - Năng lượng: 112 kcal",
            "Bảo quản nơi khô ráo, thoáng mát",
            "HSD: 25/12/2026 - Số lô: L2608",
        ],
    },
    {
        "id": "pkg_03",
        "bg": (238, 248, 240),
        "accent": (25, 115, 50),
        "lines": [
            "NƯỚC NHỎ MẮT NATRI CLORID 0,9%",
            "Dung tích: Lọ 10 ml vô trùng",
            "Cách dùng: Nhỏ 2 đến 3 giọt mỗi bên mắt",
            "Không dùng quá 15 ngày sau khi mở nắp.",
        ],
    },
    {
        "id": "pkg_04",
        "bg": (252, 252, 250),
        "accent": (40, 40, 40),
        "lines": [
            "SIÊU THỊ CO.OPMART HÀ NỘI",
            "HÓA ĐƠN BÁN LẺ HÀNG HÓA",
            "1. Gạo thơm ST25 túi 5kg: 175000 đồng",
            "2. Dầu ăn thực vật 1 lít: 52000 đồng",
            "Tổng thanh toán: 227000 đồng.",
        ],
    },
    {
        "id": "pkg_05",
        "bg": (236, 224, 204),
        "accent": (55, 38, 22),
        "lines": [
            "TRÀ GỪNG MẬT ONG THẢO DƯỢC",
            "Khối lượng tịnh: 200 g (20 gói x 10 g)",
            "Pha 1 gói với 150 ml nước ấm 80 độ C",
            "NSX: 01/06/2026 - HSD: 01/06/2028",
        ],
    },
    {
        "id": "pkg_06",
        "bg": (248, 244, 252),
        "accent": (95, 35, 135),
        "lines": [
            "VIÊN SỦI VITAMIN C 1000MG",
            "Hòa tan 1 viên trong 200 ml nước đun sôi để nguội",
            "Liều dùng: Người lớn uống 1 viên mỗi ngày",
            "Giá bán lẻ: 48000 VNĐ/tuýp 10 viên.",
        ],
    },
    {
        "id": "pkg_07",
        "bg": (250, 250, 248),
        "accent": (30, 80, 120),
        "lines": [
            "NHÀ THUỐC ĐẠT CHUẨN GPP",
            "PHIẾU HƯỚNG DẪN DÙNG THUỐC",
            "1. Amoxicillin 500mg: Sáng 1 viên, Tối 1 viên sau ăn",
            "2. Vitamin B1 50mg: Sáng 1 viên trước bữa ăn.",
        ],
    },
    {
        "id": "pkg_08",
        "bg": (245, 248, 242),
        "accent": (45, 95, 45),
        "lines": [
            "YẾN MẠCH NGUYÊN CHẤT CÁN DẸT",
            "Thành phần: 100% yến mạch nhập khẩu Úc",
            "Khối lượng tịnh: 500 g - Giá: 89000 đồng",
            "HSD: 18/09/2027 - Mã vạch: 89360124",
        ],
    },
    {
        "id": "pkg_09",
        "bg": (252, 246, 242),
        "accent": (165, 40, 30),
        "lines": [
            "CỒN Y TẾ ETHANOL 70 ĐỘ",
            "Chai 500 ml - Chỉ dùng sát khuẩn ngoài da",
            "Cảnh báo: Không được uống, tránh xa lửa",
            "Số đăng ký: VD-24819-26.",
        ],
    },
    {
        "id": "pkg_10",
        "bg": (248, 249, 250),
        "accent": (25, 25, 25),
        "lines": [
            "BƯU ĐIỆN THÀNH PHỐ HỒ CHÍ MINH",
            "PHIẾU GỬI BƯU PHẨM ĐẢM BẢO",
            "Mã vận đơn: VN908124578",
            "Khối lượng: 450 g - Cước phí: 32000 đồng.",
        ],
    },
]


def render_single_col_doc(idx: int, spec: Dict[str, Any]) -> np.ndarray:
    img = Image.new("RGB", (1080, 1440), color=(250, 249, 246))
    draw = ImageDraw.Draw(img)
    # Page border and header underline
    draw.rectangle([45, 45, 1035, 1395], outline=(190, 190, 185), width=2)

    f_agency = _font("serif", 28, bold=True)
    f_title = _font("serif", 34, bold=True)
    f_body = _font("serif", 29, bold=False)

    lines = spec["lines"]
    # Line 0: agency header
    draw.text((95, 110), lines[0], fill=(18, 18, 20), font=f_agency)
    # Decorative rule under agency
    draw.line([(95, 155), (420, 155)], fill=(30, 30, 30), width=2)

    # Line 1: title
    draw.text((95, 220), lines[1], fill=(15, 15, 18), font=f_title)
    draw.line([(95, 275), (960, 275)], fill=(80, 80, 80), width=2)

    # Body lines + realistic dotted fill-in-the-blank line at bottom
    y = 345
    for ln in lines[2:]:
        draw.text((95, y), ln, fill=(20, 20, 22), font=f_body)
        y += 95

    # Blank dotted leader line below body (must be suppressed by layout.py, not read as garbage)
    draw.text((95, y + 40), "." * 95, fill=(70, 70, 70), font=f_body)
    draw.line([(95, y + 120), (960, y + 120)], fill=(90, 90, 90), width=2)

    return _apply_paper_camera_effect(img, seed=3000 + idx)


def render_multicol_admin_doc(idx: int, spec: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    img = Image.new("RGB", (1160, 1560), color=(251, 250, 247))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 1120, 1520], outline=(185, 185, 180), width=2)

    f_hdr_b = _font("serif", 25, bold=True)
    f_hdr_r = _font("serif", 24, bold=False)
    f_hdr_i = _font("serif", 24, italic=True)
    f_title = _font("serif", 34, bold=True)
    f_body = _font("serif", 27, bold=False)
    f_body_b = _font("serif", 28, bold=True)
    f_foot_b = _font("serif", 25, bold=True)
    f_foot_r = _font("serif", 24, bold=False)

    # Top Left Column (x in 75..460)
    y_l = 105
    for i, text in enumerate(spec["left_header"]):
        font_use = f_hdr_b if i < 2 else f_hdr_r
        draw.text((80, y_l), text, fill=(18, 18, 20), font=font_use)
        y_l += 46
        if i == 1:
            # Horizontal underline below agency name
            draw.line([(130, y_l - 8), (330, y_l - 8)], fill=(25, 25, 25), width=2)
            y_l += 10

    # Top Right Column (x in 520..1080)
    y_r = 105
    for i, text in enumerate(spec["right_header"]):
        if i == 0:
            font_use = f_hdr_b
        elif i == 1:
            font_use = f_hdr_b
        else:
            font_use = f_hdr_i
        draw.text((535, y_r), text, fill=(18, 18, 20), font=font_use)
        y_r += 46
        if i == 1:
            # Horizontal underline below Tiêu ngữ
            draw.line([(620, y_r - 8), (940, y_r - 8)], fill=(25, 25, 25), width=2)
            y_r += 10

    # Center Body Band
    y_b = 370
    for i, text in enumerate(spec["body"]):
        is_heading = text.isupper() or text.endswith(":")
        font_use = f_title if (i == 0 and text.isupper()) else (f_body_b if is_heading else f_body)
        x_pos = 420 if (i == 0 and len(text) < 22) else 95
        draw.text((x_pos, y_b), text, fill=(16, 16, 18), font=font_use)
        y_b += 78

    # Blank dotted leader line between body and footer (simulating administrative form blank lines)
    draw.text((95, y_b + 15), "." * 98, fill=(85, 85, 85), font=f_hdr_r)

    # Bottom 2-Column Footer (y around 900)
    y_fl = 890
    for i, text in enumerate(spec["left_footer"]):
        font_use = f_foot_b if i == 0 else f_foot_r
        draw.text((95, y_fl), text, fill=(18, 18, 20), font=font_use)
        y_fl += 48

    y_fr = 890
    for i, text in enumerate(spec["right_footer"]):
        font_use = f_foot_b if i == 0 else f_foot_r
        draw.text((610, y_fr), text, fill=(18, 18, 20), font=font_use)
        y_fr += 52

    expected_order: List[str] = (
        list(spec["left_header"])
        + list(spec["right_header"])
        + list(spec["body"])
        + list(spec["left_footer"])
        + list(spec["right_footer"])
    )
    bgr = _apply_paper_camera_effect(img, seed=4000 + idx)
    return bgr, expected_order


def render_low_quality_doc(idx: int) -> Tuple[np.ndarray, str, str]:
    """
    Generate 10 low-quality / unreadable document images:
    - 0..4: Low-resolution thumbnail administrative forms (~360x510) with tiny 5-6px footnotes/body
            (triggers OCR low-resolution uncertainty refusal `UNCERTAIN`).
    - 5..6: Severe camera motion blur / defocus on document (`BAD_IMAGE`).
    - 7: Very dark underexposed document capture (`BAD_IMAGE`).
    - 8: Overexposed flash glare document capture (`BAD_IMAGE`).
    - 9: Ultra-tiny distant document crop (`340x480`, 5px text -> `UNCERTAIN`).
    """
    if idx in (0, 1, 2, 3, 4, 9):
        # First render a dense full page with many small footnote lines, then downsample to ~370x520
        hi = Image.new("RGB", (1140, 1580), color=(238, 238, 238))
        draw = ImageDraw.Draw(hi)
        f_big = _font("serif", 32, bold=True)
        f_med = _font("serif", 24, bold=True)
        f_tiny = _font("serif", 16, bold=False)

        draw.text((80, 90), "TÊN CƠ QUAN CHỦ QUẢN", fill=(25, 25, 25), font=f_med)
        draw.text((520, 90), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", fill=(20, 20, 20), font=f_med)
        draw.text((590, 130), "Độc lập - Tự do - Hạnh phúc", fill=(20, 20, 20), font=f_med)
        draw.text((430, 250), f"BIỂU MẪU HÀNH CHÍNH SỐ {idx + 1}", fill=(20, 20, 20), font=f_big)

        y = 340
        for k in range(18):
            draw.text(
                (80, y),
                f"Ghi chú hướng dẫn chi tiết mục ({k + 1}): Nội dung chữ rất nhỏ dưới ngưỡng phân giải camera.",
                fill=(45, 45, 45),
                font=f_tiny,
            )
            y += 52

        target_size = (365 + (idx % 3) * 8, 515 + (idx % 3) * 10)
        lo = hi.resize(target_size, resample=Image.Resampling.BILINEAR)
        bgr = cv2.cvtColor(np.array(lo), cv2.COLOR_RGB2BGR)
        return bgr, "low_res_thumbnail_doc", "UNCERTAIN"

    # Base readable document for camera defect simulations
    base_bgr = render_single_col_doc(0, SINGLE_COL_SPECS[idx % len(SINGLE_COL_SPECS)])
    if idx in (5, 6):
        # Severe out-of-focus blur
        blurred = cv2.GaussianBlur(base_bgr, (61, 61), 22.0)
        return blurred, "severe_camera_blur_doc", "BAD_IMAGE"
    if idx == 7:
        # Very dark unlit document
        dark = (base_bgr.astype(np.float32) * 0.08).astype(np.uint8)
        return dark, "underexposed_dark_doc", "BAD_IMAGE"
    # idx == 8: Overexposed flash glare
    glare = np.clip(base_bgr.astype(np.float32) * 1.15 + 235.0, 0, 255).astype(np.uint8)
    return glare, "overexposed_glare_doc", "BAD_IMAGE"


def render_packaging_image(idx: int, spec: Dict[str, Any]) -> np.ndarray:
    img = Image.new("RGB", (940, 620), color=spec["bg"])
    draw = ImageDraw.Draw(img)
    accent = spec["accent"]

    draw.rectangle([28, 28, 912, 592], outline=accent, width=3)
    draw.line([(48, 128), (892, 128)], fill=accent, width=3)

    f_title = _font("sans", 32, bold=True)
    f_body = _font("sans", 28, bold=False)

    lines = spec["lines"]
    draw.text((52, 58), lines[0], fill=accent, font=f_title)

    y = 165
    for ln in lines[1:]:
        draw.text((52, y), ln, fill=(20, 22, 25), font=f_body)
        y += 88

    return _apply_paper_camera_effect(img, seed=5000 + idx, vignette=0.05)


def generate_p1_1_dataset() -> Dict[str, Any]:
    REAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []

    # 1. 10 Single-column readable documents
    for i, spec in enumerate(SINGLE_COL_SPECS, start=1):
        fname = f"{spec['id']}.png"
        fpath = REAL_DOCS_DIR / fname
        bgr = render_single_col_doc(i, spec)
        sha = save_image_unicode(fpath, bgr)
        gt = "\n".join(spec["lines"])
        items.append({
            "id": spec["id"],
            "filename": fname,
            "sha256": sha,
            "category": "readable_doc_single",
            "layout": "single_column",
            "expected_status": "READABLE",
            "expected_lines_order": spec["lines"],
            "ground_truth": gt,
        })

    # 2. 10 Multi-column readable administrative documents
    for i, spec in enumerate(MULTICOL_ADMIN_SPECS, start=1):
        fname = f"{spec['id']}.png"
        fpath = REAL_DOCS_DIR / fname
        bgr, expected_order = render_multicol_admin_doc(i, spec)
        sha = save_image_unicode(fpath, bgr)
        gt = "\n".join(expected_order)
        items.append({
            "id": spec["id"],
            "filename": fname,
            "sha256": sha,
            "category": "readable_doc_multicol",
            "layout": "two_column_admin",
            "expected_status": "READABLE",
            "expected_lines_order": expected_order,
            "ground_truth": gt,
        })

    # 3. 10 Low-quality / unreadable documents
    for i in range(10):
        doc_id = f"doc_lowqual_{i + 1:02d}"
        fname = f"{doc_id}.png"
        fpath = REAL_DOCS_DIR / fname
        bgr, defect_kind, exp_status = render_low_quality_doc(i)
        sha = save_image_unicode(fpath, bgr)
        items.append({
            "id": doc_id,
            "filename": fname,
            "sha256": sha,
            "category": "low_quality_unreadable_doc",
            "condition": defect_kind,
            "expected_status": exp_status,
            "ground_truth": "",
        })

    # 4. 10 Product packaging / medicine box / receipt images
    for i, spec in enumerate(PACKAGING_SPECS, start=1):
        fname = f"{spec['id']}.png"
        fpath = REAL_DOCS_DIR / fname
        bgr = render_packaging_image(i, spec)
        sha = save_image_unicode(fpath, bgr)
        gt = "\n".join(spec["lines"])
        items.append({
            "id": spec["id"],
            "filename": fname,
            "sha256": sha,
            "category": "product_packaging_receipt",
            "layout": "packaging_multiline",
            "expected_status": "READABLE",
            "expected_lines_order": spec["lines"],
            "ground_truth": gt,
        })

    reg_sha = hashlib.sha256(REGRESSION_IMG_PATH.read_bytes()).hexdigest().upper()
    manifest = {
        "dataset_name": "VisionMate_Phase1_1_RealDocument_And_Packaging_Suite",
        "version": "1.1.0",
        "generated_date": "2026-11-09",
        "regression_primary_sample": {
            "path": "tests/fixtures/regression_p1_1/administrative_form_379x527.png",
            "sha256": reg_sha,
            "resolution": "379x527",
            "expected_status": "UNCERTAIN",
        },
        "counts": {
            "total": len(items),
            "readable_doc_single": 10,
            "readable_doc_multicol": 10,
            "low_quality_unreadable_doc": 10,
            "product_packaging_receipt": 10,
        },
        "items": items,
    }
    with MANIFEST_P1_1_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


if __name__ == "__main__":
    m = generate_p1_1_dataset()
    print(f"Generated {m['counts']['total']} Phase 1.1 evaluation images in {REAL_DOCS_DIR}")
    print(f"Primary regression image SHA-256: {m['regression_primary_sample']['sha256']}")
