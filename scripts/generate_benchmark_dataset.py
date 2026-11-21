# -*- coding: utf-8 -*-
"""
Generates the 70-image consent-safe synthetic benchmark dataset and manifest.json (A02..A05).
Categories:
- 30 clear_vi: Clear printed Vietnamese labels (single & multi-line, varied fonts/sizes)
- 20 challenging_vi: Challenging lighting, tilt, noise, small font, low contrast, dark bg
- 10 no_text: Non-text scenes, blank surfaces, dark/glare/blurry frames
- 10 digits_dates: Product dates (NSX/HSD), dosages (mg/ml), prices, serials, voltages
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "tests" / "fixtures" / "dataset"
MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "manifest.json"

FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
]


def get_font(idx: int, size: int) -> ImageFont.FreeTypeFont:
    for offset in range(len(FONT_PATHS)):
        candidate = FONT_PATHS[(idx + offset) % len(FONT_PATHS)]
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()  # type: ignore[return-value]


def save_image_unicode(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if ok:
        buf.tofile(str(path))


CLEAR_TEXTS: List[str] = [
    "Sữa tươi tiệt trùng không đường",
    "Nước khoáng thiên nhiên đóng chai",
    "Bánh quy bơ sữa cao cấp",
    "Trà xanh túi lọc hương nhài",
    "Cà phê rang xay nguyên chất",
    "Gạo thơm đặc sản Sóc Trăng",
    "Nước mắm truyền thống Phú Quốc",
    "Dầu ăn thực vật tinh luyện",
    "Mì khoai tây sợi dai thơm ngon",
    "Bột giặt hương hoa tự nhiên",
    "Nước rửa chén chiết xuất chanh",
    "Khăn giấy lụa mềm mại hai lớp",
    "Kem đánh răng bảo vệ nướu",
    "Dầu gội thảo dược bồ kết",
    "Sữa chua ăn men sống tự nhiên",
    "Mật ong rừng nguyên chất Tây Nguyên",
    "Hạt điều rang muối Bình Phước",
    "Ngũ cốc dinh dưỡng buổi sáng",
    "Tương ớt nguyên chất cay dịu",
    "Nước ép cam tươi nguyên chất",
    "Bảo quản nơi khô ráo thoáng mát",
    "Tránh ánh nắng trực tiếp mặt trời",
    "Lắc đều trước khi sử dụng",
    "Dùng ngay sau khi mở nắp hộp",
    "Thành phần: Sữa bò tươi nguyên chất",
    "Sản xuất tại nhà máy Việt Nam",
    "Hướng dẫn sử dụng in trên bao bì",
    "Phòng khám đa khoa khu vực tầng một",
    "Lối đi dành cho người đi bộ",
    "Quầy tiếp nhận thông tin khách hàng",
]

CHALLENGING_SPECS: List[Tuple[str, str]] = [
    ("Bánh gạo nướng vị rong biển", "small_font"),
    ("Nước tương lên men tự nhiên", "low_contrast"),
    ("Sữa đặc có đường cao cấp", "shadow_gradient"),
    ("Trà thảo mộc thanh nhiệt cơ thể", "tilt_pos"),
    ("Bột ngũ cốc dinh dưỡng ít đường", "tilt_neg"),
    ("Dầu ô liu nguyên chất nhập khẩu", "sensor_noise"),
    ("Cà phê hòa tan đậm đà hương vị", "dark_inverted"),
    ("Kẹo dẻo trái cây tổng hợp", "cardboard_bg"),
    ("Nước súc miệng diệt khuẩn bạc hà", "mild_blur"),
    ("Muối biển i-ốt tinh khiết", "small_font"),
    ("Bơ thực vật giàu vitamin", "low_contrast"),
    ("Cháo tổ yến thịt bằm dinh dưỡng", "shadow_gradient"),
    ("Nước cốt dừa đóng lon tiện lợi", "tilt_pos"),
    ("Bánh phồng tôm đặc sản miền Tây", "sensor_noise"),
    ("Sô cô la đen nguyên chất đắng nhẹ", "dark_inverted"),
    ("Giấm gạo lên men tự nhiên", "cardboard_bg"),
    ("Đường tinh luyện trắng sạch", "mild_blur"),
    ("Hạt tiêu đen xay nhuyễn thơm nồng", "tilt_neg"),
    ("Nước giải khát hương chanh muối", "shadow_gradient"),
    ("Bột canh gia vị nấu ăn hằng ngày", "low_contrast"),
]

DIGITS_DATES_TEXTS: List[str] = [
    "NSX: 15/03/2026 - HSD: 15/03/2027",
    "Paracetamol 500mg - Hộp 10 vỉ",
    "Giá bán: 145000 VNĐ - Mã SP: VN893",
    "Dung tích thực: 180 ml - 110 kcal",
    "Liều dùng: 2 viên/ngày sau bữa ăn",
    "Số lô: B260911 - Ngày: 09/11/2026",
    "Điện áp: 220V 50Hz - Công suất 1500W",
    "Nhiệt độ bảo quản: từ 2 đến 8 độ C",
    "Tổng thanh toán: 1250500 đồng",
    "Cồn y tế Ethanol 70 độ - Chai 500ml",
]


def render_clear_image(idx: int, text: str) -> np.ndarray:
    bg_colors = [(255, 255, 255), (250, 249, 246), (246, 248, 250), (252, 252, 248)]
    bg = bg_colors[idx % len(bg_colors)]
    img = Image.new("RGB", (800, 220), color=bg)
    draw = ImageDraw.Draw(img)
    # Draw subtle label border so active tile contrast is realistic
    draw.rectangle([20, 20, 780, 200], outline=(180, 185, 190), width=2)
    font_size = 30 + (idx % 3) * 3
    font = get_font(idx, font_size)
    draw.text((48, 85), text, fill=(15, 18, 22), font=font)
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_challenging_image(idx: int, text: str, mode: str) -> np.ndarray:
    rng = np.random.default_rng(1000 + idx)

    if mode == "dark_inverted":
        bg = (28, 32, 38)
        fg = (242, 244, 248)
    elif mode == "low_contrast":
        bg = (215, 215, 215)
        fg = (95, 95, 95)
    elif mode == "cardboard_bg":
        bg = (212, 194, 168)
        fg = (35, 28, 20)
    else:
        bg = (248, 248, 246)
        fg = (22, 22, 22)

    img = Image.new("RGB", (820, 240), color=bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle([24, 24, 796, 216], outline=fg, width=2)

    font_size = 22 if mode == "small_font" else 30
    font = get_font(idx, font_size)
    draw.text((52, 92), text, fill=fg, font=font)

    if mode == "tilt_pos":
        img = img.rotate(2.5, resample=Image.Resampling.BICUBIC, fillcolor=bg)
    elif mode == "tilt_neg":
        img = img.rotate(-2.5, resample=Image.Resampling.BICUBIC, fillcolor=bg)
    elif mode == "mild_blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=0.85))

    arr = np.array(img, dtype=np.float32)

    if mode == "shadow_gradient":
        grad = np.linspace(0.62, 1.05, arr.shape[1], dtype=np.float32)
        arr = arr * grad[None, :, None]
    elif mode == "sensor_noise":
        noise = rng.normal(0.0, 9.0, size=arr.shape).astype(np.float32)
        arr = arr + noise

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def render_no_text_image(idx: int) -> Tuple[np.ndarray, str]:
    rng = np.random.default_rng(2000 + idx)
    h, w = 360, 640

    if idx == 0:
        # Blank off-white wall
        arr = np.full((h, w, 3), 235, dtype=np.uint8)
        return arr, "blank_wall"
    if idx == 1:
        # Smooth wooden gradient surface
        base = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            val = int(140 + 18 * math.sin(y / 14.0))
            base[y, :] = (val - 30, val - 10, val + 20)
        return base, "wood_surface"
    if idx == 2:
        # Geometric circles only (sharp edges, zero text)
        arr = np.full((h, w, 3), 240, dtype=np.uint8)
        cv2.circle(arr, (200, 180), 80, (60, 120, 200), thickness=6)
        cv2.circle(arr, (440, 180), 70, (80, 170, 90), thickness=6)
        return arr, "geometric_circles"
    if idx == 3:
        # Strongly out-of-focus blurry scene (triggers quality gate TOO_BLURRY)
        arr = np.full((h, w, 3), 220, dtype=np.uint8)
        cv2.rectangle(arr, (120, 80), (520, 280), (60, 60, 60), -1)
        arr = cv2.GaussianBlur(arr, (51, 51), 18.0)
        return arr, "out_of_focus_blur"
    if idx == 4:
        # Very dark unlit frame (triggers quality gate TOO_DARK)
        arr = rng.integers(4, 18, size=(h, w, 3), dtype=np.uint8)
        return arr, "too_dark_frame"
    if idx == 5:
        # Overexposed glare frame (triggers quality gate TOO_BRIGHT_OR_GLARE)
        arr = np.full((h, w, 3), 253, dtype=np.uint8)
        return arr, "overexposed_glare"
    if idx == 6:
        # Soft gradient sky
        grad = np.linspace(160, 230, h, dtype=np.uint8)[:, None]
        arr = np.repeat(grad[:, :, None], w, axis=1)
        arr = np.repeat(arr, 3, axis=2)
        return arr, "soft_gradient"
    if idx == 7:
        # Blank pastel card
        arr = np.full((h, w, 3), (225, 238, 245), dtype=np.uint8)
        return arr, "blank_pastel_card"
    if idx == 8:
        # Defocused natural blobs
        noise = rng.integers(80, 190, size=(h, w, 3), dtype=np.uint8)
        arr = cv2.GaussianBlur(noise, (45, 45), 14.0)
        return arr, "defocused_blobs"
    # idx == 9: Gray gradient surface
    grad_x = np.linspace(110, 195, w, dtype=np.uint8)[None, :]
    arr = np.repeat(grad_x[:, :, None], h, axis=0)
    arr = np.repeat(arr, 3, axis=2)
    return arr, "gray_gradient"


def generate_all() -> Dict[str, Any]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []

    # 1. 30 clear Vietnamese text images
    for i, gt in enumerate(CLEAR_TEXTS, start=1):
        fname = f"clear_{i:02d}.png"
        fpath = DATASET_DIR / fname
        bgr = render_clear_image(i - 1, gt)
        save_image_unicode(fpath, bgr)
        items.append({
            "id": f"clear_{i:02d}",
            "filename": fname,
            "category": "clear_vi",
            "ground_truth": gt,
            "synthetic": True,
            "license": "CC0-1.0 (Project Synthetic Benchmark)",
            "notes": "Clear printed Vietnamese text",
        })

    # 2. 20 challenging Vietnamese text images
    for i, (gt, mode) in enumerate(CHALLENGING_SPECS, start=1):
        fname = f"challenging_{i:02d}.png"
        fpath = DATASET_DIR / fname
        bgr = render_challenging_image(i - 1, gt, mode)
        save_image_unicode(fpath, bgr)
        items.append({
            "id": f"challenging_{i:02d}",
            "filename": fname,
            "category": "challenging_vi",
            "ground_truth": gt,
            "condition": mode,
            "synthetic": True,
            "license": "CC0-1.0 (Project Synthetic Benchmark)",
            "notes": f"Challenging condition: {mode}",
        })

    # 3. 10 no-text images
    for i in range(1, 11):
        fname = f"notext_{i:02d}.png"
        fpath = DATASET_DIR / fname
        bgr, subkind = render_no_text_image(i - 1)
        save_image_unicode(fpath, bgr)
        items.append({
            "id": f"notext_{i:02d}",
            "filename": fname,
            "category": "no_text",
            "ground_truth": "",
            "condition": subkind,
            "synthetic": True,
            "license": "CC0-1.0 (Project Synthetic Benchmark)",
            "notes": f"No-text sample ({subkind})",
        })

    # 4. 10 digit/date/dosage Vietnamese images
    for i, gt in enumerate(DIGITS_DATES_TEXTS, start=1):
        fname = f"digits_{i:02d}.png"
        fpath = DATASET_DIR / fname
        bgr = render_clear_image(i + 30, gt)
        save_image_unicode(fpath, bgr)
        items.append({
            "id": f"digits_{i:02d}",
            "filename": fname,
            "category": "digits_dates",
            "ground_truth": gt,
            "synthetic": True,
            "license": "CC0-1.0 (Project Synthetic Benchmark)",
            "notes": "Digits, dates, dosages, units preservation test",
        })

    manifest = {
        "dataset_name": "VisionMate_Phase1_Benchmark_70",
        "version": "1.0.0",
        "generated_date": "2026-11-09",
        "synthetic": True,
        "consent_safe": True,
        "license": "CC0-1.0 (Synthetic images rendered locally via Windows system fonts)",
        "counts": {
            "total": len(items),
            "clear_vi": 30,
            "challenging_vi": 20,
            "no_text": 10,
            "digits_dates": 10,
        },
        "items": items,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


if __name__ == "__main__":
    m = generate_all()
    print(f"Generated {m['counts']['total']} images in {DATASET_DIR}")
