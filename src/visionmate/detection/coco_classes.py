# -*- coding: utf-8 -*-
"""
Official COCO 80-Class Labels, Vietnamese Mapping, and Natural Phrasing for VisionMate OD-1.

Follows Vietnamese grammar standards with appropriate noun classifiers:
- 'một chai nước' (bottle)
- 'một chiếc cốc' (cup)
- 'một điện thoại' (cell phone)
- '2 chiếc ghế' (chairs)
- 'Chưa xác định được đồ vật.' (when confidence is low or no detections)
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple


# Official COCO 80 classes (0 to 79)
COCO_CLASSES_EN: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)

# Standard Vietnamese translation and primary classifier
# (Vietnamese label, Vietnamese singular classifier, prefix with classifier in speech)
COCO_INFO_VI: Dict[str, Tuple[str, str, bool]] = {
    "person": ("người", "người", False),
    "bicycle": ("xe đạp", "chiếc", True),
    "car": ("ô tô", "chiếc", True),
    "motorcycle": ("xe máy", "chiếc", True),
    "airplane": ("máy bay", "chiếc", True),
    "bus": ("xe buýt", "chiếc", True),
    "train": ("tàu hỏa", "chuyến", True),
    "truck": ("xe tải", "chiếc", True),
    "boat": ("thuyền", "chiếc", True),
    "traffic light": ("đèn giao thông", "cột", True),
    "fire hydrant": ("trụ cứu hỏa", "trụ", True),
    "stop sign": ("biển báo dừng", "biển", True),
    "parking meter": ("đồng hồ đỗ xe", "chiếc", True),
    "bench": ("ghế dài", "chiếc", True),
    "bird": ("chim", "con", True),
    "cat": ("mèo", "con", True),
    "dog": ("chó", "con", True),
    "horse": ("ngựa", "con", True),
    "sheep": ("cừu", "con", True),
    "cow": ("bò", "con", True),
    "elephant": ("voi", "con", True),
    "bear": ("gấu", "con", True),
    "zebra": ("ngựa vằn", "con", True),
    "giraffe": ("hươu cao cổ", "con", True),
    "backpack": ("ba lô", "chiếc", True),
    "umbrella": ("ô", "chiếc", True),
    "handbag": ("túi xách", "chiếc", True),
    "tie": ("cà vạt", "chiếc", True),
    "suitcase": ("va li", "chiếc", True),
    "frisbee": ("đĩa bay", "chiếc", True),
    "skis": ("ván trượt tuyết", "đôi", True),
    "snowboard": ("ván trượt tuyết đơn", "chiếc", True),
    "sports ball": ("quả bóng", "quả", False),
    "kite": ("con diều", "con", False),
    "baseball bat": ("gậy bóng chày", "chiếc", True),
    "baseball glove": ("găng tay bóng chày", "chiếc", True),
    "skateboard": ("ván trượt", "chiếc", True),
    "surfboard": ("ván lướt sóng", "chiếc", True),
    "tennis racket": ("vợt tennis", "chiếc", True),
    "bottle": ("chai nước", "chai", False),
    "wine glass": ("ly rượu", "chiếc", True),
    "cup": ("chiếc cốc", "chiếc", False),
    "fork": ("nĩa", "chiếc", True),
    "knife": ("dao", "con", True),
    "spoon": ("thìa", "chiếc", True),
    "bowl": ("bát", "chiếc", True),
    "banana": ("chuối", "quả", True),
    "apple": ("táo", "quả", True),
    "sandwich": ("bánh mì kẹp", "chiếc", True),
    "orange": ("cam", "quả", True),
    "broccoli": ("súp lơ xanh", "búp", True),
    "carrot": ("cà rốt", "củ", True),
    "hot dog": ("bánh mì xúc xích", "chiếc", True),
    "pizza": ("bánh pizza", "chiếc", True),
    "donut": ("bánh donut", "chiếc", True),
    "cake": ("bánh ngọt", "chiếc", True),
    "chair": ("ghế", "chiếc", True),
    "couch": ("ghế sofa", "chiếc", True),
    "potted plant": ("chậu cây", "chậu", False),
    "bed": ("giường", "chiếc", True),
    "dining table": ("bàn", "chiếc", True),
    "toilet": ("bồn cầu", "chiếc", True),
    "tv": ("ti vi", "chiếc", True),
    "laptop": ("máy tính xách tay", "chiếc", True),
    "mouse": ("chuột máy tính", "con", True),
    "remote": ("điều khiển từ xa", "chiếc", True),
    "keyboard": ("bàn phím", "chiếc", True),
    "cell phone": ("điện thoại", "chiếc", False),
    "microwave": ("lò vi sóng", "chiếc", True),
    "oven": ("lò nướng", "chiếc", True),
    "toaster": ("máy nướng bánh mì", "chiếc", True),
    "sink": ("bồn rửa", "chiếc", True),
    "refrigerator": ("tủ lạnh", "chiếc", True),
    "book": ("quyển sách", "quyển", False),
    "clock": ("đồng hồ", "chiếc", True),
    "vase": ("bình hoa", "bình", False),
    "scissors": ("kéo", "chiếc", True),
    "teddy bear": ("gấu bông", "con", True),
    "hair drier": ("máy sấy tóc", "chiếc", True),
    "toothbrush": ("bàn chải đánh răng", "chiếc", True),
}


def get_class_name_vi(class_name_en: str) -> str:
    """Return Vietnamese label for a COCO class name."""
    info = COCO_INFO_VI.get(class_name_en.lower())
    if info:
        return info[0]
    return class_name_en


def get_class_name_vi_by_id(class_id: int) -> str:
    """Return Vietnamese label for a COCO class ID (0..79)."""
    if 0 <= class_id < len(COCO_CLASSES_EN):
        en_name = COCO_CLASSES_EN[class_id]
        return get_class_name_vi(en_name)
    return f"vật thể #{class_id}"


def format_vietnamese_noun_phrase(class_name_en: str, count: int = 1) -> str:
    """Format count + classifier + noun naturally in Vietnamese.
    
    Examples:
    - bottle, 1 -> 'một chai nước'
    - cup, 1 -> 'một chiếc cốc'
    - cell phone, 1 -> 'một điện thoại'
    - chair, 2 -> '2 chiếc ghế'
    - book, 3 -> '3 quyển sách'
    """
    en_key = class_name_en.lower()
    info = COCO_INFO_VI.get(en_key)
    if not info:
        return f"{count} {class_name_en}"
    
    label, classifier, use_classifier = info
    
    if count == 1:
        if en_key == "bottle":
            return "một chai nước"
        if en_key == "cup":
            return "một chiếc cốc"
        if en_key == "cell phone":
            return "một điện thoại"
        if en_key == "book":
            return "một quyển sách"
        if en_key == "potted plant":
            return "một chậu cây"
        if en_key == "vase":
            return "một bình hoa"
        if use_classifier:
            return f"một {classifier} {label}"
        return f"một {label}"
    else:
        # Multiple count (2, 3, 4...)
        if use_classifier:
            return f"{count} {classifier} {label}"
        # For items that already include classifier or natural phrase:
        if en_key in ("bottle", "potted plant", "vase"):
            return f"{count} {classifier} {label.split()[-1]}" if len(label.split()) > 1 else f"{count} {label}"
        if en_key == "book":
            return f"{count} quyển sách"
        if en_key == "cell phone":
            return f"{count} chiếc điện thoại"
        if en_key == "cup":
            return f"{count} chiếc cốc"
        return f"{count} {label}"


def format_vietnamese_object_summary(detected_class_names: Sequence[str]) -> str:
    """Synthesize a natural Vietnamese spoken summary from a sequence of detected class names.
    
    Examples:
    - [] -> 'Chưa xác định được đồ vật.'
    - ['bottle'] -> 'Phát hiện một chai nước.'
    - ['bottle', 'cup'] -> 'Phát hiện một chai nước và một chiếc cốc.'
    - ['bottle', 'cup', 'cell phone'] -> 'Phát hiện một chai nước, một chiếc cốc và một điện thoại.'
    - ['chair', 'chair', 'dining table'] -> 'Phát hiện 2 chiếc ghế và một chiếc bàn.'
    """
    if not detected_class_names:
        return "Chưa xác định được đồ vật."

    # Count occurrences preserving first-seen order
    counts: Counter[str] = Counter()
    first_seen: List[str] = []
    for name in detected_class_names:
        en_name = name.lower()
        if en_name not in counts:
            first_seen.append(en_name)
        counts[en_name] += 1

    phrases = [format_vietnamese_noun_phrase(en_name, counts[en_name]) for en_name in first_seen]

    if len(phrases) == 1:
        return f"Phát hiện {phrases[0]}."
    if len(phrases) == 2:
        return f"Phát hiện {phrases[0]} và {phrases[1]}."
    
    # 3 or more: 'Phát hiện A, B và C.'
    leading = ", ".join(phrases[:-1])
    return f"Phát hiện {leading} và {phrases[-1]}."
