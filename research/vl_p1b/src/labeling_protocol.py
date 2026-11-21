"""Canonical category definitions, hard-negative groups, labeling rules, and Wikimedia license policy for VL-P1b."""
from __future__ import annotations

HARD_NEGATIVE_GROUPS = [
    "glasses_case_confuser",     # Bao kính da, hộp kính cứng -> dễ nhầm Ví tiền (object_03)
    "notebook_pouch_confuser",   # Sổ tay bìa da, túi gập, hộp nhỏ -> dễ nhầm Ví tiền (object_03)
    "walking_stick_confuser",    # Gậy chống gỗ, gậy đi bộ/leo núi, gậy bảo tàng -> dễ nhầm Gậy trắng (object_01)
    "long_metal_rod_confuser",   # Thanh kim loại dài, gậy chụp ảnh/dụng cụ dài -> dễ nhầm Gậy trắng (object_01)
    "lock_key_metal_confuser",   # Ổ khóa, kéo, thìa, đồ kim loại nhỏ -> dễ nhầm Chùm chìa khóa (object_02)
    "footwear_sock_confuser",    # Vớ/tất, guốc gỗ, bao chân / đồ đi chân khác -> dễ nhầm Giày/Dép (object_04/05)
    "umbrella_handle_confuser",  # Ô dù gấp/dài, tay cầm thiết bị cầm tay -> dễ nhầm Gậy trắng/khác
]

TARGET_CONFUSER_MAP = {
    "object_01": ["walking_stick_confuser", "long_metal_rod_confuser", "umbrella_handle_confuser"],
    "object_02": ["lock_key_metal_confuser"],
    "object_03": ["glasses_case_confuser", "notebook_pouch_confuser"],
    "object_04": ["footwear_sock_confuser"],
    "object_05": ["footwear_sock_confuser"],
}

ALLOWED_OPEN_LICENSES = {
    "cc0",
    "public domain",
    "cc by",
    "cc by 2.0",
    "cc by 2.5",
    "cc by 3.0",
    "cc by 4.0",
    "cc by-sa",
    "cc by-sa 2.0",
    "cc by-sa 2.5",
    "cc by-sa 3.0",
    "cc by-sa 4.0",
    "cc-pd-mark",
}


def is_allowed_open_license(lic_str: str | None) -> bool:
    """Return True if the license metadata is explicitly an open/public-domain Wikimedia license."""
    if not lic_str:
        return False
    norm = " ".join(lic_str.strip().lower().replace("-", " ").split())
    norm_hyphen = " ".join(lic_str.strip().lower().split())
    for allowed in ALLOWED_OPEN_LICENSES:
        if allowed in norm_hyphen or allowed.replace("-", " ") in norm:
            return True
    if "public domain" in norm or "cc0" in norm or "cc by" in norm:
        return True
    return False


# Explicit labeling & boundary decision rules for VL-P1b
LABEL_BOUNDARY_RULES = {
    "object_01": {
        "include": [
            "Gậy trắng hỗ trợ người khiếm thị (white cane / long cane / guide cane / red-tipped cane), "
            "cầm tay hoặc gập/dựng đứng, kể cả ảnh đời thực cùng người sử dụng."
        ],
        "exclude_to_unknown": [
            "Gậy chống gỗ cổ/thời trang, gậy đi bộ trekking/walking stick du lịch, gậy kính viễn vọng, nạng y tế, ô dù, sào/thanh kim loại."
        ],
        "label_ambiguous": [
            "Gậy đi bộ màu trắng hoàn toàn không rõ tay cầm đặc trưng hay đầu típ đỏ/trắng có phải gậy hỗ trợ người khiếm thị hay không."
        ],
    },
    "object_02": {
        "include": [
            "Chùm chìa khóa: tối thiểu >= 2 chìa trên khoen, hoặc 1 chìa gắn móc khóa/fob rõ ràng thành bộ chìa khóa mang theo."
        ],
        "exclude_to_unknown": [
            "Ổ khóa không có chùm chìa, kéo, thìa, ốc vít, dụng cụ kim loại đơn lẻ."
        ],
        "label_ambiguous": [
            "Một chiếc chìa khóa kim loại cổ hoặc chìa nhà đơn lẻ trần trụi không có khoen/móc."
        ],
    },
    "object_03": {
        "include": [
            "Ví tiền cầm tay (bi-fold, tri-fold, long zip wallet, ví đựng tiền xu/coin purse có khóa/gập)."
        ],
        "exclude_to_unknown": [
            "Bao kính/hộp kính da, sổ tay bìa da, túi đeo chéo/ba lô lớn, hộp thuốc lá/hộp đựng cứng không phải ví tiền."
        ],
        "label_ambiguous": [
            "Túi zip/pouch đa năng nhỏ không rõ công năng đựng tiền hay đựng mỹ phẩm/cáp sạc."
        ],
    },
    "object_04": {
        "include": [
            "Đôi dép (flip-flops, dép xỏ ngón, dép quai ngang, slide sandals, house slippers hở mũi hoặc hở gót rõ kết cấu dép)."
        ],
        "exclude_to_unknown": [
            "Tất/vớ đơn thuần không có đế cứng, bao chân vải."
        ],
        "label_ambiguous": [
            "Giày sục/clog bít mũi hở gót (lai giữa giày và dép) hoặc sandal rọ kín mũi kiểu lai."
        ],
    },
    "object_05": {
        "include": [
            "Đôi giày (sneakers, running shoes, dress/leather shoes, boots bít thân chân & đế giày)."
        ],
        "exclude_to_unknown": [
            "Tất/vớ, bao giày y tế."
        ],
        "label_ambiguous": [
            "Sandal thể thao có quai hậu dày bít nhiều phần chân giống giày hở."
        ],
    },
}
