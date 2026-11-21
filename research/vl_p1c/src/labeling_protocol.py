"""Canonical category definitions, hard-negative groups, labeling rules, and Wikimedia license policy for VL-P1c."""
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
