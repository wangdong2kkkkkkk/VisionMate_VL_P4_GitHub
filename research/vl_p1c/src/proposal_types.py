"""Shared class-agnostic proposal dataclasses and geometry helpers for VL-P1c (never accepts GT boxes)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProposalBox:
    bbox_xywh: tuple[int, int, int, int]
    source: str
    raw_confidence: float
    rank_score: float = 0.0
    coco_class_id: int | None = None
    source_family: str = "other"  # F_primary | F_context | F_thin | F_small | other
    endpoints_xyxy: tuple[int, int, int, int] | None = None
    oriented_angle_deg: float | None = None
    thin_evidence_score: float = 0.0
    meta: dict = field(default_factory=dict)


def iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return float(inter / max(1, aw * ah + bw * bh - inter))
