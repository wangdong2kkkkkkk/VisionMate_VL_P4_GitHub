# -*- coding: utf-8 -*-
"""
Bounding Box and Label Visualization for VisionMate OD-1.
Draws high-contrast bounding boxes, label pills, and confidence metrics on RAM frames.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple
import cv2
import numpy as np

from visionmate.contracts import DetectedObject


# Palette of high-contrast distinct colors for object bounding boxes (BGR format)
PALETTE_BGR: Tuple[Tuple[int, int, int], ...] = (
    (46, 204, 113),   # Emerald green
    (52, 152, 219),   # Peter river blue
    (155, 89, 182),   # Amethyst purple
    (241, 196, 15),   # Sun flower yellow
    (230, 126, 34),   # Carrot orange
    (231, 76, 60),    # Alizarin red
    (26, 188, 156),   # Turquoise
    (243, 156, 18),   # Orange
    (211, 84, 0),     # Pumpkin
    (192, 57, 43),    # Pomegranate
)


def draw_detection_annotations(
    image: np.ndarray,
    objects: Sequence[DetectedObject],
    line_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw bounding boxes and Vietnamese label pills on a copy of the input frame.
    
    Args:
        image: Original BGR frame.
        objects: Sequence of DetectedObject instances.
        line_thickness: Box stroke thickness in pixels.
        font_scale: OpenCV font scale for text.
        
    Returns:
        Annotated BGR frame copy. Original image is never mutated.
    """
    if image is None or image.size == 0:
        return image

    annotated = image.copy()
    h_img, w_img = annotated.shape[:2]

    for obj in objects:
        color = PALETTE_BGR[obj.class_id % len(PALETTE_BGR)]
        x, y, w, h = obj.box_xywh

        # Clip box to image boundaries
        x1 = max(0, min(w_img - 1, x))
        y1 = max(0, min(h_img - 1, y))
        x2 = max(0, min(w_img - 1, x + w))
        y2 = max(0, min(h_img - 1, y + h))

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Label text: e.g. "Chai nước (85%)"
        conf_pct = int(round(obj.confidence * 100))
        label_text = f"{obj.class_name_vi.capitalize()} {conf_pct}%"

        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

        # Label pill placement (prefer above box; if not enough room, place inside)
        pill_y2 = y1 if y1 - th - baseline - 4 >= 0 else y1 + th + baseline + 6
        pill_y1 = pill_y2 - th - baseline - 4
        pill_x1 = x1
        pill_x2 = min(w_img - 1, x1 + tw + 8)

        # Draw filled pill background
        cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)

        # Draw text inside pill (dark text for high contrast on bright pill)
        text_y = pill_y2 - baseline - 2
        text_color = (0, 0, 0) if (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299) > 130 else (255, 255, 255)
        cv2.putText(
            annotated,
            label_text,
            (pill_x1 + 4, text_y),
            font,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return annotated
