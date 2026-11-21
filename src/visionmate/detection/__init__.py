# -*- coding: utf-8 -*-
"""VisionMate Object Detection (OD-1) Package."""
from visionmate.detection.annotator import draw_detection_annotations
from visionmate.detection.coco_classes import (
    COCO_CLASSES_EN,
    format_vietnamese_noun_phrase,
    format_vietnamese_object_summary,
    get_class_name_vi,
    get_class_name_vi_by_id,
)
from visionmate.detection.detector import (
    FakeObjectDetector,
    ObjectDetector,
    YOLOXObjectDetector,
)

__all__ = [
    "COCO_CLASSES_EN",
    "FakeObjectDetector",
    "ObjectDetector",
    "YOLOXObjectDetector",
    "draw_detection_annotations",
    "format_vietnamese_noun_phrase",
    "format_vietnamese_object_summary",
    "get_class_name_vi",
    "get_class_name_vi_by_id",
]
