# -*- coding: utf-8 -*-
"""
YOLOX ONNX Object Detector for VisionMate OD-1.

Runs completely offline on CPU via ONNX Runtime.
Complies with safety policies:
- Strict confidence filtering (default: 0.40).
- Rejects uncertain/unsupported objects with 'Chưa xác định được đồ vật.'.
- Never speculates physical distance or gives navigation/avoidance advice.
- Completely offline (zero network requests, zero telemetry).
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from visionmate.contracts import DetectedObject, ObjectDetectionResult
from visionmate.detection.annotator import draw_detection_annotations
from visionmate.detection.coco_classes import (
    COCO_CLASSES_EN,
    format_vietnamese_object_summary,
    get_class_name_vi,
    get_class_name_vi_by_id,
)


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "models" / "object_detection" / "yolox_nano.onnx"
)


class ObjectDetector(ABC):
    """Abstract base class for object detection engines."""

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        cancel_token: Optional[threading.Event] = None,
    ) -> ObjectDetectionResult:
        """Run object detection on a BGR image frame."""
        pass


_GRID_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _get_yolox_grid(img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute or retrieve cached anchor grid and strides."""
    if img_size not in _GRID_CACHE:
        grids = []
        expanded_strides = []
        strides = [8, 16, 32]
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        _GRID_CACHE[img_size] = (
            np.concatenate(grids, 1),
            np.concatenate(expanded_strides, 1),
        )
    return _GRID_CACHE[img_size]


def _yolox_grid_postprocess(
    outputs: np.ndarray,
    img_size: Tuple[int, int] = (416, 416),
) -> np.ndarray:
    """Decode YOLOX raw anchor grid offsets into absolute box coordinates."""
    grids, expanded_strides = _get_yolox_grid(img_size)
    decoded = outputs.copy()
    decoded[..., :2] = (decoded[..., :2] + grids) * expanded_strides
    decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * expanded_strides
    return decoded


class YOLOXObjectDetector(ObjectDetector):
    """Official YOLOX detector running on ONNX Runtime CPU.
    
    Trained on 80 COCO classes.
    Uses letterbox scaling with padding to preserve aspect ratio.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.40,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (416, 416),
        model_name: str = "yolox_nano",
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = input_size
        self.model_name = model_name
        self._session: Optional[ort.InferenceSession] = None
        self._session_lock = threading.Lock()
        self._load_error: Optional[str] = None
        self._init_session()

    def _init_session(self) -> None:
        with self._session_lock:
            if not self.model_path.is_file():
                self._load_error = f"Không tìm thấy file mô hình tại: {self.model_path}"
                self._session = None
                return

            try:
                # Use CPUExecutionProvider exclusively for reproducible CPU-first performance
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._load_error = None
            except Exception as exc:
                self._session = None
                self._load_error = f"Lỗi khởi tạo mô hình ONNX: {exc}"

    def is_ready(self) -> bool:
        """Return True if the ONNX session is loaded and ready for inference."""
        with self._session_lock:
            return self._session is not None

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Letterbox resize image to self.input_size preserving aspect ratio."""
        h, w = image.shape[:2]
        r = min(self.input_size[0] / h, self.input_size[1] / w)
        resized_h, resized_w = int(round(h * r)), int(round(w * r))
        resized_img = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )
        padded_img = np.full(
            (self.input_size[0], self.input_size[1], 3),
            114,
            dtype=np.uint8,
        )
        padded_img[:resized_h, :resized_w] = resized_img
        # Transpose BGR to CHW float32
        tensor_img = padded_img.transpose((2, 0, 1)).astype(np.float32)
        tensor_img = np.ascontiguousarray(tensor_img[None, ...])
        return tensor_img, r

    def detect(
        self,
        image: np.ndarray,
        cancel_token: Optional[threading.Event] = None,
    ) -> ObjectDetectionResult:
        """Run object detection on the provided BGR image frame."""
        t0 = time.perf_counter()

        if cancel_token is not None and cancel_token.is_set():
            return ObjectDetectionResult(
                objects=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="CANCELLED",
                model_name=self.model_name,
                spoken_text="Chưa xác định được đồ vật.",
                display_text="Chưa xác định được đồ vật.",
                confidence_status="NONE",
            )

        if image is None or image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return ObjectDetectionResult(
                objects=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="Ảnh trống hoặc không hợp lệ.",
                model_name=self.model_name,
                spoken_text="Chưa xác định được đồ vật.",
                display_text="Chưa xác định được đồ vật.",
                confidence_status="NONE",
            )

        with self._session_lock:
            if self._session is None:
                err_msg = self._load_error or "Mô hình nhận diện đồ vật chưa sẵn sàng."
                return ObjectDetectionResult(
                    objects=[],
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    error=err_msg,
                    model_name=self.model_name,
                    spoken_text="Chưa xác định được đồ vật.",
                    display_text="Chưa xác định được đồ vật.",
                    confidence_status="NONE",
                )
            session = self._session

        # 1. Preprocessing
        inp_tensor, scale_ratio = self._preprocess(image)

        if cancel_token is not None and cancel_token.is_set():
            return ObjectDetectionResult(
                objects=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="CANCELLED",
                model_name=self.model_name,
                spoken_text="Chưa xác định được đồ vật.",
                display_text="Chưa xác định được đồ vật.",
                confidence_status="NONE",
            )

        # 2. Inference
        try:
            input_name = session.get_inputs()[0].name
            raw_outputs = session.run(None, {input_name: inp_tensor})[0]
        except Exception as exc:
            return ObjectDetectionResult(
                objects=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"Lỗi suy luận YOLOX: {exc}",
                model_name=self.model_name,
                spoken_text="Chưa xác định được đồ vật.",
                display_text="Chưa xác định được đồ vật.",
                confidence_status="NONE",
            )

        if cancel_token is not None and cancel_token.is_set():
            return ObjectDetectionResult(
                objects=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="CANCELLED",
                model_name=self.model_name,
                spoken_text="Chưa xác định được đồ vật.",
                display_text="Chưa xác định được đồ vật.",
                confidence_status="NONE",
            )

        # 3. Postprocessing & Coordinate Decoding
        decoded = _yolox_grid_postprocess(raw_outputs, self.input_size)[0]
        boxes_xywh = decoded[:, :4]
        obj_conf = decoded[:, 4:5]
        class_probs = decoded[:, 5:]
        total_scores = obj_conf * class_probs

        h_orig, w_orig = image.shape[:2]

        # Fast vectorized filtering across 3549 anchors
        cids = np.argmax(total_scores, axis=1)
        scores = total_scores[np.arange(len(total_scores)), cids]
        mask = scores >= self.conf_threshold

        cand_boxes: List[List[int]] = []
        cand_scores: List[float] = []
        cand_cids: List[int] = []

        if np.any(mask):
            filtered_cids = cids[mask]
            filtered_scores = scores[mask]
            filtered_boxes = boxes_xywh[mask]

            cx = filtered_boxes[:, 0]
            cy = filtered_boxes[:, 1]
            bw = filtered_boxes[:, 2]
            bh = filtered_boxes[:, 3]

            x1 = np.round((cx - bw / 2.0) / scale_ratio).astype(int)
            y1 = np.round((cy - bh / 2.0) / scale_ratio).astype(int)
            bw_orig = np.round(bw / scale_ratio).astype(int)
            bh_orig = np.round(bh / scale_ratio).astype(int)

            # Clipping
            x1 = np.clip(x1, 0, w_orig - 1)
            y1 = np.clip(y1, 0, h_orig - 1)
            bw_orig = np.clip(bw_orig, 1, w_orig - x1)
            bh_orig = np.clip(bh_orig, 1, h_orig - y1)

            for i in range(len(filtered_scores)):
                cand_boxes.append([int(x1[i]), int(y1[i]), int(bw_orig[i]), int(bh_orig[i])])
                cand_scores.append(float(filtered_scores[i]))
                cand_cids.append(int(filtered_cids[i]))

        # 4. Multi-class NMS
        final_objects: List[DetectedObject] = []
        if cand_boxes:
            indices = cv2.dnn.NMSBoxes(
                cand_boxes,
                cand_scores,
                self.conf_threshold,
                self.nms_threshold,
            )
            for idx_entry in indices:
                idx = idx_entry[0] if isinstance(idx_entry, (list, tuple, np.ndarray)) else idx_entry
                cid = cand_cids[idx]
                en_name = COCO_CLASSES_EN[cid] if 0 <= cid < len(COCO_CLASSES_EN) else f"object_{cid}"
                vi_name = get_class_name_vi(en_name)
                conf = cand_scores[idx]
                box = tuple(cand_boxes[idx])  # (x, y, w, h)

                final_objects.append(
                    DetectedObject(
                        class_id=cid,
                        class_name_en=en_name,
                        class_name_vi=vi_name,
                        confidence=conf,
                        box_xywh=box,  # type: ignore[arg-type]
                    )
                )

        # Sort objects by position (top-to-bottom, left-to-right) for consistent reading
        final_objects.sort(key=lambda o: (o.box_xywh[1] // 50, o.box_xywh[0]))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Natural Speech & Display Summary
        if final_objects:
            class_names_sequence = [o.class_name_en for o in final_objects]
            spoken_text = format_vietnamese_object_summary(class_names_sequence)
            
            # Detailed display lines with bounding box and confidence
            detail_lines = [
                f"{i+1}. {obj.class_name_vi.capitalize()} — "
                f"Độ tin cậy: {int(round(obj.confidence * 100))}% "
                f"(Vị trí: x={obj.box_xywh[0]}, y={obj.box_xywh[1]}, w={obj.box_xywh[2]}, h={obj.box_xywh[3]})"
                for i, obj in enumerate(final_objects)
            ]
            display_text = f"{spoken_text}\n\nDanh sách chi tiết:\n" + "\n".join(detail_lines)
            conf_status = "HIGH" if min(o.confidence for o in final_objects) >= 0.50 else "MEDIUM"
        else:
            spoken_text = "Chưa xác định được đồ vật."
            display_text = "Chưa xác định được đồ vật."
            conf_status = "NONE"

        # 6. Generate annotated frame copy for preview
        annotated_frame = draw_detection_annotations(image, final_objects)

        return ObjectDetectionResult(
            objects=final_objects,
            elapsed_ms=elapsed_ms,
            error=None,
            model_name=self.model_name,
            spoken_text=spoken_text,
            display_text=display_text,
            confidence_status=conf_status,
            annotated_frame=annotated_frame,
        )


class FakeObjectDetector(ObjectDetector):
    """Deterministic mock object detector for unit tests, race conditions, and error tests."""

    def __init__(
        self,
        mock_objects: Optional[List[DetectedObject]] = None,
        latency_ms: float = 5.0,
        should_fail: bool = False,
        error_message: str = "Mock detector failure",
    ) -> None:
        self.mock_objects = mock_objects or []
        self.latency_ms = latency_ms
        self.should_fail = should_fail
        self.error_message = error_message
        self.detect_call_count = 0

    def detect(
        self,
        image: np.ndarray,
        cancel_token: Optional[threading.Event] = None,
    ) -> ObjectDetectionResult:
        self.detect_call_count += 1
        t0 = time.perf_counter()

        if self.latency_ms > 0:
            step = 0.005
            elapsed = 0.0
            while elapsed < (self.latency_ms / 1000.0):
                if cancel_token is not None and cancel_token.is_set():
                    return ObjectDetectionResult(
                        objects=[],
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        error="CANCELLED",
                        spoken_text="Chưa xác định được đồ vật.",
                        display_text="Chưa xác định được đồ vật.",
                        confidence_status="NONE",
                    )
                time.sleep(step)
                elapsed += step

        if self.should_fail:
            return ObjectDetectionResult(
                objects=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error=self.error_message,
                spoken_text="Chưa xác định được đồ vật.",
                display_text="Chưa xác định được đồ vật.",
                confidence_status="NONE",
            )

        if self.mock_objects:
            names = [o.class_name_en for o in self.mock_objects]
            spoken_text = format_vietnamese_object_summary(names)
            detail_lines = [
                f"{i+1}. {obj.class_name_vi.capitalize()} — "
                f"Độ tin cậy: {int(round(obj.confidence * 100))}%"
                for i, obj in enumerate(self.mock_objects)
            ]
            display_text = f"{spoken_text}\n\nDanh sách chi tiết:\n" + "\n".join(detail_lines)
            annotated = draw_detection_annotations(image, self.mock_objects)
            return ObjectDetectionResult(
                objects=list(self.mock_objects),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error=None,
                model_name="fake_yolox",
                spoken_text=spoken_text,
                display_text=display_text,
                confidence_status="HIGH",
                annotated_frame=annotated,
            )
        else:
            return ObjectDetectionResult(
                objects=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error=None,
                model_name="fake_yolox",
                spoken_text="Chưa xác định được đồ vật.",
                display_text="Chưa xác định được đồ vật.",
                confidence_status="NONE",
                annotated_frame=image.copy() if image is not None else None,
            )
