# -*- coding: utf-8 -*-
"""
ImageQualityGate (Section 4.1, Section 6.1, FR-03).
Evaluates single-frame sharpness (calibrated global + active-tile Laplacian variance),
underexposure, overexposure/glare, and frame validity before invoking OCR.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from visionmate.config import QualityThresholds
from visionmate.contracts import QualityCheckResult, QualityIssue


MSG_CORRUPT = "Không đọc được dữ liệu ảnh hợp lệ, vui lòng thử lại."
MSG_BLURRY = "Ảnh đang mờ, hãy giữ camera ổn định và đưa nội dung vào gần hơn."
MSG_DARK = "Ảnh đang quá tối, hãy bật thêm đèn hoặc hướng nội dung ra nơi đủ sáng."
MSG_GLARE = "Ảnh bị chói sáng, hãy nghiêng nhẹ nhãn hoặc đổi góc để giảm phản xạ ánh sáng."


class ImageQualityGate:
    """Evaluates whether an input frame is suitable for OCR or requires user adjustment."""

    def __init__(self, thresholds: Optional[QualityThresholds] = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(self, frame: Optional[np.ndarray]) -> QualityCheckResult:
        t0 = time.perf_counter()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return QualityCheckResult(
                passed=False,
                issue=QualityIssue.CORRUPT_OR_EMPTY,
                blur_score=0.0,
                active_blur_score=0.0,
                mean_brightness=0.0,
                glare_ratio=0.0,
                contrast_std=0.0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                reason_vi=MSG_CORRUPT,
            )

        if frame.ndim not in (2, 3):
            return QualityCheckResult(
                passed=False,
                issue=QualityIssue.CORRUPT_OR_EMPTY,
                blur_score=0.0,
                active_blur_score=0.0,
                mean_brightness=0.0,
                glare_ratio=0.0,
                contrast_std=0.0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                reason_vi=MSG_CORRUPT,
            )

        h, w = frame.shape[:2]
        if h < 24 or w < 24:
            return QualityCheckResult(
                passed=False,
                issue=QualityIssue.CORRUPT_OR_EMPTY,
                blur_score=0.0,
                active_blur_score=0.0,
                mean_brightness=0.0,
                glare_ratio=0.0,
                glare_ratio_center=0.0,
                contrast_std=0.0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                reason_vi=MSG_CORRUPT,
            ) if False else QualityCheckResult(
                passed=False,
                issue=QualityIssue.CORRUPT_OR_EMPTY,
                blur_score=0.0,
                active_blur_score=0.0,
                mean_brightness=0.0,
                glare_ratio=0.0,
                contrast_std=0.0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                reason_vi=MSG_CORRUPT,
            )

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            max_ch = np.max(frame, axis=2)
        else:
            gray = frame
            max_ch = gray

        # Normalize scale to width=640 for resolution-consistent Laplacian calibration
        norm_w = 640
        if w != norm_w and w > 0:
            norm_h = max(32, int(round(h * (norm_w / float(w)))))
            interp = cv2.INTER_AREA if w > norm_w else cv2.INTER_LINEAR
            gray_norm = cv2.resize(gray, (norm_w, norm_h), interpolation=interp)
            max_ch_norm = cv2.resize(max_ch, (norm_w, norm_h), interpolation=interp)
        else:
            gray_norm = gray
            max_ch_norm = max_ch

        mean_brightness = float(np.mean(gray_norm))
        contrast_std = float(np.std(gray_norm))
        glare_ratio = float(np.mean(gray_norm >= 250))
        dark_ink_ratio = float(np.mean(gray_norm <= 110))

        p10, p90, p95, p99 = np.percentile(gray_norm, [10, 90, 95, 99])
        p10_ch, p95_ch, p99_ch = np.percentile(max_ch_norm, [10, 95, 99])

        lap = cv2.Laplacian(gray_norm, cv2.CV_64F)
        global_blur_score = float(lap.var())
        active_blur_score, tile_glares, tile_edges = self._compute_tile_metrics(gray_norm, lap)

        # Detect specular glare / reflective band (e.g. phone screen or glossy paper reflection)
        max_tile_glare = max(tile_glares) if tile_glares else 0.0
        reflective_band = bool(glare_ratio >= 0.008 or max_tile_glare >= 0.06)

        # Detect if text or high-contrast content touches the extreme margins
        h_n, w_n = gray_norm.shape[:2]
        margin_x = max(6, int(w_n * 0.025))
        margin_y = max(6, int(h_n * 0.025))
        edge_contrast = (
            float(np.std(gray_norm[:margin_y, :])) > 28.0
            or float(np.std(gray_norm[-margin_y:, :])) > 28.0
            or float(np.std(gray_norm[:, :margin_x])) > 28.0
            or float(np.std(gray_norm[:, -margin_x:])) > 28.0
        )

        guidance_hints: List[str] = []
        if reflective_band:
            guidance_hints.append("Nghiêng nhẹ màn hình hoặc đổi góc để giảm ánh sáng phản chiếu.")
        if edge_contrast:
            guidance_hints.append("Đưa nội dung chữ vào giữa khung hình camera.")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 1. Underexposure check
        # Distinguish genuine underexposure (flat dark scene, covered lens) from dark-background
        # high-contrast documents/screens (dark mode, code syntax, white/colored text on black).
        is_high_contrast_dark_mode = bool(
            ((float(p95) >= 75.0 or float(p99) >= 115.0) or (float(p95_ch) >= 80.0 or float(p99_ch) >= 120.0))
            and ((float(p95 - p10) >= 45.0) or (float(p99 - p10) >= 75.0) or (float(p95_ch - p10_ch) >= 50.0))
            and contrast_std >= 18.0
            and active_blur_score >= self.thresholds.min_active_laplacian_var
            and global_blur_score >= 10.0
        )

        if mean_brightness < self.thresholds.min_brightness and not is_high_contrast_dark_mode:
            guidance_hints.append("Bật thêm đèn hoặc hướng nội dung ra nơi đủ sáng.")
            return QualityCheckResult(
                passed=False,
                issue=QualityIssue.TOO_DARK,
                blur_score=global_blur_score,
                active_blur_score=active_blur_score,
                mean_brightness=mean_brightness,
                glare_ratio=glare_ratio,
                contrast_std=contrast_std,
                elapsed_ms=elapsed_ms,
                reason_vi=MSG_DARK,
                reflective_band_detected=reflective_band,
                edge_clipped_text=edge_contrast,
                guidance_hints_vi=guidance_hints,
            )

        # 2. Overexposure / specular glare check
        if (
            (mean_brightness > self.thresholds.max_brightness and dark_ink_ratio < 0.005)
            or (glare_ratio > self.thresholds.max_glare_ratio and dark_ink_ratio < 0.005)
        ):
            return QualityCheckResult(
                passed=False,
                issue=QualityIssue.TOO_BRIGHT_OR_GLARE,
                blur_score=global_blur_score,
                active_blur_score=active_blur_score,
                mean_brightness=mean_brightness,
                glare_ratio=glare_ratio,
                contrast_std=contrast_std,
                elapsed_ms=elapsed_ms,
                reason_vi=MSG_GLARE,
                reflective_band_detected=reflective_band,
                edge_clipped_text=edge_contrast,
                guidance_hints_vi=guidance_hints,
            )

        # 3. Blur check on frames that have non-trivial visual content
        if contrast_std >= self.thresholds.min_contrast_std:
            if (
                active_blur_score < self.thresholds.min_active_laplacian_var
                and global_blur_score < self.thresholds.min_global_laplacian_var * 3.0
            ):
                guidance_hints.insert(
                    0, "Đưa thiết bị ra xa camera hơn (khoảng 30-40cm) để camera lấy nét rõ, và giữ yên tay."
                )
                return QualityCheckResult(
                    passed=False,
                    issue=QualityIssue.TOO_BLURRY,
                    blur_score=global_blur_score,
                    active_blur_score=active_blur_score,
                    mean_brightness=mean_brightness,
                    glare_ratio=glare_ratio,
                    contrast_std=contrast_std,
                    elapsed_ms=elapsed_ms,
                    reason_vi=MSG_BLURRY,
                    reflective_band_detected=reflective_band,
                    edge_clipped_text=edge_contrast,
                    guidance_hints_vi=guidance_hints,
                )

        # If passed but blur score is marginal or phone screen close-up
        if active_blur_score < 110.0 and contrast_std >= 25.0:
            guidance_hints.append(
                "Đưa thiết bị ra xa camera hơn (khoảng 30-40cm) để camera lấy nét tối ưu."
            )

        return QualityCheckResult(
            passed=True,
            issue=QualityIssue.OK,
            blur_score=global_blur_score,
            active_blur_score=active_blur_score,
            mean_brightness=mean_brightness,
            glare_ratio=glare_ratio,
            contrast_std=contrast_std,
            elapsed_ms=elapsed_ms,
            reason_vi="",
            reflective_band_detected=reflective_band,
            edge_clipped_text=edge_contrast,
            guidance_hints_vi=guidance_hints,
        )

    @staticmethod
    def _compute_tile_metrics(
        gray: np.ndarray, lap: np.ndarray
    ) -> Tuple[float, List[float], List[float]]:
        """Compute Laplacian variance across the top 25% highest-contrast 64x64 tiles,
        along with per-tile glare and edge metrics.
        """
        h, w = gray.shape[:2]
        tile_h, tile_w = 64, 64
        if h < tile_h or w < tile_w:
            g = float(np.mean(gray >= 245))
            return float(lap.var()), [g], [0.0]

        tile_vars: List[float] = []
        tile_glares: List[float] = []
        tile_edges: List[float] = []
        for y in range(0, h - tile_h + 1, tile_h):
            for x in range(0, w - tile_w + 1, tile_w):
                patch_gray = gray[y : y + tile_h, x : x + tile_w]
                tile_glares.append(float(np.mean(patch_gray >= 245)))
                if float(np.std(patch_gray)) < 4.0:
                    continue
                patch_lap = lap[y : y + tile_h, x : x + tile_w]
                tile_vars.append(float(patch_lap.var()))

        if not tile_vars:
            return float(lap.var()), tile_glares, tile_edges

        tile_vars.sort(reverse=True)
        top_k = max(1, len(tile_vars) // 4)
        return float(np.mean(tile_vars[:top_k])), tile_glares, tile_edges
