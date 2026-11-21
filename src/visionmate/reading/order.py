# -*- coding: utf-8 -*-
"""
ReadingOrder (Section 4.1, FR-04, A05, Phase 1.1 Multi-Column ROI Support).
Orders recognized OCR lines respecting multi-column ROI order_index (Left Column top-to-bottom,
then Right Column top-to-bottom), and joins text WITHOUT modifying or rewriting characters,
numbers, dates, dosages, or units.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List

from visionmate.contracts import OCRLine


class ReadingOrder:
    """Deterministic spatial and ROI-aware line sorter that preserves exact OCR characters."""

    @classmethod
    def sort_lines(cls, lines: List[OCRLine]) -> List[OCRLine]:
        """Sort lines respecting ROI order_index if present, then top-to-bottom, left-to-right."""
        if not lines:
            return []

        normalized: List[OCRLine] = [
            OCRLine(
                text=unicodedata.normalize("NFC", ln.text.strip()),
                bbox=ln.bbox,
                raw_score=ln.raw_score,
                order_index=ln.order_index,
                column_role=ln.column_role,
            )
            for ln in lines
            if ln.text and ln.text.strip()
        ]
        if len(normalized) <= 1:
            return normalized

        has_roi_order = any(ln.order_index is not None for ln in normalized)
        if has_roi_order:
            grouped_by_roi: Dict[int, List[OCRLine]] = {}
            unindexed: List[OCRLine] = []
            for ln in normalized:
                if ln.order_index is not None:
                    grouped_by_roi.setdefault(int(ln.order_index), []).append(ln)
                else:
                    unindexed.append(ln)

            ordered_all: List[OCRLine] = []
            for roi_idx in sorted(grouped_by_roi.keys()):
                ordered_all.extend(cls._sort_within_region(grouped_by_roi[roi_idx]))
            if unindexed:
                ordered_all.extend(cls._sort_within_region(unindexed))
            return ordered_all

        return cls._sort_within_region(normalized)

    @staticmethod
    def _sort_within_region(lines: List[OCRLine]) -> List[OCRLine]:
        if len(lines) <= 1:
            return list(lines)

        with_bbox = [ln for ln in lines if ln.bbox is not None]
        without_bbox = [ln for ln in lines if ln.bbox is None]

        if not with_bbox:
            return without_bbox

        with_bbox.sort(key=lambda ln: (ln.bbox[1] + ln.bbox[3] / 2.0, ln.bbox[0]))  # type: ignore[index]

        rows: List[List[OCRLine]] = []
        for ln in with_bbox:
            assert ln.bbox is not None
            x, y, w, h = ln.bbox
            y_center = y + h / 2.0
            placed = False
            for row in rows:
                ref_bbox = row[0].bbox
                assert ref_bbox is not None
                ref_y_center = ref_bbox[1] + ref_bbox[3] / 2.0
                row_height = max(ref_bbox[3], h, 12)
                if abs(y_center - ref_y_center) <= 0.55 * row_height:
                    row.append(ln)
                    placed = True
                    break
            if not placed:
                rows.append([ln])

        ordered: List[OCRLine] = []
        for row in rows:
            row.sort(key=lambda ln: ln.bbox[0])  # type: ignore[index]
            ordered.extend(row)

        ordered.extend(without_bbox)
        return ordered

    @classmethod
    def join_for_display(cls, lines: List[OCRLine]) -> str:
        """Join ordered lines with newline for UI display, preserving exact characters."""
        ordered = cls.sort_lines(lines)
        return "\n".join(ln.text for ln in ordered)

    @classmethod
    def join_for_speech(cls, lines: List[OCRLine]) -> str:
        """Join ordered lines for TTS readout without altering any digits or characters."""
        ordered = cls.sort_lines(lines)
        return ". ".join(ln.text for ln in ordered)
