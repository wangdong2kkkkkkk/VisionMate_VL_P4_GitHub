# -*- coding: utf-8 -*-
"""
SafetyPolicy & MessagePolicy (Section 4.1, Section 6.1, FR-03, FR-04, A04, A05, Phase 1.1 & Phase 1.2).
Enforces fail-closed behavior and explicitly separates:
1. Image Quality Status (`image_quality_status`: `"OK"` | `"LOW_RESOLUTION"` | `QualityIssue` value)
2. OCR Confidence Status (`ocr_confidence_status`: `"HIGH"` | `"MEDIUM"` | `"LOW"` | `"NONE"`)
3. Speech Readout Policy (`speech_policy`: `"ALLOW_FULL_SPEECH"` | `"SPEAK_WARNING_ONLY"` | `"SPEAK_NO_TEXT"` | `"SPEAK_ERROR"`)

Safety principles:
- Never invents scene descriptions when no text is detected.
- Never alters numbers, dates, dosages, or units.
- Never provides medical advice or mobility/navigation guidance.
- Never labels partial or low-resolution OCR fragments as "đã xác minh" (verified).
- Multi-signal uncertainty refusal: when source image text is genuinely sub-Nyquist / too small or OCR
  uncertainty/rejection ratio is high, classifies as UNCERTAIN and announces:
  'Ảnh có chữ quá nhỏ hoặc không đủ rõ. Hãy đưa camera lại gần hoặc dùng ảnh độ phân giải cao hơn'
  instead of reading garbled text with false confidence.
"""
from __future__ import annotations

import re
from typing import List, Optional

from visionmate.contracts import (
    OCRLine,
    OCRResult,
    PolicyDecision,
    QualityCheckResult,
    SafetyStatus,
)
from visionmate.reading.order import ReadingOrder


MSG_NO_TEXT = "Mình chưa nhìn thấy chữ rõ để đọc."
MSG_BLURRY_DEFAULT = "Ảnh đang mờ, hãy giữ camera ổn định và đưa nội dung vào gần hơn."
MSG_LOW_RES_OR_UNCERTAIN = (
    "Ảnh có chữ quá nhỏ hoặc không đủ rõ. "
    "Hãy đưa camera lại gần hoặc dùng ảnh độ phân giải cao hơn"
)
MSG_UNCERTAIN = MSG_LOW_RES_OR_UNCERTAIN
MSG_ENGINE_ERROR = "Hệ thống nhận dạng chữ gặp lỗi, vui lòng thử lại."
MSG_SENSITIVE_CAUTION = (
    "Lưu ý: Nội dung có chứa con số, hạn dùng hoặc thông tin nhạy cảm, "
    "vui lòng kiểm tra lại với người hỗ trợ hoặc nguồn tin cậy."
)

# Sensitive keywords (medicine, dosage, chemicals, safety warnings, dates)
_SENSITIVE_KEYWORDS = (
    "thuốc",
    "liều dùng",
    "liều lượng",
    "viên/ngày",
    "viên",
    "chống chỉ định",
    "tác dụng phụ",
    "kê đơn",
    "hạn sử dụng",
    "hạn dùng",
    "hsd",
    "nsx",
    "ngày sản xuất",
    "hóa chất",
    "độc hại",
    "nguy hiểm",
    "cảnh báo",
    "cồn",
    "axit",
    "tẩy rửa",
    "mg",
    "ml",
    "mcg",
)

# Date or numeric dosage patterns (for caution flagging only — never rewrites digits!)
_DATE_OR_DOSAGE_RE = re.compile(
    r"(\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b|\b\d+\s*(?:mg|ml|mcg|g|kg|%|độ|viên)\b)",
    re.IGNORECASE,
)


class SafetyPolicy:
    """Evaluates QualityCheckResult and OCRResult using multiple signals to produce a fail-closed PolicyDecision."""

    def __init__(self, min_ocr_raw_score: float = 42.0) -> None:
        self.min_ocr_raw_score = float(min_ocr_raw_score)

    def evaluate(
        self,
        quality: Optional[QualityCheckResult],
        ocr: Optional[OCRResult],
    ) -> PolicyDecision:
        lang = getattr(ocr, "language", "vi") if ocr is not None else "vi"

        # 1. Quality gate rejection -> BAD_IMAGE
        if quality is not None and not quality.passed:
            msg = quality.reason_vi or MSG_BLURRY_DEFAULT
            warns = [f"Gate chất lượng ảnh từ chối: {quality.issue.value}"]
            if quality.guidance_hints_vi:
                warns.extend(quality.guidance_hints_vi)
            return PolicyDecision(
                status=SafetyStatus.BAD_IMAGE,
                ordered_lines=[],
                spoken_text=msg,
                display_text=msg,
                warnings=warns,
                contains_sensitive_content=False,
                image_quality_status=quality.issue.value,
                ocr_confidence_status="NONE",
                speech_policy="SPEAK_WARNING_ONLY",
                language=lang,
            )

        # 2. OCR engine failure -> ENGINE_ERROR
        if ocr is None or ocr.error is not None:
            err_detail = ocr.error if ocr and ocr.error else "Không có kết quả từ OCR backend."
            return PolicyDecision(
                status=SafetyStatus.ENGINE_ERROR,
                ordered_lines=[],
                spoken_text=MSG_ENGINE_ERROR,
                display_text=f"{MSG_ENGINE_ERROR}\n({err_detail})",
                warnings=[err_detail],
                contains_sensitive_content=False,
                image_quality_status="OK",
                ocr_confidence_status="NONE",
                speech_policy="SPEAK_ERROR",
                language=lang,
            )

        # Guidance hints collected from quality check and glyph analysis
        guidance: List[str] = list(quality.guidance_hints_vi) if quality is not None else []
        if ocr is not None and ocr.low_resolution_detected and 0.0 < ocr.median_char_height_px < 9.5:
            guidance.append("Tăng cỡ chữ trên thiết bị hoặc đưa tài liệu lại gần hơn nếu hình vẫn nét.")

        # 3. Sort lines without altering characters
        ordered_lines = ReadingOrder.sort_lines(ocr.lines)

        # Check if candidate lines existed on the page, but all were rejected due to low resolution / gibberish
        if not ordered_lines:
            if ocr.low_resolution_detected:
                warns = [
                    f"Phát hiện chữ quá nhỏ hoặc không đủ rõ (chiều cao chữ trung vị ~{ocr.median_char_height_px:.1f}px)."
                ]
                if guidance:
                    warns.extend(guidance)
                return PolicyDecision(
                    status=SafetyStatus.UNCERTAIN,
                    ordered_lines=[],
                    spoken_text=MSG_LOW_RES_OR_UNCERTAIN,
                    display_text=MSG_LOW_RES_OR_UNCERTAIN,
                    warnings=warns,
                    contains_sensitive_content=False,
                    image_quality_status="LOW_RESOLUTION",
                    ocr_confidence_status="LOW",
                    speech_policy="SPEAK_WARNING_ONLY",
                    language=lang,
                )
            return PolicyDecision(
                status=SafetyStatus.NO_TEXT,
                ordered_lines=[],
                spoken_text=MSG_NO_TEXT,
                display_text=MSG_NO_TEXT,
                warnings=[],
                contains_sensitive_content=False,
                image_quality_status="OK",
                ocr_confidence_status="NONE",
                speech_policy="SPEAK_NO_TEXT",
                language=lang,
            )

        # 4. Separate evaluation of Image Resolution, OCR Confidence, and Speech Readout Policy
        mean_score = self._compute_mean_score(ordered_lines)
        joined_display = "\n".join(ln.text for ln in ordered_lines)
        joined_speech = ". ".join(ln.text for ln in ordered_lines)
        alnum_count = sum(1 for ch in joined_display if ch.isalnum())

        rejected_ratio = (
            float(ocr.rejected_line_count) / float(max(1, ocr.total_candidate_lines))
            if ocr.total_candidate_lines > 0
            else 0.0
        )

        img_qual_status = "LOW_RESOLUTION" if ocr.low_resolution_detected else "OK"
        if mean_score is None or alnum_count == 0 or mean_score < self.min_ocr_raw_score:
            ocr_conf_status = "LOW"
        elif mean_score < 65.0 or rejected_ratio >= 0.30:
            ocr_conf_status = "MEDIUM"
        else:
            ocr_conf_status = "HIGH"

        # If source document has tiny glyphs (e.g. 379x527 full-page form) and high line rejection ratio
        if ocr.low_resolution_detected and rejected_ratio >= 0.30:
            display_with_headings = (
                f"{MSG_LOW_RES_OR_UNCERTAIN}\n\n"
                f"[Các dòng chữ lớn có điểm OCR cao tách được từ ảnh (chưa đủ điều kiện đọc toàn trang) — "
                f"Đã từ chối {ocr.rejected_line_count}/{ocr.total_candidate_lines} "
                f"dòng chữ nhỏ/không chắc chắn (cỡ chữ trung vị ~{ocr.median_char_height_px:.1f}px)]:\n"
                f"{joined_display}"
            )
            warns = [
                MSG_LOW_RES_OR_UNCERTAIN,
                f"Cỡ chữ trung vị quá nhỏ ({ocr.median_char_height_px:.1f}px); từ chối đọc tự động toàn trang.",
            ]
            if guidance:
                warns.extend(guidance)
            return PolicyDecision(
                status=SafetyStatus.UNCERTAIN,
                ordered_lines=ordered_lines,
                spoken_text=MSG_LOW_RES_OR_UNCERTAIN,
                display_text=display_with_headings,
                warnings=warns,
                contains_sensitive_content=False,
                image_quality_status=img_qual_status,
                ocr_confidence_status="LOW",
                speech_policy="SPEAK_WARNING_ONLY",
                language=lang,
            )

        if alnum_count == 0 or (mean_score is not None and mean_score < self.min_ocr_raw_score):
            warns = [MSG_LOW_RES_OR_UNCERTAIN]
            if guidance:
                warns.extend(guidance)
            return PolicyDecision(
                status=SafetyStatus.UNCERTAIN,
                ordered_lines=ordered_lines,
                spoken_text=MSG_LOW_RES_OR_UNCERTAIN,
                display_text=f"{MSG_LOW_RES_OR_UNCERTAIN}\n[Văn bản thô độ tin cậy thấp]: {joined_display}",
                warnings=warns,
                contains_sensitive_content=False,
                image_quality_status=img_qual_status,
                ocr_confidence_status="LOW",
                speech_policy="SPEAK_WARNING_ONLY",
                language=lang,
            )

        # 5. Readable text — check for sensitive numbers/dates/medicine/chemicals
        warnings: List[str] = []
        is_sensitive = self._detect_sensitive_content(joined_display)
        spoken_out = joined_speech

        if mean_score is not None and mean_score < 65.0:
            warnings.append(
                "Độ rõ của chữ ở mức trung bình; một số ký tự hoặc dấu thanh có thể chưa chính xác."
            )
            spoken_out = f"Kết quả đọc có thể chưa hoàn toàn chắc chắn: {spoken_out}"

        if is_sensitive:
            warnings.append(MSG_SENSITIVE_CAUTION)
            spoken_out = f"{spoken_out}. {MSG_SENSITIVE_CAUTION}"

        if quality is not None and quality.edge_clipped_text:
            warnings.append("Nội dung văn bản chạm sát viền ảnh; một số ký tự ở mép có thể bị khuyết.")
            spoken_out = f"{spoken_out}. Lưu ý: Một số ký tự ở viền khung hình có thể bị khuyết."

        if guidance:
            warnings.extend(guidance)

        return PolicyDecision(
            status=SafetyStatus.READABLE,
            ordered_lines=ordered_lines,
            spoken_text=spoken_out,
            display_text=joined_display,
            warnings=warnings,
            contains_sensitive_content=is_sensitive,
            image_quality_status=img_qual_status,
            ocr_confidence_status=ocr_conf_status,
            speech_policy="ALLOW_FULL_SPEECH",
            language=lang,
        )

    @staticmethod
    def _compute_mean_score(lines: List[OCRLine]) -> Optional[float]:
        scores = [ln.raw_score for ln in lines if ln.raw_score is not None]
        if not scores:
            return None
        return float(sum(scores) / len(scores))

    @staticmethod
    def _detect_sensitive_content(text: str) -> bool:
        lower = text.lower()
        for kw in _SENSITIVE_KEYWORDS:
            if kw in lower:
                return True
        if _DATE_OR_DOSAGE_RE.search(text):
            return True
        if re.search(r"\d{3,}", text):
            return True
        return False
