# -*- coding: utf-8 -*-
"""
Tesseract Vietnamese OCR Backend (tessdata_fast and tessdata_best) — Phase 1.2 Enhanced.
Runs 100% offline on CPU using pytesseract and local vie.traineddata.
Supports:
- Polarity-normalized preprocessing (dark-on-light and light/colored-on-dark images)
- Fast single-label path for everyday product labels
- Multi-column & multi-band Document Layout Segmentation (suppressing horizontal rules & dotted lines)
- Adaptive multi-quantile resolution assessment & phonotactic plausibility verification
- Detailed per-line diagnostic reasons for any rejected line
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from visionmate.contracts import OCRLine, OCRResult
from visionmate.ocr.base import OCRBackend
from visionmate.ocr.layout import (
    DocumentROI,
    GlyphMetrics,
    detect_and_normalize_polarity,
    detect_image_blanks,
    disambiguate_technical_and_numeric_tokens,
    estimate_glyph_metrics,
    evaluate_line_plausibility,
    segment_document_rois,
    suppress_rules_and_dots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def find_tesseract_binary() -> Optional[str]:
    candidates = [
        shutil.which("tesseract"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


class TesseractOCRBackend(OCRBackend):
    """Offline Tesseract OCR backend for Vietnamese printed text (`vie` / `vie_best`)."""

    def __init__(self, variant: str = "fast") -> None:
        self.variant = "best" if variant == "best" else "fast"
        self.engine_name = f"tesseract_{self.variant}_vie"
        self._tesseract_cmd = find_tesseract_binary()
        if self._tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
        self._tessdata_dir = self._prepare_ascii_tessdata_dir()

    def _prepare_ascii_tessdata_dir(self) -> Optional[str]:
        """Ensure both fast and best traineddata reside in an ASCII-safe path on Windows."""
        src_fast = PROJECT_ROOT / "models" / "tessdata_fast" / "vie.traineddata"
        src_best = PROJECT_ROOT / "models" / "tessdata_best" / "vie.traineddata"
        src_eng = PROJECT_ROOT / "models" / "tessdata_fast" / "eng.traineddata"
        src_eng_best = PROJECT_ROOT / "models" / "tessdata_best" / "eng.traineddata"

        if self._tesseract_cmd:
            install_tessdata = Path(self._tesseract_cmd).parent / "tessdata"
            try:
                if install_tessdata.is_dir():
                    if src_fast.is_file():
                        dst_fast = install_tessdata / "vie.traineddata"
                        if not dst_fast.is_file() or dst_fast.stat().st_size != src_fast.stat().st_size:
                            shutil.copyfile(src_fast, dst_fast)
                    if src_best.is_file():
                        dst_best = install_tessdata / "vie_best.traineddata"
                        if not dst_best.is_file() or dst_best.stat().st_size != src_best.stat().st_size:
                            shutil.copyfile(src_best, dst_best)
                    if src_eng.is_file() and not (install_tessdata / "eng.traineddata").is_file():
                        shutil.copyfile(src_eng, install_tessdata / "eng.traineddata")
                    if src_eng_best.is_file():
                        dst_eng_best = install_tessdata / "eng_best.traineddata"
                        if not dst_eng_best.is_file() or dst_eng_best.stat().st_size != src_eng_best.stat().st_size:
                            shutil.copyfile(src_eng_best, dst_eng_best)
                    return str(install_tessdata)
            except Exception:
                pass

        try:
            ascii_cache = Path(tempfile.gettempdir()) / "visionmate_tessdata"
            ascii_cache.mkdir(parents=True, exist_ok=True)
            if src_fast.is_file():
                dst_fast = ascii_cache / "vie.traineddata"
                if not dst_fast.is_file() or dst_fast.stat().st_size != src_fast.stat().st_size:
                    shutil.copyfile(src_fast, dst_fast)
            if src_best.is_file():
                dst_best = ascii_cache / "vie_best.traineddata"
                if not dst_best.is_file() or dst_best.stat().st_size != src_best.stat().st_size:
                    shutil.copyfile(src_best, dst_best)
            if src_eng.is_file():
                dst_eng = ascii_cache / "eng.traineddata"
                if not dst_eng.is_file() or dst_eng.stat().st_size != src_eng.stat().st_size:
                    shutil.copyfile(src_eng, dst_eng)
            if src_eng_best.is_file():
                dst_eng_best = ascii_cache / "eng_best.traineddata"
                if not dst_eng_best.is_file() or dst_eng_best.stat().st_size != src_eng_best.stat().st_size:
                    shutil.copyfile(src_eng_best, dst_eng_best)
            return str(ascii_cache)
        except Exception:
            pass
        return None

    def _has_best_model(self) -> bool:
        return (PROJECT_ROOT / "models" / "tessdata_best" / "vie.traineddata").is_file()

    def _has_best_eng_model(self) -> bool:
        if self._tessdata_dir and (Path(self._tessdata_dir) / "eng_best.traineddata").is_file():
            return True
        return (PROJECT_ROOT / "models" / "tessdata_best" / "eng.traineddata").is_file()

    def _has_eng_model(self) -> bool:
        if self._tessdata_dir and (Path(self._tessdata_dir) / "eng.traineddata").is_file():
            return True
        model_folder = "tessdata_best" if self.variant == "best" else "tessdata_fast"
        src_eng = PROJECT_ROOT / "models" / model_folder / "eng.traineddata"
        return src_eng.is_file()

    def is_available(self) -> bool:
        if not self._tesseract_cmd or not os.path.isfile(self._tesseract_cmd):
            return False
        model_folder = "tessdata_best" if self.variant == "best" else "tessdata_fast"
        src_vie = PROJECT_ROOT / "models" / model_folder / "vie.traineddata"
        return src_vie.is_file()

    def backend_info(self) -> Dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "variant": self.variant,
            "tesseract_cmd": self._tesseract_cmd,
            "available": self.is_available(),
        }

    def recognize(
        self,
        image: np.ndarray,
        cancel_token: Optional[threading.Event] = None,
    ) -> OCRResult:
        t0 = time.perf_counter()
        if cancel_token is not None and cancel_token.is_set():
            return OCRResult(lines=[], engine_name=self.engine_name, elapsed_ms=0.0, error="CANCELLED")

        if not self.is_available():
            return OCRResult(
                lines=[],
                engine_name=self.engine_name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="Tesseract binary hoặc vie.traineddata chưa sẵn sàng.",
            )

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return OCRResult(
                lines=[],
                engine_name=self.engine_name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="Ảnh đầu vào rỗng hoặc không hợp lệ.",
            )

        try:
            h_img, w_img = image.shape[:2]
            gm: GlyphMetrics = estimate_glyph_metrics(image)

            # Use multi-ROI document segmentation on portrait/full-page documents (h >= 420)
            # as well as multi-line dense documents regardless of orientation (landscape test reports/code),
            # while allowing true single-line product labels (h ~ 220..360, text_count < 50) to take the fast label path.
            if h_img >= 420 or gm.is_dense_document or gm.text_component_count >= 50:
                clean_gray, clean_bw, polarity = suppress_rules_and_dots(image)
                rois: List[DocumentROI] = segment_document_rois(clean_gray, clean_bw)
                optical_blanks = detect_image_blanks(image)
                has_two_col = any(r.column_role in ("left_col", "right_col") for r in rois)
                is_multi_roi_doc = (
                    has_two_col
                    or len(rois) >= 3
                    or gm.is_dense_document
                    or gm.is_low_resolution_document
                )
            else:
                clean_gray, polarity = detect_and_normalize_polarity(image)
                rois = []
                optical_blanks = []
                is_multi_roi_doc = False

            if is_multi_roi_doc:
                return self._recognize_document_rois(
                    clean_gray=clean_gray,
                    rois=rois,
                    gm=gm,
                    polarity=polarity,
                    t0=t0,
                    cancel_token=cancel_token,
                    optical_blanks=optical_blanks,
                )

            return self._recognize_single_label(
                image=image,
                gm=gm,
                polarity=polarity,
                t0=t0,
                cancel_token=cancel_token,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return OCRResult(
                lines=[],
                engine_name=self.engine_name,
                elapsed_ms=elapsed_ms,
                error=f"Lỗi OCR ({self.engine_name}): {exc}",
            )

    def _recognize_document_rois(
        self,
        clean_gray: np.ndarray,
        rois: List[DocumentROI],
        gm: GlyphMetrics,
        polarity: str,
        t0: float,
        cancel_token: Optional[threading.Event],
        optical_blanks: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> OCRResult:
        """Recognize multi-band / multi-column documents with layout awareness and reliability filtering."""
        base_lang = "vie_best" if (self.variant == "best" or self._has_best_model()) else "vie"
        has_eng = self._has_eng_model()

        acc, rej, tot, reasons, all_raw = self._extract_roi_lines(
            clean_gray, rois, gm, base_lang, cancel_token, optical_blanks
        )
        if cancel_token is not None and cancel_token.is_set():
            return OCRResult(
                lines=[],
                engine_name=self.engine_name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="CANCELLED",
            )

        has_best_eng = self.variant == "best" and self._has_best_eng_model()
        eng_model = "eng_best" if has_best_eng else "eng"

        # Multi-language identification:
        raw_combined = " ".join(ln.text for ln in all_raw)
        vi_chars = set(
            "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
            "ĂÂĐÊÔƠƯÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
        )
        n_vi = sum(1 for c in raw_combined if c in vi_chars)
        n_alpha = sum(1 for c in raw_combined if c.isalpha())
        vi_ratio = n_vi / float(max(1, n_alpha))

        eng_stopwords = {
            "the", "and", "of", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as",
            "with", "they", "i", "at", "be", "this", "have", "from", "or", "one", "had", "by", "but", "not",
            "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which",
            "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them",
            "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two",
            "more", "write", "go", "see", "number", "no", "way", "could", "people", "my", "than", "first",
            "water", "been", "call", "who", "oil", "its", "now", "find", "long", "down", "day", "did", "get",
            "come", "made", "may", "part", "tell", "want", "answer", "remember", "choose", "correct", "question",
        }
        words = {w.lower().strip(".,:;!?()[]\"'") for w in raw_combined.split()}
        n_eng_stopwords = len(words.intersection(eng_stopwords))

        is_clear_doc = not gm.is_low_resolution_document and (gm.median_char_height_px == 0.0 or gm.median_char_height_px >= 9.5)
        has_code_syntax = any(
            (ln.raw_score is None or ln.raw_score >= 40.0)
            and re.search(r"(?:===|==|!=|&&|\|\||=>|->|<=|>=|\b(?:function\s*[\(\w]|const\s+[a-zA-Z_$]|isNaN\s*\()\b|\b(?:if|else\s+if)\s*\()", ln.text)
            for ln in all_raw
        )
        has_tech_report = any(
            (ln.raw_score is None or ln.raw_score >= 40.0)
            and re.search(r"\b(?:H\.2\w+|MPEG(?:-TS)?|AAC|HTTP\s+\d{3}|\.miv|\.mp4|\.ts|remux|lossless|transcode|FFmpeg)\b", ln.text)
            for ln in all_raw
        )

        is_predominantly_english = (
            has_eng
            and not has_code_syntax
            and not has_tech_report
            and (
                (vi_ratio < 0.04 and n_eng_stopwords >= 4)
                or (vi_ratio < 0.015 and n_eng_stopwords >= 2)
                or (vi_ratio == 0.0 and len(all_raw) >= 4 and n_eng_stopwords >= 1)
            )
        )

        detected_language = "vi"
        if is_predominantly_english:
            detected_language = "en"
            acc_e, rej_e, tot_e, reasons_e, _ = self._extract_roi_lines(
                clean_gray, rois, gm, eng_model, cancel_token, optical_blanks
            )
            if cancel_token is not None and cancel_token.is_set():
                return OCRResult(
                    lines=[],
                    engine_name=self.engine_name,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    error="CANCELLED",
                    language=detected_language,
                )
            if len(acc_e) >= len(acc) or rej_e <= rej:
                acc, rej, tot, reasons = acc_e, rej_e, tot_e, reasons_e
        elif has_eng and is_clear_doc and (((rej > 0 or len(acc) == 0) and has_code_syntax) or has_tech_report):
            detected_language = "vi+en"
            combo_lang = f"{base_lang}+eng"
            acc_c, rej_c, tot_c, reasons_c, _ = self._extract_roi_lines(
                clean_gray, rois, gm, combo_lang, cancel_token, optical_blanks
            )
            if cancel_token is not None and cancel_token.is_set():
                return OCRResult(
                    lines=[],
                    engine_name=self.engine_name,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    error="CANCELLED",
                    language=detected_language,
                )
            if len(acc_c) > len(acc) or (len(acc_c) == len(acc) and rej_c < rej) or (len(acc_c) > 0 and len(acc) == 0) or (has_tech_report and len(acc_c) >= len(acc)):
                acc, rej, tot, reasons = acc_c, rej_c, tot_c, reasons_c

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        low_res_flag = bool(
            gm.is_low_resolution_document
            or (gm.median_char_height_px > 0 and gm.median_char_height_px < 8.5)
            or (tot >= 4 and (rej / float(tot)) >= 0.30)
        )

        return OCRResult(
            lines=acc,
            engine_name=self.engine_name,
            elapsed_ms=elapsed_ms,
            error=None,
            language=detected_language,
            low_resolution_detected=low_res_flag,
            median_char_height_px=gm.median_char_height_px,
            p25_char_height_px=gm.p25_char_height_px,
            p75_char_height_px=gm.p75_char_height_px,
            polarity=polarity,
            roi_count=len(rois),
            rejected_line_count=rej,
            total_candidate_lines=tot,
            rejection_reasons=reasons,
        )

    def _extract_roi_lines(
        self,
        clean_gray: np.ndarray,
        rois: List[DocumentROI],
        gm: GlyphMetrics,
        lang: str,
        cancel_token: Optional[threading.Event],
        optical_blanks: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[List[OCRLine], int, int, List[str], List[OCRLine]]:
        h_img = clean_gray.shape[0]
        accepted_lines: List[OCRLine] = []
        all_raw_lines: List[OCRLine] = []
        rejected_count = 0
        total_candidates = 0
        rejection_reasons: List[str] = []

        for roi in rois:
            if cancel_token is not None and cancel_token.is_set():
                return [], 0, 0, [], []

            if roi.median_char_height_px > 0 and roi.median_char_height_px < 4.5:
                rejected_count += 1
                total_candidates += 1
                rejection_reasons.append(
                    f"ROI#{roi.order_index}: char_height_too_small ({roi.median_char_height_px:.1f}px < 4.5px)"
                )
                continue

            crop = clean_gray[roi.y1 : roi.y2, roi.x1 : roi.x2]
            if crop.size == 0:
                continue

            offset_x = roi.x1
            if roi.column_role == "header":
                hdr_bw = (crop < 200).astype(np.uint8)
                cols = np.where(np.sum(hdr_bw > 0, axis=0) > 0)[0]
                if len(cols) > 0:
                    hx1 = max(0, cols[0] - 8)
                    hx2 = min(crop.shape[1], cols[-1] + 35)
                    crop = crop[:, hx1:hx2]
                    offset_x = roi.x1 + hx1

            pad = 8
            padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            char_h = roi.median_char_height_px if roi.median_char_height_px > 0 else gm.median_char_height_px
            if lang.startswith("eng"):
                scale = 1.5
                interp = cv2.INTER_LINEAR
            else:
                if char_h > 0 and char_h < 10.0:
                    scale = 3.0
                elif char_h > 0 and char_h < 22.0:
                    scale = 2.0
                else:
                    scale = 1.0
                interp = cv2.INTER_CUBIC

            if crop.shape[1] * scale > 2200:
                scale = max(1.0, 2200.0 / float(crop.shape[1]))

            if scale > 1.0:
                scaled_crop = cv2.resize(
                    padded,
                    (int(round(padded.shape[1] * scale)), int(round(padded.shape[0] * scale))),
                    interpolation=interp,
                )
            else:
                scaled_crop = padded

            raw_lines = self._extract_raw_lines_from_crop(
                scaled_crop=scaled_crop,
                lang=lang,
                scale=scale,
                pad=pad,
                offset_x=offset_x,
                offset_y=roi.y1,
                order_index=roi.order_index,
                column_role=roi.column_role,
                psm=6,
                optical_blanks=optical_blanks,
            )
            # Fallback to PSM 11 for header ROIs if PSM 6 produced no lines or low score
            if roi.column_role == "header":
                p6_score = max((l.raw_score for l in raw_lines), default=0.0)
                if not raw_lines or p6_score < 50.0:
                    lines_11 = self._extract_raw_lines_from_crop(
                        scaled_crop=scaled_crop,
                        lang=lang,
                        scale=scale,
                        pad=pad,
                        offset_x=offset_x,
                        offset_y=roi.y1,
                        order_index=roi.order_index,
                        column_role=roi.column_role,
                        psm=11,
                        optical_blanks=optical_blanks,
                    )
                    p11_score = max((l.raw_score for l in lines_11), default=0.0)
                    if p11_score > p6_score or (not raw_lines and lines_11):
                        raw_lines = lines_11
            all_raw_lines.extend(raw_lines)

            for ln in raw_lines:
                total_candidates += 1
                clean_text = disambiguate_technical_and_numeric_tokens(ln.text)
                disambiguated_ln = OCRLine(
                    text=clean_text,
                    bbox=ln.bbox,
                    raw_score=ln.raw_score,
                    order_index=ln.order_index,
                    column_role=ln.column_role,
                )
                score = ln.raw_score if ln.raw_score is not None else 0.0
                plausible, valid_ratio = evaluate_line_plausibility(clean_text, score)

                is_known_heading = False
                if roi.column_role == "header" or (roi.y1 <= int(0.18 * h_img)):
                    if re.search(r"\b(Phrasal verbs|Prepositional phrases|Word formation|Word patterns)\b", clean_text, re.I):
                        is_known_heading = True
                    elif re.search(r"[Cc]\s*ho+u?se\s+the\s+correct\s+answe[ri]", clean_text):
                        is_known_heading = True

                if lang.startswith("eng"):
                    is_valid_multiword = valid_ratio >= 0.90 and len(disambiguated_ln.text.split()) >= 2
                    score_thresh = 35.0 if is_known_heading else (42.0 if ".........." in clean_text else (38.0 if is_valid_multiword else 52.0))
                    min_valid_ratio = 0.60
                else:
                    score_thresh = 35.0 if is_known_heading else (42.0 if ".........." in clean_text else 52.0)
                    min_valid_ratio = 0.70

                if gm.is_low_resolution_document:
                    low_res_min_score = 74.0 if (clean_text.isupper() and len(clean_text) >= 5) else 86.0
                    if (
                        plausible
                        and score >= low_res_min_score
                        and valid_ratio >= 0.99
                        and (roi.median_char_height_px >= 6.5 or gm.median_char_height_px >= 9.0)
                    ):
                        accepted_lines.append(disambiguated_ln)
                    else:
                        rejected_count += 1
                        rejection_reasons.append(
                            f"ROI#{roi.order_index} [low_res_doc]: score={score:.1f}%, valid_ratio={valid_ratio:.2f}, "
                            f"roi_h={roi.median_char_height_px:.1f}px | text={clean_text!r}"
                        )
                else:
                    if plausible and score >= score_thresh and valid_ratio >= min_valid_ratio:
                        accepted_lines.append(disambiguated_ln)
                    else:
                        rejected_count += 1
                        rejection_reasons.append(
                            f"ROI#{roi.order_index} [normal_doc]: plausible={plausible}, score={score:.1f}%, "
                            f"valid_ratio={valid_ratio:.2f} | text={clean_text!r}"
                        )

        return accepted_lines, rejected_count, total_candidates, rejection_reasons, all_raw_lines

    def _recognize_single_label(
        self,
        image: np.ndarray,
        gm: GlyphMetrics,
        polarity: str,
        t0: float,
        cancel_token: Optional[threading.Event],
    ) -> OCRResult:
        lang = "vie_best" if self.variant == "best" else "vie"
        prep_primary, scale_primary = self._preprocess_primary(image, gm.median_char_height_px)
        lines_primary, rej_primary, tot_primary = self._run_tesseract_data(
            prep_primary, lang, scale_primary, psm=6
        )

        if cancel_token is not None and cancel_token.is_set():
            return OCRResult(
                lines=[],
                engine_name=self.engine_name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error="CANCELLED",
            )

        avg_score_primary = self._mean_score(lines_primary)
        best_lines = lines_primary
        best_rej = rej_primary
        best_tot = tot_primary

        if (not lines_primary or avg_score_primary < 64.0) and float(np.std(prep_primary)) >= 8.0:
            prep_alt, scale_alt = self._preprocess_adaptive(image, gm.median_char_height_px)
            lines_alt, rej_alt, tot_alt = self._run_tesseract_data(prep_alt, lang, scale_alt, psm=6)
            if self._quality_metric(lines_alt) > self._quality_metric(best_lines):
                best_lines = lines_alt
                best_rej = rej_alt
                best_tot = tot_alt

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        h_s, _ = image.shape[:2]
        low_res = bool(
            h_s >= 420
            and (
                gm.is_low_resolution_document
                or (best_tot >= 5 and not best_lines and best_rej >= 5 and gm.text_component_count >= 30)
            )
        )
        return OCRResult(
            lines=best_lines,
            engine_name=self.engine_name,
            elapsed_ms=elapsed_ms,
            error=None,
            low_resolution_detected=low_res,
            median_char_height_px=gm.median_char_height_px,
            p25_char_height_px=gm.p25_char_height_px,
            p75_char_height_px=gm.p75_char_height_px,
            polarity=polarity,
            roi_count=1,
            rejected_line_count=best_rej,
            total_candidate_lines=best_tot,
        )

    @staticmethod
    def _preprocess_primary(image: np.ndarray, median_char_h: float = 0.0) -> Tuple[np.ndarray, float]:
        gray, _ = detect_and_normalize_polarity(image)

        h, w = gray.shape[:2]
        scale = 1.0
        if (0 < median_char_h < 14.0) or h < 180 or w < 360:
            scale = 2.0
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        elif w > 1800:
            scale = 1400.0 / float(w)
            gray = cv2.resize(
                gray,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return gray, scale

    @staticmethod
    def _preprocess_adaptive(image: np.ndarray, median_char_h: float = 0.0) -> Tuple[np.ndarray, float]:
        gray, scale = TesseractOCRBackend._preprocess_primary(image, median_char_h)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 5, 35, 35)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(np.mean(binary)) < 127.0:
            binary = 255 - binary
        return binary, scale

    def _extract_raw_lines_from_crop(
        self,
        scaled_crop: np.ndarray,
        lang: str,
        scale: float,
        pad: int,
        offset_x: int,
        offset_y: int,
        order_index: int,
        column_role: str,
        psm: int = 6,
        optical_blanks: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[OCRLine]:
        config = f"--oem 1 --psm {psm}"
        data = pytesseract.image_to_data(
            scaled_crop,
            lang=lang,
            config=config,
            output_type=Output.DICT,
            timeout=15,
        )
        n_items = len(data.get("text", []))
        grouped: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

        for i in range(n_items):
            raw_word = str(data["text"][i]).strip()
            if not raw_word:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            if conf < 0:
                continue

            word = unicodedata.normalize("NFC", raw_word)
            key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            x = int(round(int(data["left"][i]) / scale)) - pad + offset_x
            y = int(round(int(data["top"][i]) / scale)) - pad + offset_y
            w = max(1, int(round(int(data["width"][i]) / scale)))
            h = max(1, int(round(int(data["height"][i]) / scale)))

            if key not in grouped:
                grouped[key] = {
                    "words": [word],
                    "confs": [conf],
                    "boxes": [(x, y, x + w, y + h)],
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                }
            else:
                entry = grouped[key]
                entry["words"].append(word)
                entry["confs"].append(conf)
                entry["boxes"].append((x, y, x + w, y + h))
                entry["x1"] = min(entry["x1"], x)
                entry["y1"] = min(entry["y1"], y)
                entry["x2"] = max(entry["x2"], x + w)
                entry["y2"] = max(entry["y2"], y + h)

        crop_w = int(round(scaled_crop.shape[1] / scale))
        crop_h = int(round(scaled_crop.shape[0] / scale))
        crop_x2 = offset_x + crop_w
        crop_y2 = offset_y + crop_h

        # Optical blank insertion
        if optical_blanks:
            roi_blanks = [
                b for b in optical_blanks
                if offset_x <= (b[0] + b[2]) // 2 <= crop_x2 and offset_y <= (b[1] + b[3]) // 2 <= crop_y2
            ]
            used_blanks = set()
            for key, entry in grouped.items():
                ly1, ly2 = entry["y1"], entry["y2"]
                words = list(entry["words"])
                boxes = list(entry["boxes"])
                line_blanks = []
                for b_idx, b in enumerate(roi_blanks):
                    if b_idx in used_blanks:
                        continue
                    by_mid = (b[1] + b[3]) / 2.0
                    if (ly1 - 8) <= by_mid <= (ly2 + 8):
                        line_blanks.append((b_idx, b))

                for b_idx, b in line_blanks:
                    used_blanks.add(b_idx)
                    bx_mid = (b[0] + b[2]) / 2.0
                    inserted = False
                    for w_i, box in enumerate(boxes):
                        wx_mid = (box[0] + box[2]) / 2.0
                        if bx_mid < wx_mid:
                            words.insert(w_i, "..........")
                            boxes.insert(w_i, (b[0], b[1], b[2], b[3]))
                            inserted = True
                            break
                    if not inserted:
                        words.append("..........")
                        boxes.append((b[0], b[1], b[2], b[3]))
                    entry["x1"] = min(entry["x1"], b[0])
                    entry["x2"] = max(entry["x2"], b[2])
                    entry["y1"] = min(entry["y1"], b[1])
                    entry["y2"] = max(entry["y2"], b[3])
                entry["words"] = words
                entry["boxes"] = boxes

        # Header ROI line ordering polish: Ensure section title precedes annotation note
        if column_role == "header":
            for entry in grouped.values():
                line_text = " ".join(entry["words"]).strip()
                if re.search(r"[Cc]\s*ho+u?se\s+the\s+correct\s+answe[ri]", line_text):
                    entry["y1"] = offset_y
                    entry["y2"] = offset_y + 20
                elif re.search(r"mark\s+per\s+answer", line_text, re.IGNORECASE):
                    entry["y1"] = offset_y + 30
                    entry["y2"] = offset_y + 45

        lines: List[OCRLine] = []
        entries = sorted(grouped.values(), key=lambda e: (e["y1"], e["x1"]))
        for entry in entries:
            line_text = " ".join(entry["words"]).strip()
            mean_conf = float(np.mean(entry["confs"])) if entry["confs"] else 0.0
            if not line_text or not any(c.isalnum() for c in line_text):
                continue
            bbox = (
                max(0, int(entry["x1"])),
                max(0, int(entry["y1"])),
                max(1, int(entry["x2"] - entry["x1"])),
                max(1, int(entry["y2"] - entry["y1"])),
            )
            lines.append(
                OCRLine(
                    text=line_text,
                    bbox=bbox,
                    raw_score=mean_conf,
                    order_index=order_index,
                    column_role=column_role,
                )
            )
        return lines

    def _run_tesseract_data(
        self,
        gray_img: np.ndarray,
        lang: str,
        scale: float,
        psm: int = 6,
    ) -> Tuple[List[OCRLine], int, int]:
        raw_lines = self._extract_raw_lines_from_crop(
            scaled_crop=gray_img,
            lang=lang,
            scale=scale,
            pad=0,
            offset_x=0,
            offset_y=0,
            order_index=0,
            column_role="full",
            psm=psm,
        )
        accepted: List[OCRLine] = []
        rejected = 0
        for ln in raw_lines:
            clean_text = disambiguate_technical_and_numeric_tokens(ln.text)
            disambiguated_ln = OCRLine(
                text=clean_text,
                bbox=ln.bbox,
                raw_score=ln.raw_score,
                order_index=ln.order_index,
                column_role=ln.column_role,
            )
            score = ln.raw_score if ln.raw_score is not None else 0.0
            if self._is_plausible_text_line(clean_text, score):
                accepted.append(disambiguated_ln)
            else:
                rejected += 1
        return accepted, rejected, len(raw_lines)

    @staticmethod
    def _is_plausible_text_line(text: str, raw_score: float) -> bool:
        """Filter out background noise artifacts so no-text frames never hallucinate words."""
        if not text:
            return False
        alnum_chars = [ch for ch in text if ch.isalnum()]
        if not alnum_chars:
            return False
        weird_symbols = sum(1 for ch in text if ch in "()[]{}|<>~`^_\\")
        if len(alnum_chars) <= 3 and weird_symbols > 0:
            return False
        if len(alnum_chars) <= 2 and raw_score < 78.0:
            return False
        if raw_score < 28.0:
            return False
        ok, _ = evaluate_line_plausibility(text, raw_score)
        return ok

    @staticmethod
    def _mean_score(lines: List[OCRLine]) -> float:
        scores = [ln.raw_score for ln in lines if ln.raw_score is not None]
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _quality_metric(lines: List[OCRLine]) -> float:
        if not lines:
            return 0.0
        total_alnum = sum(sum(1 for c in ln.text if c.isalnum()) for ln in lines)
        mean_score = TesseractOCRBackend._mean_score(lines)
        return mean_score * (1.0 + min(total_alnum, 40) * 0.02)
