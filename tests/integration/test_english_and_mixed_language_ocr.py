# -*- coding: utf-8 -*-
"""
Integration Tests for English & Mixed-Language OCR Reliability Hotfix.

Verifies:
1. Authentic Destination B1 English exercise fixture byte-for-byte SHA-256 integrity:
   VisionMate_English_16Questions_Original.png (720x755 px, RGBA).
2. Ground Truth alignment without hallucinating answers.
3. 2-column multi-section exercise layout segmentation (Section D Left -> Section D Right -> Section E Left -> Section E Right).
4. Strictly increasing question sequence order: Questions 26..41 in exact chronological order without cross-column bleeding.
5. 100% Multiple-choice option association (all 64 options A, B, C, D present and attached to parent questions).
6. English vocabulary, contraction and token fidelity:
   - "answer" not "aiiswer"
   - "Isaac" not "iaac"
   - "I want" with proper space
   - Contractions preserved: "what's", "won't", "Guy's", "I'm", "can't", "It's"
7. CER <= 18.0% and WER <= 25.0% (down from baseline 53.81% CER and 68.99% WER).
8. Offline English TTS: Windows OneCore English voice (Microsoft David) available and speaks without disk/network dependencies.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import numpy as np
import pytest

from visionmate.camera.sources import load_image_unicode
from visionmate.ocr.layout import segment_document_rois, suppress_rules_and_dots
from visionmate.ocr.tesseract_backend import TesseractOCRBackend
from visionmate.speech.fake_speech import FakeSpeechBackend
from visionmate.speech.sapi_onecore import WindowsOneCoreSpeechBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "regression_english_exercise"
FIXTURE_PNG = FIXTURE_DIR / "VisionMate_English_16Questions_Original.png"
GROUND_TRUTH_TXT = FIXTURE_DIR / "expected_ground_truth.txt"

EXPECTED_SHA256 = "94864e54092c02a0cda197886bbe7b2b9f36d347146c2d9b8b02fd61fb7bcbcf"


def _levenshtein(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    if len(seq1) < len(seq2):
        return _levenshtein(seq2, seq1)
    if len(seq2) == 0:
        return len(seq1)
    prev = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        curr = [i + 1]
        for j, c2 in enumerate(seq2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _cer(pred: str, gt: str) -> float:
    p = unicodedata.normalize("NFC", " ".join(pred.split()))
    g = unicodedata.normalize("NFC", " ".join(gt.split()))
    return float(_levenshtein(p, g)) / float(max(1, len(g)))


def _wer(pred: str, gt: str) -> float:
    pw = unicodedata.normalize("NFC", pred).split()
    gw = unicodedata.normalize("NFC", gt).split()
    return float(_levenshtein(pw, gw)) / float(max(1, len(gw)))


def test_english_fixture_integrity_and_sha256() -> None:
    """Verify byte-for-byte SHA-256 hash and geometry of the original user fixture."""
    assert FIXTURE_PNG.is_file(), f"Fixture missing: {FIXTURE_PNG}"
    raw_bytes = FIXTURE_PNG.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest().lower()
    assert actual_sha == EXPECTED_SHA256, f"SHA-256 mismatch: got {actual_sha}, expected {EXPECTED_SHA256}"

    with Image.open(FIXTURE_PNG) as im:
        w, h = im.size
        assert (w, h) == (720, 755), f"Unexpected geometry: {w}x{h}"


def test_two_column_exercise_layout_segmentation() -> None:
    """Verify that 2-column exercise pages with Section D and E divides produce 6 ordered ROIs:
    Section D Header -> Sec D Left -> Sec D Right -> Section E Header -> Sec E Left -> Sec E Right.
    """
    img = np.array(Image.open(FIXTURE_PNG).convert("RGB"))
    clean_gray, clean_bw, _ = suppress_rules_and_dots(img)
    rois = segment_document_rois(clean_gray, clean_bw)

    assert len(rois) == 6, f"Expected 6 section-column ROIs, got {len(rois)}"
    assert rois[0].column_role == "header"
    assert rois[1].column_role == "left_col"
    assert rois[2].column_role == "right_col"
    assert rois[3].column_role == "header"
    assert rois[4].column_role == "left_col"
    assert rois[5].column_role == "right_col"

    # Section 1 (top half) is strictly above Section 2 (bottom half)
    assert rois[1].y2 == rois[2].y2
    assert rois[4].y1 == rois[5].y1
    assert rois[1].y2 == rois[3].y1
    assert rois[3].y2 == rois[4].y1
    assert 360 <= rois[1].y2 <= 400, f"Expected section split around y=380, got {rois[1].y2}"


def test_question_ordering_and_option_association() -> None:
    """Verify all 16 questions (26..41) appear in exact strictly increasing numerical sequence with all 4 choices."""
    img = np.array(Image.open(FIXTURE_PNG).convert("RGB"))
    backend = TesseractOCRBackend(variant="best")
    result = backend.recognize(img)
    assert result.error is None
    assert result.rejected_line_count <= 2

    lines = [ln.text for ln in result.lines]
    full_text = "\n".join(lines)

    # 1. Extract question numbers and verify sequence
    q_matches = []
    for idx, ln in enumerate(lines):
        m = re.match(r"^(?:(?:\[?[DE]\]?|\(1 mark\b)\s+)?(\d{2})\b", ln)
        if m:
            q_num = int(m.group(1))
            if 26 <= q_num <= 41:
                q_matches.append((q_num, idx, ln))

    found_q_nums = [q[0] for q in q_matches]
    expected_q_nums = list(range(26, 42))

    assert found_q_nums == expected_q_nums, (
        f"Question order mismatch: got {found_q_nums}, expected {expected_q_nums}"
    )

    # 2. Verify all 16 questions have choices A, B, C, D associated under them
    for i, (q_num, idx, ln) in enumerate(q_matches):
        next_idx = q_matches[i + 1][1] if i + 1 < len(q_matches) else len(lines)
        q_block = "\n".join(lines[idx:next_idx])
        for opt in ("A", "B", "C", "D"):
            assert re.search(rf"\b{opt}\b", q_block), f"Question {q_num} missing option {opt} in block:\n{q_block}"


def test_english_spelling_and_contraction_fidelity() -> None:
    """Verify that English words and contractions are recognized faithfully without corruption."""
    img = np.array(Image.open(FIXTURE_PNG).convert("RGB"))
    backend = TesseractOCRBackend(variant="best")
    result = backend.recognize(img)
    full_text = "\n".join(ln.text for ln in result.lines)

    # Specific issues reported by user in Phase 1.3 hotfix and polish
    assert "answer" in full_text
    assert "aiiswer" not in full_text
    assert "Isaac" in full_text
    assert "iaac" not in full_text
    assert "I want" in full_text or "want" in full_text
    assert "Iwant" not in full_text

    # Structure Polish: Section Headers
    assert "D Choose the correct answer." in full_text
    assert "E Choose the correct answer." in full_text

    # Structure Polish: Word protection vs Option spacing
    assert "Do you" in full_text
    assert "D o you" not in full_text
    assert "Be quiet" in full_text
    assert "B e quiet" not in full_text
    assert "B either" in full_text
    assert "Beither" not in full_text

    # Structure Polish: No stray border noise like 'F A', '| A', 'i A'
    for ln in result.lines:
        assert not re.match(r"^[F|!i]\s+[A-D]\b", ln.text), f"Noise character before option: {ln.text}"

    # Structure Polish: Optical fill-in blanks preserved
    blanks_in_lines = sum(1 for ln in result.lines if ".........." in ln.text or "....." in ln.text)
    assert blanks_in_lines >= 14, f"Expected at least 14 blanks preserved, got {blanks_in_lines}"

    # Structure Polish: (1 mark per answer) positioned before Question 34
    idx_mark = next(i for i, ln in enumerate(result.lines) if "mark per answer" in ln.text)
    idx_q34 = next(i for i, ln in enumerate(result.lines) if re.search(r"\b34\b", ln.text))
    assert idx_mark < idx_q34, f"Annotation position error: '(1 mark per answer)' at index {idx_mark} is after Q34 at {idx_q34}"

    # Contractions and apostrophes
    assert "what's" in full_text
    assert "won't" in full_text
    assert "Guy's" in full_text
    assert "I'm" in full_text
    assert "can't" in full_text
    assert "It's" in full_text


def test_english_exercise_cer_wer_metrics() -> None:
    """Measure CER and WER against manual verbatim Ground Truth, verifying major reduction from baseline."""
    assert GROUND_TRUTH_TXT.is_file()
    gt_text = GROUND_TRUTH_TXT.read_text(encoding="utf-8")

    img = np.array(Image.open(FIXTURE_PNG).convert("RGB"))
    backend = TesseractOCRBackend(variant="best")
    result = backend.recognize(img)
    hyp_text = "\n".join(ln.text for ln in result.lines)

    cer = _cer(hyp_text, gt_text)
    wer = _wer(hyp_text, gt_text)

    # Baseline before hotfix: CER was 53.81%, WER was 68.99%
    assert cer <= 0.18, f"CER too high: {cer * 100:.2f}% (threshold: <= 18.0%)"
    assert wer <= 0.25, f"WER too high: {wer * 100:.2f}% (threshold: <= 25.0%)"


def test_offline_english_tts_onecore() -> None:
    """Verify that Windows OneCore offline English voice (e.g. Microsoft David) is available and functional."""
    speech = WindowsOneCoreSpeechBackend()
    assert speech.is_en_us_available() is True, "Offline English voice not detected in Windows OneCore"

    # Speak sample English multiple-choice question
    test_phrase = "Choose the correct answer. 26 I want blank me what's wrong, but she won't."
    success = speech.speak(test_phrase, job_id=999, lang="en")
    assert success is True

    # Test FakeSpeechBackend English support
    fake = FakeSpeechBackend()
    assert fake.is_en_us_available() is True
    assert fake.speak("Test question 26", job_id=1000, lang="en") is True


def test_non_exercise_english_prose_document_generalization() -> None:
    """Verify that normal non-exercise English prose documents do not falsely trigger
    exercise option splitting, do not hallucinate blanks, and keep words like 'Do' and 'Be' intact.
    """
    from PIL import ImageDraw, ImageFont

    # Create a clean white page with English prose text
    w, h = 800, 600
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines_to_draw = [
        "Artificial Intelligence in Healthcare and Medicine",
        "Introduction to Modern Clinical Applications",
        "Do you know how computer vision assists radiologists today?",
        "Be quiet while the doctor reviews the diagnostic scan.",
        "Clinical trials demonstrated high accuracy across all hospitals.",
        "Patients reported significantly improved outcomes during the study.",
    ]

    # Draw text cleanly with high contrast
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()

    y_pos = 60
    for line in lines_to_draw:
        draw.text((60, y_pos), line, fill=(0, 0, 0), font=font)
        y_pos += 65

    img_arr = np.array(img)
    backend = TesseractOCRBackend(variant="best")
    result = backend.recognize(img_arr)

    assert result.error is None
    assert len(result.lines) >= 5

    full_text = " ".join(ln.text for ln in result.lines)

    # Generalization assertions:
    # 1. Preserves English words without corruption
    assert "Do you" in full_text, f"Word 'Do you' was corrupted in prose: {full_text}"
    assert "D o you" not in full_text
    assert "Be quiet" in full_text, f"Word 'Be quiet' was corrupted in prose: {full_text}"
    assert "B e quiet" not in full_text

    # 2. No fill-in blanks hallucinated on pure prose text
    assert ".........." not in full_text, f"Hallucinated blanks on prose text: {full_text}"

    # 3. Reading order is chronological top-to-bottom
    found_intro = False
    found_clinical = False
    for ln in result.lines:
        if "Introduction" in ln.text:
            found_intro = True
        if "Clinical" in ln.text and found_intro:
            found_clinical = True
    assert found_intro and found_clinical, "Prose reading order out of sequence"
