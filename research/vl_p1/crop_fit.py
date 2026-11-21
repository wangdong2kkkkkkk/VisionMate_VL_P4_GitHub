"""Select an ROI on a fit-to-screen preview, save full-resolution crop.

VisionMate VL-P1 research only. Does not modify the original image or production UI.
Usage: python crop_fit.py <source-image> <destination-crop>
"""
from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path

import cv2
from orb_baseline import load_image, save_image


def preview_size(image_width: int, image_height: int, screen_width: int, screen_height: int) -> tuple[int, int]:
    """Fit within desktop limits while preserving aspect ratio and never upscaling."""
    if min(image_width, image_height, screen_width, screen_height) <= 0:
        raise ValueError("Image and screen sizes must be positive")
    # Leave room for window chrome, taskbar, and selection instructions.
    max_w = max(320, min(1200, int(screen_width * 0.82)))
    max_h = max(240, min(720, int(screen_height * 0.65)))
    scale = min(1.0, max_w / image_width, max_h / image_height)
    return max(1, int(round(image_width * scale))), max(1, int(round(image_height * scale)))


def roi_to_original(box: tuple[int, int, int, int], preview_wh: tuple[int, int], original_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    """Map selection (x,y,w,h) back to original pixel coordinates."""
    x, y, w, h = map(int, box)
    pw, ph = preview_wh
    ow, oh = original_wh
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0
    x0 = max(0, min(ow, math.floor(x * ow / pw)))
    y0 = max(0, min(oh, math.floor(y * oh / ph)))
    x1 = max(0, min(ow, math.ceil((x + w) * ow / pw)))
    y1 = max(0, min(oh, math.ceil((y + h) * oh / ph)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def screen_size() -> tuple[int, int]:
    try:
        # Matches logical desktop sizes used by ordinary Windows UI processes.
        user32 = ctypes.windll.user32
        sw, sh = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        if sw > 0 and sh > 0:
            return sw, sh
    except (AttributeError, OSError):
        pass
    return 1366, 768


def crop(src: Path, dst: Path) -> None:
    if dst.exists():
        raise SystemExit(f"Refusing to overwrite existing crop: {dst}")
    frame = load_image(src)
    oh, ow = frame.shape[:2]
    pw, ph = preview_size(ow, oh, *screen_size())
    shown = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA) if (pw, ph) != (ow, oh) else frame
    name = "VisionMate VL-P1 | ROI fitted to screen"
    print(f"Original image: {ow}x{oh}; selection preview: {pw}x{ph}")
    print("Drag around ONE full object. ENTER/SPACE = save; ESC = cancel.")
    try:
        roi = cv2.selectROI(name, shown, showCrosshair=True, fromCenter=False)
    finally:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass
    x, y, w, h = roi_to_original(tuple(map(int, roi)), (pw, ph), (ow, oh))
    if w <= 0 or h <= 0:
        print("Canceled; no file created.")
        return
    save_image(dst, frame[y:y + h, x:x + w])
    print(f"Saved original-resolution crop: {dst} ({w}x{h})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    crop(args.source, args.output)
