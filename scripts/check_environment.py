# -*- coding: utf-8 -*-
"""
Environment Audit & Preflight Checker for VisionMate Glasses Phase 1.
Checks Windows OS, CPU/RAM, Python, Webcam (UVC), Audio output,
Vietnamese Offline TTS (Windows OneCore SAPI), and Offline OCR backends.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import psutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def check_os_and_hardware() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "ram_available_gb": round(vm.available / (1024 ** 3), 2),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }


def check_cameras(max_index: int = 2) -> List[Dict[str, Any]]:
    cameras: List[Dict[str, Any]] = []
    for idx in range(max_index):
        for backend_name, backend_id in [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]:
            cap = cv2.VideoCapture(idx, backend_id)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if ok and frame is not None and frame.size > 0:
                cameras.append({
                    "index": idx,
                    "backend": backend_name,
                    "opened": True,
                    "frame_read_ok": True,
                    "resolution": f"{width}x{height}",
                })
                break
    return cameras


def check_tts_voices() -> Dict[str, Any]:
    voices_found: List[Dict[str, str]] = []
    vi_voice_id: str | None = None
    synth_test_bytes = 0
    error_msg: str | None = None

    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        speaker = win32com.client.Dispatch("SAPI.SpVoice")

        categories = [
            ("SAPI_Desktop", r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices"),
            ("SAPI_OneCore", r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"),
        ]
        seen_ids = set()
        vi_token = None

        for source_name, reg_path in categories:
            try:
                cat = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
                cat.SetId(reg_path, False)
                for token in cat.EnumerateTokens():
                    tid = str(token.Id)
                    if tid in seen_ids:
                        continue
                    seen_ids.add(tid)
                    desc = str(token.GetDescription())
                    voices_found.append({
                        "source": source_name,
                        "id": tid,
                        "description": desc,
                    })
                    if ("vivn" in tid.lower() or "vietnamese" in desc.lower()) and vi_token is None:
                        vi_token = token
                        vi_voice_id = tid
            except Exception:
                continue

        if vi_token is not None:
            speaker.Voice = vi_token
            stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
            stream.Format.Type = 22  # SAFT22kHz16BitMono
            speaker.AudioOutputStream = stream
            speaker.Speak("Kiểm tra giọng đọc tiếng Việt ngoại tuyến.")
            pcm = bytes(stream.GetData())
            synth_test_bytes = len(pcm)
    except Exception as exc:
        error_msg = str(exc)

    return {
        "vi_vn_available": bool(vi_voice_id and synth_test_bytes > 0),
        "vi_voice_id": vi_voice_id,
        "synth_test_ram_bytes": synth_test_bytes,
        "installed_voices": voices_found,
        "error": error_msg,
    }


def check_audio_output() -> Dict[str, Any]:
    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        default_out = None
        try:
            info = pa.get_default_output_device_info()
            default_out = {
                "index": info.get("index"),
                "name": info.get("name"),
                "max_output_channels": info.get("maxOutputChannels"),
                "default_sample_rate": info.get("defaultSampleRate"),
            }
        finally:
            pa.terminate()
        return {"pyaudio_ready": default_out is not None, "default_output_device": default_out}
    except Exception as exc:
        return {"pyaudio_ready": False, "error": str(exc)}


def check_ocr_backends() -> Dict[str, Any]:
    candidates = [
        shutil.which("tesseract"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    tess_bin = next((c for c in candidates if c and os.path.isfile(c)), None)
    tess_version = None
    if tess_bin:
        try:
            proc = subprocess.run(
                [tess_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            tess_version = proc.stdout.splitlines()[0] if proc.stdout else None
        except Exception:
            tess_version = None

    models_info: Dict[str, Any] = {}
    model_paths = {
        "tessdata_fast_vie": PROJECT_ROOT / "models" / "tessdata_fast" / "vie.traineddata",
        "tessdata_best_vie": PROJECT_ROOT / "models" / "tessdata_best" / "vie.traineddata",
        "paddleocr_latin_rec": PROJECT_ROOT / "models" / "paddleocr_latin" / "latin_rec.onnx",
        "paddleocr_latin_dict": PROJECT_ROOT / "models" / "paddleocr_latin" / "latin_dict.txt",
    }
    for key, path in model_paths.items():
        if path.is_file():
            models_info[key] = {
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            models_info[key] = {"exists": False}

    return {
        "tesseract_binary": tess_bin,
        "tesseract_version": tess_version,
        "models": models_info,
    }


def run_audit() -> Dict[str, Any]:
    return {
        "system": check_os_and_hardware(),
        "cameras": check_cameras(),
        "audio": check_audio_output(),
        "tts": check_tts_voices(),
        "ocr": check_ocr_backends(),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = run_audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
