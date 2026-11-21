# -*- coding: utf-8 -*-
"""Fake SpeechBackend for unit, integration, race-condition, and TTS_BLOCKED testing."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from visionmate.speech.base import SpeechBackend, VoiceInfo


@dataclass(frozen=True)
class SpokenRecord:
    """Record of a completed or interrupted speech call in FakeSpeechBackend."""
    job_id: int
    text: str
    completed: bool


class FakeSpeechBackend(SpeechBackend):
    """Deterministic fake TTS backend supporting simulated playback duration and instant stop."""

    def __init__(
        self,
        vi_available: bool = True,
        speak_duration_s: float = 0.05,
    ) -> None:
        self._vi_available = bool(vi_available)
        self._speak_duration_s = float(speak_duration_s)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active_job_id: int = -1
        self._shutting_down: bool = False
        self.rate: int = 0
        self.volume: int = 90
        self.history: List[SpokenRecord] = []
        self.started_jobs: List[int] = []

    def set_vi_available(self, available: bool) -> None:
        self._vi_available = bool(available)

    def list_voices(self) -> List[VoiceInfo]:
        voices = [
            VoiceInfo(
                voice_id="FAKE_EN_US",
                description="Fake English Voice",
                source="Fake",
                is_vietnamese=False,
            )
        ]
        if self._vi_available:
            voices.append(
                VoiceInfo(
                    voice_id="FAKE_VI_VN_AN",
                    description="Fake Microsoft An - Vietnamese (Vietnam)",
                    source="Fake_OneCore",
                    is_vietnamese=True,
                )
            )
        return voices

    def is_vi_vn_available(self) -> bool:
        return self._vi_available

    def set_rate(self, rate: int) -> None:
        self.rate = max(-10, min(10, int(rate)))

    def set_volume(self, volume: int) -> None:
        self.volume = max(0, min(100, int(volume)))

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            self._active_job_id = -1

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            self._stop_event.set()
            self._active_job_id = -1

    def _is_cancelled(self, job_id: int, cancel_token: Optional[threading.Event]) -> bool:
        if self._shutting_down or self._stop_event.is_set():
            return True
        if cancel_token is not None and cancel_token.is_set():
            return True
        with self._lock:
            return self._active_job_id != job_id

    def is_en_us_available(self) -> bool:
        return True

    def speak(
        self,
        text: str,
        job_id: int,
        cancel_token: Optional[threading.Event] = None,
        on_speech_started: Optional[Callable[[int, float], None]] = None,
        lang: str = "auto",
    ) -> bool:
        is_en = (lang == "en") or (lang == "auto" and not any(c in text for c in "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"))
        if is_en:
            pass  # English available
        elif not self._vi_available:
            return False

        with self._lock:
            if self._shutting_down:
                return False
            self._stop_event.clear()
            self._active_job_id = job_id

        if self._is_cancelled(job_id, cancel_token):
            return False

        t0 = time.perf_counter()
        with self._lock:
            self.started_jobs.append(job_id)

        if on_speech_started is not None:
            on_speech_started(job_id, (time.perf_counter() - t0) * 1000.0)

        step = 0.005
        waited = 0.0
        while waited < self._speak_duration_s:
            if self._is_cancelled(job_id, cancel_token):
                with self._lock:
                    self.history.append(SpokenRecord(job_id=job_id, text=text, completed=False))
                return False
            time.sleep(min(step, self._speak_duration_s - waited))
            waited += step

        completed = not self._is_cancelled(job_id, cancel_token)
        with self._lock:
            self.history.append(SpokenRecord(job_id=job_id, text=text, completed=completed))
        return completed
