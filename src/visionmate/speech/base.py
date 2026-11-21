# -*- coding: utf-8 -*-
"""Abstract SpeechBackend interface (Section 4.1, FR-05)."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class VoiceInfo:
    """Metadata for an installed offline TTS voice."""
    voice_id: str
    description: str
    source: str
    is_vietnamese: bool
    is_english: bool = False


class SpeechBackend(ABC):
    """Interface for offline Text-to-Speech backends (Vietnamese vi-VN & English en-US)."""

    @abstractmethod
    def list_voices(self) -> List[VoiceInfo]:
        """Enumerate installed offline TTS voices on the system."""

    @abstractmethod
    def is_vi_vn_available(self) -> bool:
        """Return True only if a verified offline Vietnamese (vi-VN) voice is ready."""

    def is_en_us_available(self) -> bool:
        """Return True only if a verified offline English (en-US) voice is ready."""
        return False

    @abstractmethod
    def speak(
        self,
        text: str,
        job_id: int,
        cancel_token: Optional[threading.Event] = None,
        on_speech_started: Optional[Callable[[int, float], None]] = None,
        lang: str = "auto",
    ) -> bool:
        """Speak text synchronously on the calling worker thread while supporting instant stop().
        Returns True if speech completed without error or cancellation, False otherwise.
        """

    @abstractmethod
    def stop(self) -> None:
        """Immediately halt any active speech synthesis or playback."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop playback and release audio/TTS resources before app exit."""

    @abstractmethod
    def set_rate(self, rate: int) -> None:
        """Set speech rate (-10 to +10)."""

    @abstractmethod
    def set_volume(self, volume: int) -> None:
        """Set speech volume (0 to 100)."""
