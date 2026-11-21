# -*- coding: utf-8 -*-
"""Offline speech synthesis package."""
from .base import SpeechBackend, VoiceInfo
from .fake_speech import FakeSpeechBackend
from .sapi_onecore import WindowsOneCoreSpeechBackend

__all__ = [
    "SpeechBackend",
    "VoiceInfo",
    "FakeSpeechBackend",
    "WindowsOneCoreSpeechBackend",
]
