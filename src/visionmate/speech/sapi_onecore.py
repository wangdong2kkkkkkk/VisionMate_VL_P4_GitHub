# -*- coding: utf-8 -*-
"""
Windows OneCore / SAPI Offline Vietnamese SpeechBackend (Section 5.1, FR-05, FR-07, FR-08).
- Uses Microsoft An - Vietnamese (Vietnam) (`MSTTS_V110_viVN_An`) installed in
  `HKLM\\SOFTWARE\\Microsoft\\Speech_OneCore\\Voices`.
- Synthesizes PCM audio in RAM (`SAPI.SpMemoryStream`, zero disk files).
- Plays audio in 2048-byte chunks via PyAudio so `stop()`, `shutdown()`, or a newer `job_id`
  cancels playback within <50ms without blocking or leaving stale audio.
- Fail-closed: if no `vi-VN` voice is installed, reports `is_vi_vn_available() == False`
  and never substitutes an English voice or cloud TTS.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np

from visionmate.speech.base import SpeechBackend, VoiceInfo


class WindowsOneCoreSpeechBackend(SpeechBackend):
    """Offline Windows SAPI / OneCore TTS backend for vi-VN."""

    def __init__(
        self,
        rate: int = 0,
        volume: int = 90,
        play_audio: bool = True,
    ) -> None:
        self._rate = max(-10, min(10, int(rate)))
        self._volume = max(0, min(100, int(volume)))
        self._play_audio = bool(play_audio)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active_job_id: int = -1
        self._shutting_down: bool = False
        self._voices_cache: Optional[List[VoiceInfo]] = None
        self._vi_voice_id: Optional[str] = None
        self._vi_category_path: str = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"
        self._en_voice_id: Optional[str] = None
        self._en_category_path: str = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"
        self._last_spoken_voice_id: Optional[str] = None
        self._last_spoken_voice_desc: Optional[str] = None
        self._last_spoken_lang: Optional[str] = None
        self._probe_voices()

    def _probe_voices(self) -> None:
        voices: List[VoiceInfo] = []
        vi_id: Optional[str] = None
        vi_cat: str = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"
        en_id: Optional[str] = None
        en_cat: str = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            categories = [
                ("SAPI_OneCore", r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"),
                ("SAPI_Desktop", r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices"),
            ]
            seen = set()
            for src_name, reg_path in categories:
                try:
                    cat = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
                    cat.SetId(reg_path, False)
                    for token in cat.EnumerateTokens():
                        tid = str(token.Id)
                        if tid in seen:
                            continue
                        seen.add(tid)
                        desc = str(token.GetDescription())
                        is_vi = ("vivn" in tid.lower()) or ("vietnamese" in desc.lower())
                        is_en = ("enus" in tid.lower()) or ("english" in desc.lower())
                        voices.append(
                            VoiceInfo(
                                voice_id=tid,
                                description=desc,
                                source=src_name,
                                is_vietnamese=is_vi,
                                is_english=is_en,
                            )
                        )
                        if is_vi and vi_id is None:
                            vi_id = tid
                            vi_cat = reg_path
                        if is_en and en_id is None:
                            en_id = tid
                            en_cat = reg_path
                        elif is_en and "david" in desc.lower() and "david" not in (en_id or "").lower():
                            # Prefer Microsoft David for clear English speech
                            en_id = tid
                            en_cat = reg_path
                except Exception:
                    continue
        except Exception:
            pass

        self._voices_cache = voices
        self._vi_voice_id = vi_id
        self._vi_category_path = vi_cat
        self._en_voice_id = en_id
        self._en_category_path = en_cat

    def list_voices(self) -> List[VoiceInfo]:
        if self._voices_cache is None:
            self._probe_voices()
        return list(self._voices_cache or [])

    def is_vi_vn_available(self) -> bool:
        if self._voices_cache is None:
            self._probe_voices()
        return self._vi_voice_id is not None

    def is_en_us_available(self) -> bool:
        if self._voices_cache is None:
            self._probe_voices()
        return self._en_voice_id is not None

    def set_rate(self, rate: int) -> None:
        with self._lock:
            self._rate = max(-10, min(10, int(rate)))

    def set_volume(self, volume: int) -> None:
        with self._lock:
            self._volume = max(0, min(100, int(volume)))

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
            if self._active_job_id != job_id:
                return True
        return False

    def speak(
        self,
        text: str,
        job_id: int,
        cancel_token: Optional[threading.Event] = None,
        on_speech_started: Optional[Callable[[int, float], None]] = None,
        lang: str = "auto",
    ) -> bool:
        clean_text = (text or "").strip()
        if not clean_text:
            return False

        vi_chars = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
        n_vi_chars = sum(1 for c in clean_text.lower() if c in vi_chars)
        vi_ratio = n_vi_chars / float(max(1, len(clean_text)))
        eng_stop = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
            "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
            "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
            "if", "about", "who", "get", "which", "go", "me", "when", "make", "can",
            "like", "time", "no", "just", "him", "know", "take", "people", "into", "year",
            "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
            "look", "only", "come", "its", "over", "think", "also", "back", "after", "use",
            "two", "how", "our", "work", "first", "well", "way", "even", "new", "want",
            "because", "any", "these", "give", "day", "most", "us", "verb", "phrase", "word",
            "patterns", "formation", "verbs", "phrases"
        }
        words = {w.lower().strip(".,:;!?()[]\"'") for w in clean_text.split()}
        has_eng_words = len(words.intersection(eng_stop)) >= 2
        is_english = (lang == "en") or (lang == "auto" and vi_ratio < 0.03 and (has_eng_words or n_vi_chars == 0) and self.is_en_us_available())

        if is_english:
            if not self.is_en_us_available():
                return False
            # Natural blank pronunciation for English exercises
            import re
            clean_text = re.sub(r"[._~-]{3,}", " blank ", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
        else:
            if not self.is_vi_vn_available():
                return False
            import re
            clean_text = re.sub(r"[._~-]{3,}", " chỗ trống ", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

        with self._lock:
            if self._shutting_down:
                return False
            self._stop_event.clear()
            self._active_job_id = job_id
            rate = self._rate
            volume = self._volume
            if is_english:
                target_voice_id = self._en_voice_id
                target_cat_path = self._en_category_path
            else:
                target_voice_id = self._vi_voice_id
                target_cat_path = self._vi_category_path

        if self._is_cancelled(job_id, cancel_token):
            return False

        t0 = time.perf_counter()
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            cat = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
            cat.SetId(target_cat_path, False)

            selected_token = None
            for token in cat.EnumerateTokens():
                if str(token.Id) == target_voice_id:
                    selected_token = token
                    break

            if selected_token is None:
                # Fallback search within category
                for token in cat.EnumerateTokens():
                    tid_lower = str(token.Id).lower()
                    if is_english and ("enus" in tid_lower or "david" in tid_lower):
                        selected_token = token
                        break
                    elif not is_english and "vivn" in tid_lower:
                        selected_token = token
                        break

            if selected_token is None:
                return False

            speaker.Voice = selected_token
            with self._lock:
                self._last_spoken_voice_id = str(selected_token.Id)
                self._last_spoken_voice_desc = str(selected_token.GetDescription())
                self._last_spoken_lang = "en" if is_english else "vi"
            speaker.Rate = rate
            speaker.Volume = volume

            stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
            stream.Format.Type = 22  # SAFT22kHz16BitMono (22050 Hz, 16-bit mono PCM)
            speaker.AudioOutputStream = stream

            if self._is_cancelled(job_id, cancel_token):
                return False

            speaker.Speak(clean_text)
            pcm_bytes = bytes(stream.GetData())
        except Exception:
            return False

        if not pcm_bytes or self._is_cancelled(job_id, cancel_token):
            return False

        # Apply software volume scaling in RAM if volume < 100
        if volume < 100:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            samples *= (volume / 100.0)
            pcm_bytes = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

        synth_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._is_cancelled(job_id, cancel_token):
            return False

        if on_speech_started is not None:
            on_speech_started(job_id, synth_elapsed_ms)

        if not self._play_audio:
            return not self._is_cancelled(job_id, cancel_token)

        # Play PCM in small 2048-byte (~46ms) chunks via PyAudio for instant cancellation
        pa = None
        out_stream = None
        try:
            import pyaudio

            pa = pyaudio.PyAudio()
            out_stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=22050,
                output=True,
                frames_per_buffer=1024,
            )
            chunk_size = 2048  # 1024 int16 frames = ~46.4 ms
            for offset in range(0, len(pcm_bytes), chunk_size):
                if self._is_cancelled(job_id, cancel_token):
                    return False
                out_stream.write(pcm_bytes[offset : offset + chunk_size])
            return not self._is_cancelled(job_id, cancel_token)
        except Exception:
            return False
        finally:
            if out_stream is not None:
                try:
                    out_stream.stop_stream()
                    out_stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass
