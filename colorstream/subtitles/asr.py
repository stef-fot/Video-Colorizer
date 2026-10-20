"""Speech recognition backends that turn audio into timed cues (all major languages, Greek included).

The default backend is faster-whisper (multilingual Whisper on CTranslate2), installed separately:
    pip install -r requirements-subtitles.txt
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

import numpy as np

from .cues import Cue
from .languages import NO_SPACE_SCRIPTS, script_of

log = logging.getLogger("colorstream.asr")

SENTENCE_END = re.compile(r"[.!?;…·。！？]$")
SOFT_BREAK = re.compile(r"[,:，、]$")

# Whisper tends to invent these over silence or music.
HALLUCINATIONS = {
    "thanks for watching", "thank you for watching", "subtitles by the amara.org community",
    "υποτιτλισμός", "υπότιτλοι", "ευχαριστώ που παρακολουθήσατε", "ευχαριστούμε που παρακολουθήσατε",
    "sous-titres réalisés para la communauté d'amara.org", "untertitel der amara.org-community",
    "gracias por ver el video", "subtítulos realizados por la comunidad de amara.org",
}


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class AsrResult:
    cues: list = field(default_factory=list)  # times relative to the start of the audio window
    language: str = "auto"
    language_probability: float = 0.0


def words_to_cues(words: list[Word], language: str = "en", max_chars: int = 74, max_duration: float = 6.0,
                  max_gap: float = 0.8) -> list[Cue]:
    """Group timed words into readable cues, preferring to break at sentence ends and pauses."""
    joiner = "" if script_of(language) in NO_SPACE_SCRIPTS else " "
    cues: list[Cue] = []
    cur: list[Word] = []

    def flush():
        nonlocal cur
        if cur:
            text = joiner.join(w.text.strip() for w in cur).strip()
            if text:
                cues.append(Cue(cur[0].start, cur[-1].end, text))
        cur = []

    for w in words:
        if not w.text.strip():
            continue
        if cur:
            gap = w.start - cur[-1].end
            length = len(joiner.join(x.text.strip() for x in cur)) + len(w.text.strip()) + 1
            span = w.end - cur[0].start
            if gap > max_gap or length > max_chars or span > max_duration:
                flush()
            elif length > max_chars * 0.6 and SOFT_BREAK.search(cur[-1].text.strip()):
                flush()
        cur.append(w)
        if SENTENCE_END.search(w.text.strip()):
            flush()
    flush()
    return cues


def is_hallucination(text: str, no_speech_prob: float, avg_logprob: float, compression_ratio: float) -> bool:
    t = text.strip().lower().rstrip(".!")
    if t in HALLUCINATIONS:
        return True
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return True
    if compression_ratio > 2.4:  # repetitive loops
        return True
    return False


def drop_repeats(cues: list[Cue], limit: int = 3) -> list[Cue]:
    """Remove runs of the same text longer than `limit` (a typical Whisper failure mode)."""
    out, run = [], 0
    for c in cues:
        run = run + 1 if out and out[-1].text == c.text else 1
        if run <= limit:
            out.append(c)
    return out


class AsrBackend:
    name = "asr"

    def transcribe(self, samples: np.ndarray, language: str = "auto") -> AsrResult:  # pragma: no cover
        raise NotImplementedError


class FasterWhisperBackend(AsrBackend):
    name = "faster-whisper"

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "auto",
                 prompt: str = ""):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "Speech recognition needs faster-whisper. Install it with: "
                "pip install -r requirements-subtitles.txt") from e
        log.info("Loading speech model '%s' (device=%s, compute=%s)...", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.prompt = prompt
        self.lock = threading.Lock()

    def transcribe(self, samples: np.ndarray, language: str = "auto") -> AsrResult:
        with self.lock:
            segments, info = self.model.transcribe(
                samples, language=None if language == "auto" else language, word_timestamps=True,
                vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
                condition_on_previous_text=False, initial_prompt=self.prompt or None, beam_size=5)
            lang = info.language
            cues: list[Cue] = []
            for seg in segments:
                if is_hallucination(seg.text, seg.no_speech_prob, seg.avg_logprob, seg.compression_ratio):
                    continue
                words = [Word(w.start, w.end, w.word) for w in (seg.words or [])]
                if words:
                    cues.extend(words_to_cues(words, lang))
                elif seg.text.strip():
                    cues.append(Cue(seg.start, seg.end, seg.text.strip()))
        return AsrResult(drop_repeats(cues), lang, float(info.language_probability))


_backends: dict[tuple, AsrBackend] = {}
_lock = threading.Lock()


def get_backend(model: str, device: str, compute_type: str, prompt: str = "") -> AsrBackend:
    key = (model, device, compute_type, prompt)
    with _lock:
        if key not in _backends:
            _backends[key] = FasterWhisperBackend(model, device, compute_type, prompt)
        return _backends[key]
