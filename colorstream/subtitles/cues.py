"""Subtitle cues: parsing (SRT, WebVTT, ASS), writing, cleaning and frame snapping.

All times are seconds on the media timeline. Pure functions, no I/O except `read_file`.
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .languages import LANGUAGES, normalize_lang, script_of_text


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------- decoding

def decode_bytes(data: bytes, lang_hint: str = "auto") -> str:
    """Decode subtitle bytes. Old Greek files are usually cp1253/ISO-8859-7, so a language
    hint (or detection) matters: a wrong guess turns Greek into mojibake."""
    for bom, enc in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if data.startswith(bom):
            return data.decode(enc, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    lang = normalize_lang(lang_hint)
    if lang in LANGUAGES:
        order = LANGUAGES[lang][3] + ["cp1252"]
    else:
        # unknown language: try every legacy encoding and keep the most plausible text
        order = ["cp1253", "cp1251", "cp1250", "cp1254", "cp1256", "cp1255", "cp1252"]
    best, best_score = None, -1.0
    for enc in order:
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        letters = [c for c in text if c.isalpha()]
        if not letters:
            continue
        script = script_of_text(text)
        wanted = LANGUAGES[lang][2] if lang in LANGUAGES else script
        score = sum(1 for c in letters if script_of_text(c) == wanted) / len(letters)
        if score > best_score + 1e-9:
            best, best_score = text, score
    return best if best is not None else data.decode("latin-1")


# ---------------------------------------------------------------- parsing

_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_TIME_LINE = re.compile(rf"{_TS}\s*-->\s*{_TS}")


def _sec(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt_vtt(text: str) -> list[Cue]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    cues = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip("\n").split("\n")
        for i, ln in enumerate(lines):
            m = _TIME_LINE.search(ln)
            if m:
                g = m.groups()
                start, end = _sec(*g[0:4]), _sec(*g[4:8])
                body = "\n".join(lines[i + 1:])
                if body.strip():
                    cues.append(Cue(start, end, body))
                break
    return cues


def parse_ass(text: str) -> list[Cue]:
    cues = []
    fmt: list[str] = []
    in_events = False
    for ln in text.replace("\r\n", "\n").split("\n"):
        s = ln.strip()
        if s.lower().startswith("[events]"):
            in_events = True
            continue
        if s.startswith("[") and in_events:
            break
        if not in_events:
            continue
        if s.lower().startswith("format:"):
            fmt = [x.strip().lower() for x in s.split(":", 1)[1].split(",")]
        elif s.lower().startswith("dialogue:") and fmt:
            parts = s.split(":", 1)[1].lstrip().split(",", len(fmt) - 1)
            row = dict(zip(fmt, parts))
            try:
                h1, m1, s1 = row["start"].split(":")
                h2, m2, s2 = row["end"].split(":")
                start = int(h1) * 3600 + int(m1) * 60 + float(s1)
                end = int(h2) * 3600 + int(m2) * 60 + float(s2)
            except (KeyError, ValueError):
                continue
            body = row.get("text", "").replace("\\N", "\n").replace("\\n", "\n")
            if body.strip():
                cues.append(Cue(start, end, body))
    return cues


def parse(text: str) -> list[Cue]:
    head = text.lstrip("﻿")[:4000].lower()
    if "[script info]" in head or "[events]" in text.lower()[:20000]:
        return parse_ass(text)
    return parse_srt_vtt(text)


def read_file(path: str | Path, lang_hint: str = "auto") -> list[Cue]:
    return parse(decode_bytes(Path(path).read_bytes(), lang_hint))


# ---------------------------------------------------------------- cleaning

_TAGS = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}|\[[^\]]*\]\s*$")


def clean_text(text: str) -> str:
    """Strip markup, entities and music/annotation noise. Keeps line breaks."""
    t = html.unescape(_TAGS.sub("", text))
    t = re.sub(r"[♪♫♬]+", "", t)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def normalize(cues: list[Cue], fps: float | None = None, min_duration: float = 0.7, max_duration: float = 7.0,
              chars_per_second: float = 20.0, gap: float = 0.04,
              extend_until: float | None = None, extend: bool = True) -> list[Cue]:
    """Clean text, order, remove overlaps and give each cue a readable duration.

    `extend=False` keeps authored end times (existing subtitle files); recognizer output is extended
    to a comfortable reading time where there is room.

    With `fps`, times are snapped to whole frames so a cue appears and disappears on exact frames.
    """
    items = []
    for c in cues:
        t = clean_text(c.text)
        if t and c.end > c.start - 1e-9 and c.start >= 0:
            items.append(Cue(c.start, c.end, t))
    items.sort(key=lambda c: (c.start, c.end))
    out: list[Cue] = []
    for i, c in enumerate(items):
        nxt = items[i + 1].start if i + 1 < len(items) else math.inf
        need_end = c.start + min(max_duration, max(min_duration, len(c.text.replace("\n", "")) / chars_per_second))
        limit = nxt - gap
        if extend_until is not None:  # never grow a cue into a region that is not transcribed yet
            limit = min(limit, max(extend_until, c.end))
        end = max(c.end, min(need_end, limit)) if extend else c.end  # extend short cues only where there is room
        if end > limit:  # overlap with the next cue
            end = max(limit, c.start + 0.02)
        out.append(Cue(c.start, max(end, c.start + 0.02), c.text))
    # merge consecutive identical text (ASR often repeats)
    merged: list[Cue] = []
    for c in out:
        if merged and merged[-1].text == c.text and c.start - merged[-1].end < 0.5:
            merged[-1] = replace(merged[-1], end=c.end)
        else:
            merged.append(c)
    return snap_to_frames(merged, fps) if fps else merged


def snap_to_frames(cues: list[Cue], fps: float) -> list[Cue]:
    out = []
    last_end_f = -1
    for c in cues:
        s = max(int(round(c.start * fps)), last_end_f)
        e = max(int(round(c.end * fps)), s + 1)
        out.append(Cue(s / fps, e / fps, c.text))
        last_end_f = e
    return out


def frame_range(cue: Cue, fps: float) -> tuple[int, int]:
    """Frame indices [first, last_exclusive) in which this cue is visible."""
    return int(round(cue.start * fps)), int(round(cue.end * fps))


# ---------------------------------------------------------------- transforms

def shift(cues: list[Cue], offset: float) -> list[Cue]:
    return [Cue(max(0.0, c.start + offset), max(0.0, c.end + offset), c.text) for c in cues]


def retime(cues: list[Cue], scale: float, offset: float) -> list[Cue]:
    """t' = scale * t + offset (fixes both a constant delay and fps-mismatch drift)."""
    return [Cue(max(0.0, c.start * scale + offset), max(0.0, c.end * scale + offset), c.text) for c in cues]


def window(cues: list[Cue], t0: float, t1: float) -> list[Cue]:
    return [c for c in cues if c.end > t0 and c.start < t1]


# ---------------------------------------------------------------- writing

def _fmt(t: float, sep: str) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    return "".join(f"{i}\n{_fmt(c.start, ',')} --> {_fmt(c.end, ',')}\n{c.text}\n\n" for i, c in enumerate(cues, 1))


def to_vtt(cues: list[Cue]) -> str:
    body = "".join(f"{_fmt(c.start, '.')} --> {_fmt(c.end, '.')}\n{html.escape(c.text, quote=False)}\n\n" for c in cues)
    return "WEBVTT\n\n" + body
