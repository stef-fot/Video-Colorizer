"""Frame-exact subtitle rendering (burned into the picture) with Pillow.

Because the text is drawn on the same frames that get encoded, it is correct on every frame, in any
player, at any playback speed (2x, 4x ...). Works for Greek, Cyrillic, CJK and, with libraqm, Arabic/Hebrew.
"""
from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np

from .languages import NO_SPACE_SCRIPTS, script_of, script_of_text

log = logging.getLogger("colorstream.subtitles")

_WIN = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
FONT_CANDIDATES = {
    # first existing file wins
    "latin": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf",
              "Helvetica.ttc", "Arial.ttf"],
    "greek": ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
              "NotoSans-Regular.ttf", "Arial.ttf", "Helvetica.ttc"],
    "cyrillic": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf",
                 "Arial.ttf"],
    "arabic": ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "NotoSansArabic-Regular.ttf", "DejaVuSans.ttf", "Arial.ttf"],
    "hebrew": ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "NotoSansHebrew-Regular.ttf", "DejaVuSans.ttf", "Arial.ttf"],
    "indic": ["Nirmala.ttf", "NotoSansDevanagari-Regular.ttf", "Mangal.ttf", "Kohinoor.ttc", "DejaVuSans.ttf"],
    "thai": ["leelawui.ttf", "tahoma.ttf", "NotoSansThai-Regular.ttf", "Thonburi.ttc", "DejaVuSans.ttf"],
    "cjk": ["msyh.ttc", "YuGothM.ttc", "malgun.ttf", "meiryo.ttc", "NotoSansCJK-Regular.ttc",
            "NotoSansCJKsc-Regular.otf", "wqy-microhei.ttc", "PingFang.ttc", "Hiragino Sans GB.ttc",
            "AppleSDGothicNeo.ttc"],
}
FONT_DIRS = [_WIN, Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
             Path("/Library/Fonts"), Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"),
             Path.home() / "Library/Fonts"]

_font_path_cache: dict[tuple, str | None] = {}
_font_index: dict[str, str] | None = None


def _index_fonts() -> dict[str, str]:
    global _font_index
    if _font_index is None:
        _font_index = {}
        for d in FONT_DIRS:
            if d.exists():
                for p in d.rglob("*"):
                    if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
                        _font_index.setdefault(p.name.lower(), str(p))
    return _font_index


def find_font(script: str, override: str = "") -> str | None:
    """Path of a font that covers the script, or None."""
    if override and Path(override).exists():
        return override
    key = (script, override)
    if key in _font_path_cache:
        return _font_path_cache[key]
    found = None
    wanted = FONT_CANDIDATES.get(script, FONT_CANDIDATES["latin"])
    index = _index_fonts()
    for name in wanted:
        if name.lower() in index:
            found = index[name.lower()]
            break
    if found is None:
        log.warning("No font found for %s text. Set subtitles.font in config.yaml to a TTF/OTF that covers it.", script)
    _font_path_cache[key] = found
    return found


def wrap_text(text: str, measure: Callable[[str], float], max_width: float, script: str = "latin",
              max_lines: int = 3) -> list[str]:
    """Greedy line wrapping by pixel width. Breaks between words, or between any characters
    for scripts without spaces (Chinese, Japanese, Thai). Existing line breaks are kept."""
    out: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            continue
        if measure(para) <= max_width:
            out.append(para)
            continue
        tokens = list(para) if script in NO_SPACE_SCRIPTS else para.split(" ")
        joiner = "" if script in NO_SPACE_SCRIPTS else " "
        line = ""
        for tok in tokens:
            trial = tok if not line else line + joiner + tok
            if line and measure(trial) > max_width:
                out.append(line)
                line = tok
            else:
                line = trial
        if line:
            out.append(line)
    if len(out) > max_lines:  # keep the text, fold the overflow into the last line
        out = out[:max_lines - 1] + [" ".join(out[max_lines - 1:])]
    return out


class SubtitleRenderer:
    """Draws cue text at the bottom centre of BGR frames."""

    def __init__(self, frame_w: int, frame_h: int, size_fraction: float = 0.045, font_override: str = "",
                 lang: str = "auto"):
        from PIL import ImageFont, features

        self.w, self.h = frame_w, frame_h
        self.font_size = max(12, int(frame_h * size_fraction))
        self.override = font_override
        self.lang = lang
        self.raqm = features.check("raqm")
        self._fonts: dict[str, object] = {}
        self._ImageFont = ImageFont
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    def _font(self, script: str):
        if script not in self._fonts:
            path = find_font(script, self.override)
            engine = self._ImageFont.Layout.RAQM if self.raqm else self._ImageFont.Layout.BASIC
            try:
                self._fonts[script] = (self._ImageFont.truetype(path, self.font_size, layout_engine=engine)
                                       if path else self._ImageFont.load_default())
            except OSError:
                self._fonts[script] = self._ImageFont.load_default()
        return self._fonts[script]

    def _build(self, text: str):
        from PIL import Image, ImageDraw

        script = script_of(self.lang) if self.lang != "auto" else script_of_text(text)
        if script in ("arabic", "hebrew") and not self.raqm:
            log.warning("Pillow was built without libraqm: %s text will not be shaped correctly. "
                        "pip install --upgrade pillow (wheels include raqm).", script)
        font = self._font(script)
        stroke = max(2, self.font_size // 12)
        measure_img = ImageDraw.Draw(Image.new("L", (1, 1)))
        measure = lambda s: measure_img.textlength(s, font=font)  # noqa: E731
        max_w = self.w * 0.9 - 2 * stroke
        lines = wrap_text(text, measure, max_w, script)
        line_h = int(self.font_size * 1.2)
        img_w = int(min(self.w, max(measure(ln) for ln in lines) + 4 * stroke + 16))
        img_h = line_h * len(lines) + 2 * stroke + 8
        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, img_w - 1, img_h - 1], radius=8, fill=(0, 0, 0, 110))
        for i, ln in enumerate(lines):
            lw = measure(ln)
            x = (img_w - lw) / 2
            d.text((x, 4 + stroke + i * line_h), ln, font=font, fill=(255, 255, 255, 255),
                   stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        arr = np.asarray(img)
        bgr = np.ascontiguousarray(arr[..., 2::-1]).astype(np.float32)
        alpha = (arr[..., 3:4].astype(np.float32) / 255.0)
        return bgr * alpha, 1.0 - alpha  # premultiplied color, inverse alpha

    def overlay_for(self, text: str):
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        item = self._build(text)
        self._cache[text] = item
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return item

    def draw(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Blend text onto `frame` (BGR uint8, modified in place and returned)."""
        pre, inv = self.overlay_for(text)
        oh, ow = pre.shape[:2]
        x = max(0, (self.w - ow) // 2)
        y = max(0, self.h - oh - int(self.h * 0.05))
        region = frame[y:y + oh, x:x + ow]
        h, w = region.shape[:2]
        region[...] = np.clip(region.astype(np.float32) * inv[:h, :w] + pre[:h, :w], 0, 255).astype(np.uint8)
        return frame
