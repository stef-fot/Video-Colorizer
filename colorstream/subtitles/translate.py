"""Machine translation of subtitle text between the 21 supported languages.

Backends (config `subs.translator`):
  nllb  : Meta NLLB-200 distilled 600M through CTranslate2 (int8, runs on CPU or GPU). One model covers all
          languages, so no pivot through English. Files: scripts/download_translation.py
  argos : Argos Translate (pip install argostranslate). Smaller, installs language packs on demand,
          pivots through English, and does not cover every language (Marathi, Telugu, Tamil, Swahili are weak/missing).

Translation only changes the text: cue times are copied from the source, so every cue stays on its frame.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from .cues import Cue
from .languages import NLLB_CODES, normalize_lang

log = logging.getLogger("colorstream.translate")

_gpu_lock = threading.Lock()  # one translation job at a time: it is CPU/GPU heavy


class Translator:
    name = "translator"

    def supports(self, src: str, tgt: str) -> bool:  # pragma: no cover
        return True

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError


class NllbTranslator(Translator):
    name = "nllb"

    def __init__(self, model_dir: str, device: str = "auto", compute_type: str = "int8", beam_size: int = 3):
        try:
            import ctranslate2
            import sentencepiece as spm
        except ImportError as e:
            raise RuntimeError("Translation needs ctranslate2 and sentencepiece: "
                               "pip install -r requirements-subtitles.txt") from e
        d = Path(model_dir)
        spm_path = d / "sentencepiece.bpe.model"
        if not (d / "model.bin").exists() or not spm_path.exists():
            raise RuntimeError(
                f"Translation model not found in {d}. Download it once with: "
                f"python scripts/download_translation.py   (about 620 MB), or set subs.nllb_dir.")
        self.translator = ctranslate2.Translator(str(d), device=device, compute_type=compute_type)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(spm_path))
        self.beam = beam_size

    def supports(self, src: str, tgt: str) -> bool:
        return src in NLLB_CODES and tgt in NLLB_CODES

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if not self.supports(src, tgt):
            raise ValueError(f"NLLB has no code for {src} -> {tgt}")
        s_code, t_code = NLLB_CODES[src], NLLB_CODES[tgt]
        sources = [[s_code] + self.sp.encode(t, out_type=str) + ["</s>"] for t in texts]
        with _gpu_lock:
            results = self.translator.translate_batch(
                sources, target_prefix=[[t_code]] * len(sources), beam_size=self.beam,
                max_batch_size=16, max_decoding_length=256, repetition_penalty=1.1, no_repeat_ngram_size=4)
        out = []
        for r in results:
            tokens = r.hypotheses[0]
            if tokens and tokens[0] == t_code:
                tokens = tokens[1:]
            out.append(self.sp.decode(tokens))
        return out


class ArgosTranslator(Translator):
    name = "argos"

    def __init__(self):
        try:
            import argostranslate.package as pkg
            import argostranslate.translate as tr
        except ImportError as e:
            raise RuntimeError("pip install argostranslate  (or use the nllb translator)") from e
        self.pkg, self.tr = pkg, tr
        self._ready: set[tuple[str, str]] = set()

    def _ensure(self, a: str, b: str) -> None:
        if (a, b) in self._ready:
            return
        installed = {(p.from_code, p.to_code) for p in self.pkg.get_installed_packages()}
        if (a, b) not in installed:
            self.pkg.update_package_index()
            cand = next((p for p in self.pkg.get_available_packages() if p.from_code == a and p.to_code == b), None)
            if cand is None:
                raise ValueError(f"Argos has no language pack for {a} -> {b}")
            self.pkg.install_from_path(cand.download())
        self._ready.add((a, b))

    def supports(self, src: str, tgt: str) -> bool:
        return True  # decided when the packs are looked up

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        with _gpu_lock:
            if src == "en" or tgt == "en":
                self._ensure(src, tgt)
            else:  # pivot through English
                self._ensure(src, "en")
                self._ensure("en", tgt)
            return [self.tr.translate(t, src, tgt) for t in texts]


_translators: dict[tuple, Translator] = {}
_lock = threading.Lock()


def get_translator(cfg) -> Translator:
    sc = cfg.subs
    key = (sc.translator, sc.nllb_dir, sc.translate_device)
    with _lock:
        if key not in _translators:
            if sc.translator == "argos":
                _translators[key] = ArgosTranslator()
            else:
                _translators[key] = NllbTranslator(sc.nllb_dir, sc.translate_device, sc.translate_compute_type)
        return _translators[key]


# ------------------------------------------------------------------ cue level

_SPACE = re.compile(r"\s+")


def _one_line(text: str) -> str:
    return _SPACE.sub(" ", text.replace("\n", " ")).strip()


def sane(src: str, out: str) -> bool:
    """Reject empty output and runaway repetition, which small translation models can produce."""
    if not out.strip():
        return False
    return len(out) <= 4 * len(src) + 30


def translate_cues(cues: list[Cue], src: str, tgt: str, translator: Translator, batch: int = 24) -> list[Cue]:
    """Translate the text of every cue, keep its timing. A cue the model fails on keeps the source text."""
    src, tgt = normalize_lang(src), normalize_lang(tgt)
    if src == tgt or not cues:
        return list(cues)
    lines = [_one_line(c.text) for c in cues]
    out: list[str] = []
    for i in range(0, len(lines), batch):
        part = lines[i:i + batch]
        try:
            out += translator.translate(part, src, tgt)
        except Exception as e:  # noqa: BLE001
            log.warning("Translation %s->%s failed for %d cues (%s), keeping source text", src, tgt, len(part), e)
            out += part
    return [Cue(c.start, c.end, o if sane(l, o) else l) for c, l, o in zip(cues, lines, out)]
