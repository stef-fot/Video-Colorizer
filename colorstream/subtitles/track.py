"""SubtitleTrack: the subtitles of one video, from existing files or from speech recognition.

The track fills itself in a background thread, working ahead of the playhead (and faster when the
viewer plays at 2x or 4x). Cues are stored as exact frame ranges, so burning them in is correct on
every single frame.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import re
import subprocess
import threading
import time
from pathlib import Path

from ..cache import SegmentCache
from ..config import Config, ProcSettings
from ..ffmpeg import MediaInfo, input_args, is_remote
from . import cues as cu
from .align import Alignment, audio_energy_db, estimate, mask_from_db, decode_audio
from .cues import Cue
from .languages import iso3, normalize_lang, script_of_text, LANGUAGES

log = logging.getLogger("colorstream.subtitles")

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "subviewer", "microdvd"}
SIDECAR_EXT = (".srt", ".vtt", ".ass", ".ssa")


def _script_to_lang(script: str) -> str:
    return {"greek": "el", "cyrillic": "ru", "hebrew": "he", "arabic": "ar", "indic": "hi", "thai": "th"}.get(script, "auto")


def find_sidecars(video: Path) -> list[tuple[str, Path]]:
    """Subtitle files next to a local video: (language code or 'auto', path)."""
    found = []
    stem = video.stem.lower()
    for p in sorted(video.parent.iterdir()):
        if p.suffix.lower() not in SIDECAR_EXT or not p.stem.lower().startswith(stem):
            continue
        tokens = re.split(r"[^a-z0-9]+", p.stem.lower()[len(stem):])
        lang = next((normalize_lang(t) for t in tokens if t and normalize_lang(t) in LANGUAGES), "auto")
        found.append((lang, p))
    return found


def list_subtitle_streams(cfg: Config, src: str, headers: dict | None) -> list[dict]:
    cmd = [cfg.ffprobe, "-v", "error", "-select_streams", "s", "-print_format", "json", "-show_streams"]
    if is_remote(src) and headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    res = subprocess.run(cmd + [src], capture_output=True, timeout=60)
    if res.returncode != 0:
        return []
    out = []
    for i, st in enumerate(json.loads(res.stdout or b"{}").get("streams", [])):
        tags = st.get("tags", {})
        out.append({"pos": i, "codec": st.get("codec_name", ""), "lang": normalize_lang(tags.get("language")),
                    "title": tags.get("title", ""), "forced": bool(st.get("disposition", {}).get("forced"))})
    return out


def extract_stream(cfg: Config, src: str, headers: dict | None, pos: int) -> list[Cue]:
    cmd = [cfg.ffmpeg, "-v", "error", "-nostdin"] + input_args(src, headers)
    cmd += ["-i", src, "-map", f"0:s:{pos}", "-f", "srt", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, timeout=1800)
    if res.returncode != 0:
        return []
    return cu.parse(cu.decode_bytes(res.stdout))


class SubtitleTrack:
    def __init__(self, cfg: Config, settings: ProcSettings, src: str, headers: dict | None, info: MediaInfo,
                 cache: SegmentCache, src_key: str, backend=None, subtitle_file: str = ""):
        self.cfg, self.s, self.src, self.headers, self.info = cfg, settings, src, headers, info
        self.fps = info.fps
        self.duration = info.duration
        self.file = subtitle_file
        self.lang = normalize_lang(settings.subtitle_lang)
        self.key = ""
        self.kind = ""  # 'existing' or 'asr' once decided
        self.status = "idle"
        self.error: str | None = None
        self.complete = False
        self.managed = False  # True when a streaming session sets the focus
        self.want_all = False
        self.speed = 1.0
        self.playhead = 0.0
        self.cond = threading.Condition()
        self.stop_ev = threading.Event()
        self.thread: threading.Thread | None = None
        self.backend = backend
        self.chunk = max(5.0, cfg.subs.chunk_seconds)
        self.nchunks = int(math.ceil(self.duration / self.chunk))
        self.done: set[int] = set()
        self._chunk_cues: dict[int, list[Cue]] = {}
        self._flat: list[Cue] = []
        self._f0: list[int] = []
        self._f1: list[int] = []
        self.alignment: Alignment | None = None
        sig = json.dumps([settings.subtitle_lang, settings.subtitle_source, subtitle_file, settings.subtitle_asr_model,
                          settings.subtitle_align, cfg.subs.chunk_seconds])
        if subtitle_file and Path(subtitle_file).exists():
            sig += str(Path(subtitle_file).stat().st_mtime_ns)
        self._path = cache.root / src_key / f"subs_{hashlib.sha1(sig.encode()).hexdigest()[:12]}.json"
        self._load()

    # ------------------------------------------------------------- persistence
    def _load(self) -> None:
        try:
            d = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.kind, self.lang, self.complete = d["kind"], d["lang"], d["complete"]
        self.done = set(d["done"])
        for k, v in d["chunks"].items():
            self._chunk_cues[int(k)] = [Cue(*c) for c in v]
        self._rebuild()
        if self.complete:
            self.status = "ready"

    def _save(self) -> None:
        d = {"kind": self.kind, "lang": self.lang, "complete": self.complete, "done": sorted(self.done),
             "chunks": {str(k): [[c.start, c.end, c.text] for c in v] for k, v in self._chunk_cues.items()}}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    # ------------------------------------------------------------- cue store
    def _rebuild(self) -> None:
        flat = [c for k in sorted(self._chunk_cues) for c in self._chunk_cues[k]]
        flat.sort(key=lambda c: c.start)
        out: list[Cue] = []
        for c in flat:  # cues from neighbouring chunks may touch: keep them strictly sequential
            if out and c.start < out[-1].end:
                prev = out[-1]
                new_end = max(prev.start + 1 / self.fps, c.start)
                out[-1] = Cue(prev.start, new_end, prev.text)
            out.append(c)
        self._flat = out
        self._f0 = [cu.frame_range(c, self.fps)[0] for c in out]
        self._f1 = [cu.frame_range(c, self.fps)[1] for c in out]

    def _add_chunk(self, idx: int, new: list[Cue]) -> None:
        s = self.cfg.subs
        c1 = (idx + 1) * self.chunk
        norm = cu.normalize(new, self.fps, s.min_duration, s.max_duration, s.chars_per_second, extend_until=c1)
        with self.cond:
            self._chunk_cues[idx] = norm
            self.done.add(idx)
            self._rebuild()
            self.cond.notify_all()

    def text_at_frame(self, idx: int) -> str | None:
        i = bisect.bisect_right(self._f0, idx) - 1
        if i >= 0 and idx < self._f1[i]:
            return self._flat[i].text
        return None

    def cues(self) -> list[Cue]:
        return list(self._flat)

    def covered(self, t0: float, t1: float) -> bool:
        if self.complete:
            return True
        a = int(max(t0, 0) // self.chunk)
        b = int(min(max(t1, t0), self.duration - 1e-6) // self.chunk)
        return all(i in self.done for i in range(a, b + 1))

    # ------------------------------------------------------------- control
    def start(self, want_all: bool = False) -> None:
        with self.cond:
            self.want_all = self.want_all or want_all
            if self.complete or (self.thread and self.thread.is_alive()):
                self.cond.notify_all()
                return
            self.stop_ev.clear()
            self.thread = threading.Thread(target=self._run, daemon=True, name="cs-subtitles")
            self.thread.start()

    def stop(self) -> None:
        self.stop_ev.set()
        with self.cond:
            self.cond.notify_all()

    def focus(self, t: float, speed: float | None = None) -> None:
        with self.cond:
            self.playhead = t
            if speed:
                self.speed = speed
            self.cond.notify_all()

    def ensure(self, t0: float, t1: float, cancel=None, timeout: float | None = None) -> bool:
        """Block until subtitles for [t0, t1] are available. False on timeout, error or cancel."""
        timeout = self.cfg.subs.wait_seconds if timeout is None else timeout
        if not self.managed:
            self.focus(t0)
        self.start()
        deadline = time.time() + timeout
        with self.cond:
            while not self.covered(t0, t1):
                if self.error or self.stop_ev.is_set() or (cancel is not None and cancel.is_set()):
                    return False
                if time.time() > deadline:
                    return False
                self.cond.wait(timeout=0.25)
        return True

    def wait_complete(self, timeout: float | None = None, on_progress=None) -> bool:
        self.start(want_all=True)
        deadline = None if timeout is None else time.time() + timeout
        with self.cond:
            while not self.complete:
                if self.error or self.stop_ev.is_set():
                    return False
                if deadline and time.time() > deadline:
                    return False
                if on_progress:
                    on_progress(len(self.done) / max(self.nchunks, 1), self.status)
                self.cond.wait(timeout=0.5)
        return True

    def summary(self) -> dict:
        return {"kind": self.kind or "pending", "language": self.lang, "status": self.status,
                "progress": round(len(self.done) / max(self.nchunks, 1), 3) if self.kind == "asr" else
                (1.0 if self.complete else 0.0), "cues": len(self._flat), "error": self.error,
                "aligned": ({"offset": round(self.alignment.offset, 2), "scale": round(self.alignment.scale, 5),
                             "score": round(self.alignment.score, 2)} if self.alignment else None)}

    # ------------------------------------------------------------- worker
    def _run(self) -> None:
        try:
            if not self.kind:
                self._decide_source()
            if self.kind == "asr" and not self.complete:
                self._run_asr()
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.status = "error"
            log.error("Subtitles failed: %s", self.error)
        finally:
            with self.cond:
                self.cond.notify_all()

    def _decide_source(self) -> None:
        mode = self.s.subtitle_source
        if mode in ("auto", "existing"):
            self.status = "looking for existing subtitles"
            found = self._find_existing()
            if found:
                cues, lang, origin = found
                self._use_existing(cues, lang, origin)
                return
            if mode == "existing":
                raise RuntimeError("No existing subtitles found (looked for subtitle_file, files next to the video "
                                   "and embedded text tracks). Use subtitle_source: asr to generate them.")
        self.kind = "asr"

    def _find_existing(self):
        want = self.lang
        if self.file:
            return cu.read_file(self.file, want), want, self.file
        local = not is_remote(self.src)
        if local:
            side = find_sidecars(Path(self.src))
            pick = [p for l, p in side if want == "auto" or l == want] or ([] if want != "auto" else [p for _, p in side])
            if pick:
                cues = cu.read_file(pick[0], want)
                if cues:
                    return cues, want, str(pick[0])
        if local or self.cfg.subs.extract_remote:
            streams = [s for s in list_subtitle_streams(self.cfg, self.src, self.headers)
                       if s["codec"] in TEXT_CODECS and not s["forced"]]
            pick = [s for s in streams if want == "auto" or s["lang"] == want]
            if pick:
                cues = extract_stream(self.cfg, self.src, self.headers, pick[0]["pos"])
                if cues:
                    return cues, pick[0]["lang"] if pick[0]["lang"] != "auto" else want, f"embedded #{pick[0]['pos']}"
        return None

    def _use_existing(self, cues: list[Cue], lang: str, origin: str) -> None:
        self.kind = "existing"
        cues = cu.normalize(cues, None, extend=False)
        if lang == "auto":
            lang = _script_to_lang(script_of_text(" ".join(c.text for c in cues[:200]))) 
            if lang == "auto":
                lang = "en"
        self.lang = lang
        if self.s.subtitle_align and (not is_remote(self.src) or self.cfg.subs.extract_remote):
            self.status = "aligning to audio"
            try:
                db = audio_energy_db(self.cfg, self.src, self.headers, cancel=self.stop_ev)
                self.alignment = estimate(mask_from_db(db), cues)
                a = self.alignment
                if a.significant:
                    log.info("Subtitles re-synced: scale %.5f offset %+.2fs (match %.0f%% -> %.0f%%)",
                             a.scale, a.offset, 100 * a.score_before, 100 * a.score)
                    cues = a.apply(cues)
                else:
                    log.info("Subtitles already in sync (match %.0f%%).", 100 * a.score)
            except Exception as e:  # noqa: BLE001
                log.warning("Could not align subtitles (%s), using them as they are.", e)
        cues = cu.normalize(cues, self.fps, extend=False)
        with self.cond:
            self._chunk_cues = {0: cues}
            self.done = set(range(self.nchunks))
            self._rebuild()
            self.complete = True
            self.status = "ready"
        self._save()
        log.info("Using %d existing subtitle cues from %s (language %s)", len(cues), origin, self.lang)

    def _asr(self):
        if self.backend is None:
            from .asr import get_backend
            sc = self.cfg.subs
            self.backend = get_backend(self.s.subtitle_asr_model, sc.device, sc.compute_type, sc.prompt)
        return self.backend

    def _next_chunk(self) -> int | None:
        horizon = self.cfg.subs.lookahead_seconds * max(1.0, self.speed)
        first = int(min(self.playhead, self.duration - 1e-3) // self.chunk)
        last = int(min(self.playhead + horizon, self.duration - 1e-3) // self.chunk)
        for i in range(first, last + 1):
            if i not in self.done:
                return i
        if self.want_all or self.cfg.subs.background_full:
            for i in range(self.nchunks):
                if i not in self.done:
                    return i
        return None

    def _run_asr(self) -> None:
        self.status = "loading speech model"
        backend = self._asr()
        while not self.stop_ev.is_set():
            with self.cond:
                idx = self._next_chunk()
                if idx is None:
                    if len(self.done) >= self.nchunks:
                        self.complete = True
                        self.status = "ready"
                        break
                    self.status = "waiting"
                    self.cond.wait(timeout=1.0)
                    continue
                self.status = f"recognizing {idx + 1}/{self.nchunks}"
            c0 = idx * self.chunk
            pad = 1.0 if c0 > 0 else 0.0
            samples = decode_audio(self.cfg, self.src, self.headers, c0 - pad, self.chunk + pad + 1.0)
            new: list[Cue] = []
            if len(samples) > 1600:
                res = backend.transcribe(samples, self.lang)
                if self.lang == "auto" and res.language_probability >= 0.6 and res.language != "auto":
                    self.lang = normalize_lang(res.language)
                    log.info("Detected language: %s (%.0f%%)", self.lang, 100 * res.language_probability)
                for c in res.cues:
                    start = c.start + c0 - pad
                    if c0 <= start < c0 + self.chunk:
                        new.append(Cue(start, c.end + c0 - pad, c.text))
            self._add_chunk(idx, new)
            self._save()
        if self.complete:
            self._save()

    # ------------------------------------------------------------- export
    def export(self, fmt: str = "vtt") -> str:
        return cu.to_vtt(self._flat) if fmt == "vtt" else cu.to_srt(self._flat)
