"""The core pipeline, shared by the HTTP server and the CLI.

decode (FFmpeg pipe, gray) -> downscale -> model chroma -> temporal smoothing
-> upscale chroma + original luminance -> encode (FFmpeg pipe).

`run_pass` processes one contiguous time range and writes either HLS
segments or a single MPEG-TS file.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .color import merge_luma_chroma
from .config import Config, ProcSettings
from .engine import ChromaEngine
from .ffmpeg import (
    Decoder, Encoder, FFmpegError, MediaInfo, audio_args_for_ts, decode_candidates, encoder_args,
    input_args, output_size, pick_encoder, raw_input_args,
)
from .models import load_model

log = logging.getLogger("colorstream.pipeline")


class Cancelled(Exception):
    pass


@dataclass
class PassSpec:
    src: str
    headers: dict | None
    info: MediaInfo
    settings: ProcSettings
    start: float  # first emitted frame time (seconds)
    # Exactly one of the two output kinds:
    hls_dir: Path | None = None
    hls_start_number: int = 0
    out_file: Path | None = None
    end: float | None = None  # stop emitting at this time (None = to the end)
    frame_limit: int | None = None  # alternative to end: emit exactly this many frames
    with_audio: bool = False  # HLS only: mux the source audio into the segments
    compare: bool = False  # side by side original | colorized
    cancel: threading.Event = field(default_factory=threading.Event)
    # Called with the media time of the next frame; may block (look-ahead throttle).
    throttle: Callable[[float], None] | None = None
    # HLS: called with the index of the segment about to start; True means stop.
    stop_at_segment: Callable[[int], bool] | None = None
    on_progress: Callable[[float, float], None] | None = None  # (media_time, fps)


@dataclass
class PassResult:
    frames: int = 0
    seconds: float = 0.0
    stopped_at: float = 0.0
    timings: dict = field(default_factory=dict)


def _cancel_aware_put(q: queue.Queue, item, cancel: threading.Event) -> None:
    while True:
        if cancel.is_set():
            raise Cancelled()
        try:
            q.put(item, timeout=0.25)
            return
        except queue.Full:
            continue


def _cancel_aware_get(q: queue.Queue, cancel: threading.Event):
    while True:
        if cancel.is_set():
            raise Cancelled()
        try:
            return q.get(timeout=0.25)
        except queue.Empty:
            continue


def build_encoder_cmd(cfg: Config, spec: PassSpec, frame_w: int, frame_h: int) -> list[str]:
    s = spec.settings
    fps = spec.info.fps
    encoder = pick_encoder(cfg)
    hls = spec.hls_dir is not None
    pre, post = encoder_args(cfg, encoder, s.crf, fps, s.segment_seconds if hls else None)
    cmd = [cfg.ffmpeg, "-v", "error", "-nostdin", "-y"] + pre
    cmd += raw_input_args(frame_w, frame_h, fps)
    audio = spec.with_audio and bool(spec.info.audio_codecs)
    if audio:
        cmd += input_args(spec.src, spec.headers)
        if spec.start > 0:
            cmd += ["-ss", f"{spec.start:.3f}"]
        cmd += ["-i", spec.src]
    cmd += ["-map", "0:v:0"]
    if audio:
        cmd += ["-map", "1:a?", "-sn"] + audio_args_for_ts(spec.info.audio_codecs) + ["-shortest"]
    else:
        cmd += ["-an"]
    cmd += post
    if hls:
        d = Path(spec.hls_dir)
        cmd += [
            "-output_ts_offset", f"{spec.hls_start_number * s.segment_seconds:.3f}",
            "-f", "hls", "-hls_time", f"{s.segment_seconds}", "-hls_list_size", "0",
            "-hls_flags", "temp_file+independent_segments", "-hls_segment_type", "mpegts",
            "-start_number", str(spec.hls_start_number),
            "-hls_segment_filename", str(d / "seg_%05d.ts"),
            str(d / f"_pass_{spec.hls_start_number}.m3u8"),
        ]
    else:
        cmd += ["-output_ts_offset", f"{spec.start:.3f}", "-f", "mpegts", str(spec.out_file)]
    return cmd


def _open_decoder(cfg: Config, spec: PassSpec, w: int, h: int, dstart: float, dur: float | None):
    last_err: Exception | None = None
    for hw in decode_candidates(cfg):
        dec = None
        try:
            dec = Decoder(cfg, spec.src, spec.headers, w, h, spec.info.fps, dstart, dur, hwaccel=hw)
            first = dec.read_frame()
            if first is None:
                dec.check()
                raise FFmpegError("No video frames could be decoded from this position.")
            return dec, first
        except FFmpegError as e:
            last_err = e
            if dec:
                dec.kill()
            if hw is not None:
                log.info("Hardware decode '%s' failed, retrying in software: %s", hw, e)
    raise last_err or FFmpegError("Decoder could not be started.")


def run_pass(cfg: Config, spec: PassSpec) -> PassResult:
    s = spec.settings
    info = spec.info
    fps = info.fps
    out_w, out_h = output_size(info, s.output_height)
    model = load_model(cfg, s.resolved_model())
    size = model.resolve_size(s.model_resolution)
    engine = ChromaEngine(model, s, size)

    warm = max(0, s.warmup_frames) if spec.start > 0 else 0
    dstart = max(0.0, spec.start - warm / fps)
    warm = int(round((spec.start - dstart) * fps))
    dur = None
    if spec.end is not None:
        dur = spec.end - dstart
    elif spec.frame_limit is not None:
        dur = (spec.frame_limit + warm + 2) / fps  # decoder may run slightly over, we stop by count

    batch_size = max(1, s.batch_size)
    q_in: queue.Queue = queue.Queue(maxsize=batch_size * 3)
    q_out: queue.Queue = queue.Queue(maxsize=batch_size * 3)
    errors: list = []
    cancel = spec.cancel
    abort = threading.Event()  # set on any failure so helper threads exit
    reader_stop = threading.Event()  # set when the main loop no longer needs frames
    writer_fail = threading.Event()
    reader_ev = _Either(cancel, abort, reader_stop)
    writer_ev = _Either(cancel, abort)
    main_put_ev = _Either(cancel, abort, writer_fail)

    decoder, first = _open_decoder(cfg, spec, out_w, out_h, dstart, dur)
    frame_w = out_w * (2 if spec.compare else 1)
    try:
        encoder = Encoder(build_encoder_cmd(cfg, spec, frame_w, out_h))
    except Exception:
        decoder.kill()
        raise

    def reader():
        try:
            _cancel_aware_put(q_in, first, reader_ev)
            while True:
                fr = decoder.read_frame()
                if fr is None:
                    if not reader_stop.is_set():
                        decoder.check()
                    break
                _cancel_aware_put(q_in, fr, reader_ev)
        except Cancelled:
            pass
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            try:
                q_in.put(None, timeout=1)
            except queue.Full:
                pass

    def writer():
        try:
            while True:
                item = _cancel_aware_get(q_out, writer_ev)
                if item is None:
                    break
                encoder.write(item)
        except Cancelled:
            pass
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            writer_fail.set()

    rt = threading.Thread(target=reader, daemon=True, name="cs-reader")
    wt = threading.Thread(target=writer, daemon=True, name="cs-writer")
    rt.start()
    wt.start()

    n = 0  # frames consumed since decode start
    emitted = 0
    t_start = time.perf_counter()
    t_render = 0.0
    last_report = 0.0
    seg_dur = s.segment_seconds
    cur_seg = spec.hls_start_number
    try:
        batch: list = []
        eof = False
        while not eof:
            frame = _cancel_aware_get(q_in, _Either(cancel, abort))
            if frame is None:
                eof = True
            else:
                batch.append(frame)
            if not batch or not (eof or len(batch) >= batch_size):
                continue
            if spec.throttle is not None and n >= warm:
                spec.throttle(dstart + n / fps)
            if cancel.is_set():
                raise Cancelled()
            smalls = [cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA) for g in batch]
            chromas = engine.process(smalls)
            t0 = time.perf_counter()
            stop_now = False
            for g, ab in zip(batch, chromas):
                t_frame = dstart + n / fps
                n += 1
                if n - 1 < warm:
                    continue
                if spec.frame_limit is not None and emitted >= spec.frame_limit:
                    stop_now = True
                    break
                if spec.end is not None and t_frame >= spec.end - 0.5 / fps:
                    stop_now = True
                    break
                if spec.hls_dir is not None:
                    si = int(t_frame / seg_dur + 1e-6)
                    if si > cur_seg:
                        cur_seg = si
                        if spec.stop_at_segment is not None and spec.stop_at_segment(si):
                            stop_now = True
                            break
                color = merge_luma_chroma(g, ab, s.saturation)
                if spec.compare:
                    color = np.hstack([cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), color])
                _cancel_aware_put(q_out, color.tobytes(), main_put_ev)
                emitted += 1
            t_render += time.perf_counter() - t0
            batch = []
            now = time.perf_counter()
            if spec.on_progress and now - last_report > 0.5:
                last_report = now
                spec.on_progress(dstart + n / fps, emitted / max(now - t_start, 1e-6))
            if stop_now:
                break
    except BaseException as e:
        abort.set()
        decoder.kill()
        encoder.kill()
        if errors and isinstance(e, Cancelled) and not cancel.is_set():
            raise errors[0]
        raise
    reader_stop.set()
    decoder.kill()
    if errors:
        abort.set()
        encoder.kill()
        raise errors[0]
    try:
        _cancel_aware_put(q_out, None, main_put_ev)
    except Cancelled:
        abort.set()
        encoder.kill()
        if errors:
            raise errors[0]
        raise
    wt.join(timeout=300)
    if errors:
        encoder.kill()
        raise errors[0]
    encoder.finish()
    elapsed = max(time.perf_counter() - t_start, 1e-6)
    res = PassResult(
        frames=emitted, seconds=elapsed, stopped_at=dstart + n / fps,
        timings={"infer": engine.stats.infer_seconds, "temporal": engine.stats.temporal_seconds,
                 "render": t_render, "cuts": engine.stats.cuts, "inferred_frames": engine.stats.inferred})
    if spec.on_progress:
        spec.on_progress(res.stopped_at, emitted / elapsed)
    return res


class _Either:
    """Event-like object that is set when any of the given events is set."""

    def __init__(self, *events: threading.Event):
        self.events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self.events)
