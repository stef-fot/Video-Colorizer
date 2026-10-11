"""FFmpeg helpers: probing, hardware detection, decoder and encoder processes.

Frames travel over pipes (raw gray in, raw BGR out). No temporary images.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import Config

log = logging.getLogger("colorstream.ffmpeg")


class FFmpegError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    vcodec: str
    audio_codecs: list = field(default_factory=list)
    subtitle_count: int = 0
    is_remote: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d: dict) -> "MediaInfo":
        return MediaInfo(**d)


def check_binaries(cfg: Config) -> None:
    for exe in (cfg.ffmpeg, cfg.ffprobe):
        if shutil.which(exe) is None and not Path(exe).exists():
            raise FFmpegError(
                f"'{exe}' was not found. Install FFmpeg and make sure it is on your PATH "
                f"(Windows: 'winget install Gyan.FFmpeg'), or set ffmpeg/ffprobe in config.yaml.")


def is_remote(src: str) -> bool:
    return src.lower().startswith(("http://", "https://"))


def input_args(src: str, headers: dict | None = None) -> list[str]:
    args: list[str] = []
    if is_remote(src):
        args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
        if headers:
            blob = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            args += ["-headers", blob]
    return args


def probe(cfg: Config, src: str, headers: dict | None = None, timeout: float = 45.0) -> MediaInfo:
    cmd = [cfg.ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format"]
    if is_remote(src) and headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd.append(src)
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"Timed out probing the source after {timeout:.0f}s. Is the URL reachable?") from e
    if res.returncode != 0:
        msg = res.stderr.decode("utf-8", "replace").strip().splitlines()
        raise FFmpegError(f"Cannot open source: {msg[-1] if msg else 'ffprobe failed'}")
    data = json.loads(res.stdout or b"{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"
                  and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
    if video is None:
        raise FFmpegError("The source has no video stream.")
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or 25.0
    fps = min(max(fps, 5.0), 60.0)
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise FFmpegError("Cannot determine the duration of this source (live streams are not supported).")
    return MediaInfo(
        duration=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        vcodec=video.get("codec_name", ""),
        audio_codecs=[s.get("codec_name", "") for s in data["streams"] if s.get("codec_type") == "audio"],
        subtitle_count=sum(1 for s in data["streams"] if s.get("codec_type") == "subtitle"),
        is_remote=is_remote(src),
    )


def _parse_rate(v) -> float:
    try:
        f = float(Fraction(v))
        return f if f > 0 else 0.0
    except Exception:
        return 0.0


def output_size(info: MediaInfo, output_height: int) -> tuple[int, int]:
    """Even width and height keeping the aspect ratio. 0 keeps source height."""
    h = info.height if output_height <= 0 else min(output_height, info.height)
    w = int(round(info.width * h / info.height))
    return max(2, w // 2 * 2), max(2, h // 2 * 2)


# -- hardware detection ------------------------------------------------------

_hw_lock = threading.Lock()
_encoder_cache: dict[str, str] = {}


def _smoke_test_encoder(cfg: Config, encoder: str) -> bool:
    cmd = [cfg.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=256x256:d=0.2:r=10"]
    if encoder == "h264_vaapi":
        cmd = [cfg.ffmpeg, "-v", "error", "-vaapi_device", cfg.encode.vaapi_device, "-f", "lavfi",
               "-i", "color=c=gray:s=256x256:d=0.2:r=10", "-vf", "format=nv12,hwupload"]
    cmd += ["-frames:v", "2", "-c:v", encoder, "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def pick_encoder(cfg: Config) -> str:
    """Choose a working H.264 encoder. Hardware first, libx264 as the safe fallback."""
    wanted = cfg.encode.video_encoder
    with _hw_lock:
        if wanted in _encoder_cache:
            return _encoder_cache[wanted]
        if wanted != "auto":
            chosen = wanted
        else:
            chosen = "libx264"
            for cand in ("h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_amf", "h264_vaapi"):
                if _smoke_test_encoder(cfg, cand):
                    chosen = cand
                    break
        log.info("Video encoder: %s", chosen)
        _encoder_cache[wanted] = chosen
        return chosen


def encoder_args(cfg: Config, encoder: str, crf: int, fps: float, keyint_seconds: float | None) -> tuple[list, list]:
    """Returns (pre_input_args, codec_args). keyint_seconds forces IDR frames for HLS."""
    pre: list[str] = []
    post: list[str] = []
    if encoder == "libx264":
        post = ["-c:v", "libx264", "-preset", cfg.encode.preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    elif encoder == "h264_nvenc":
        post = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf + 2), "-b:v", "0",
                "-pix_fmt", "yuv420p"]
        if keyint_seconds:
            post += ["-forced-idr", "1"]
    elif encoder == "h264_videotoolbox":
        post = ["-c:v", "h264_videotoolbox", "-q:v", str(max(30, 100 - crf * 2)), "-pix_fmt", "yuv420p"]
    elif encoder == "h264_qsv":
        post = ["-c:v", "h264_qsv", "-global_quality", str(crf + 2), "-preset", "veryfast"]
    elif encoder == "h264_amf":
        post = ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf + 2),
                "-pix_fmt", "yuv420p"]
    elif encoder == "h264_vaapi":
        pre = ["-vaapi_device", cfg.encode.vaapi_device]
        post = ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", str(crf + 1)]
    else:
        post = ["-c:v", encoder, "-pix_fmt", "yuv420p"]
    if keyint_seconds:
        post += ["-force_key_frames", f"expr:gte(t,n_forced*{keyint_seconds})", "-sc_threshold", "0"] \
            if encoder == "libx264" else ["-force_key_frames", f"expr:gte(t,n_forced*{keyint_seconds})"]
    return pre, post


TS_COPYABLE_AUDIO = {"aac", "ac3", "eac3", "mp3", "mp2", "dts"}


def audio_args_for_ts(audio_codecs: list) -> list[str]:
    """Copy audio when MPEG-TS can carry it, otherwise transcode to AAC."""
    if all(c in TS_COPYABLE_AUDIO for c in audio_codecs):
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", "256k"]


# -- processes ---------------------------------------------------------------

class _StderrTail:
    """Drain stderr in a thread so the pipe never blocks; keep the last lines."""

    def __init__(self, stream):
        self.lines: deque = deque(maxlen=40)
        self._t = threading.Thread(target=self._run, args=(stream,), daemon=True)
        self._t.start()

    def _run(self, stream):
        for line in iter(stream.readline, b""):
            self.lines.append(line.decode("utf-8", "replace").rstrip())

    def text(self) -> str:
        return "\n".join(self.lines)


class Decoder:
    """Decodes `src` to raw gray frames at a constant frame rate and fixed size."""

    def __init__(self, cfg: Config, src: str, headers: dict | None, width: int, height: int, fps: float,
                 start: float, duration: float | None, hwaccel: str | None = None):
        self.cfg = cfg
        self.width, self.height = width, height
        self.frame_bytes = width * height
        cmd = [cfg.ffmpeg, "-v", "error", "-nostdin"]
        if hwaccel and hwaccel != "none":
            cmd += ["-hwaccel", hwaccel]
        cmd += input_args(src, headers)
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", src]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-an", "-sn", "-dn", "-vf", f"fps={fps:.5f},scale={width}:{height}:flags=bicubic,format=gray",
                "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.err = _StderrTail(self.proc.stderr)

    def read_frame(self) -> np.ndarray | None:
        buf = bytearray(self.frame_bytes)
        view = memoryview(buf)
        got = 0
        while got < self.frame_bytes:
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                return None  # end of stream (or decoder died, see check())
            got += n
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width)

    def check(self) -> None:
        """Call after EOF: raise if the decoder failed (as opposed to finishing normally)."""
        try:
            rc = self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.kill()
            return
        if rc != 0:
            raise FFmpegError(f"Decoder failed: {self.err.text() or 'unknown error'}")

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass


class Encoder:
    """Takes raw BGR frames on stdin and writes HLS segments or a single file."""

    def __init__(self, cmd: list):
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, bufsize=0)
        self.err = _StderrTail(self.proc.stderr)

    def write(self, data: bytes) -> None:
        try:
            self.proc.stdin.write(data)
        except (BrokenPipeError, OSError) as e:
            raise FFmpegError(f"Encoder stopped unexpectedly: {self.err.text() or e}") from e

    def finish(self, timeout: float = 120.0) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise FFmpegError("Encoder did not finish in time.")
        if rc != 0:
            raise FFmpegError(f"Encoder failed: {self.err.text() or 'unknown error'}")

    def kill(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass


def raw_input_args(width: int, height: int, fps: float) -> list[str]:
    return ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", f"{fps:.5f}", "-i", "pipe:0"]


def decode_candidates(cfg: Config) -> list:
    """Hardware decode first, software as the fallback."""
    mode = cfg.encode.hwaccel_decode
    if mode == "none":
        return [None]
    if mode == "auto":
        return ["auto", None]
    return [mode, None]
