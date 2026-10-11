"""Standalone file conversion with resume, built on the shared pipeline."""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .config import Config, ProcSettings
from .detect import source_is_color
from .ffmpeg import FFmpegError, MediaInfo, probe
from .pipeline import PassSpec, run_pass

log = logging.getLogger("colorstream.convert")

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".ts", ".webm", ".wmv", ".flv"}


class AlreadyColor(Exception):
    pass


@dataclass
class ConvertOptions:
    chunk_seconds: float = 120.0
    resume: bool = True
    compare: bool = False
    keep_audio: bool = True
    keep_subs: bool = True
    force: bool = False
    keep_temp: bool = False


def _job_signature(src: Path, settings: ProcSettings, opts: ConvertOptions, chunk_frames: int) -> dict:
    st = src.stat()
    return {"src": str(src.resolve()), "size": st.st_size, "mtime": int(st.st_mtime),
            "settings": settings.signature(), "compare": opts.compare, "chunk_frames": chunk_frames}


def convert_file(cfg: Config, src: str | Path, dst: str | Path, settings: ProcSettings,
                 opts: ConvertOptions | None = None,
                 progress: Callable[[float, float, float], None] | None = None,
                 cancel: threading.Event | None = None) -> Path:
    """Colorize `src` into `dst`. progress(done_seconds, total_seconds, fps)."""
    opts = opts or ConvertOptions()
    src, dst = Path(src), Path(dst)
    cancel = cancel or threading.Event()
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")
    if src.resolve() == dst.resolve():
        raise ValueError("Output must be a different file than the input.")
    info = probe(cfg, str(src))

    if settings.detect_color and not (opts.force or settings.force_colorize):
        is_color, spreads = source_is_color(cfg, str(src), None, info, settings.color_threshold)
        if is_color:
            raise AlreadyColor(f"{src.name} already looks like color footage (use --force to colorize anyway).")

    fps = info.fps
    chunk_frames = max(int(round(opts.chunk_seconds * fps)), int(fps))
    total_frames = int(math.ceil(info.duration * fps - 1e-6))
    nchunks = int(math.ceil(total_frames / chunk_frames))
    job = dst.with_name(dst.name + ".colorstream-job")
    job.mkdir(parents=True, exist_ok=True)
    meta_path = job / "job.json"
    sig = _job_signature(src, settings, opts, chunk_frames)
    done: set[int] = set()
    if opts.resume and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("signature") == sig:
                done = {i for i in meta.get("done", []) if (job / f"part_{i:05d}.ts").exists()}
                if done:
                    log.info("Resuming: %d of %d chunks already finished", len(done), nchunks)
            else:
                log.info("Settings or input changed since the interrupted job, starting over.")
        except Exception:
            pass
    if not done:
        for f in job.glob("part_*.ts"):
            f.unlink()
    meta_path.write_text(json.dumps({"signature": sig, "done": sorted(done)}))

    t_total = info.duration
    base_done = 0.0
    for i in range(nchunks):
        start_frame = i * chunk_frames
        n_frames = min(chunk_frames, total_frames - start_frame)
        chunk_dur = n_frames / fps
        if i in done:
            base_done += chunk_dur
            continue
        part = job / f"part_{i:05d}.ts"
        tmp = job / f"part_{i:05d}.tmp.ts"
        spec = PassSpec(
            src=str(src), headers=None, info=info, settings=settings, start=start_frame / fps,
            out_file=tmp, frame_limit=n_frames, compare=opts.compare, cancel=cancel,
            on_progress=(lambda t, f, b=base_done, s=start_frame / fps:
                         progress(b + max(0.0, t - s), t_total, f)) if progress else None)
        run_pass(cfg, spec)
        tmp.replace(part)
        done.add(i)
        base_done += chunk_dur
        meta_path.write_text(json.dumps({"signature": sig, "done": sorted(done)}))

    _mux(cfg, info, src, dst, job, nchunks, opts)
    if not opts.keep_temp:
        shutil.rmtree(job, ignore_errors=True)
    if progress:
        progress(t_total, t_total, 0.0)
    return dst


def _mux(cfg: Config, info: MediaInfo, src: Path, dst: Path, job: Path, nchunks: int, opts: ConvertOptions) -> None:
    listing = job / "parts.txt"
    def entry(i: int) -> str:
        path = (job / f"part_{i:05d}.ts").resolve().as_posix().replace("'", "'\\''")
        return f"file '{path}'\n"

    listing.write_text("".join(entry(i) for i in range(nchunks)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    is_mp4 = dst.suffix.lower() in (".mp4", ".m4v", ".mov")

    def attempt(audio_mode: str, subs: bool) -> subprocess.CompletedProcess:
        cmd = [cfg.ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
               "-i", str(src), "-map", "0:v:0"]
        if opts.keep_audio and info.audio_codecs:
            cmd += ["-map", "1:a?"]
        if subs and opts.keep_subs and info.subtitle_count:
            cmd += ["-map", "1:s?"]
        cmd += ["-map_metadata", "1", "-map_chapters", "1", "-c:v", "copy"]
        cmd += ["-c:a", "copy"] if audio_mode == "copy" else ["-c:a", "aac", "-b:a", "256k"]
        if subs and opts.keep_subs and info.subtitle_count:
            cmd += ["-c:s", "mov_text" if is_mp4 else "copy"]
        if is_mp4:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(dst))
        return subprocess.run(cmd, capture_output=True)

    res = attempt("copy", True)
    if res.returncode != 0:
        log.warning("Muxing with original audio/subtitles copied failed (%s). Retrying with AAC audio and "
                    "without subtitles.", res.stderr.decode("utf-8", "replace").strip().splitlines()[-1:])
        res = attempt("aac", False)
    if res.returncode != 0:
        raise FFmpegError("Final muxing failed: " + res.stderr.decode("utf-8", "replace").strip())


def find_videos(folder: Path, recursive: bool = False) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
