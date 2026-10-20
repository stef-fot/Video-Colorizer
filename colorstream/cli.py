"""Command line interface: serve, convert, batch, benchmark, models, check."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from . import __version__
from .config import Config, ProcSettings, apply_overrides, load_config
from .convert import AlreadyColor, ConvertOptions, convert_file, find_videos
from .ffmpeg import FFmpegError, check_binaries, pick_encoder
from .models import ModelNotFoundError, available_models

log = logging.getLogger("colorstream")

# Quality presets for the CLI. Streaming uses config.yaml instead.
PRESETS = {
    "fast": dict(mode="fast", model_resolution=256, inference_stride=2, optical_flow=True, smoothing=0.7, crf=23),
    "balanced": dict(mode="fast", model_resolution=384, inference_stride=1, optical_flow=False, smoothing=0.7, crf=21),
    "high": dict(mode="quality", model_resolution=512, inference_stride=1, optical_flow=True, smoothing=0.75, crf=18),
}


def settings_from_args(cfg: Config, a: argparse.Namespace) -> ProcSettings:
    s = replace(cfg.processing, **PRESETS[a.quality])
    # Files are converted offline: quality matters more than latency.
    s = replace(s, output_height=0, detect_color=True)
    ov = {"subtitles": a.subs, "subtitle_lang": a.sub_lang, "subtitle_source": a.sub_source,
          "subtitle_asr_model": a.asr_model, "subtitle_align": False if a.no_sub_align else None,
          "model": a.model, "saturation": a.saturation, "output_height": a.resolution,
          "model_resolution": a.model_res, "smoothing": a.smoothing, "inference_stride": a.stride,
          "optical_flow": a.flow, "batch_size": a.batch}
    s = apply_overrides(s, ov)
    if a.crf is not None:
        s = replace(s, crf=a.crf)
    if a.sub_size:
        s = replace(s, subtitle_size=a.sub_size)
    if a.encoder:
        cfg.encode.video_encoder = a.encoder
    return s


def _add_proc_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--quality", choices=sorted(PRESETS), default="balanced", help="preset (default: balanced)")
    p.add_argument("--model", help="model name, see 'colorstream models' (default: from quality preset)")
    p.add_argument("--saturation", type=float, help="chroma multiplier (default 0.85)")
    p.add_argument("--resolution", type=int, help="output height in pixels, 0 keeps the source (default: keep)")
    p.add_argument("--model-res", type=int, help="side of the grayscale copy given to the model")
    p.add_argument("--smoothing", type=float, help="temporal smoothing 0..0.98 (default 0.7)")
    p.add_argument("--stride", type=int, help="run the model every N frames")
    p.add_argument("--flow", action=argparse.BooleanOptionalAction, default=None, help="optical flow warping")
    p.add_argument("--batch", type=int, help="frames per model call")
    p.add_argument("--crf", type=int, help="video quality, lower is better (default depends on preset)")
    p.add_argument("--encoder", help="force a video encoder, e.g. h264_nvenc or libx264")
    p.add_argument("--force", action="store_true", help="colorize even if the source looks like color")
    p.add_argument("--no-audio", action="store_true", help="drop audio tracks")
    p.add_argument("--no-subs", action="store_true", help="drop the subtitle tracks that are already in the file")
    p.add_argument("--subs", choices=["off", "soft", "burn", "both"], default=None,
                   help="create subtitles: soft = extra track + .srt file, burn = drawn on every frame")
    p.add_argument("--sub-lang", help="subtitle language: el, en, es, fr, de, it, ru ... (default: auto-detect)")
    p.add_argument("--sub-source", choices=["auto", "existing", "asr"],
                   help="auto = use existing subtitles (re-synced to the audio) else generate with speech recognition")
    p.add_argument("--sub-file", help="use this .srt/.vtt/.ass (it is re-synced to the audio)")
    p.add_argument("--asr-model", help="speech model size: tiny, base, small, medium, large-v3 (default small)")
    p.add_argument("--no-sub-align", action="store_true", help="do not re-sync existing subtitles")
    p.add_argument("--sub-size", type=float, help="burned text size as fraction of the height (default 0.045)")
    p.add_argument("--compare", action="store_true", help="write original and colorized side by side")
    p.add_argument("--chunk-seconds", type=float, default=120.0, help="resume granularity (default 120)")
    p.add_argument("--no-resume", action="store_true", help="ignore a previous interrupted job")
    p.add_argument("--keep-temp", action="store_true", help="keep temporary parts after finishing")


def _options(a: argparse.Namespace) -> ConvertOptions:
    return ConvertOptions(chunk_seconds=a.chunk_seconds, resume=not a.no_resume, compare=a.compare,
                          keep_audio=not a.no_audio, keep_subs=not a.no_subs, force=a.force, keep_temp=a.keep_temp,
                          subtitle_file=a.sub_file or "")


def _convert_one(cfg: Config, src: Path, dst: Path, settings: ProcSettings, opts: ConvertOptions) -> None:
    bar = tqdm(total=100.0, unit="%", bar_format="{l_bar}{bar}| {n:.1f}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
               desc=src.name[:40], dynamic_ncols=True)
    last = [0.0]

    def prog(done: float, total: float, fps: float) -> None:
        pct = min(100.0, 100.0 * done / max(total, 1e-6))
        bar.update(max(0.0, pct - last[0]))
        last[0] = pct
        bar.set_postfix_str(f"{fps:.1f} fps")

    sub_bar = None

    def sub_prog(frac: float, status: str) -> None:
        nonlocal sub_bar
        if sub_bar is None:
            sub_bar = tqdm(total=100.0, unit="%", desc="subtitles", dynamic_ncols=True)
        sub_bar.n = min(100.0, 100.0 * frac)
        sub_bar.set_postfix_str(status)
        sub_bar.refresh()

    opts.on_subtitle_progress = sub_prog
    try:
        convert_file(cfg, src, dst, settings, opts, progress=prog)
    finally:
        bar.close()
        if sub_bar is not None:
            sub_bar.close()


def cmd_convert(cfg: Config, a: argparse.Namespace) -> int:
    check_binaries(cfg)
    settings = settings_from_args(cfg, a)
    src = Path(a.input)
    dst = Path(a.output) if a.output else src.with_name(src.stem + ".colorized.mkv")
    print(f"Encoder: {pick_encoder(cfg)} | model: {settings.resolved_model()} @ {settings.model_resolution}px | "
          f"saturation {settings.saturation}")
    try:
        _convert_one(cfg, src, dst, settings, _options(a))
    except AlreadyColor as e:
        print(f"Skipped: {e}")
        return 3
    print(f"Done: {dst}")
    return 0


def cmd_batch(cfg: Config, a: argparse.Namespace) -> int:
    check_binaries(cfg)
    settings = settings_from_args(cfg, a)
    folder = Path(a.folder)
    out_dir = Path(a.output_dir) if a.output_dir else folder / "colorized"
    files = [f for f in find_videos(folder, a.recursive) if out_dir not in f.parents]
    if not files:
        print(f"No video files found in {folder}")
        return 1
    failures = 0
    for i, f in enumerate(files, 1):
        dst = out_dir / (f.stem + ".mkv")
        if dst.exists() and not a.overwrite:
            print(f"[{i}/{len(files)}] skip (exists): {dst.name}")
            continue
        print(f"[{i}/{len(files)}] {f.name}")
        try:
            _convert_one(cfg, f, dst, settings, _options(a))
        except AlreadyColor as e:
            print(f"  skipped: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: {e}")
    return 1 if failures else 0


def cmd_serve(cfg: Config, a: argparse.Namespace) -> int:
    from .server import serve

    try:
        asyncio.run(serve(cfg))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_benchmark(cfg: Config, a: argparse.Namespace) -> int:
    from .benchmark import choose_settings, run_benchmark

    check_binaries(cfg)
    bench = run_benchmark(cfg, a.height, force=True)
    print(f"Provider/hardware benchmark, output height {a.height}p:")
    print(f"  CPU per frame: render {bench['cpu']['render'] * 1000:.1f} ms, flow {bench['cpu']['flow'] * 1000:.1f} ms")
    for name, m in bench["models"].items():
        row = ", ".join(f"{k}px: {1 / v:.0f} fps" for k, v in sorted(m["per_frame"].items(), key=lambda kv: int(kv[0])))
        print(f"  {name:18s} [{m['provider']}] {row}")
    try:
        chosen, rep = choose_settings(cfg, bench, replace(cfg.processing, mode="fast"), a.fps)
        print(f"Auto choice for {a.fps} fps source: {rep}")
    except RuntimeError as e:
        print(f"Auto choice: {e}")
    return 0


def cmd_models(cfg: Config, a: argparse.Namespace) -> int:
    for m in available_models(cfg):
        print(f"{'[x]' if m['available'] else '[ ]'} {m['name']:18s} {m['description']}")
    print(f"\nWeights folder: {Path(cfg.paths.models_dir).resolve()}")
    return 0


def cmd_check(cfg: Config, a: argparse.Namespace) -> int:
    ok = True
    try:
        check_binaries(cfg)
        print(f"[ok] ffmpeg/ffprobe found, encoder that will be used: {pick_encoder(cfg)}")
    except FFmpegError as e:
        ok = False
        print(f"[!!] {e}")
    try:
        import onnxruntime as ort
        print(f"[ok] onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")
        if ort.get_available_providers() == ["CPUExecutionProvider"]:
            print("[!!] Only CPU provider available. See README: install onnxruntime-gpu or onnxruntime-directml.")
    except ImportError:
        ok = False
        print("[!!] onnxruntime is not installed (pip install -r requirements.txt)")
    from .sysinfo import gpu_stats
    g = gpu_stats(0)
    print(f"[ok] GPU: {g['name']} {g['memory_total_mb']:.0f} MB" if g else "[--] nvidia-smi not found or no NVIDIA GPU")
    avail = [m for m in available_models(cfg) if m["available"]]
    print(f"[ok] models available: {', '.join(m['name'] for m in avail)}")
    if not [m for m in avail if m["name"] != "demo"]:
        print("[!!] No real model weights installed, only 'demo'. See README section 'Model weights'.")
    return 0 if ok else 1


def cmd_clear_cache(cfg: Config, a: argparse.Namespace) -> int:
    from .cache import SegmentCache

    SegmentCache(cfg.paths.cache_dir).clear()
    print("Cache cleared.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colorstream", description="Colorize black and white video with AI.")
    p.add_argument("--version", action="version", version=f"colorstream {__version__}")
    p.add_argument("-c", "--config", help="path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="start the HLS server, dashboard and API")
    s.set_defaults(fn=cmd_serve)

    c = sub.add_parser("convert", help="colorize a local video file")
    c.add_argument("input")
    c.add_argument("-o", "--output")
    _add_proc_args(c)
    c.set_defaults(fn=cmd_convert)

    b = sub.add_parser("batch", help="colorize every video in a folder")
    b.add_argument("folder")
    b.add_argument("-o", "--output-dir")
    b.add_argument("-r", "--recursive", action="store_true")
    b.add_argument("--overwrite", action="store_true")
    _add_proc_args(b)
    b.set_defaults(fn=cmd_batch)

    k = sub.add_parser("benchmark", help="measure this machine and show the auto-selected settings")
    k.add_argument("--height", type=int, default=720)
    k.add_argument("--fps", type=float, default=24.0)
    k.set_defaults(fn=cmd_benchmark)

    sub.add_parser("models", help="list models and whether weights are installed").set_defaults(fn=cmd_models)
    sub.add_parser("check", help="diagnose FFmpeg, GPU providers and models").set_defaults(fn=cmd_check)
    sub.add_parser("clear-cache", help="delete cached segments").set_defaults(fn=cmd_clear_cache)
    return p


def main(argv: list | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not a.cmd:
        parser.print_help()
        return 2
    try:
        cfg = load_config(a.config)
        return a.fn(cfg, a)
    except (FFmpegError, ModelNotFoundError, FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command again to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
