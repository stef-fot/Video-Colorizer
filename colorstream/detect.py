"""Detect whether a source is already in color by sampling a few frames."""
from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .color import chroma_spread, is_color_content
from .config import Config
from .ffmpeg import MediaInfo, input_args

log = logging.getLogger("colorstream.detect")

W, H = 192, 108


def _grab(cfg: Config, src: str, headers: dict | None, t: float) -> np.ndarray | None:
    cmd = [cfg.ffmpeg, "-v", "error", "-nostdin"] + input_args(src, headers)
    cmd += ["-ss", f"{t:.2f}", "-i", src, "-frames:v", "1", "-vf", f"scale={W}:{H}", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "pipe:1"]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=40)
    except subprocess.TimeoutExpired:
        return None
    if res.returncode != 0 or len(res.stdout) < W * H * 3:
        return None
    return np.frombuffer(res.stdout[: W * H * 3], dtype=np.uint8).reshape(H, W, 3)


def sample_spreads(cfg: Config, src: str, headers: dict | None, info: MediaInfo, samples: int = 6) -> list[float]:
    # Skip the first and last 5 percent (titles, credits are often special).
    times = [info.duration * (0.05 + 0.9 * (i + 0.5) / samples) for i in range(samples)]
    with ThreadPoolExecutor(max_workers=min(samples, 4)) as ex:
        frames = list(ex.map(lambda t: _grab(cfg, src, headers, t), times))
    return [chroma_spread(f) for f in frames if f is not None]


def source_is_color(cfg: Config, src: str, headers: dict | None, info: MediaInfo, threshold: float) -> tuple[bool, list[float]]:
    spreads = sample_spreads(cfg, src, headers, info)
    if not spreads:
        log.warning("Could not sample any frame to check for color, assuming black and white.")
        return False, []
    verdict = is_color_content(spreads, threshold)
    log.info("Color check: spreads=%s threshold=%.1f -> %s", [round(x, 1) for x in spreads], threshold,
             "COLOR" if verdict else "black and white")
    return verdict, spreads
