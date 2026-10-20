"""Re-synchronize existing subtitles to the audio.

A speech/no-speech mask is computed from the audio (energy based, no models) and compared with the
"a cue is visible" mask of the subtitles. Cross-correlation gives the delay, and doing that on several
parts of the film gives a straight line t' = scale * t + offset, which also fixes frame rate mismatch
(for example subtitles made for 25 fps on a 23.976 fps video).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..ffmpeg import input_args
from .cues import Cue, retime

SR = 16000
HOP = 0.1  # seconds per mask step


@dataclass
class Alignment:
    scale: float = 1.0
    offset: float = 0.0
    score: float = 0.0  # 0..1, how well the subtitles match speech after alignment
    score_before: float = 0.0

    @property
    def significant(self) -> bool:
        return abs(self.offset) > 0.25 or abs(self.scale - 1.0) > 2e-4

    def apply(self, cues: list[Cue]) -> list[Cue]:
        return retime(cues, self.scale, self.offset)


def decode_audio(cfg: Config, src: str, headers: dict | None, start: float = 0.0, duration: float | None = None,
                 timeout: float = 600) -> np.ndarray:
    """Mono 16 kHz float32 samples. Used by alignment and by speech recognition."""
    cmd = [cfg.ffmpeg, "-v", "error", "-nostdin"] + input_args(src, headers)
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", src]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-sn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if res.returncode != 0 and not res.stdout:
        raise RuntimeError("Could not decode audio: " + res.stderr.decode("utf-8", "replace").strip()[-300:])
    return np.frombuffer(res.stdout[: len(res.stdout) // 4 * 4], dtype=np.float32)


def frame_db(samples: np.ndarray, sr: int = SR, hop: float = HOP) -> np.ndarray:
    """Loudness in dB for every `hop` seconds."""
    step = int(sr * hop)
    n = len(samples) // step
    if n <= 0:
        return np.zeros(0)
    frames = samples[: n * step].reshape(n, step).astype(np.float64)
    return 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)


def mask_from_db(db: np.ndarray) -> np.ndarray:
    """Boolean per-hop mask: True where the audio is clearly louder than the noise floor."""
    n = len(db)
    if n == 0:
        return np.zeros(0, bool)
    floor, top = np.percentile(db, 10), np.percentile(db, 95)
    if top - floor < 6.0:  # almost constant level: no usable speech structure
        return np.zeros(n, bool)
    mask = db > floor + 0.35 * (top - floor)
    return np.convolve(mask.astype(float), np.ones(3), mode="same") >= 2  # close tiny gaps, drop clicks


def speech_mask(samples: np.ndarray, sr: int = SR, hop: float = HOP) -> np.ndarray:
    return mask_from_db(frame_db(samples, sr, hop))


def audio_energy_db(cfg: Config, src: str, headers: dict | None, sr: int = 8000, hop: float = HOP,
                    cancel=None) -> np.ndarray:
    """Stream the whole audio through FFmpeg and return per-hop loudness, using little memory."""
    cmd = [cfg.ffmpeg, "-v", "error", "-nostdin"] + input_args(src, headers)
    cmd += ["-i", src, "-vn", "-sn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    step_bytes = int(sr * hop) * 4
    block = step_bytes * 600
    out: list[np.ndarray] = []
    rest = b""
    try:
        while True:
            if cancel is not None and cancel.is_set():
                break
            data = proc.stdout.read(block)
            if not data:
                break
            data = rest + data
            use = len(data) // step_bytes * step_bytes
            rest = data[use:]
            if use:
                out.append(frame_db(np.frombuffer(data[:use], dtype=np.float32), sr, hop))
    finally:
        proc.kill()
        proc.wait()
    return np.concatenate(out) if out else np.zeros(0)


def cue_mask(cues: list[Cue], n: int, hop: float = HOP) -> np.ndarray:
    m = np.zeros(n, bool)
    for c in cues:
        a, b = int(c.start / hop), int(np.ceil(c.end / hop))
        if b > 0 and a < n:
            m[max(a, 0):min(b, n)] = True
    return m


def _best_lag(x: np.ndarray, cm: np.ndarray, start: int, max_lag: int) -> tuple[int, float]:
    """Lag L (in hops) so that speech x (which starts at hop `start`) best matches cm[start+L : ...].
    L > 0 means the cues sit L hops LATER than the speech they belong to."""
    m = len(x)
    y = np.zeros(m + 2 * max_lag)
    lo, hi = start - max_lag, start + m + max_lag
    a, b = max(lo, 0), min(hi, len(cm))
    if b > a:
        y[a - lo:b - lo] = cm[a:b]
    xc = x.astype(float) - x.mean()
    yc = y - y.mean()
    corr = np.correlate(yc, xc, mode="valid")  # index j <-> lag j - max_lag
    j = int(np.argmax(corr))
    q = float(corr[j] / (np.linalg.norm(xc) * np.linalg.norm(yc) + 1e-9))
    return j - max_lag, q


def match_score(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    return float((a == b).mean())


# Frame rate conversions that commonly cause subtitle drift.
COMMON_SCALES = [1.0, 25 / 23.976, 23.976 / 25, 24 / 23.976, 23.976 / 24, 25 / 24, 24 / 25,
                 30 / 29.97, 29.97 / 30, 29.97 / 25, 25 / 29.97, 30 / 25, 25 / 30]


def estimate(speech: np.ndarray, cues: list[Cue], max_shift: float = 120.0, parts: int = 6,
             hop: float = HOP) -> Alignment:
    """Find scale and offset so that `speech_time = scale * cue_time + offset`."""
    n = len(speech)
    if n < 450 or not cues or speech.sum() < 20:
        return Alignment()
    ms = int(max_shift / hop)
    base = match_score(speech, cue_mask(cues, n, hop))
    # 1) pick the most plausible frame rate conversion by global correlation quality
    pre = 1.0
    best_q = -1.0
    for cand in COMMON_SCALES:
        _, q = _best_lag(speech, cue_mask(retime(cues, cand, 0.0), n + 2 * ms, hop), 0, ms)
        if q > best_q + 0.01 * (cand != 1.0):  # prefer 1.0 unless another one is clearly better
            best_q, pre = q, cand
    original = cues
    cues = retime(cues, pre, 0.0)
    cm = cue_mask(cues, n + 2 * ms, hop)
    g_lag, _ = _best_lag(speech, cm, 0, ms)
    points = []  # (cue time of the part, speech time of the part, quality)
    size = n // parts
    for p in range(parts):
        a, b = p * size, ((p + 1) * size if p < parts - 1 else n)
        if speech[a:b].sum() < 20:
            continue
        lag, q = _best_lag(speech[a:b], cm, a, ms)
        if q > 0.1:
            centre = (a + b) / 2
            points.append(((centre + lag) * hop, centre * hop, q))
    scale, offset = 1.0, -g_lag * hop
    if len(points) >= 3:
        tau = np.array([p[0] for p in points])
        t = np.array([p[1] for p in points])
        w = np.array([p[2] for p in points])
        for _ in range(3):  # drop outlier parts (music, silence) and refit
            k, c = np.polyfit(tau, t, 1, w=w)
            resid = np.abs(t - (k * tau + c))
            keep = resid <= max(1.0, 2.5 * float(np.median(resid)))
            if keep.all() or keep.sum() < 3:
                break
            tau, t, w = tau[keep], t[keep], w[keep]
        k, c = np.polyfit(tau, t, 1, w=w)
        if abs(k - 1.0) < 0.05:
            scale, offset = float(k), float(c)
    scale = scale * pre  # total mapping from the ORIGINAL cue times
    best = Alignment(scale, offset)
    best.score = match_score(speech, cue_mask(best.apply(original), n, hop))
    best.score_before = base
    if best.score < base + 0.02:  # only accept a clear improvement
        return Alignment(1.0, 0.0, base, base)
    return best
