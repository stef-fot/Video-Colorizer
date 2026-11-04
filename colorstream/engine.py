"""ChromaEngine: model inference plus everything that makes colors look natural and steady.

Per frame stream, at model resolution (cheap):
  * scene cut detection (luma histogram)
  * shot memory: when a shot returns (shot / reverse shot, jump cuts) it starts from the colors it had before
  * adaptive inference: the model runs only when the picture changed (or every `max_stride` frames)
  * temporal smoothing: running average (a plain average at the start of a shot), optional optical flow warp
  * a limit on how fast a chroma value may change, so colors cannot jump
  * shot level consistency: each shot is gently pulled toward the film's overall color strength and tint
  * soft ceiling on chroma strength against neon colors
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .color import soft_limit_chroma
from .config import ProcSettings
from .models.base import ColorModel
from .temporal import farneback_flow, histogram_difference, luma_histogram, warp_chroma

THUMB = 16
RECALL_MAX_DIFF = 14.0  # mean abs gray difference (0..255) between thumbnails to call two shots "the same"
RECALL_MAX_HIST = 0.40
BANK_SIZE = 8
MIN_SHOT_FRAMES = 12  # shorter shots do not update the film's palette


@dataclass
class EngineStats:
    frames: int = 0
    inferred: int = 0
    cuts: int = 0
    recalls: int = 0
    infer_seconds: float = 0.0
    temporal_seconds: float = 0.0

    @property
    def inferred_fraction(self) -> float:
        return self.inferred / self.frames if self.frames else 0.0


@dataclass
class Shot:
    first_thumb: np.ndarray
    last_thumb: np.ndarray
    hist: np.ndarray
    chroma: np.ndarray | None = None  # latest smoothed chroma (before per-shot adjustments)
    frames: int = 0
    mag: float = 0.0  # running mean chroma strength
    mean_ab: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    gain: float = 1.0  # adjustment currently applied (ramped)
    shift: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    inferred: int = 0


class ChromaEngine:
    """Turns small grayscale frames into temporally stable chroma.

    Frames must be fed in display order. State carries across calls; `reset()` forgets everything
    (use it when seeking).
    """

    def __init__(self, model: ColorModel, settings: ProcSettings, size: int):
        self.model = model
        self.s = settings
        self.size = size
        self.stats = EngineStats()
        self.reset()

    def reset(self) -> None:
        self.state: np.ndarray | None = None  # smoothed chroma (2,S,S)
        self.prev_gray: np.ndarray | None = None
        self.prev_hist: np.ndarray | None = None
        self.since_infer = 0
        self.last_infer_thumb: np.ndarray | None = None
        self.shot: Shot | None = None
        self.bank: list[Shot] = []
        self.pal_mag = 0.0
        self.pal_mean = np.zeros(2, np.float32)
        self.pal_n = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _thumb(gray: np.ndarray) -> np.ndarray:
        return cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _recall(self, thumb: np.ndarray, hist: np.ndarray) -> Shot | None:
        if not self.s.shot_memory:
            return None
        best, best_d = None, 1e9
        for sh in self.bank:
            if sh.chroma is None:
                continue
            d = min(float(np.abs(thumb - sh.first_thumb).mean()), float(np.abs(thumb - sh.last_thumb).mean()))
            if d < best_d and histogram_difference(hist, sh.hist) < RECALL_MAX_HIST:
                best, best_d = sh, d
        return best if best is not None and best_d < RECALL_MAX_DIFF else None

    def _close_shot(self) -> None:
        sh = self.shot
        if sh is None or sh.chroma is None:
            return
        if sh.frames >= MIN_SHOT_FRAMES:
            if self.pal_n == 0:
                self.pal_mag, self.pal_mean = sh.mag, sh.mean_ab.copy()
            else:
                self.pal_mag = 0.8 * self.pal_mag + 0.2 * sh.mag
                self.pal_mean = 0.8 * self.pal_mean + 0.2 * sh.mean_ab
            self.pal_n += 1
        self.bank.append(sh)
        del self.bank[:-BANK_SIZE]

    def _finish(self, cur: np.ndarray, sh: Shot) -> np.ndarray:
        """Per-shot consistency and the soft chroma ceiling. Does not change the stored state."""
        s = self.s
        mag_now = float(np.sqrt(cur[0] ** 2 + cur[1] ** 2).mean())
        mean_now = cur.reshape(2, -1).mean(axis=1)
        a = 0.85 if sh.frames > 1 else 0.0  # running stats of this shot
        sh.mag = a * sh.mag + (1 - a) * mag_now
        sh.mean_ab = (a * sh.mean_ab + (1 - a) * mean_now).astype(np.float32)
        out = cur
        k = float(s.shot_consistency)
        if k > 0 and self.pal_n >= 2 and sh.mag > 1.0:
            target_gain = float(np.clip((self.pal_mag / sh.mag) ** k, 0.75, 1.35))
            target_shift = np.clip(k * (self.pal_mean - sh.mean_ab), -6.0, 6.0).astype(np.float32)
            sh.gain += 0.12 * (target_gain - sh.gain)  # ramp, never a jump
            sh.shift += 0.12 * (target_shift - sh.shift)
            out = cur * np.float32(sh.gain) + sh.shift[:, None, None]
        return soft_limit_chroma(out, float(s.chroma_limit)) if s.chroma_limit > 0 else out

    # ------------------------------------------------------------------ main entry
    def process(self, smalls: list[np.ndarray]) -> list[np.ndarray]:
        """smalls: list of (S,S) uint8. Returns list of (2,S,S) float32 chroma."""
        n = len(smalls)
        if n == 0:
            return []
        s = self.s
        thr = s.scene_cut_threshold
        hists = [luma_histogram(g) for g in smalls]
        thumbs = [self._thumb(g) for g in smalls]

        # Pass 1: cuts, shot recall, and which frames need the model.
        cuts: list[bool] = []
        recalls: list[Shot | None] = []
        infer: list[bool] = []
        prev_h = self.prev_hist
        have = self.state is not None
        since = self.since_infer
        last_t = self.last_infer_thumb
        bank_view = self.bank  # recall only needs shots that already ended
        for h, t in zip(hists, thumbs):
            cut = prev_h is not None and histogram_difference(prev_h, h) > thr
            rec = None
            if cut:
                rec = self._recall(t, h)
                have = rec is not None
                if rec is not None:
                    last_t, since = rec.last_thumb, 0
            if not have:
                need = True
            elif s.adaptive_stride and last_t is not None:
                motion = float(np.abs(t - last_t).mean())
                need = motion > s.motion_threshold or since + 1 >= max(1, s.max_stride, s.inference_stride)
            else:
                need = since + 1 >= max(1, s.inference_stride)
            if need:
                since, last_t, have = 0, t, True
            else:
                since += 1
            cuts.append(cut)
            recalls.append(rec)
            infer.append(need)
            prev_h = h

        # Pass 2: one batched model call for the frames that need it.
        idx = [i for i, need in enumerate(infer) if need]
        raw: dict[int, np.ndarray] = {}
        if idx:
            t0 = time.perf_counter()
            out = self.model.predict(np.stack([smalls[i] for i in idx]))
            self.stats.infer_seconds += time.perf_counter() - t0
            self.stats.inferred += len(idx)
            for k, i in enumerate(idx):
                raw[i] = out[k]

        # Pass 3: sequential temporal ops.
        t0 = time.perf_counter()
        results: list[np.ndarray] = []
        for i, gray in enumerate(smalls):
            if cuts[i]:
                self._close_shot()
                self.stats.cuts += 1
                rec = recalls[i]
                if rec is not None and rec.chroma is not None:
                    self.stats.recalls += 1
                    self.state = rec.chroma.copy()
                    self.shot = Shot(thumbs[i], thumbs[i], hists[i], None, 0, rec.mag, rec.mean_ab.copy(),
                                     rec.gain, rec.shift.copy(), 8)  # counts as "already averaged"
                else:
                    self.state = None
                    self.shot = None
            if self.shot is None:
                self.shot = Shot(thumbs[i], thumbs[i], hists[i])
            sh = self.shot
            prev = self.state
            if prev is not None and s.optical_flow and self.prev_gray is not None and not cuts[i]:
                prev = warp_chroma(prev, farneback_flow(self.prev_gray, gray))
            if i not in raw and prev is None:  # recalled shot whose chroma vanished: compute now
                raw[i] = self.model.predict(gray[None])[0]
                self.stats.inferred += 1
            if i in raw:
                if prev is None:
                    cur = raw[i]
                else:
                    # Weight of the new model output: at least 1/(k+1) at the start of a shot, so the first
                    # frames are a plain average (a lucky or unlucky first frame cannot set the look),
                    # then the exponential average takes over.
                    w_new = max(1.0 - s.smoothing if s.smoothing > 0 else 1.0, 1.0 / (sh.inferred + 1))
                    cur = prev + w_new * (raw[i] - prev)
                    if s.max_chroma_step > 0 and sh.inferred >= 2:
                        step = s.max_chroma_step * (self.since_infer_for(infer, i) + 1)
                        cur = prev + np.clip(cur - prev, -step, step)
                sh.inferred += 1
            else:
                cur = prev  # propagated (warped) chroma between model calls
            cur = cur.astype(np.float32, copy=False)
            self.state = cur
            sh.chroma = cur
            sh.last_thumb = thumbs[i]
            sh.frames += 1
            self.prev_gray = gray
            results.append(self._finish(cur, sh))
        self.prev_hist = hists[-1]
        self.since_infer = since
        self.last_infer_thumb = last_t
        self.stats.frames += n
        self.stats.temporal_seconds += time.perf_counter() - t0
        return results

    @staticmethod
    def since_infer_for(infer: list[bool], i: int) -> int:
        """Frames since the previous model call inside this batch (0 if the previous frame was one)."""
        k = 0
        j = i - 1
        while j >= 0 and not infer[j]:
            k += 1
            j -= 1
        return k
