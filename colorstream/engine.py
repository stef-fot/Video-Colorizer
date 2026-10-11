"""ChromaEngine: model inference plus temporal consistency, per frame stream."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import ProcSettings
from .models.base import ColorModel
from .temporal import (
    ema_blend,
    farneback_flow,
    histogram_difference,
    luma_histogram,
    warp_chroma,
)


@dataclass
class EngineStats:
    frames: int = 0
    inferred: int = 0
    cuts: int = 0
    infer_seconds: float = 0.0
    temporal_seconds: float = 0.0


class ChromaEngine:
    """Turns small grayscale frames into temporally stable chroma.

    Frames must be fed in display order. State is carried across calls and
    reset on scene cuts or by `reset()` (use it when seeking).
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

    def process(self, smalls: list[np.ndarray]) -> list[np.ndarray]:
        """smalls: list of (S,S) uint8. Returns list of (2,S,S) float32 chroma."""
        n = len(smalls)
        if n == 0:
            return []
        stride = max(1, int(self.s.inference_stride))
        thr = self.s.scene_cut_threshold

        # Pass 1: cut detection and decide which frames need the model.
        hists = [luma_histogram(g) for g in smalls]
        cuts: list[bool] = []
        infer: list[bool] = []
        prev_h = self.prev_hist
        have = self.state is not None
        since = self.since_infer
        for h in hists:
            cut = prev_h is not None and histogram_difference(prev_h, h) > thr
            if cut:
                have = False
            need = (not have) or (since + 1 >= stride)
            since = 0 if need else since + 1
            have = have or need
            cuts.append(cut)
            infer.append(need)
            prev_h = h

        # Pass 2: one batched model call for the frames that need it.
        idx = [i for i, need in enumerate(infer) if need]
        raw: dict[int, np.ndarray] = {}
        if idx:
            t0 = time.perf_counter()
            batch = np.stack([smalls[i] for i in idx])
            out = self.model.predict(batch)
            self.stats.infer_seconds += time.perf_counter() - t0
            self.stats.inferred += len(idx)
            for k, i in enumerate(idx):
                raw[i] = out[k]

        # Pass 3: sequential temporal ops.
        t0 = time.perf_counter()
        results: list[np.ndarray] = []
        for i, gray in enumerate(smalls):
            if cuts[i]:
                self.state = None
                self.stats.cuts += 1
            prev = self.state
            if prev is not None and self.s.optical_flow and self.prev_gray is not None:
                prev = warp_chroma(prev, farneback_flow(self.prev_gray, gray))
            if i in raw:
                cur = ema_blend(prev, raw[i], self.s.smoothing)
            else:
                cur = prev  # propagated (warped) chroma between model calls
            self.state = cur
            self.prev_gray = gray
            results.append(cur)
        self.prev_hist = hists[-1]
        self.since_infer = since
        self.stats.frames += n
        self.stats.temporal_seconds += time.perf_counter() - t0
        return results
