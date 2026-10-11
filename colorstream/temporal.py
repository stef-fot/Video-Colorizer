"""Temporal consistency primitives: EMA, scene cuts, optical flow warping.

All functions are pure and operate on small (model resolution) arrays so the
cost per frame stays tiny compared with the model itself.
"""
from __future__ import annotations

import cv2
import numpy as np

HIST_BINS = 64


def luma_histogram(gray: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Normalized (sums to 1) luminance histogram of a uint8 image."""
    h = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float32)
    total = float(h.sum())
    return h / total if total > 0 else h


def histogram_difference(h1: np.ndarray, h2: np.ndarray) -> float:
    """Total variation distance between two normalized histograms, 0..1."""
    return float(0.5 * np.abs(h1 - h2).sum())


def is_scene_cut(h1: np.ndarray, h2: np.ndarray, threshold: float = 0.5) -> bool:
    return histogram_difference(h1, h2) > threshold


def ema_blend(previous: np.ndarray | None, new: np.ndarray, smoothing: float) -> np.ndarray:
    """Exponential moving average on chroma.

    smoothing is the weight kept on the history: 0 returns `new`, values near 1
    react slowly (very stable, but colors lag behind motion).
    """
    if previous is None or smoothing <= 0.0:
        return new
    s = float(min(max(smoothing, 0.0), 0.999))
    return s * previous + (1.0 - s) * new


def farneback_flow(prev_gray: np.ndarray, cur_gray: np.ndarray, max_side: int = 192) -> np.ndarray:
    """Backward flow (current -> previous), shape (h,w,2), in pixels of the input size.

    Computed on a reduced copy for speed and scaled back.
    """
    h, w = cur_gray.shape
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        size = (max(8, int(w * scale)), max(8, int(h * scale)))
        p = cv2.resize(prev_gray, size, interpolation=cv2.INTER_AREA)
        c = cv2.resize(cur_gray, size, interpolation=cv2.INTER_AREA)
    else:
        p, c = prev_gray, cur_gray
    flow = cv2.calcOpticalFlowFarneback(c, p, None, 0.5, 3, 15, 2, 5, 1.1, 0)
    if scale < 1.0:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    return flow


def warp_chroma(ab: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp previous chroma (2,h,w) into the current frame using backward flow."""
    _, h, w = ab.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    out = np.empty_like(ab)
    for c in range(2):
        out[c] = cv2.remap(ab[c], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out
