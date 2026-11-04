"""Luma/chroma helpers: gray to Lab lightness, chroma merge, color detection.

The key idea: the original luminance is never touched. The model only
predicts chroma (Lab a/b) on a small image, which is upscaled and merged
with the full resolution luminance.
"""
from __future__ import annotations

import cv2
import numpy as np


def _build_lut() -> np.ndarray:
    ramp = np.arange(256, dtype=np.uint8)
    bgr = np.repeat(ramp[None, :, None], 3, axis=2)  # (1,256,3) neutral grays
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return np.ascontiguousarray(lab[0, :, 0])


# Gray value (0..255) to 8-bit OpenCV Lab lightness (0..255 meaning L 0..100).
GRAY_TO_L8 = _build_lut()


def gray_to_L8(gray: np.ndarray) -> np.ndarray:
    return cv2.LUT(gray, GRAY_TO_L8)


def gray_to_L100(gray: np.ndarray) -> np.ndarray:
    """Gray uint8 image (any shape) to Lab lightness in 0..100 as float32."""
    return gray_to_L8(gray).astype(np.float32) * (100.0 / 255.0)


def upscale_chroma(ab: np.ndarray, width: int, height: int) -> np.ndarray:
    """ab: (2,h,w) float32 -> (height,width,2) float32."""
    ab_hw = np.ascontiguousarray(ab.transpose(1, 2, 0), dtype=np.float32)
    if ab_hw.shape[0] != height or ab_hw.shape[1] != width:
        ab_hw = cv2.resize(ab_hw, (width, height), interpolation=cv2.INTER_LINEAR)
    return ab_hw


GUIDE_MAX_SIDE = 640  # chroma of a 720p H.264 picture is 640x360 anyway, so refine at this size


def _box(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REPLICATE)


def guided_chroma(gray: np.ndarray, ab: np.ndarray, eps: float = 2e-3) -> np.ndarray:
    """Edge-aware chroma: (2,h,w) model chroma -> (h2,w2,2) aligned to the luminance edges.

    A fast guided filter (He et al.) with the original gray frame as the guide. The model's chroma is
    blurry and its boundaries sit a few pixels off the real object edges, which shows as color leaking
    over edges. Here every pixel's chroma is a local linear function of its luminance, so color edges
    snap to luminance edges. The filter coefficients are estimated at half the working size and
    upsampled, which keeps it cheap. Working size is at most GUIDE_MAX_SIDE.
    """
    height, width = gray.shape
    scale = min(1.0, GUIDE_MAX_SIDE / float(max(height, width)))
    w2, h2 = max(16, int(width * scale)), max(16, int(height * scale))
    guide = cv2.resize(gray, (w2, h2), interpolation=cv2.INTER_AREA).astype(np.float32) * np.float32(1 / 255.0)
    p_all = cv2.resize(np.ascontiguousarray(ab.transpose(1, 2, 0), dtype=np.float32), (w2, h2),
                       interpolation=cv2.INTER_LINEAR)
    ws, hs = w2 // 2, h2 // 2  # coefficient estimation grid
    g_s = cv2.resize(guide, (ws, hs), interpolation=cv2.INTER_AREA)
    p_s = cv2.resize(p_all, (ws, hs), interpolation=cv2.INTER_AREA)
    r = max(2, ws // 12)  # wide enough to span the blur of a low resolution model output
    mean_i = _box(g_s, r)
    var_i = _box(g_s * g_s, r) - mean_i * mean_i + np.float32(eps)
    out = np.empty((h2, w2, 2), np.float32)
    for c in range(2):
        p = p_s[..., c]
        mean_p = _box(p, r)
        a = (_box(g_s * p, r) - mean_i * mean_p) / var_i
        b = mean_p - a * mean_i
        a_full = cv2.resize(_box(a, r), (w2, h2), interpolation=cv2.INTER_LINEAR)
        b_full = cv2.resize(_box(b, r), (w2, h2), interpolation=cv2.INTER_LINEAR)
        out[..., c] = a_full * guide + b_full
    return out


def soft_limit_chroma(ab: np.ndarray, limit: float) -> np.ndarray:
    """Compress strong chroma smoothly (tanh on the magnitude). Weak colors are left almost untouched,
    the strongest are capped near `limit`, which avoids neon colors without dulling the picture."""
    if limit <= 0:
        return ab
    mag = np.sqrt(ab[0] * ab[0] + ab[1] * ab[1]) + np.float32(1e-6)
    factor = np.float32(limit) * np.tanh(mag / np.float32(limit)) / mag
    return ab * factor


def merge_luma_chroma(gray: np.ndarray, ab: np.ndarray, saturation: float = 1.0, guided: bool = False) -> np.ndarray:
    """Merge full resolution original luminance with low resolution chroma.

    gray: (H,W) uint8 original frame (luminance is preserved exactly).
    ab:   (2,h,w) float32 predicted chroma in Lab units (about -110..110).
    saturation: multiplier on chroma magnitude (1.0 = model output).
    guided: refine the chroma with the luminance edges first (see guided_chroma).
    Returns BGR uint8 (H,W,3).

    Saturation and 8-bit quantization happen at the small chroma resolution, so only
    two cheap uint8 planes are interpolated at full resolution.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a 2D uint8 array")
    height, width = gray.shape
    if guided:
        chroma = guided_chroma(gray, ab * np.float32(saturation))  # (h2,w2,2)
        ab8 = np.clip(chroma + np.float32(128.5), 0, 255).astype(np.uint8)
        planes_src = [ab8[..., 0], ab8[..., 1]]
    else:
        ab8 = np.clip(ab * np.float32(saturation) + np.float32(128.5), 0, 255).astype(np.uint8)
        planes_src = [ab8[0], ab8[1]]
    planes = [gray_to_L8(gray)]
    for p in planes_src:
        if p.shape != (height, width):
            p = cv2.resize(p, (width, height), interpolation=cv2.INTER_LINEAR)
        planes.append(p)
    return cv2.cvtColor(cv2.merge(planes), cv2.COLOR_LAB2BGR)


def chroma_spread(bgr: np.ndarray) -> float:
    """How much real color a BGR frame contains.

    Standard deviation of Lab a/b over the frame. A pure gray frame gives 0.
    A uniformly tinted (sepia, cyanotype) black and white frame stays small
    because the tint barely varies, while real color varies a lot.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return float(np.sqrt(a.var() + b.var()))


def is_color_content(spreads: list[float], threshold: float = 7.0) -> bool:
    """Decide from several sampled frames whether the video is already color.

    The median is used so a single colored title card does not flip the verdict.
    """
    if not spreads:
        return False
    return float(np.median(spreads)) > threshold
