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


def merge_luma_chroma(gray: np.ndarray, ab: np.ndarray, saturation: float = 1.0) -> np.ndarray:
    """Merge full resolution original luminance with low resolution chroma.

    gray: (H,W) uint8 original frame (luminance is preserved exactly).
    ab:   (2,h,w) float32 predicted chroma in Lab units (about -110..110).
    saturation: multiplier on chroma magnitude (1.0 = model output).
    Returns BGR uint8 (H,W,3).

    Saturation and 8-bit quantization happen at the small chroma resolution, so only
    two cheap uint8 planes are interpolated at full resolution.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a 2D uint8 array")
    height, width = gray.shape
    ab8 = np.clip(ab * np.float32(saturation) + np.float32(128.5), 0, 255).astype(np.uint8)
    planes = [gray_to_L8(gray)]
    for c in range(2):
        p = ab8[c]
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
