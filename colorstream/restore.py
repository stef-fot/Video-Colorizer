"""Restoration of the picture: sharpen, denoise, deflicker, stabilize, remove hardcoded text.

Works on the luminance only (gray frames, or the Y channel of color frames), so chroma is never touched.
All operations are plain OpenCV/NumPy and meant to run on an ordinary CPU. Stages run in this order:
stabilize, deflicker, text removal, denoise, sharpen. Temporal state resets at scene cuts.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import ProcSettings

CUT_THUMB = (32, 18)
CUT_LEVEL = 30.0  # mean absolute thumbnail difference (0..255) that counts as a cut


def parse_region(spec: str, w: int, h: int) -> tuple[int, int, int, int] | None:
    """'bottom' | 'top' | 'x,y,w,h' (fractions) -> pixel box (x0, y0, x1, y1)."""
    if not spec:
        return None
    if spec == "bottom":
        fx, fy, fw, fh = 0.0, 0.74, 1.0, 0.26
    elif spec == "top":
        fx, fy, fw, fh = 0.0, 0.0, 1.0, 0.22
    else:
        fx, fy, fw, fh = [float(x) for x in spec.split(",")]
    x0, y0 = int(fx * w), int(fy * h)
    x1, y1 = min(w, int((fx + fw) * w)), min(h, int((fy + fh) * h))
    return (x0, y0, x1, y1) if x1 - x0 >= 8 and y1 - y0 >= 8 else None


# ------------------------------------------------------------------ single operations

def sharpen(y: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask with noise coring and halo limiting.

    amount 0..1. Detail below a small threshold is ignored (so grain is not boosted) and the result may not
    leave the local min/max of the original by more than a few levels (so no bright or dark halos).
    """
    if amount <= 0:
        return y
    h = y.shape[0]
    sigma = max(1.2, h / 600.0 * 1.5)  # about 1.8 px at 720p, 2.7 px at 1080p: soft sources need a wide radius
    yf = y.astype(np.float32)
    detail = yf - cv2.GaussianBlur(yf, (0, 0), sigma)
    core = 1.2 + 1.5 * amount
    detail = np.sign(detail) * np.maximum(np.abs(detail) - core, 0.0)
    out = yf + (0.6 + 2.0 * amount) * detail
    k = 2 * int(round(sigma)) + 1
    kernel = np.ones((k, k), np.uint8)
    lo = cv2.erode(y, kernel).astype(np.float32) - 6.0
    hi = cv2.dilate(y, kernel).astype(np.float32) + 6.0
    return np.clip(np.clip(out, lo, hi), 0, 255).astype(np.uint8)


class TemporalDenoiser:
    """Blend with the previous output where nothing moved, keep the new frame where something did."""

    def __init__(self, strength: float):
        self.s = float(strength)
        self.prev: np.ndarray | None = None

    def reset(self) -> None:
        self.prev = None

    def apply(self, y: np.ndarray) -> np.ndarray:
        s = self.s
        if self.prev is not None and self.prev.shape == y.shape:
            sigma = 5.0 + 16.0 * s
            diff = np.abs(y.astype(np.float32) - self.prev)
            alpha = (0.45 + 0.45 * s) * np.exp(-(diff / sigma) ** 2)
            out = alpha * self.prev + (1.0 - alpha) * y.astype(np.float32)
        else:
            out = y.astype(np.float32)
        self.prev = out
        res = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        if s >= 0.25:  # light spatial pass, edge preserving
            res = cv2.bilateralFilter(res, 5, 8 + 30 * s, 3)
        return res


class Deflicker:
    def __init__(self):
        self.target: float | None = None

    def reset(self) -> None:
        self.target = None

    def apply(self, y: np.ndarray, mean: float) -> np.ndarray:
        if self.target is None or mean < 3:
            self.target = mean
            return y
        gain = float(np.clip(self.target / max(mean, 1.0), 0.92, 1.08))
        self.target += 0.08 * (mean - self.target)  # follows slow, real brightness changes
        return y if abs(gain - 1.0) < 0.004 else cv2.convertScaleAbs(y, alpha=gain)


class Stabilizer:
    """Removes small camera shake: measures the translation of each frame, low-passes the camera path
    and shifts the frame by the difference. Translation only, no rotation or zoom."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.scale = 192.0 / max(w, h)
        self.reset()

    def reset(self) -> None:
        self.prev: np.ndarray | None = None
        self.path = np.zeros(2)
        self.smooth = np.zeros(2)

    def apply(self, y: np.ndarray) -> np.ndarray:
        small = cv2.resize(y, (max(16, int(self.w * self.scale)), max(16, int(self.h * self.scale))),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.prev is None:
            self.prev = small
            return y
        (dx, dy), _ = cv2.phaseCorrelate(self.prev, small)
        self.prev = small
        self.path += np.array([dx, dy]) / self.scale
        self.smooth += 0.3 * (self.path - self.smooth)
        corr = self.smooth - self.path  # shift that puts the frame on the smoothed path
        limit = np.array([0.03 * self.w, 0.03 * self.h])
        corr = np.clip(corr, -limit, limit)
        if np.abs(corr).max() < 0.15:
            return y
        m = np.float32([[1, 0, corr[0]], [0, 1, corr[1]]])
        return cv2.warpAffine(y, m, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class TextRemover:
    """Removes bright hardcoded text (subtitles, captions) inside a chosen region.

    Text pixels are found as thin bright structures (top-hat) and dilated to take the dark outline too,
    then filled from the surroundings with Telea inpainting. Works best on light text over calm backgrounds.
    If the region is mostly bright (a white wall) or has no text, the frame is left alone.
    """

    def __init__(self, region: tuple[int, int, int, int]):
        self.box = region
        x0, y0, x1, y1 = region
        self.k_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, (y1 - y0) // 9) | 1,) * 2)
        self.k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def mask_for(self, crop: np.ndarray) -> np.ndarray | None:
        top = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, self.k_tophat)
        text = ((top > 45) & (crop > 165)).astype(np.uint8)
        share = float(text.mean())
        if share < 0.0015 or share > 0.20:
            return None
        return cv2.dilate(text, self.k_grow, iterations=2) * 255

    def apply(self, y: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        crop = np.ascontiguousarray(y[y0:y1, x0:x1])
        mask = self.mask_for(crop)
        if mask is None:
            return y
        out = y.copy()
        out[y0:y1, x0:x1] = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
        return out


# ------------------------------------------------------------------ the pipeline stage

class Restorer:
    def __init__(self, s: ProcSettings, w: int, h: int):
        self.s, self.w, self.h = s, w, h
        self.stab = Stabilizer(w, h) if s.restore_stabilize else None
        self.defl = Deflicker() if s.restore_deflicker else None
        region = parse_region(s.restore_text, w, h)
        self.text = TextRemover(region) if region else None
        self.dn = TemporalDenoiser(s.restore_denoise) if s.restore_denoise > 0 else None
        self.prev_thumb: np.ndarray | None = None

    def reset(self) -> None:
        for part in (self.stab, self.defl, self.dn):
            if part is not None:
                part.reset()
        self.prev_thumb = None

    def _y(self, y: np.ndarray) -> np.ndarray:
        thumb = cv2.resize(y, CUT_THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.prev_thumb is not None and float(np.abs(thumb - self.prev_thumb).mean()) > CUT_LEVEL:
            self.reset()
        self.prev_thumb = thumb
        if self.stab is not None:
            y = self.stab.apply(y)
        if self.defl is not None:
            y = self.defl.apply(y, float(thumb.mean()))
        if self.text is not None:
            y = self.text.apply(y)
        if self.dn is not None:
            y = self.dn.apply(y)
        if self.s.restore_sharpen > 0:
            y = sharpen(y, self.s.restore_sharpen)
        return y

    def process(self, frame: np.ndarray) -> np.ndarray:
        """frame: (H,W) gray, or (H,W,3) BGR (only luminance is changed)."""
        if frame.ndim == 2:
            return self._y(frame)
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycc[..., 0] = self._y(np.ascontiguousarray(ycc[..., 0]))
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
