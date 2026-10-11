"""Demo backend: a fake, deterministic tint so the pipeline can be tested
without model weights. It is NOT a colorization model."""
from __future__ import annotations

import numpy as np

from .base import ColorModel


class DemoModel(ColorModel):
    name = "demo"
    provider = "numpy"

    def predict(self, gray: np.ndarray) -> np.ndarray:
        n, h, w = gray.shape
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None]  # 0 top, 1 bottom
        l = gray.astype(np.float32) / 255.0
        a = 18.0 * (l - 0.5) + 6.0 * y
        b = -30.0 * (1.0 - y) * (1.0 - l) + 25.0 * y * l
        out = np.empty((n, 2, h, w), dtype=np.float32)
        out[:, 0] = np.broadcast_to(a, (n, h, w))
        out[:, 1] = np.broadcast_to(b, (n, h, w))
        return out
