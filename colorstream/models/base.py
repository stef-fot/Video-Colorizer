"""Model abstraction. Every backend maps small grayscale frames to Lab chroma."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModelNotFoundError(RuntimeError):
    """Raised with a human readable message when weights are missing."""


class ColorModel(ABC):
    name: str = "model"
    #: Fixed square input side, or 0 when any side works.
    fixed_size: int = 0
    provider: str = "cpu"

    def resolve_size(self, requested: int) -> int:
        return self.fixed_size or int(requested)

    @abstractmethod
    def predict(self, gray: np.ndarray) -> np.ndarray:
        """gray: (N,S,S) uint8 -> chroma (N,2,S,S) float32 in Lab a/b units."""

    def warmup(self, size: int) -> None:
        self.predict(np.full((1, size, size), 128, dtype=np.uint8))

    def info(self) -> dict:
        return {"name": self.name, "provider": self.provider, "fixed_size": self.fixed_size}
