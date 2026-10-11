"""Model registry: name to loaded backend, cached for the process lifetime."""
from __future__ import annotations

import logging
import threading

from ..config import Config
from .base import ColorModel
from .demo import DemoModel

log = logging.getLogger("colorstream.models")

_cache: dict[str, ColorModel] = {}
_lock = threading.Lock()


def load_model(cfg: Config, name: str) -> ColorModel:
    with _lock:
        if name in _cache:
            return _cache[name]
        spec = cfg.model_spec(name)
        if spec.backend == "demo":
            model: ColorModel = DemoModel()
        elif spec.backend in ("ddcolor", "eccv16"):
            from .onnx_backend import DDColorModel, Eccv16Model

            cls = DDColorModel if spec.backend == "ddcolor" else Eccv16Model
            model = cls(spec, cfg)
        else:
            raise ValueError(f"Unknown backend '{spec.backend}' for model '{name}'")
        _cache[name] = model
        return model


def unload_all() -> None:
    with _lock:
        _cache.clear()


def available_models(cfg: Config) -> list[dict]:
    """List models with a flag telling whether weights are present."""
    from pathlib import Path

    out = []
    for m in cfg.models:
        present = m.backend == "demo" or (Path(cfg.paths.models_dir) / m.file).exists()
        out.append({"name": m.name, "backend": m.backend, "available": present, "description": m.description})
    return out
