"""ONNX Runtime backends (DDColor and the Zhang et al. ECCV16/SIGGRAPH17 nets).

Providers are chosen in this order: CUDA, DirectML, CoreML, ROCm, CPU.
Falls back to CPU with a loud warning.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from ..color import gray_to_L100
from ..config import Config, ModelSpec
from .base import ColorModel, ModelNotFoundError

log = logging.getLogger("colorstream.models")

_AUTO_ORDER = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
]


def select_providers(cfg: Config) -> list:
    import onnxruntime as ort

    available = ort.get_available_providers()
    wanted = cfg.onnx.providers or ["auto"]
    if "auto" in wanted:
        chosen = [p for p in _AUTO_ORDER if p in available]
    else:
        chosen = [p for p in wanted if p in available]
        missing = [p for p in wanted if p not in available]
        if missing:
            log.warning("Requested ONNX providers not available: %s (available: %s)", missing, available)
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    if chosen == ["CPUExecutionProvider"]:
        log.warning(
            "No GPU provider for ONNX Runtime was found (available: %s). Running on CPU, "
            "which is far too slow for real time. Install onnxruntime-gpu (NVIDIA CUDA) or "
            "onnxruntime-directml (Windows, any GPU). See README.", available)
    out = []
    for p in chosen:
        if p == "CUDAExecutionProvider":
            out.append((p, {"device_id": cfg.onnx.device_id}))
        elif p == "DmlExecutionProvider":
            out.append((p, {"device_id": cfg.onnx.device_id}))
        else:
            out.append(p)
    return out


class OnnxModel(ColorModel):
    def __init__(self, spec: ModelSpec, cfg: Config):
        import onnxruntime as ort

        path = Path(cfg.paths.models_dir) / spec.file
        if not path.exists():
            raise ModelNotFoundError(
                f"Model weights for '{spec.name}' not found at {path}. "
                f"Run 'python scripts/download_models.py {spec.name}' and then "
                f"'python scripts/export_onnx.py {spec.name}', or see README section 'Model weights'. "
                f"To test the pipeline without weights use model 'demo'.")
        self.name = spec.name
        self.spec = spec
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if cfg.onnx.intra_threads:
            opts.intra_op_num_threads = cfg.onnx.intra_threads
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=select_providers(cfg))
        self.provider = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_is_fp16 = "float16" in inp.type
        shape = inp.shape  # [N, C, H, W], entries may be str or None for dynamic dims
        self.fixed_batch = shape[0] if isinstance(shape[0], int) else 0
        h, w = shape[2], shape[3]
        self.fixed_size = int(h) if isinstance(h, int) and isinstance(w, int) and h == w else (spec.size or 0)
        self._warned_nan = False
        self._lock = threading.Lock()
        log.info("Loaded %s on %s (fixed_size=%s, fixed_batch=%s)",
                 spec.name, self.provider, self.fixed_size or "dynamic", self.fixed_batch or "dynamic")

    # -- to be provided by subclasses --------------------------------------
    def _prepare(self, gray: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _run(self, x: np.ndarray) -> np.ndarray:
        if self.input_is_fp16:
            x = x.astype(np.float16)
        out = self.session.run(None, {self.input_name: x})[0]
        return out.astype(np.float32, copy=False)

    def predict(self, gray: np.ndarray) -> np.ndarray:
        n, s, _ = gray.shape
        x = self._prepare(gray)
        if self.fixed_batch:
            parts = [self._run(x[i:i + self.fixed_batch]) for i in range(0, n, self.fixed_batch)]
            out = np.concatenate(parts, axis=0)
        else:
            out = self._run(x)
        if out.shape[2] != s or out.shape[3] != s:
            out = np.stack([
                cv2.resize(np.ascontiguousarray(o.transpose(1, 2, 0)), (s, s),
                           interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                for o in out])
        if not np.isfinite(out).all():
            if not self._warned_nan:
                log.warning("Model '%s' produced NaN/Inf (FP16 problem on some GPUs). "
                            "Use an FP32 ONNX file. Replacing with zeros.", self.name)
                self._warned_nan = True
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -128.0, 127.0)


class DDColorModel(OnnxModel):
    """Input: gray image replicated to RGB in 0..1. Output: ab (N,2,H,W)."""

    def _prepare(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float32) / 255.0
        return np.ascontiguousarray(np.repeat(g[:, None], 3, axis=1))


class Eccv16Model(OnnxModel):
    """Input: Lab lightness (N,1,H,W) in 0..100. Output: ab in Lab units."""

    def _prepare(self, gray: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(gray_to_L100(gray)[:, None])
