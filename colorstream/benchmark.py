"""Auto-benchmark: measure what this machine can do and pick settings so
playback never stalls. Results are cached on disk and re-used."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .color import merge_luma_chroma
from .config import Config, ProcSettings
from .models import ModelNotFoundError, available_models, load_model
from .temporal import farneback_flow, luma_histogram, warp_chroma

log = logging.getLogger("colorstream.benchmark")

# Quality ladder, best first: (model_resolution, inference_stride)
LADDER = [(512, 1), (384, 1), (256, 1), (256, 2), (256, 3), (192, 3), (192, 4)]


def _path(cfg: Config) -> Path:
    return Path(cfg.paths.cache_dir) / "benchmark.json"


def _signature(cfg: Config, out_h: int) -> str:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception:
        providers = []
    files = []
    for m in cfg.models:
        p = Path(cfg.paths.models_dir) / m.file if m.file else None
        if p and p.exists():
            files.append((m.name, p.stat().st_size, int(p.stat().st_mtime)))
    blob = json.dumps([providers, files, out_h, platform.node(), cfg.onnx.fp16], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _time(fn, iters: int) -> float:
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def measure_model(cfg: Config, name: str, batch: int) -> dict:
    """Seconds per frame of model inference for each resolution in the ladder."""
    model = load_model(cfg, name)
    out = {"provider": model.provider, "per_frame": {}}
    for res in sorted({r for r, _ in LADDER}):
        size = model.resolve_size(res)
        x = np.random.randint(0, 255, (batch, size, size), dtype=np.uint8)
        out["per_frame"][str(size)] = _time(lambda: model.predict(x), 3) / batch
    return out


def measure_cpu_costs(out_h: int) -> dict:
    w = int(out_h * 16 / 9) // 2 * 2
    gray = np.random.randint(0, 255, (out_h, w), dtype=np.uint8)
    ab = np.random.randn(2, 256, 256).astype(np.float32) * 20
    small = cv2.resize(gray, (256, 256))
    small2 = np.roll(small, 3, axis=1)
    return {
        "render": _time(lambda: merge_luma_chroma(gray, ab, 0.85), 5)
        + _time(lambda: cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA), 5),
        "temporal": _time(lambda: luma_histogram(small), 20),
        "flow": _time(lambda: warp_chroma(ab, farneback_flow(small, small2)), 5),
    }


def run_benchmark(cfg: Config, output_height: int = 720, batch: int = 4, force: bool = False) -> dict:
    sig = _signature(cfg, output_height)
    p = _path(cfg)
    if p.exists() and not force:
        try:
            cached = json.loads(p.read_text())
            if cached.get("signature") == sig:
                return cached
        except Exception:
            pass
    log.info("Running benchmark (first run, takes about 30 seconds)...")
    result = {"signature": sig, "output_height": output_height, "batch": batch, "models": {}, "cpu": {}}
    result["cpu"] = measure_cpu_costs(output_height)
    for m in available_models(cfg):
        if not m["available"]:
            continue
        try:
            result["models"][m["name"]] = measure_model(cfg, m["name"], batch)
        except Exception as e:  # noqa: BLE001
            log.warning("Benchmark of %s failed: %s", m["name"], e)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    return result


def estimate_fps(bench: dict, model: str, res: int, stride: int, flow: bool) -> float:
    m = bench["models"][model]["per_frame"]
    key = str(min((int(k) for k in m), key=lambda k: abs(k - res)))
    cpu = bench["cpu"]
    per_frame = m[key] / stride + cpu["render"] + cpu["temporal"]
    if flow:
        per_frame += cpu["flow"]
    return 1.0 / per_frame


def choose_settings(cfg: Config, bench: dict, settings: ProcSettings, source_fps: float,
                    margin: float = 1.25) -> tuple[ProcSettings, dict]:
    """Pick model, resolution and stride so that estimated fps >= source fps * margin."""
    wanted = settings.model or (settings.model_quality if settings.mode == "quality" else settings.model_fast)
    # The fake 'demo' model is never picked silently, only when asked for explicitly.
    candidates = [wanted] + [n for n in ("ddcolor-tiny", "siggraph17", "eccv16") if n != wanted]
    model = next((n for n in candidates if n in bench["models"]), None)
    if model is None:
        raise ModelNotFoundError(
            f"No colorization model is installed (wanted '{wanted}'). Download and export weights, see README "
            f"section 'Model weights', or set processing.model: demo to test the pipeline with a fake tint.")
    if model != wanted:
        log.warning("Model '%s' is not installed, falling back to '%s'.", wanted, model)
    fixed = load_model(cfg, model).fixed_size
    target = source_fps * margin
    best = None
    for res, stride in LADDER:
        if fixed and res != LADDER[2][0] and stride == 1:
            continue
        flow = stride > 1
        est = estimate_fps(bench, model, fixed or res, stride, flow)
        best = (res, stride, flow, est)
        if est >= target:
            break
    res, stride, flow, est = best
    chosen = replace(settings, model=model, model_resolution=res, inference_stride=stride,
                     optical_flow=settings.optical_flow or flow, batch_size=bench.get("batch", 4))
    report = {"model": model, "model_resolution": fixed or res, "inference_stride": stride,
              "optical_flow": chosen.optical_flow, "estimated_fps": round(est, 1),
              "realtime": est >= source_fps * 1.05, "target_fps": round(target, 1)}
    if not report["realtime"]:
        log.warning("This machine is estimated at %.1f fps, below the source rate of %.1f fps. "
                    "Playback will pre-buffer, and may stall. Lower output_height or use a lighter model.",
                    est, source_fps)
    return chosen, report


def auto_tune(cfg: Config, settings: ProcSettings, source_fps: float, output_height: int | None = None):
    bench = run_benchmark(cfg, output_height or settings.output_height or 720)
    return choose_settings(cfg, bench, settings, source_fps)
