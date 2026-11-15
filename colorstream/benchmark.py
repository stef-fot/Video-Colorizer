"""Auto-benchmark: measure what this machine can do and pick settings so
playback never stalls. Results are cached on disk and re-used."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import threading
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


def measure_model(cfg: Config, name: str, batch: int, budget_s: float = 25.0) -> dict:
    """Seconds per frame of model inference for the sizes in the ladder.

    A model with a fixed input size is measured once, not once per ladder entry. Measuring stops when the
    time budget is used up (a slow CPU fallback must not turn the first start into a long wait).
    """
    model = load_model(cfg, name)
    out = {"provider": model.provider, "per_frame": {}}
    sizes = sorted({model.resolve_size(r) for r, _ in LADDER})
    t_start = time.perf_counter()
    for size in sizes:
        x = np.random.randint(0, 255, (batch, size, size), dtype=np.uint8)
        out["per_frame"][str(size)] = _time(lambda: model.predict(x), 2) / batch
        if time.perf_counter() - t_start > budget_s:
            log.warning("Benchmark of %s stopped early (%.0fs): measured sizes %s", name, budget_s,
                        list(out["per_frame"]))
            break
    return out


def measure_cpu_costs(out_h: int) -> dict:
    w = int(out_h * 16 / 9) // 2 * 2
    gray = np.random.randint(0, 255, (out_h, w), dtype=np.uint8)
    ab = np.random.randn(2, 256, 256).astype(np.float32) * 20
    small = cv2.resize(gray, (256, 256))
    small2 = np.roll(small, 3, axis=1)
    return {
        "render": _time(lambda: merge_luma_chroma(gray, ab, 0.85, guided=True), 5)
        + _time(lambda: cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA), 5),
        "temporal": _time(lambda: luma_histogram(small), 20),
        "flow": _time(lambda: warp_chroma(ab, farneback_flow(small, small2)), 5),
    }


_bench_lock = threading.Lock()  # one benchmark at a time, a second caller waits for the first


def needed_models(cfg: Config, settings: ProcSettings) -> list[str]:
    """The models auto-tuning may pick: the wanted one and its light fallbacks, never the heavy ones by accident."""
    wanted = settings.model or (settings.model_quality if settings.mode == "quality" else settings.model_fast)
    avail = {m["name"] for m in available_models(cfg) if m["available"]}
    names = [wanted] + [n for n in ("ddcolor-tiny", "siggraph17", "eccv16") if n != wanted]
    return [n for n in names if n in avail]


def cached_benchmark(cfg: Config, output_height: int, models: list[str] | None = None) -> dict | None:
    """A saved benchmark for this machine that covers `models`, or None."""
    p = _path(cfg)
    try:
        cached = json.loads(p.read_text())
    except Exception:
        return None
    if cached.get("signature") != _signature(cfg, output_height):
        return None
    if models and not all(m in cached.get("models", {}) for m in models):
        return None
    return cached


def run_benchmark(cfg: Config, output_height: int = 720, batch: int = 4, force: bool = False,
                  models: list[str] | None = None) -> dict:
    with _bench_lock:
        sig = _signature(cfg, output_height)
        if not force:
            cached = cached_benchmark(cfg, output_height, models)
            if cached is not None:
                return cached
        old = cached_benchmark(cfg, output_height) or {}
        log.info("Running benchmark (first run, about 30 seconds)...")
        t0 = time.perf_counter()
        result = {"signature": sig, "output_height": output_height, "batch": batch,
                  "models": dict(old.get("models", {})), "cpu": measure_cpu_costs(output_height)}
        names = models if models is not None else [m["name"] for m in available_models(cfg) if m["available"]]
        for name in names:
            if name in result["models"] and not force:
                continue
            try:
                result["models"][name] = measure_model(cfg, name, batch)
            except Exception as e:  # noqa: BLE001
                log.warning("Benchmark of %s failed: %s", name, e)
        p = _path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
        log.info("Benchmark finished in %.0fs", time.perf_counter() - t0)
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
    """Settings for this machine. Never blocks a stream on a first-time benchmark: if there is no saved
    benchmark yet, one starts in the background and sensible defaults are used for now."""
    out_h = output_height or settings.output_height or 720
    names = needed_models(cfg, settings)
    bench = cached_benchmark(cfg, out_h, names)
    if bench is not None:
        return choose_settings(cfg, bench, settings, source_fps)
    start_background_benchmark(cfg, out_h, names)
    model = names[0] if names else (settings.model or settings.model_fast)
    chosen = replace(settings, model=model, model_resolution=min(settings.model_resolution, 256),
                     adaptive_stride=True, max_stride=max(settings.max_stride, 6))
    return chosen, {"model": model, "model_resolution": chosen.model_resolution, "pending": True,
                    "note": "benchmark running in the background, safe defaults for now"}


_bg_started: set = set()


def start_background_benchmark(cfg: Config, out_h: int, models: list[str]) -> None:
    key = (out_h, tuple(models))
    if key in _bg_started or not models:
        return
    _bg_started.add(key)

    def work():
        try:
            run_benchmark(cfg, out_h, models=models)
        except Exception as e:  # noqa: BLE001
            log.warning("Background benchmark failed: %s", e)

    threading.Thread(target=work, daemon=True, name="cs-benchmark").start()
