"""GPU statistics (NVIDIA via nvidia-smi). Returns None when unavailable."""
from __future__ import annotations

import shutil
import subprocess
import time

_cache: tuple[float, dict | None] = (0.0, None)


def gpu_stats(max_age: float = 1.0) -> dict | None:
    global _cache
    now = time.time()
    if now - _cache[0] < max_age:
        return _cache[1]
    result = None
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            if out.returncode == 0 and out.stdout.strip():
                name, util, used, total, temp = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
                result = {"name": name, "utilization": float(util), "memory_used_mb": float(used),
                          "memory_total_mb": float(total), "temperature_c": float(temp)}
        except Exception:
            result = None
    _cache = (now, result)
    return result
