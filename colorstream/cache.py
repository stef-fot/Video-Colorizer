"""On-disk segment cache keyed by source hash, settings signature and segment index."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path


def source_key(src: str, ignore_query: bool = False) -> str:
    """Stable id for a source. Local files include size and mtime so edits invalidate."""
    if src.lower().startswith(("http://", "https://")):
        ident = src.split("?", 1)[0] if ignore_query else src
    else:
        p = Path(src)
        try:
            st = p.stat()
            ident = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            ident = str(p)
    return hashlib.sha1(ident.encode()).hexdigest()[:16]


class SegmentCache:
    def __init__(self, root: str | Path, max_gb: float = 20.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * 1024 ** 3)
        self._lock = threading.Lock()
        self._size_cache: tuple[float, int] = (0.0, 0)

    def entry_dir(self, src_key: str, signature: str) -> Path:
        d = self.root / src_key / signature
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def segment_name(index: int) -> str:
        return f"seg_{index:05d}.ts"

    def segment_path(self, entry_dir: Path, index: int) -> Path:
        return entry_dir / self.segment_name(index)

    def has(self, entry_dir: Path, index: int) -> bool:
        return self.segment_path(entry_dir, index).exists()

    def touch(self, path: Path) -> None:
        try:
            os.utime(path, None)
        except OSError:
            pass

    # -- per-source metadata (probe result, color verdict) -------------------
    def read_meta(self, src_key: str) -> dict:
        p = self.root / src_key / "meta.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def write_meta(self, src_key: str, meta: dict) -> None:
        d = self.root / src_key
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp, d / "meta.json")

    # -- size and eviction ----------------------------------------------------
    def size_bytes(self, max_age: float = 5.0) -> int:
        now = time.time()
        with self._lock:
            if now - self._size_cache[0] < max_age:
                return self._size_cache[1]
        total = 0
        for dirpath, _, files in os.walk(self.root):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        with self._lock:
            self._size_cache = (now, total)
        return total

    def evict_if_needed(self, protected: set | None = None) -> int:
        """Delete least recently used settings directories until under the limit."""
        protected = {str(p) for p in (protected or set())}
        if self.size_bytes(0) <= self.max_bytes:
            return 0
        entries = []
        for src_dir in self.root.iterdir():
            if not src_dir.is_dir():
                continue
            for sig_dir in src_dir.iterdir():
                if sig_dir.is_dir() and str(sig_dir) not in protected:
                    newest = max((f.stat().st_mtime for f in sig_dir.glob("*.ts")), default=sig_dir.stat().st_mtime)
                    entries.append((newest, sig_dir))
        entries.sort()
        removed = 0
        for _, d in entries:
            if self.size_bytes(0) <= self.max_bytes * 0.9:
                break
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            with self._lock:
                self._size_cache = (0.0, 0)
        return removed

    def clear(self, protected: set | None = None) -> None:
        protected = {str(p) for p in (protected or set())}
        for src_dir in self.root.iterdir():
            if not src_dir.is_dir():
                continue
            for sig_dir in src_dir.iterdir():
                if sig_dir.is_dir() and str(sig_dir) not in protected:
                    shutil.rmtree(sig_dir, ignore_errors=True)
        with self._lock:
            self._size_cache = (0.0, 0)
