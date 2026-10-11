"""Streaming sessions: look-ahead worker, seeking, pause/resume, session limits."""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from pathlib import Path

from .benchmark import auto_tune
from .cache import SegmentCache, source_key
from .config import Config, ProcSettings, apply_overrides
from .detect import source_is_color
from .ffmpeg import FFmpegError, MediaInfo, probe
from .models import load_model
from .pipeline import Cancelled, PassSpec, run_pass

log = logging.getLogger("colorstream.session")


class TooManySessions(RuntimeError):
    pass


class Passthrough:
    """Marker: the source is already in color, send the original."""

    def __init__(self, src: str, spreads: list):
        self.src = src
        self.spreads = spreads


def build_playlist(duration: float, seg: float, url_for_index) -> str:
    """VOD playlist listing every segment so players can seek anywhere."""
    nseg = int(math.ceil(duration / seg - 1e-9))
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{int(math.ceil(seg))}",
             "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-INDEPENDENT-SEGMENTS"]
    for i in range(nseg):
        d = min(seg, duration - i * seg)
        lines.append(f"#EXTINF:{d:.3f},")
        lines.append(url_for_index(i))
    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


class Session:
    def __init__(self, mgr: "SessionManager", sid: str, src: str, headers: dict | None,
                 settings: ProcSettings, info: MediaInfo, entry_dir: Path, tune_report: dict | None):
        self.mgr = mgr
        self.cfg = mgr.cfg
        self.cache = mgr.cache
        self.id = sid
        self.src = src
        self.headers = headers
        self.settings = settings
        self.info = info
        self.dir = entry_dir
        self.tune = tune_report or {}
        self.seg = settings.segment_seconds
        self.nseg = int(math.ceil(info.duration / self.seg - 1e-9))
        self.lookahead = max(1, int(math.ceil(settings.buffer_seconds / self.seg)))
        self.cond = threading.Condition()
        self.playhead = 0
        self.paused = False
        self.stopped = False
        self.worker: threading.Thread | None = None
        self.worker_alive = False
        self.pass_cancel = threading.Event()
        self.pass_start = 0
        self.pass_cur = 0
        self.fps = 0.0
        self.status = "starting"
        self.error: str | None = None
        self.created = time.time()
        self.last_access = time.time()

    # -- public API used by the HTTP layer ---------------------------------
    @property
    def ratio(self) -> float | None:
        """Processing speed relative to real time (1.0 = exactly real time)."""
        return self.fps / self.info.fps if self.fps > 0 else None

    def segment_path(self, i: int) -> Path:
        return self.cache.segment_path(self.dir, i)

    def ready_ahead(self) -> int:
        n = 0
        i = self.playhead
        while i < self.nseg and self.cache.has(self.dir, i):
            n += 1
            i += 1
        return n

    def cached_count(self) -> int:
        return sum(1 for i in range(self.nseg) if self.cache.has(self.dir, i))

    def request_segment(self, i: int) -> None:
        """Called whenever the player asks for segment i."""
        with self.cond:
            if self.stopped:
                raise RuntimeError("session stopped")
            self.last_access = time.time()
            self.playhead = max(0, min(i, self.nseg - 1))
            self.paused = False
            if not self.worker_alive:
                self._start_worker()
            elif not self.cache.has(self.dir, i):
                far = i < self.pass_start or i > self.pass_cur + self.lookahead + 1
                if far:
                    log.info("[%s] seek to segment %d, restarting processing", self.id[:6], i)
                    self.pass_cancel.set()
            self.cond.notify_all()

    def prebuffer_target(self) -> int:
        r = self.ratio
        if r is None:
            return 1
        if r >= 1.1:
            return min(2, self.nseg)
        need = int(math.ceil(self.lookahead * (1.0 - r / 1.1)))
        return max(2, min(self.lookahead, need, self.nseg))

    def pause(self) -> None:
        with self.cond:
            self.paused = True
            self.status = "paused"

    def resume(self) -> None:
        with self.cond:
            self.paused = False
            self.cond.notify_all()

    def stop(self) -> None:
        with self.cond:
            self.stopped = True
            self.status = "stopped"
            self.pass_cancel.set()
            self.cond.notify_all()

    def summary(self) -> dict:
        r = self.ratio
        return {
            "id": self.id, "source": _shorten(self.src), "status": self.status, "error": self.error,
            "fps": round(self.fps, 1), "speed": round(r, 2) if r else None, "source_fps": round(self.info.fps, 2),
            "model": self.settings.resolved_model(), "model_resolution": self.settings.model_resolution,
            "stride": self.settings.inference_stride, "saturation": self.settings.saturation,
            "output_height": self.settings.output_height, "playhead": self.playhead, "segments": self.nseg,
            "cached_segments": self.cached_count(),
            "buffer_seconds": round(self.ready_ahead() * self.seg, 1), "buffer_target": self.settings.buffer_seconds,
            "duration": round(self.info.duration, 1), "paused": self.paused,
            "tuning": self.tune, "idle_seconds": round(time.time() - self.last_access, 1),
        }

    # -- worker -------------------------------------------------------------
    def _start_worker(self) -> None:
        self.worker_alive = True
        self.pass_cancel = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True, name=f"cs-worker-{self.id[:6]}")
        self.worker.start()

    def _next_missing(self) -> int | None:
        for i in range(self.playhead, min(self.nseg, self.playhead + self.lookahead + 1)):
            if not self.cache.has(self.dir, i):
                return i
        return None

    def _worker_loop(self) -> None:
        failures = 0
        idle_since = time.time()
        try:
            while True:
                with self.cond:
                    if self.stopped:
                        return
                    target = self._next_missing()
                    if target is None:
                        if self.cached_count() >= self.nseg:
                            self.status = "complete"
                            return
                        self.status = "buffered"
                        if time.time() - idle_since > 120:
                            return  # idle, new requests restart the worker
                        self.cond.wait(timeout=1.0)
                        continue
                    idle_since = time.time()
                    self.pass_cancel = threading.Event()
                    self.pass_start = self.pass_cur = target
                    cancel = self.pass_cancel
                    self.status = "processing"
                try:
                    self._run(target, cancel)
                    failures = 0
                except Cancelled:
                    continue
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    self.error = f"{type(e).__name__}: {e}"
                    log.error("[%s] processing failed: %s", self.id[:6], self.error)
                    if failures >= 2:
                        self.status = "error"
                        return
                    time.sleep(1.0)
        finally:
            with self.cond:
                self.worker_alive = False
                self.cond.notify_all()

    def _run(self, start_idx: int, cancel: threading.Event) -> None:
        log.info("[%s] processing from segment %d (t=%.1fs)", self.id[:6], start_idx, start_idx * self.seg)
        spec = PassSpec(
            src=self.src, headers=self.headers, info=self.info, settings=self.settings,
            start=start_idx * self.seg, hls_dir=self.dir, hls_start_number=start_idx, with_audio=True,
            cancel=cancel, throttle=self._throttle, stop_at_segment=self._stop_at, on_progress=self._progress)
        run_pass(self.cfg, spec)
        self.mgr.cache.evict_if_needed(protected={self.dir})

    def _stop_at(self, idx: int) -> bool:
        self.pass_cur = idx
        return self.cache.has(self.dir, idx)

    def _progress(self, t_media: float, fps: float) -> None:
        self.fps = fps if self.fps == 0 else 0.6 * self.fps + 0.4 * fps
        self.pass_cur = max(self.pass_cur, int(t_media / self.seg))

    def _throttle(self, t_media: float) -> None:
        idx = int(t_media / self.seg)
        self.pass_cur = idx
        with self.cond:
            while not self.stopped and not self.pass_cancel.is_set():
                if self.paused:
                    self.status = "paused"
                elif idx > self.playhead + self.lookahead:
                    self.status = "buffered"
                else:
                    self.status = "processing"
                    return
                self.cond.wait(timeout=0.5)
        raise Cancelled()


def _shorten(src: str, n: int = 80) -> str:
    src = src.split("?", 1)[0]
    return src if len(src) <= n else src[:n // 2] + "..." + src[-n // 2:]


class SessionManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache = SegmentCache(cfg.paths.cache_dir, cfg.paths.cache_max_gb)
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._prep_locks: dict[str, threading.Lock] = {}
        self._reaper = threading.Thread(target=self._reap, daemon=True, name="cs-reaper")
        self._reaper.start()

    def get(self, sid: str) -> Session | None:
        return self.sessions.get(sid)

    def open(self, src: str, headers: dict | None, overrides: dict | None) -> Session | Passthrough:
        """Probe, check color, tune, and return a (possibly existing) session. Blocking."""
        base = apply_overrides(self.cfg.processing, overrides or {})
        key = source_key(src)
        prep_key = hashlib.sha1((key + base.signature()).encode()).hexdigest()[:16]
        with self._lock:
            plock = self._prep_locks.setdefault(prep_key, threading.Lock())
        with plock:
            meta = self.cache.read_meta(key)
            if "info" in meta:
                info = MediaInfo.from_dict(meta["info"])
            else:
                info = probe(self.cfg, src, headers)
                meta["info"] = info.to_dict()
                self.cache.write_meta(key, meta)
            if base.detect_color and not base.force_colorize:
                if "color" not in meta:
                    verdict, spreads = source_is_color(self.cfg, src, headers, info, base.color_threshold)
                    if spreads:  # only remember a verdict that was based on real samples
                        meta["color"] = {"is_color": verdict, "spreads": spreads}
                        self.cache.write_meta(key, meta)
                else:
                    verdict, spreads = meta["color"]["is_color"], meta["color"]["spreads"]
                if verdict:
                    log.info("Source is already in color, passing it through unchanged.")
                    return Passthrough(src, spreads)
            tune = None
            settings = base
            if base.mode == "auto" and not base.model:
                settings, tune = auto_tune(self.cfg, base, info.fps)
            load_model(self.cfg, settings.resolved_model())  # fail early with a readable error
            sid = hashlib.sha1((key + settings.signature()).encode()).hexdigest()[:16]
            with self._lock:
                existing = self.sessions.get(sid)
                if existing and not existing.stopped:
                    existing.last_access = time.time()
                    return existing
                self._make_room()
                entry = self.cache.entry_dir(key, settings.signature())
                sess = Session(self, sid, src, headers, settings, info, entry, tune)
                self.sessions[sid] = sess
                log.info("[%s] new session: %s model=%s res=%d stride=%d (%.1f fps, %.0fs)", sid[:6], _shorten(src),
                         settings.resolved_model(), settings.model_resolution, settings.inference_stride,
                         info.fps, info.duration)
                return sess

    def _make_room(self) -> None:
        live = [s for s in self.sessions.values() if not s.stopped]
        if len(live) < self.cfg.server.max_sessions:
            return
        victim = max(live, key=lambda s: time.time() - s.last_access)
        if time.time() - victim.last_access < 20:
            raise TooManySessions(
                f"Too many active sessions (limit {self.cfg.server.max_sessions}). Stop one in the dashboard "
                f"or raise server.max_sessions in config.yaml.")
        log.info("[%s] evicting idle session to make room", victim.id[:6])
        victim.stop()
        self.sessions.pop(victim.id, None)

    def stop_session(self, sid: str) -> bool:
        s = self.sessions.pop(sid, None)
        if s:
            s.stop()
        return s is not None

    def shutdown(self) -> None:
        for s in list(self.sessions.values()):
            s.stop()

    def protected_dirs(self) -> set:
        return {s.dir for s in self.sessions.values()}

    def _reap(self) -> None:
        while True:
            time.sleep(15)
            now = time.time()
            for sid, s in list(self.sessions.items()):
                if now - s.last_access > self.cfg.server.session_idle_seconds:
                    log.info("[%s] idle for %ds, closing session (cache is kept)", sid[:6], now - s.last_access)
                    self.stop_session(sid)
