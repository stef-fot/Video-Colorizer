"""FastAPI server: HLS endpoint for Stremio plus JSON API for the dashboard."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import platform
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

from . import __version__
from .benchmark import run_benchmark
from .config import Config, RUNTIME_SETTINGS, apply_overrides, save_config
from .ffmpeg import FFmpegError, check_binaries, pick_encoder
from .models import ModelNotFoundError, available_models
from .session import Passthrough, SessionManager, TooManySessions, build_playlist
from .sysinfo import gpu_stats

log = logging.getLogger("colorstream.server")
M3U8 = "application/vnd.apple.mpegurl"


def validate_source(cfg: Config, src: str) -> str:
    """Only direct HTTP(S) URLs (and optionally allow-listed local files) are accepted."""
    if not src:
        raise HTTPException(400, "Missing 'src' parameter.")
    low = src.lower()
    if low.startswith(("magnet:", "torrent:")):
        raise HTTPException(400, "Torrent and magnet links are not supported. Only direct HTTP(S) URLs "
                                 "that Stremio has already resolved are accepted.")
    if low.startswith(("http://", "https://")):
        return src
    if "://" in src:
        raise HTTPException(400, f"Unsupported URL scheme in '{src.split('://')[0]}://'. Use http or https.")
    if not cfg.server.allow_local_files:
        raise HTTPException(403, "Local files are disabled. Set server.allow_local_files: true and list "
                                 "folders in server.allowed_roots in config.yaml.")
    p = Path(src).resolve()
    roots = [Path(r).resolve() for r in cfg.server.allowed_roots]
    if not any(p == r or r in p.parents for r in roots):
        raise HTTPException(403, "This file is outside server.allowed_roots.")
    if not p.is_file():
        raise HTTPException(404, "File not found.")
    return str(p)


def create_api(cfg: Config, manager: SessionManager) -> FastAPI:
    app = FastAPI(title="ColorStream", version=__version__)
    dash = f"http://127.0.0.1:{cfg.server.dashboard_port}"
    app.add_middleware(
        CORSMiddleware, allow_origins=[dash, f"http://localhost:{cfg.server.dashboard_port}"],
        allow_methods=["GET", "POST", "PUT", "OPTIONS"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__}

    # ---- the stream entry point (this is what Stremio plays) ----------------
    @app.api_route("/stream", methods=["GET", "HEAD"])
    @app.api_route("/stream.m3u8", methods=["GET", "HEAD"])
    async def stream(request: Request, src: str = "", hdr: str = "", mode: str | None = None,
                     model: str | None = None, res: int | None = None, sat: float | None = None,
                     force: int | None = None, subs: str | None = None, lang: str | None = None,
                     subsrc: str | None = None, asr: str | None = None):
        src = validate_source(cfg, src)
        if request.method == "HEAD":
            return Response(status_code=200, media_type=M3U8, headers={"Cache-Control": "no-store"})
        headers = _parse_headers(hdr)
        overrides = {"mode": mode, "model": model, "model_resolution": res, "saturation": sat,
                     "force_colorize": bool(force) if force is not None else None,
                     "subtitles": subs, "subtitle_lang": lang, "subtitle_source": subsrc,
                     "subtitle_asr_model": asr}
        try:
            sess = await asyncio.to_thread(manager.open, src, headers, overrides)
        except TooManySessions as e:
            raise HTTPException(429, str(e))
        except ModelNotFoundError as e:
            raise HTTPException(503, str(e))
        except FFmpegError as e:
            raise HTTPException(502, str(e))
        if isinstance(sess, Passthrough):
            # Already in color: let the player talk to the origin directly (it handles Range itself).
            if sess.src.lower().startswith(("http://", "https://")):
                return RedirectResponse(sess.src, status_code=307, headers={"X-ColorStream": "passthrough-color"})
            return FileResponse(sess.src, headers={"X-ColorStream": "passthrough-color"})  # handles Range
        sess.request_segment(0)
        waited = await _prebuffer(sess, cfg.server.prebuffer_max_wait)
        base = str(request.base_url).rstrip("/")
        text = build_playlist(sess.info.duration, sess.seg, lambda i: f"{base}/hls/{sess.id}/seg_{i:05d}.ts")
        r = sess.ratio
        log.info("[%s] playlist served after %.1fs, speed=%s", sess.id[:6], waited,
                 f"{r:.2f}x realtime" if r else "unknown")
        return Response(text, media_type=M3U8, headers={
            "Cache-Control": "no-store", "X-ColorStream-Speed": f"{r:.2f}" if r else "unknown"})

    @app.get("/subs.{fmt}")
    async def subtitles(fmt: str, src: str = "", hdr: str = "", lang: str | None = None,
                        subsrc: str | None = None, asr: str | None = None):
        """Soft subtitles for the whole video as WebVTT or SRT (Stremio attaches them to the stream).

        Existing subtitles are returned right away (re-synced to the audio). Generated ones are
        returned as far as speech recognition got within subs.soft_wait_seconds; use burned-in
        subtitles (subs=burn) when you need them progressively while watching.
        """
        if fmt not in ("vtt", "srt"):
            raise HTTPException(404, "Use /subs.vtt or /subs.srt")
        src = validate_source(cfg, src)
        headers = _parse_headers(hdr)
        try:
            track = await asyncio.to_thread(
                manager.subtitle_track, src, headers,
                {"subtitle_lang": lang, "subtitle_source": subsrc, "subtitle_asr_model": asr})
        except FFmpegError as e:
            raise HTTPException(502, str(e))
        await asyncio.to_thread(track.wait_complete, cfg.subs.soft_wait_seconds)
        if track.error:
            raise HTTPException(503, f"Subtitles failed: {track.error}")
        ctype = "text/vtt; charset=utf-8" if fmt == "vtt" else "application/x-subrip; charset=utf-8"
        state = "complete" if track.complete else "partial"
        return Response(track.export(fmt), media_type=ctype, headers={
            "X-ColorStream-Subtitles": state, "Access-Control-Allow-Origin": "*",
            "Cache-Control": "max-age=3600" if track.complete else "no-store"})

    @app.get("/hls/{sid}/seg_{index}.ts")
    async def segment(sid: str, index: int):
        sess = manager.get(sid)
        if sess is None:
            raise HTTPException(404, "Unknown or expired session. Press play again to restart it.")
        if not 0 <= index < sess.nseg:
            raise HTTPException(404, "Segment out of range.")
        sess.request_segment(index)
        path = sess.segment_path(index)
        deadline = time.time() + cfg.server.segment_wait_timeout
        while not path.exists():
            if sess.status == "error":
                raise HTTPException(500, f"Colorization failed: {sess.error}")
            if sess.stopped or time.time() > deadline:
                raise HTTPException(504, "Segment not ready in time. The machine may be too slow for "
                                         "real time, see the dashboard.")
            await asyncio.sleep(0.15)
        manager.cache.touch(path)
        return FileResponse(path, media_type="video/mp2t", headers={"Cache-Control": "max-age=3600"})

    # ---- dashboard API --------------------------------------------------------
    @app.get("/api/status")
    def status():
        bench = None
        try:
            bp = Path(cfg.paths.cache_dir) / "benchmark.json"
            bench = json.loads(bp.read_text()) if bp.exists() else None
        except Exception:
            pass
        proc = cfg.processing
        return {
            "version": __version__, "platform": platform.platform(),
            "sessions": [s.summary() for s in manager.sessions.values()],
            "gpu": gpu_stats(),
            "cache": {"bytes": manager.cache.size_bytes(), "max_bytes": manager.cache.max_bytes},
            "settings": {k: getattr(proc, k) for k in RUNTIME_SETTINGS},
            "models": available_models(cfg),
            "encoder": pick_encoder(cfg),
            "benchmark": bench,
            "max_sessions": cfg.server.max_sessions,
        }

    @app.put("/api/config")
    async def update_config(request: Request):
        body = await request.json()
        try:
            new = apply_overrides(cfg.processing, body)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        cfg.processing = new
        saved = None
        if request.query_params.get("save") == "1":
            saved = save_config(cfg)
        return {"ok": True, "saved_to": saved}

    @app.post("/api/sessions/{sid}/{action}")
    def session_action(sid: str, action: str):
        s = manager.get(sid)
        if s is None:
            raise HTTPException(404, "No such session.")
        if action == "pause":
            s.pause()
        elif action == "resume":
            s.resume()
        elif action == "stop":
            manager.stop_session(sid)
        else:
            raise HTTPException(400, "Unknown action.")
        return {"ok": True}

    @app.post("/api/cache/clear")
    def clear_cache():
        manager.cache.clear(protected=manager.protected_dirs())
        return {"ok": True}

    @app.post("/api/benchmark")
    async def benchmark():
        result = await asyncio.to_thread(run_benchmark, cfg, cfg.processing.output_height or 720, 4, True)
        return result

    return app


def _parse_headers(hdr: str) -> dict | None:
    if not hdr:
        return None
    try:
        return {str(k): str(v) for k, v in json.loads(hdr).items()}
    except Exception:
        raise HTTPException(400, "Invalid 'hdr' parameter, expected a JSON object of request headers.")


async def _prebuffer(sess, max_wait: float) -> float:
    """Block until the first segments exist so playback starts smoothly."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if sess.status == "error":
            raise HTTPException(500, f"Colorization failed: {sess.error}")
        if sess.ready_ahead() >= sess.prebuffer_target() or sess.ready_ahead() >= sess.nseg:
            break
        await asyncio.sleep(0.25)
    return time.time() - t0


def create_dashboard(cfg: Config) -> FastAPI:
    app = FastAPI(title="ColorStream dashboard")
    html = (Path(__file__).parent / "dashboard" / "index.html").read_text(encoding="utf-8")
    html = html.replace("__API_PORT__", str(cfg.server.api_port))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html

    return app


async def serve(cfg: Config) -> None:
    import uvicorn

    check_binaries(cfg)
    manager = SessionManager(cfg)
    api = create_api(cfg, manager)
    dash = create_dashboard(cfg)
    host = cfg.server.host
    servers = [
        uvicorn.Server(uvicorn.Config(api, host=host, port=cfg.server.api_port, log_level="warning")),
        uvicorn.Server(uvicorn.Config(dash, host=host, port=cfg.server.dashboard_port, log_level="warning")),
    ]
    log.info("ColorStream API:       http://%s:%d  (Stremio plays /stream?src=...)", host, cfg.server.api_port)
    log.info("ColorStream dashboard: http://%s:%d", host, cfg.server.dashboard_port)
    try:
        await asyncio.gather(*(s.serve() for s in servers))
    finally:
        manager.shutdown()
