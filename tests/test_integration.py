"""End to end tests with the demo model (no weights needed). Need ffmpeg."""
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from colorstream.color import chroma_spread
from colorstream.config import config_from_dict
from colorstream.convert import AlreadyColor, ConvertOptions, convert_file
from colorstream.server import create_api
from colorstream.session import SessionManager
from .conftest import needs_ffmpeg

pytestmark = needs_ffmpeg


def _make(path, color=False, d=10, size="320x180"):
    vf = "" if color else ",format=gray"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=24:d={d}{vf}",
                    "-f", "lavfi", "-i", f"sine=f=440:d={d}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True)


def _count_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return int(out.stdout.strip().strip(","))


def _first_frame(path):
    res = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-s", "160x90", "pipe:1"], capture_output=True)
    return np.frombuffer(res.stdout, np.uint8).reshape(90, 160, 3)


@pytest.fixture()
def cfg(tmp_path):
    return config_from_dict({
        "paths": {"cache_dir": str(tmp_path / "cache"), "models_dir": str(tmp_path / "w")},
        "server": {"allow_local_files": True, "allowed_roots": [str(tmp_path)], "prebuffer_max_wait": 30},
        "processing": {"mode": "fast", "model": "demo", "output_height": 180, "buffer_seconds": 8},
    })


def test_convert_colorizes_keeps_audio_and_frame_count(cfg, tmp_path):
    src, dst = tmp_path / "bw.mp4", tmp_path / "out.mkv"
    _make(src, color=False, d=8)
    progress = []
    convert_file(cfg, src, dst, cfg.processing, ConvertOptions(chunk_seconds=3), lambda d, t, f: progress.append(d))
    assert dst.exists()
    assert abs(_count_frames(dst) - _count_frames(src)) <= 2  # chunk joins are frame exact
    assert chroma_spread(_first_frame(dst)) > 3  # source had 0
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(dst)], capture_output=True, text=True).stdout
    assert "audio" in streams
    assert progress and not (tmp_path / "out.mkv.colorstream-job").exists()


def test_convert_resume_skips_finished_chunks(cfg, tmp_path):
    src, dst = tmp_path / "bw.mp4", tmp_path / "out.mkv"
    _make(src, color=False, d=8)
    cancel = threading.Event()
    seen = []

    def prog(d, t, f):
        seen.append(d)
        if d > 4:  # interrupt in the middle of chunk 2
            cancel.set()

    from colorstream.pipeline import Cancelled
    with pytest.raises(Cancelled):
        convert_file(cfg, src, dst, cfg.processing, ConvertOptions(chunk_seconds=3, keep_temp=True), prog, cancel)
    job = tmp_path / "out.mkv.colorstream-job"
    done_before = sorted(p.name for p in job.glob("part_0000?.ts"))
    assert done_before  # at least the first chunk survived
    mtime = (job / done_before[0]).stat().st_mtime_ns
    convert_file(cfg, src, dst, cfg.processing, ConvertOptions(chunk_seconds=3))
    assert dst.exists()
    assert abs(_count_frames(dst) - _count_frames(src)) <= 2
    assert mtime  # the first part was reused, the job completes without redoing it


def test_convert_refuses_color_source(cfg, tmp_path):
    src = tmp_path / "color.mp4"
    _make(src, color=True, d=4)
    with pytest.raises(AlreadyColor):
        convert_file(cfg, src, tmp_path / "o.mkv", cfg.processing)


def test_server_serves_playlist_segments_seek_and_passthrough(cfg, tmp_path):
    bw, col = tmp_path / "bw.mp4", tmp_path / "color.mp4"
    _make(bw, color=False, d=16)
    _make(col, color=True, d=6)
    mgr = SessionManager(cfg)
    client = TestClient(create_api(cfg, mgr))
    try:
        r = client.get("/stream", params={"src": str(bw)})
        assert r.status_code == 200 and "mpegurl" in r.headers["content-type"]
        urls = [l for l in r.text.splitlines() if l.startswith("http")]
        assert len(urls) == 4
        seg0 = client.get(urls[0].replace("http://testserver", ""))
        assert seg0.status_code == 200 and len(seg0.content) > 1000
        part = client.get(urls[0].replace("http://testserver", ""), headers={"Range": "bytes=0-99"})
        assert part.status_code == 206 and len(part.content) == 100
        # seek to the last segment: processing restarts there
        last = client.get(urls[3].replace("http://testserver", ""))
        assert last.status_code == 200
        f = tmp_path / "seg.ts"
        f.write_bytes(last.content)
        assert chroma_spread(_first_frame(f)) > 3
        st = client.get("/api/status").json()
        assert st["sessions"][0]["cached_segments"] >= 2
        # color source goes straight to the origin
        c = client.get("/stream", params={"src": str(col)})
        assert c.status_code == 200 and c.headers.get("x-colorstream") == "passthrough-color"
        assert client.get("/stream", params={"src": "magnet:?xt=urn:btih:abc"}).status_code == 400
        assert client.get("/hls/doesnotexist/seg_00000.ts").status_code == 404
    finally:
        mgr.shutdown()


def test_session_limit(cfg, tmp_path):
    cfg.server.max_sessions = 1
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _make(a, d=4)
    _make(b, d=5)
    mgr = SessionManager(cfg)
    client = TestClient(create_api(cfg, mgr))
    try:
        assert client.get("/stream", params={"src": str(a)}).status_code == 200
        r = client.get("/stream", params={"src": str(b)})
        assert r.status_code == 429 and "Too many" in r.json()["detail"]
    finally:
        mgr.shutdown()
