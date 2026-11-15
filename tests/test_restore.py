"""Restoration: sharpen, denoise, deflicker, stabilize, text removal. Synthetic frames, no models."""
import subprocess

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from colorstream.config import ProcSettings, apply_overrides, config_from_dict
from colorstream.convert import AlreadyColor, ConvertOptions, convert_file
from colorstream.color import chroma_spread
from colorstream.restore import (Deflicker, Restorer, Stabilizer, TemporalDenoiser, TextRemover, parse_region,
                                 sharpen)
from colorstream.server import create_api
from colorstream.session import SessionManager
from .conftest import needs_ffmpeg


def texture(h=180, w=320, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 215, (h // 12, w // 12)).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 1.5)


def soft_shapes(h=180, w=320, blur=2.0):
    """High contrast shapes that were blurred: what a soft picture looks like."""
    y = np.full((h, w), 60, np.uint8)
    cv2.rectangle(y, (40, 30), (140, 120), 200, -1)
    cv2.circle(y, (230, 90), 45, 190, -1)
    cv2.line(y, (10, 160), (310, 150), 220, 3)
    return cv2.GaussianBlur(y, (0, 0), blur)


def lap_var(y):
    return float(cv2.Laplacian(y.astype(np.float32), cv2.CV_32F).var())


# ------------------------------------------------------------------ sharpen

def test_sharpen_increases_detail_without_halos():
    y = soft_shapes()
    out = sharpen(y, 0.6)
    assert lap_var(out) > 1.5 * lap_var(y)
    k = np.ones((7, 7), np.uint8)
    lo, hi = cv2.erode(y, k).astype(int) - 7, cv2.dilate(y, k).astype(int) + 7
    assert (out.astype(int) >= lo).all() and (out.astype(int) <= hi).all()  # never over or undershoots locally


def test_sharpen_leaves_flat_areas_and_zero_amount_alone():
    flat = np.full((90, 160), 120, np.uint8)
    assert np.array_equal(sharpen(flat, 0.9), flat)
    y = texture()
    assert sharpen(y, 0.0) is y


def test_sharpen_does_not_amplify_fine_noise():
    rng = np.random.default_rng(1)
    noisy = np.clip(120 + rng.normal(0, 1.0, (180, 320)), 0, 255).astype(np.uint8)
    out = sharpen(noisy, 0.5)
    assert float(out.astype(float).std()) < 1.6 * float(noisy.astype(float).std())


# ------------------------------------------------------------------ denoise

def test_temporal_denoise_reduces_noise_on_a_still_scene():
    rng = np.random.default_rng(2)
    clean = texture()
    dn = TemporalDenoiser(0.6)
    ins, outs = [], []
    for _ in range(12):
        noisy = np.clip(clean + rng.normal(0, 6, clean.shape), 0, 255).astype(np.uint8)
        ins.append(noisy.astype(float) - clean)
        outs.append(dn.apply(noisy).astype(float) - clean)
    assert np.std(outs[-1]) < 0.6 * np.std(ins[-1])


def test_temporal_denoise_leaves_no_ghost_behind_motion():
    dn = TemporalDenoiser(0.9)
    for i in range(10):
        f = np.full((90, 160), 60, np.uint8)
        f[:, 20:40] = 220
        dn.apply(f)
    f2 = np.full((90, 160), 60, np.uint8)
    f2[:, 100:120] = 220  # the bar jumped away
    out = dn.apply(f2)
    assert abs(int(out[45, 30]) - 60) < 12  # old position is already background
    assert abs(int(out[45, 110]) - 220) < 12


# ------------------------------------------------------------------ deflicker

def test_deflicker_evens_out_brightness_flicker_but_follows_slow_changes():
    rng = np.random.default_rng(3)
    base = texture().astype(np.float32)
    d = Deflicker()
    raw, fixed = [], []
    for i in range(60):
        g = 1.0 + rng.uniform(-0.07, 0.07)
        y = np.clip(base * g, 0, 255).astype(np.uint8)
        raw.append(y.mean())
        fixed.append(d.apply(y, float(y.mean())).mean())
    assert np.std(fixed[10:]) < 0.5 * np.std(raw[10:])
    d2 = Deflicker()
    means = []
    for i in range(80):  # a real fade, 1.0 -> 0.6 over 80 frames
        y = np.clip(base * (1.0 - 0.005 * i), 0, 255).astype(np.uint8)
        means.append(d2.apply(y, float(y.mean())).mean())
    assert means[-1] < 0.75 * means[0]  # the fade survives


# ------------------------------------------------------------------ stabilize

def test_stabilizer_reduces_shake():
    rng = np.random.default_rng(4)
    big = texture(260, 440, 9)
    h, w = 180, 320
    st = Stabilizer(w, h)
    raw, out = [], []
    for i in range(40):
        dx, dy = rng.integers(-4, 5), rng.integers(-4, 5)
        y = np.ascontiguousarray(big[40 + dy:40 + dy + h, 60 + dx:60 + dx + w])
        raw.append(y)
        out.append(st.apply(y))
    jitter = lambda fr: np.mean([np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(fr[10:], fr[11:])])  # noqa: E731
    assert jitter(out) < 0.75 * jitter(raw)


def test_stabilizer_survives_flat_frames():
    st = Stabilizer(160, 90)
    flat = np.full((90, 160), 100, np.uint8)
    for _ in range(5):
        assert st.apply(flat).shape == flat.shape


# ------------------------------------------------------------------ text removal

def frame_with_text():
    h, w = 360, 640
    bg = np.tile(np.linspace(70, 130, w, dtype=np.float32), (h, 1))
    bg = cv2.GaussianBlur(bg + np.random.default_rng(5).normal(0, 2, bg.shape).astype(np.float32), (0, 0), 1.0)
    clean = np.clip(bg, 0, 255).astype(np.uint8)
    y = clean.copy()
    for thick, col in ((6, 20), (2, 245)):  # dark outline, then white fill: how subtitles look
        cv2.putText(y, "SUBTITLE TEXT HERE", (90, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, thick, cv2.LINE_AA)
    return clean, y


def test_text_removal_cleans_the_region_and_touches_nothing_else():
    clean, y = frame_with_text()
    box = parse_region("bottom", 640, 360)
    out = TextRemover(box).apply(y)
    x0, y0, x1, y1 = box
    before = np.abs(y[y0:y1, x0:x1].astype(int) - clean[y0:y1, x0:x1]).mean()
    after = np.abs(out[y0:y1, x0:x1].astype(int) - clean[y0:y1, x0:x1]).mean()
    assert after < 0.35 * before and after < 6
    assert np.array_equal(out[:y0], y[:y0])  # outside the region: identical


def test_text_removal_skips_frames_without_text_and_bright_walls():
    clean, _ = frame_with_text()
    box = parse_region("bottom", 640, 360)
    tr = TextRemover(box)
    assert tr.apply(clean) is clean
    wall = np.full((360, 640), 235, np.uint8)
    assert tr.apply(wall) is wall


def test_region_parsing():
    assert parse_region("", 640, 360) is None
    assert parse_region("top", 640, 360)[1] == 0
    assert parse_region("0.1,0.5,0.5,0.25", 640, 360) == (64, 180, 384, 270)
    with pytest.raises(ValueError):
        apply_overrides(ProcSettings(), {"restore_text": "left"})
    with pytest.raises(ValueError):
        apply_overrides(ProcSettings(), {"restore_text": "0,0,2,1"})
    assert apply_overrides(ProcSettings(), {"restore_text": "off"}).restore_text == ""


# ------------------------------------------------------------------ the stage

def test_scene_cut_resets_temporal_state():
    r = Restorer(ProcSettings(restore_denoise=0.9), 320, 180)
    dark = np.full((180, 320), 40, np.uint8)
    bright = np.full((180, 320), 200, np.uint8)
    for _ in range(5):
        r.process(dark)
    out = r.process(bright)
    assert np.abs(out.astype(int) - 200).max() <= 2  # not blended with the previous scene


def test_color_frames_keep_their_chroma():
    base = soft_shapes()
    bgr = np.dstack([base, np.roll(base, 25, axis=1), 255 - base]).astype(np.uint8)
    out = Restorer(ProcSettings(restore_sharpen=0.8), 320, 180).process(bgr)
    a, b = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb), cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
    assert lap_var(b[..., 0]) > 1.3 * lap_var(a[..., 0])  # luma got sharper
    assert np.abs(a[..., 1:].astype(int) - b[..., 1:].astype(int)).mean() < 3.0  # chroma did not


def test_presets_and_flags():
    s = apply_overrides(ProcSettings(), {"restore": "medium", "restore_deflicker": True})
    assert (s.restore_sharpen, s.restore_denoise, s.restore_deflicker) == (0.5, 0.4, True) and s.restoring()
    assert not ProcSettings().restoring()
    assert ProcSettings().signature() != s.signature()  # restored frames are cached separately
    with pytest.raises(ValueError):
        apply_overrides(ProcSettings(), {"restore": "ultra"})


# ------------------------------------------------------------------ end to end

@pytest.fixture()
def cfg(tmp_path):
    return config_from_dict({
        "paths": {"cache_dir": str(tmp_path / "cache"), "models_dir": str(tmp_path / "w")},
        "server": {"allow_local_files": True, "allowed_roots": [str(tmp_path)], "prebuffer_max_wait": 30},
        "processing": {"mode": "fast", "model": "demo", "output_height": 180, "buffer_seconds": 8},
    })


def make(path, color, d=4, size="320x180"):
    vf = "" if color else ",format=gray"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=24:d={d}{vf}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def first_frame(path):
    res = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt",
                          "bgr24", "-s", "160x90", "pipe:1"], capture_output=True)
    return np.frombuffer(res.stdout, np.uint8).reshape(90, 160, 3)


@needs_ffmpeg
def test_restore_only_keeps_the_colors_of_a_color_file(cfg, tmp_path):
    from dataclasses import replace
    src, dst = tmp_path / "color.mp4", tmp_path / "out.mkv"
    make(src, True)
    s = replace(cfg.processing, colorize=False, restore_sharpen=0.5, restore_denoise=0.3, restore_deflicker=True,
                restore_stabilize=True, restore_text="bottom", detect_color=True)
    convert_file(cfg, src, dst, s, ConvertOptions(chunk_seconds=10))  # a color source must not be refused
    assert chroma_spread(first_frame(dst)) > 0.7 * chroma_spread(first_frame(src))
    with pytest.raises(AlreadyColor):  # while colorizing it still is
        convert_file(cfg, src, tmp_path / "o2.mkv", replace(cfg.processing, colorize=True))


@needs_ffmpeg
def test_gray_source_is_restored_and_colorized_in_one_pass(cfg, tmp_path):
    from dataclasses import replace
    src = tmp_path / "bw.mp4"
    make(src, False)
    s = replace(cfg.processing, restore_sharpen=0.7, restore_denoise=0.5, restore_stabilize=True)
    convert_file(cfg, src, tmp_path / "a.mkv", s, ConvertOptions(chunk_seconds=10))
    assert chroma_spread(first_frame(tmp_path / "a.mkv")) > 3


@needs_ffmpeg
def test_restoration_actually_changes_the_picture_and_compare_shows_the_original(cfg, tmp_path):
    from dataclasses import replace
    src = tmp_path / "color.mp4"
    make(src, True, d=3)
    plain = replace(cfg.processing, colorize=False)
    sharp = replace(cfg.processing, colorize=False, restore_sharpen=0.9)
    convert_file(cfg, src, tmp_path / "p.mkv", plain, ConvertOptions(chunk_seconds=10))
    convert_file(cfg, src, tmp_path / "s.mkv", sharp, ConvertOptions(chunk_seconds=10, compare=True))
    p = cv2.cvtColor(first_frame(tmp_path / "p.mkv"), cv2.COLOR_BGR2GRAY)
    res = subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp_path / "s.mkv"), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "pipe:1"], capture_output=True)
    wide = np.frombuffer(res.stdout, np.uint8).reshape(180, 640, 3)
    left, right = wide[:, :320], wide[:, 320:]
    assert lap_var(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)) > 1.15 * lap_var(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    assert p.shape == (90, 160)


@needs_ffmpeg
def test_server_enhances_a_color_stream_instead_of_passing_it_through(cfg, tmp_path):
    src = tmp_path / "color.mp4"
    make(src, True, d=6)
    mgr = SessionManager(cfg)
    client = TestClient(create_api(cfg, mgr))
    try:
        assert client.get("/stream", params={"src": str(src)}).headers.get("x-colorstream") == "passthrough-color"
        r = client.get("/stream", params={"src": str(src), "colorize": 0, "restore": "medium", "deflicker": 1})
        assert r.status_code == 200 and r.text.startswith("#EXTM3U")
        url = [l for l in r.text.splitlines() if l.startswith("http")][0].replace("http://testserver", "")
        seg = tmp_path / "e.ts"
        seg.write_bytes(client.get(url).content)
        assert chroma_spread(first_frame(seg)) > 15  # still the original colors
        bad = client.get("/stream", params={"src": str(src), "restore": "ultra"})
        assert bad.status_code == 400 and "restore" in bad.json()["detail"]
        info = client.get("/api/probe", params={"src": str(src)}).json()
        assert info["height"] == 180 and info["width"] == 320
        assert client.get("/api/probe", params={"src": "magnet:?x"}).status_code == 400
    finally:
        mgr.shutdown()
