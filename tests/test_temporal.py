import numpy as np
import pytest

from colorstream.config import ProcSettings
from colorstream.engine import ChromaEngine
from colorstream.models.base import ColorModel
from colorstream.temporal import (
    ema_blend, farneback_flow, histogram_difference, is_scene_cut, luma_histogram, warp_chroma,
)


# ---------- EMA temporal smoothing ----------

def test_ema_first_frame_passes_through():
    new = np.ones((2, 4, 4), np.float32)
    assert np.array_equal(ema_blend(None, new, 0.8), new)


def test_ema_zero_smoothing_is_passthrough():
    prev, new = np.zeros((2, 4, 4), np.float32), np.ones((2, 4, 4), np.float32)
    assert np.array_equal(ema_blend(prev, new, 0.0), new)


def test_ema_weights():
    prev, new = np.full((2, 3, 3), 10.0, np.float32), np.full((2, 3, 3), 20.0, np.float32)
    out = ema_blend(prev, new, 0.75)
    assert np.allclose(out, 0.75 * 10 + 0.25 * 20)


def test_ema_converges_to_constant_input_and_damps_flicker():
    rng = np.random.default_rng(0)
    state = None
    outs = []
    for i in range(60):
        raw = np.full((2, 2, 2), 40.0 if i % 2 == 0 else -40.0, np.float32)  # flickering model output
        state = ema_blend(state, raw, 0.8)
        outs.append(float(state[0, 0, 0]))
    assert max(abs(x) for x in outs[10:]) < 12.0  # amplitude reduced from 40
    state = None
    for _ in range(80):
        state = ema_blend(state, np.full((2, 2, 2), 25.0, np.float32), 0.7)
    assert np.allclose(state, 25.0, atol=1e-3)


# ---------- scene cut detection ----------

def _frame(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "dark":
        return np.clip(rng.normal(40, 10, (64, 64)), 0, 255).astype(np.uint8)
    if kind == "bright":
        return np.clip(rng.normal(200, 20, (64, 64)), 0, 255).astype(np.uint8)
    raise ValueError(kind)


def test_histogram_normalized_and_identical_frames_zero_distance():
    h = luma_histogram(_frame("dark"))
    assert h.sum() == pytest.approx(1.0)
    assert histogram_difference(h, h) == 0.0


def test_no_cut_between_similar_frames():
    assert not is_scene_cut(luma_histogram(_frame("dark", 1)), luma_histogram(_frame("dark", 2)))


def test_cut_between_dark_and_bright_scene():
    assert is_scene_cut(luma_histogram(_frame("dark")), luma_histogram(_frame("bright")))


def test_no_cut_on_gradual_fade():
    base = _frame("bright").astype(np.float32)
    prev = luma_histogram(base.astype(np.uint8))
    for k in range(1, 31):  # fade to 25 percent brightness over 30 frames
        cur = luma_histogram((base * (1 - k / 40)).astype(np.uint8))
        assert not is_scene_cut(prev, cur, 0.5)
        prev = cur


def test_threshold_is_respected():
    a, b = luma_histogram(_frame("dark")), luma_histogram(_frame("bright"))
    assert not is_scene_cut(a, b, 1.01)


# ---------- warping ----------

def test_zero_flow_warp_is_identity():
    ab = np.random.default_rng(1).normal(0, 10, (2, 16, 16)).astype(np.float32)
    out = warp_chroma(ab, np.zeros((16, 16, 2), np.float32))
    assert np.allclose(out, ab, atol=1e-4)


def test_flow_of_translated_image_follows_motion():
    rng = np.random.default_rng(2)
    base = cv2_blur(rng.integers(0, 255, (96, 96)).astype(np.uint8))
    moved = np.roll(base, 4, axis=1)
    flow = farneback_flow(base, moved)
    assert flow.shape == (96, 96, 2)
    assert abs(float(np.median(flow[20:-20, 20:-20, 0])) + 4.0) < 1.5  # backward flow ~ -4 px in x


def cv2_blur(img):
    import cv2
    return cv2.GaussianBlur(img, (0, 0), 3)


# ---------- engine: cut resets state, stride reuses chroma ----------

class CountingModel(ColorModel):
    name = "counting"

    def __init__(self):
        self.calls = 0
        self.frames = 0

    def predict(self, gray):
        self.calls += 1
        self.frames += len(gray)
        mean = gray.reshape(len(gray), -1).mean(axis=1)
        out = np.zeros((len(gray), 2) + gray.shape[1:], np.float32)
        out[:, 0] = (mean / 255.0 * 50)[:, None, None]
        return out


def test_engine_resets_state_on_scene_cut():
    model = CountingModel()
    s = ProcSettings(smoothing=0.9, optical_flow=False, scene_cut_threshold=0.5)
    eng = ChromaEngine(model, s, 64)
    dark = [_frame("dark", i) for i in range(5)]
    bright = [_frame("bright", i) for i in range(5)]
    out = eng.process(dark + bright)
    expected_bright = float(bright[0].mean()) / 255.0 * 50
    # right after the cut the chroma is the fresh model output, not a blend with the dark scene
    assert out[5][0].mean() == pytest.approx(expected_bright, rel=0.02)
    assert eng.stats.cuts == 1


def test_engine_smooths_within_scene():
    model = CountingModel()
    eng = ChromaEngine(model, ProcSettings(smoothing=0.9, optical_flow=False), 64)
    levels = (100, 106, 95, 104, 98)  # same scene, flickering brightness
    frames = [np.clip(np.random.default_rng(i).normal(v, 25, (64, 64)), 0, 255).astype(np.uint8)
              for i, v in enumerate(levels)]
    out = [o[0].mean() for o in eng.process(frames)]
    raw = [f.mean() / 255.0 * 50 for f in frames]
    assert eng.stats.cuts == 0
    assert np.std(out) < np.std(raw)


def test_engine_stride_skips_model_between_keyframes():
    model = CountingModel()
    eng = ChromaEngine(model, ProcSettings(inference_stride=3, optical_flow=False, smoothing=0.0), 64)
    eng.process([_frame("dark", i) for i in range(9)])
    assert model.frames == 3  # frames 0, 3, 6
    eng.process([_frame("dark", i) for i in range(3)])
    assert model.frames == 4  # stride phase carries over between calls


def test_engine_reset_forces_inference():
    model = CountingModel()
    eng = ChromaEngine(model, ProcSettings(inference_stride=4, smoothing=0.5), 64)
    eng.process([_frame("dark")])
    eng.reset()
    eng.process([_frame("dark")])
    assert model.frames == 2
