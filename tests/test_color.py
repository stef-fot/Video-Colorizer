import cv2
import numpy as np
import pytest

from colorstream.color import (
    chroma_spread, gray_to_L8, is_color_content, merge_luma_chroma, upscale_chroma,
)


def _gray(h=72, w=128, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8)).astype(np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def _ab(h=16, w=16, a=30.0, b=-20.0):
    out = np.empty((2, h, w), np.float32)
    out[0], out[1] = a, b
    return out


def test_merge_preserves_original_luminance():
    gray = _gray()
    bgr = merge_luma_chroma(gray, _ab(), 1.0)
    assert bgr.shape == (72, 128, 3) and bgr.dtype == np.uint8
    L_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(int)
    L_in = gray_to_L8(gray).astype(int)
    assert np.abs(L_out - L_in).mean() < 2.0  # lightness survives the round trip


def test_zero_chroma_gives_neutral_gray():
    gray = _gray()
    bgr = merge_luma_chroma(gray, np.zeros((2, 8, 8), np.float32))
    assert np.abs(bgr.astype(int) - gray[..., None].astype(int)).max() <= 3


def test_saturation_zero_equals_gray_and_scales_chroma():
    gray = np.full((32, 32), 128, np.uint8)
    ab = _ab(8, 8, 40, 40)
    assert chroma_spread(merge_luma_chroma(gray, ab, 0.0)) < 0.5
    lab1 = cv2.cvtColor(merge_luma_chroma(gray, ab, 1.0), cv2.COLOR_BGR2LAB).astype(float)[..., 1] - 128
    lab5 = cv2.cvtColor(merge_luma_chroma(gray, ab, 0.5), cv2.COLOR_BGR2LAB).astype(float)[..., 1] - 128
    assert lab5.mean() == pytest.approx(lab1.mean() * 0.5, abs=3)


def test_chroma_is_upscaled_to_frame_size_and_luma_untouched_by_chroma_resolution():
    gray = _gray(90, 160)
    a = merge_luma_chroma(gray, _ab(8, 8))
    b = merge_luma_chroma(gray, _ab(64, 64))
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)[..., 0].astype(int)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)[..., 0].astype(int)
    assert np.abs(la - lb).max() <= 2
    assert upscale_chroma(_ab(8, 8), 160, 90).shape == (90, 160, 2)


def test_out_of_gamut_values_are_clipped_not_wrapped():
    gray = _gray()
    bgr = merge_luma_chroma(gray, _ab(8, 8, 500, -500))
    assert bgr.dtype == np.uint8  # no exception, no overflow wrap


def test_merge_rejects_non_gray_input():
    with pytest.raises(ValueError):
        merge_luma_chroma(np.zeros((8, 8, 3), np.uint8), _ab())


def test_chroma_spread_gray_vs_color_vs_tint():
    gray_bgr = cv2.cvtColor(_gray(), cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(3)
    color = cv2.resize(rng.integers(0, 255, (9, 16, 3)).astype(np.uint8), (128, 72), interpolation=cv2.INTER_NEAREST)
    sepia = np.clip(gray_bgr.astype(float) * np.array([0.8, 0.95, 1.1]), 0, 255).astype(np.uint8)
    assert chroma_spread(gray_bgr) < 1
    assert chroma_spread(color) > 20
    assert chroma_spread(sepia) < 7  # a uniform tint is still "black and white"


def test_is_color_content_uses_median():
    assert not is_color_content([0.2, 0.3, 0.1, 60.0, 0.2], 7.0)  # one colored title card
    assert is_color_content([40, 35, 50, 0.2], 7.0)
    assert not is_color_content([], 7.0)
