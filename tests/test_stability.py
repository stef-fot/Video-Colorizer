"""Stable, natural colors and efficiency: shot memory, averaging, step limit, palette, adaptive inference, guided chroma."""
import cv2
import numpy as np
import pytest

from colorstream.color import guided_chroma, merge_luma_chroma, soft_limit_chroma
from colorstream.config import ProcSettings
from colorstream.engine import ChromaEngine
from colorstream.models.base import ColorModel


def scene(kind, seed):
    """Noisy texture frames; `kind` fixes the overall look so the same kind of shot matches again."""
    rng = np.random.default_rng(seed)
    base = {"A": 60, "B": 190, "C": 120, "D": 25}[kind]
    img = np.full((64, 64), base, np.float32)
    if kind == "A":
        img[:, :32] -= 25
    if kind == "B":
        img[:32] += 30
    if kind == "C":
        img[::2] += 40
    if kind == "D":
        img[16:48, 16:48] += 45
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


class NoisyModel(ColorModel):
    """Colors that depend on the shot (mean luma) plus a random wobble on every call: the 'dancing' of real models."""

    name = "noisy"

    def __init__(self, wobble=12.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.wobble = wobble
        self.frames = 0

    def predict(self, gray):
        self.frames += len(gray)
        out = np.zeros((len(gray), 2) + gray.shape[1:], np.float32)
        for k, g in enumerate(gray):
            m = float(g.mean())
            out[k, 0] = (m - 128) / 4 + self.rng.normal(0, self.wobble)
            out[k, 1] = (128 - m) / 6 + self.rng.normal(0, self.wobble)
        return out


def run(settings, frames, model=None, batch=4):
    model = model or NoisyModel()
    eng = ChromaEngine(model, settings, 64)
    out = []
    for i in range(0, len(frames), batch):
        out += eng.process(frames[i:i + batch])
    return eng, model, out


def jitter(chromas):
    """Mean absolute frame to frame change of the average chroma."""
    m = np.array([c.reshape(2, -1).mean(axis=1) for c in chromas])
    return float(np.abs(np.diff(m, axis=0)).mean())


OLD = dict(smoothing=0.7, shot_memory=False, shot_consistency=0.0, max_chroma_step=0, chroma_limit=0,
           adaptive_stride=False, optical_flow=False)


def test_new_defaults_dance_much_less_than_old_behaviour():
    frames = [scene("A", i) for i in range(60)]
    _, _, old = run(ProcSettings(**OLD), frames)
    _, _, new = run(ProcSettings(), frames)
    assert jitter(new) < 0.5 * jitter(old)


def test_start_of_a_shot_is_an_average_not_one_lucky_frame():
    frames = [scene("A", i) for i in range(12)]
    _, _, new = run(ProcSettings(**{**OLD, "smoothing": 0.88}), frames)
    _, _, first_frame_only = run(ProcSettings(**{**OLD, "smoothing": 0.88}), frames[:1])
    truth = (60 - 128 + 0) / 4  # noise free mean value of channel a for shot A (average of the two halves)
    err_avg = abs(np.mean([c[0].mean() for c in new[8:]]) - truth)
    # an average over several frames is nearer the truth than a single noisy frame on average
    errs_single = []
    for seed in range(20):
        _, _, one = run(ProcSettings(**OLD), frames[:1], NoisyModel(seed=seed))
        errs_single.append(abs(one[0][0].mean() - truth))
    assert err_avg < np.mean(errs_single)


def test_returning_shot_recalls_its_colors_and_skips_the_model():
    a1 = [scene("A", i) for i in range(20)]
    b = [scene("B", 100 + i) for i in range(20)]
    a2 = [scene("A", 200 + i) for i in range(20)]
    eng, model, out = run(ProcSettings(), a1 + b + a2)
    assert eng.stats.recalls >= 1
    end_a1 = out[19].reshape(2, -1).mean(axis=1)
    start_a2 = out[40].reshape(2, -1).mean(axis=1)
    assert np.abs(end_a1 - start_a2).max() < 2.5  # same set, same colors as before the cut
    # without memory the returning shot gets a new random look
    _, _, nomem = run(ProcSettings(shot_memory=False), a1 + b + a2)
    assert np.abs(nomem[19].reshape(2, -1).mean(axis=1) - nomem[40].reshape(2, -1).mean(axis=1)).max() > \
        np.abs(end_a1 - start_a2).max()


def test_different_looking_shots_are_not_confused():
    a = [scene("A", i) for i in range(15)]
    b = [scene("B", 50 + i) for i in range(15)]
    c = [scene("C", 90 + i) for i in range(15)]
    eng, _, _ = run(ProcSettings(), a + b + c)
    assert eng.stats.recalls == 0


def test_adaptive_stride_runs_the_model_less_on_static_pictures_but_not_on_motion():
    static = [scene("A", 1)] * 60
    eng, model, _ = run(ProcSettings(), static)
    assert model.frames <= 60 / 6 + 2  # every max_stride frames at most
    moving = []
    for i in range(60):  # a bright bar sweeping across: large change every frame
        g = np.full((64, 64), 80, np.uint8)
        g[:, (i * 5) % 50:(i * 5) % 50 + 14] = 230
        moving.append(g)
    eng2, model2, _ = run(ProcSettings(scene_cut_threshold=0.99), moving)
    assert model2.frames > 40
    # a scene cut always triggers the model
    eng3, model3, _ = run(ProcSettings(), [scene("A", 1)] * 10 + [scene("B", 1)] * 10)
    assert model3.frames >= 2


def test_stats_report_the_saving():
    eng, _, _ = run(ProcSettings(), [scene("A", 1)] * 48)
    assert 0 < eng.stats.inferred_fraction < 0.3


def test_chroma_cannot_jump_faster_than_the_step_limit():
    class Jumpy(ColorModel):
        name = "jumpy"
        n = 0

        def predict(self, gray):
            self.n += 1
            return np.full((len(gray), 2) + gray.shape[1:], 60.0 if self.n % 2 else -60.0, np.float32)

    frames = [scene("A", i) for i in range(40)]
    s = ProcSettings(max_chroma_step=3.0, smoothing=0.0, shot_consistency=0, chroma_limit=0, adaptive_stride=False)
    _, _, out = run(s, frames, Jumpy(), batch=1)
    diffs = [float(np.abs(out[i + 1] - out[i]).max()) for i in range(5, 39)]
    assert max(diffs) <= 3.0 + 1e-3


def test_shot_consistency_pulls_a_washed_out_shot_toward_the_film_level():
    class Level(ColorModel):
        name = "level"

        def predict(self, gray):
            lvl = 40.0 if gray[0].mean() < 100 else 8.0  # dark shots strongly colored, bright ones pale
            return np.full((len(gray), 2) + gray.shape[1:], lvl, np.float32)

    seq = []
    for k in range(4):  # establish a palette of strongly colored shots
        seq += [scene("A", 10 * k + i) for i in range(16)]
        seq += [scene("D", 500 + 10 * k + i) for i in range(16)]
    pale = [scene("B", 900 + i) for i in range(40)]
    base = dict(shot_memory=False, chroma_limit=0, smoothing=0.5, adaptive_stride=False)
    _, _, off = run(ProcSettings(shot_consistency=0.0, **base), seq + pale, Level())
    _, _, on = run(ProcSettings(shot_consistency=1.0, **base), seq + pale, Level())
    assert on[-1][0].mean() > off[-1][0].mean() + 1.5  # moved toward the film's level, gently
    assert on[-1][0].mean() < 40  # but not forced to it


def test_soft_limit_keeps_weak_colors_and_caps_strong_ones():
    ab = np.zeros((2, 4, 4), np.float32)
    ab[0] = 5
    ab[0, 0, 0] = 200
    out = soft_limit_chroma(ab, 80)
    assert out[0, 1, 1] == pytest.approx(5, rel=0.01)
    assert out[0, 0, 0] < 85
    assert np.array_equal(soft_limit_chroma(ab, 0), ab)


def test_guided_chroma_snaps_blurry_color_edges_to_the_luminance_edge():
    gray = np.full((256, 256), 50, np.uint8)
    gray[:, 128:] = 200
    step = np.zeros((64, 64), np.float32)
    step[:, 32:] = 40
    ab = np.stack([cv2.GaussianBlur(step, (0, 0), 3.0), np.zeros_like(step)])

    def row(guided):
        out = merge_luma_chroma(gray, ab, 1.0, guided=guided)
        return cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 1].astype(int)[128] - 128

    plain, guided = row(False), row(True)
    # just left of the edge the color should be near 0 and just right near 40
    assert guided[124] < plain[124] - 3 and guided[132] > plain[132] + 3
    assert guided[110] < 8 and guided[146] > 31
    # luminance is still the original one
    L = cv2.cvtColor(merge_luma_chroma(gray, ab, 1.0, guided=True), cv2.COLOR_BGR2LAB)[..., 0].astype(int)
    assert np.abs(L - cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)[..., 0]).mean() < 2


def test_guided_chroma_shape_and_flat_regions_unchanged():
    gray = np.random.default_rng(0).integers(0, 255, (180, 320), dtype=np.uint8)
    ab = np.full((2, 32, 32), 12.0, np.float32)
    out = guided_chroma(gray, ab)
    assert out.shape[2] == 2 and out.shape[0] <= 180
    assert abs(float(out.mean()) - 12.0) < 0.5  # a constant chroma stays constant
