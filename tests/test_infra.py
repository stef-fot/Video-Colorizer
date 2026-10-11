import os
import time

import pytest
from fastapi import HTTPException

from colorstream.cache import SegmentCache, source_key
from colorstream.config import ProcSettings, apply_overrides, config_from_dict
from colorstream.server import validate_source
from colorstream.session import build_playlist


def test_source_key_stable_and_distinct():
    assert source_key("https://a/b.mkv") == source_key("https://a/b.mkv")
    assert source_key("https://a/b.mkv") != source_key("https://a/c.mkv")
    assert source_key("https://a/b.mkv?t=1", True) == source_key("https://a/b.mkv?t=2", True)


def test_local_source_key_changes_when_file_changes(tmp_path):
    f = tmp_path / "v.mkv"
    f.write_bytes(b"a")
    k1 = source_key(str(f))
    f.write_bytes(b"abc")
    assert source_key(str(f)) != k1


def test_settings_signature_depends_on_pixels_not_buffer():
    a = ProcSettings()
    assert a.signature() == ProcSettings(buffer_seconds=99).signature()
    assert a.signature() != ProcSettings(saturation=0.5).signature()
    assert a.signature() != ProcSettings(model="eccv16").signature()


def test_overrides_are_validated():
    s = apply_overrides(ProcSettings(), {"saturation": 9, "smoothing": -1, "unknown": 1})
    assert s.saturation == 2.0 and s.smoothing == 0.0
    with pytest.raises(ValueError):
        apply_overrides(ProcSettings(), {"mode": "nope"})


def test_cache_eviction_removes_oldest_first(tmp_path):
    c = SegmentCache(tmp_path, max_gb=0)
    c.max_bytes = 2500
    old, new = c.entry_dir("src1", "sigA"), c.entry_dir("src2", "sigB")
    for d in (old, new):
        (d / "seg_00000.ts").write_bytes(b"x" * 2000)
    os.utime(old / "seg_00000.ts", (time.time() - 1000,) * 2)
    removed = c.evict_if_needed()
    assert removed == 1 and not old.exists() and new.exists()


def test_cache_eviction_skips_protected(tmp_path):
    c = SegmentCache(tmp_path, max_gb=0)
    c.max_bytes = 10
    d = c.entry_dir("s", "sig")
    (d / "seg_00000.ts").write_bytes(b"x" * 1000)
    c.evict_if_needed(protected={d})
    assert d.exists()


def test_playlist_lists_all_segments_with_last_shorter():
    text = build_playlist(10.0, 4.0, lambda i: f"seg_{i}.ts")
    assert text.count("#EXTINF") == 3
    assert "#EXTINF:2.000," in text and text.rstrip().endswith("#EXT-X-ENDLIST")
    assert "#EXT-X-PLAYLIST-TYPE:VOD" in text


def test_playlist_exact_multiple():
    assert build_playlist(8.0, 4.0, lambda i: str(i)).count("#EXTINF") == 2


def test_validate_source_rules(tmp_path):
    cfg = config_from_dict({"server": {"allow_local_files": True, "allowed_roots": [str(tmp_path)]}})
    assert validate_source(cfg, "https://example.com/v.mp4")
    for bad in ("magnet:?xt=urn:btih:1", "ftp://x/y", "", "file:///etc/passwd"):
        with pytest.raises(HTTPException):
            validate_source(cfg, bad)
    inside = tmp_path / "ok.mkv"
    inside.write_bytes(b"x")
    assert validate_source(cfg, str(inside)) == str(inside.resolve())
    with pytest.raises(HTTPException) as e:
        validate_source(cfg, "/etc/passwd")
    assert e.value.status_code == 403
    off = config_from_dict({})
    with pytest.raises(HTTPException) as e:
        validate_source(off, str(inside))
    assert e.value.status_code == 403


def _bench(per_frame_256=0.01, per_frame_512=0.05):
    return {"batch": 4, "cpu": {"render": 0.008, "temporal": 0.0005, "flow": 0.01},
            "models": {"ddcolor-tiny": {"provider": "x", "per_frame": {
                "192": per_frame_256 * 0.5, "256": per_frame_256, "384": (per_frame_256 + per_frame_512) / 2, "512": per_frame_512}},
                "demo": {"provider": "numpy", "per_frame": {"256": 0.0001, "192": 0.0001, "384": 0.0001, "512": 0.0001}}}}


def test_auto_tune_picks_best_quality_that_is_realtime(monkeypatch):
    from colorstream import benchmark
    from colorstream.models.base import ColorModel

    class Dyn(ColorModel):
        fixed_size = 0

        def predict(self, g):
            raise NotImplementedError

    monkeypatch.setattr(benchmark, "load_model", lambda cfg, n: Dyn())
    cfg = config_from_dict({})
    fast, rep = benchmark.choose_settings(cfg, _bench(0.002, 0.003), cfg.processing, 24.0)
    assert fast.model_resolution == 512 and fast.inference_stride == 1 and rep["realtime"]
    slow, rep = benchmark.choose_settings(cfg, _bench(0.03, 0.2), cfg.processing, 24.0)
    assert slow.inference_stride > 1 and slow.optical_flow  # stride needs flow to stay coherent
    hopeless, rep = benchmark.choose_settings(cfg, _bench(0.5, 1.0), cfg.processing, 24.0)
    assert not rep["realtime"]


def test_auto_tune_never_silently_uses_demo(monkeypatch):
    from colorstream import benchmark
    from colorstream.models import ModelNotFoundError

    cfg = config_from_dict({})
    bench = _bench()
    del bench["models"]["ddcolor-tiny"]
    with pytest.raises(ModelNotFoundError):
        benchmark.choose_settings(cfg, bench, cfg.processing, 24.0)
