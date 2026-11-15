"""Subtitles in 21 languages: translation keeps timing, follows the source, and soft tracks are frame exact."""
import time

import pytest
from fastapi.testclient import TestClient

from colorstream.cache import SegmentCache, source_key
from colorstream.config import config_from_dict
from colorstream.ffmpeg import probe
from colorstream.server import create_api
from colorstream.session import SessionManager
from colorstream.subtitles import cues as cu
from colorstream.subtitles.asr import AsrResult
from colorstream.subtitles.cues import Cue
from colorstream.subtitles.languages import DEFAULT_TARGETS, LANGUAGES, NLLB_CODES, iso3, script_of
from colorstream.subtitles.render import find_font
from colorstream.subtitles.track import SubtitleTrack, TranslatedTrack
from colorstream.subtitles.translate import Translator, translate_cues, sane
from .conftest import needs_ffmpeg
from .test_subtitles import _bursts


class FakeMT(Translator):
    """Pretends to translate: marks the text with the language pair and counts calls."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def translate(self, texts, src, tgt):
        self.calls.append((src, tgt, len(texts)))
        return [f"[{tgt}] {t}" for t in texts]


class FakeEnglishAsr:
    def __init__(self):
        self.calls = 0

    def transcribe(self, samples, language="auto"):
        self.calls += 1
        return AsrResult([Cue(1.0, 2.2, "Good morning, how are you today?")], "en", 0.97)


# ------------------------------------------------------------------ the 21 languages


def test_default_targets_are_20_hot_languages_plus_greek():
    assert len(DEFAULT_TARGETS) == 21 and len(set(DEFAULT_TARGETS)) == 21 and "el" in DEFAULT_TARGETS
    for code in DEFAULT_TARGETS:
        assert code in LANGUAGES and code in NLLB_CODES, code
        assert len(iso3(code)) == 3
    cfg = config_from_dict({})
    assert cfg.subs.languages == DEFAULT_TARGETS


def test_scripts_of_new_languages():
    assert script_of("bn") == script_of("ta") == script_of("te") == script_of("mr") == "indic"
    assert script_of("ur") == "arabic" and script_of("sw") == "latin"


@pytest.mark.parametrize("code", DEFAULT_TARGETS)
def test_every_target_language_has_a_font_for_burning(code):
    if find_font(script_of(code)) is None:
        pytest.skip(f"no font for {code} on this machine")


# ------------------------------------------------------------------ cue translation


def test_translate_cues_keeps_timing_and_joins_lines():
    cues = [Cue(1.0, 2.0, "Hello\nworld"), Cue(3.0, 4.5, "Bye")]
    mt = FakeMT()
    out = translate_cues(cues, "en", "el", mt)
    assert [(c.start, c.end) for c in out] == [(1.0, 2.0), (3.0, 4.5)]
    assert out[0].text == "[el] Hello world"


def test_same_language_is_not_translated():
    mt = FakeMT()
    cues = [Cue(0, 1, "x")]
    assert translate_cues(cues, "el", "el", mt) == cues and not mt.calls


def test_bad_translation_falls_back_to_source_text():
    class Broken(Translator):
        def translate(self, texts, src, tgt):
            return ["", "a" * 500][: len(texts)] + ["ok"] * max(0, len(texts) - 2)

    out = translate_cues([Cue(0, 1, "one"), Cue(1, 2, "two"), Cue(2, 3, "three")], "en", "fr", Broken())
    assert [c.text for c in out] == ["one", "two", "ok"]
    assert sane("hello", "bonjour") and not sane("hi", "x" * 100)

    class Crash(Translator):
        def translate(self, texts, src, tgt):
            raise RuntimeError("boom")

    assert translate_cues([Cue(0, 1, "keep")], "en", "fr", Crash())[0].text == "keep"


# ------------------------------------------------------------------ soft tracks: frame exact times


def test_soft_cue_edges_sit_half_a_frame_early_so_they_land_on_the_exact_frames():
    fps = 23.976
    cue = cu.snap_to_frames([Cue(2.0, 3.0, "x")], fps)[0]
    f0, f1 = cu.frame_range(cue, fps)
    vtt = cu.to_vtt([cue], fps).splitlines()[2]
    a, b = [cu.parse("WEBVTT\n\n" + vtt + "\nx\n")[0].start, cu.parse("WEBVTT\n\n" + vtt + "\nx\n")[0].end]
    for k in range(f0 - 3, f1 + 3):  # which frames does a player show it on?
        shown = a <= k / fps < b
        assert shown == (f0 <= k < f1), k


def test_balance_lines():
    assert cu.balance_lines("short") == "short"
    two = cu.balance_lines("Αυτή είναι μια αρκετά μεγάλη πρόταση για δοκιμή των υπότιτλων").split("\n")
    assert len(two) == 2 and max(map(len, two)) <= 36
    assert "\n" in cu.balance_lines("这是一个很长的中文字幕句子需要分成两行来显示出来", script="cjk")
    assert cu.balance_lines("a\nb") == "a\nb"


# ------------------------------------------------------------------ tracks


@pytest.fixture()
def cfg(tmp_path):
    return config_from_dict({
        "paths": {"cache_dir": str(tmp_path / "cache"), "models_dir": str(tmp_path / "w")},
        "server": {"allow_local_files": True, "allowed_roots": [str(tmp_path)], "prebuffer_max_wait": 30},
        "processing": {"mode": "fast", "model": "demo", "output_height": 360, "buffer_seconds": 8},
        "subs": {"chunk_seconds": 5, "lookahead_seconds": 10, "background_full": False, "wait_seconds": 20},
    })


def _tracks(cfg, v, mt, asr, target):
    from dataclasses import replace
    s = replace(cfg.processing, subtitles="soft", subtitle_source="asr", subtitle_lang=target)
    info = probe(cfg, str(v))
    cache = SegmentCache(cfg.paths.cache_dir)
    key = source_key(str(v))
    source = SubtitleTrack(cfg, s, str(v), None, info, cache, key, backend=asr)
    return source, TranslatedTrack(cfg, s, str(v), None, info, cache, key, source, target, lambda: mt)


@needs_ffmpeg
def test_english_audio_gets_greek_subtitles_chunk_by_chunk_on_the_same_frames(cfg, tmp_path):
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 20)
    mt, asr = FakeMT(), FakeEnglishAsr()
    source, tr = _tracks(cfg, v, mt, asr, "el")
    assert tr.ensure(0.0, 3.0, timeout=60)
    fps = tr.fps
    f = int(round(1.5 * fps))
    assert tr.text_at_frame(f) == "[el] Good morning, how are you today?"
    assert source.text_at_frame(f) == "Good morning, how are you today?"  # same frame, source language
    assert [(s, t) for s, t, _ in mt.calls][0] == ("en", "el")
    assert tr.lang == "el" and source.lang == "en"
    n_after_first = len(mt.calls)
    tr.focus(11.0, speed=2.0)  # seek ahead, viewer plays at 2x: only the needed chunks are translated
    assert tr.ensure(11.0, 14.0, timeout=60)
    assert len(mt.calls) > n_after_first and asr.calls < tr.nchunks + 1
    assert tr.text_at_frame(int(round(11.6 * fps))) == "[el] Good morning, how are you today?"
    assert tr.export("vtt").startswith("WEBVTT") and "[el]" in tr.export("srt")
    tr.stop(); source.stop()


@needs_ffmpeg
def test_target_equal_to_spoken_language_mirrors_the_source(cfg, tmp_path):
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 10)
    mt, asr = FakeMT(), FakeEnglishAsr()
    source, tr = _tracks(cfg, v, mt, asr, "en")
    assert tr.wait_complete(timeout=60)
    assert not mt.calls
    assert tr.cues() == source.cues()


@needs_ffmpeg
def test_translation_is_cached_on_disk(cfg, tmp_path):
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 10)
    mt, asr = FakeMT(), FakeEnglishAsr()
    source, tr = _tracks(cfg, v, mt, asr, "ja")
    assert tr.wait_complete(timeout=60)
    calls = len(mt.calls)
    source2, tr2 = _tracks(cfg, v, FakeMT(), FakeEnglishAsr(), "ja")
    assert tr2.complete and tr2.cues() == tr.cues() and len(mt.calls) == calls
    other = _tracks(cfg, v, FakeMT(), FakeEnglishAsr(), "de")[1]
    assert not other.complete  # a different language is its own track


@needs_ffmpeg
def test_translation_errors_surface_instead_of_hanging(cfg, tmp_path):
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 10)

    def no_model():
        raise RuntimeError("Translation model not found in weights/x")

    from dataclasses import replace
    s = replace(cfg.processing, subtitles="soft", subtitle_source="asr", subtitle_lang="el")
    info = probe(cfg, str(v))
    cache = SegmentCache(cfg.paths.cache_dir)
    key = source_key(str(v))
    source = SubtitleTrack(cfg, s, str(v), None, info, cache, key, backend=FakeEnglishAsr())
    tr = TranslatedTrack(cfg, s, str(v), None, info, cache, key, source, "el", no_model)
    assert not tr.wait_complete(timeout=30)
    assert "Translation model not found" in tr.error


# ------------------------------------------------------------------ server + manager


@needs_ffmpeg
def test_server_serves_every_language_from_one_recognition_pass(cfg, tmp_path, monkeypatch):
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 10)
    asr, mt = FakeEnglishAsr(), FakeMT()
    from colorstream.subtitles import asr as asr_mod
    monkeypatch.setattr(asr_mod, "get_backend", lambda *a, **k: asr)
    mgr = SessionManager(cfg)
    mgr.translator_factory = lambda: mt
    client = TestClient(create_api(cfg, mgr))
    try:
        langs = client.get("/api/subs/languages").json()
        assert langs["defaults"] == DEFAULT_TARGETS and langs["languages"]["el"]["name"] == "Greek"
        r = client.post("/api/subs/prepare", json={"src": str(v), "langs": ["el", "es"]})
        assert r.status_code == 200 and set(r.json()) == {"source", "el", "es"}
        calls_after_prepare = asr.calls
        for lang in ("el", "es", "zh"):
            t = client.get("/subs.vtt", params={"src": str(v), "lang": lang})
            assert t.status_code == 200 and f"[{lang}]" in t.text, lang
            assert t.headers["x-colorstream-subtitles"] == "complete"
        original = client.get("/subs.vtt", params={"src": str(v)})
        assert "Good morning" in original.text and "[" not in original.text.split("\n", 2)[2]
        assert asr.calls <= calls_after_prepare + 1 or asr.calls <= 3  # the film was recognized once
        st = client.get("/api/status").json()["subtitle_tracks"]
        assert {t["language"] for t in st} >= {"en", "el", "es", "zh"}
        assert client.post("/api/subs/prepare", json={"src": "magnet:?x"}).status_code == 400
    finally:
        mgr.shutdown()


@needs_ffmpeg
def test_hls_segments_have_no_b_frames_so_soft_subtitle_clocks_agree(cfg, tmp_path):
    import subprocess
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 10)
    mgr = SessionManager(cfg)
    client = TestClient(create_api(cfg, mgr))
    try:
        r = client.get("/stream", params={"src": str(v)})
        url = [l for l in r.text.splitlines() if l.startswith("http")][0].replace("http://testserver", "")
        seg = tmp_path / "s.ts"
        seg.write_bytes(client.get(url).content)
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts_time,dts_time",
                              "-of", "csv=p=0", str(seg)], capture_output=True, text=True).stdout.split()
        assert out and all(line.split(",")[0] == line.split(",")[1] for line in out[:30])
    finally:
        mgr.shutdown()


@needs_ffmpeg
def test_cli_subtitles_writes_one_srt_per_language(tmp_path, monkeypatch):
    from colorstream import cli
    from colorstream.subtitles import asr as asr_mod, translate as tr_mod
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 10)
    (tmp_path / "c.yaml").write_text(
        f"paths: {{cache_dir: {tmp_path / 'cache'}, models_dir: {tmp_path / 'w'}}}\n"
        "subs: {chunk_seconds: 5, lookahead_seconds: 10}\n")
    monkeypatch.setattr(asr_mod, "get_backend", lambda *a, **k: FakeEnglishAsr())
    monkeypatch.setattr(tr_mod, "get_translator", lambda cfg: FakeMT())
    rc = cli.main(["-c", str(tmp_path / "c.yaml"), "subtitles", str(v), "--langs", "en,el,ja,hi"])
    assert rc == 0
    for lang in ("el", "ja", "hi"):
        text = (tmp_path / f"film.{lang}.srt").read_text(encoding="utf-8")
        assert f"[{lang}]" in text and "-->" in text
    en = (tmp_path / "film.en.srt").read_text(encoding="utf-8")
    assert "Good morning" in en and "[" not in en
