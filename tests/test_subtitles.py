import subprocess
import threading
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from colorstream.cache import SegmentCache, source_key
from colorstream.config import config_from_dict
from colorstream.convert import ConvertOptions, convert_file
from colorstream.ffmpeg import probe
from colorstream.server import create_api
from colorstream.session import Session, SessionManager
from colorstream.subtitles import cues as cu
from colorstream.subtitles.align import cue_mask, estimate, mask_from_db
from colorstream.subtitles.asr import AsrResult, Word, drop_repeats, is_hallucination, words_to_cues
from colorstream.subtitles.cues import Cue
from colorstream.subtitles.languages import normalize_lang, script_of_text
from colorstream.subtitles.render import SubtitleRenderer, find_font, wrap_text
from colorstream.subtitles.track import SubtitleTrack
from .conftest import needs_ffmpeg

GREEK = "Καλημέρα κόσμε! Πώς είσαι σήμερα; Ένα ταξίδι στην Ελλάδα."

# ------------------------------------------------------------------ languages / encodings


def test_language_codes_greek_variants():
    for c in ("el", "el-GR", "ell", "gre", "gr", "Greek", "EL_gr"):
        assert normalize_lang(c) == "el"
    assert normalize_lang(None) == "auto" and normalize_lang("auto") == "auto"
    assert normalize_lang("es-419") == "es" and normalize_lang("deu") == "de"


def test_script_detection():
    assert script_of_text(GREEK) == "greek"
    assert script_of_text("Привет мир") == "cyrillic"
    assert script_of_text("你好，世界") == "cjk"
    assert script_of_text("Hello") == "latin"


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "utf-16", "cp1253", "iso8859-7"])
def test_greek_subtitles_decode_in_every_common_encoding(enc):
    data = f"1\n00:00:01,000 --> 00:00:02,500\n{GREEK}\n".encode(enc)
    text = cu.decode_bytes(data, "el")
    assert GREEK in text


def test_legacy_greek_is_detected_without_a_language_hint():
    data = f"1\n00:00:01,000 --> 00:00:02,500\n{GREEK}\n".encode("cp1253")
    assert GREEK in cu.decode_bytes(data, "auto")


def test_cp1251_russian_not_confused():
    data = "Привет, как дела?".encode("cp1251")
    assert cu.decode_bytes(data, "ru") == "Привет, как дела?"


# ------------------------------------------------------------------ parsing / cleaning / frames

SRT = "1\n00:00:01,000 --> 00:00:03,000\n<i>Γεια σου</i>\n\n2\n00:01:00,500 --> 00:01:02,000\nδεύτερο {\\an8}\nκείμενο\n"


def test_parse_srt_vtt_ass():
    c = cu.parse(SRT)
    assert len(c) == 2 and c[1].start == pytest.approx(60.5) and c[1].text.startswith("δεύτερο")
    v = cu.parse("WEBVTT\n\n00:01.000 --> 00:02.000\nHello\n\n01:00:00.000 --> 01:00:01.000\nLate\n")
    assert v[0].start == 1.0 and v[1].start == 3600.0
    ass = ("[Script Info]\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           "Dialogue: 0,0:00:01.50,0:00:03.00,Default,,0,0,0,,Καλά\\Nνέα, {\\b1}φίλε{\\b0}\n")
    a = cu.parse(ass)
    assert a[0].start == 1.5 and a[0].end == 3.0 and "Καλά\nνέα" in a[0].text


def test_roundtrip_srt_vtt():
    cues = [Cue(1.0, 2.5, GREEK), Cue(3661.25, 3662.0, "x")]
    assert cu.parse(cu.to_srt(cues)) == cues
    assert cu.parse(cu.to_vtt(cues)) == cues
    assert cu.to_vtt(cues).startswith("WEBVTT")


def test_clean_text_removes_markup_and_music():
    assert cu.clean_text("<i>Hi</i> {\\an8}there ♪♪\n\n  ok ") == "Hi there\nok"


def test_normalize_orders_removes_overlap_and_extends_short_cues():
    raw = [Cue(5.0, 5.1, "short but long enough text to read"), Cue(1.0, 4.9, "first"), Cue(5.05, 6.0, "overlap")]
    out = cu.normalize(raw)
    assert [c.text for c in out] == ["first", "short but long enough text to read", "overlap"]
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start + 1e-9  # never overlapping
    assert out[0].duration >= 3.0  # a short, readable cue is kept as it was


def test_snap_to_frames_is_exact_and_gapless_in_frame_space():
    fps = 23.976
    out = cu.normalize([Cue(1.01, 2.03, "a"), Cue(2.04, 3.5, "b")], fps)
    for c in out:
        f0, f1 = cu.frame_range(c, fps)
        assert f0 / fps == pytest.approx(c.start) and f1 / fps == pytest.approx(c.end)
        assert f1 > f0
    assert cu.frame_range(out[0], fps)[1] <= cu.frame_range(out[1], fps)[0]


def test_retime_fixes_delay_and_drift():
    c = cu.retime([Cue(100, 101, "x")], 23.976 / 25, 2.0)
    assert c[0].start == pytest.approx(100 * 23.976 / 25 + 2.0)


# ------------------------------------------------------------------ alignment


def _speech_cues(total=3600, seed=3):
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    while t < total:
        t += rng.uniform(1, 12)
        d = rng.uniform(1, 5)
        out.append(Cue(t, t + d, "x"))
        t += d
    return out


@pytest.mark.parametrize("scale,offset", [(1.0, -3.2), (1.0, 7.5), (25 / 23.976, 0.0), (23.976 / 25, 4.0)])
def test_alignment_recovers_delay_and_fps_drift(scale, offset):
    true = _speech_cues()
    speech = cue_mask(true, 36000)
    bad = [Cue((c.start - offset) / scale, (c.end - offset) / scale, "x") for c in true]
    a = estimate(speech, bad)
    fixed = a.apply(bad)
    err = np.mean([abs(f.start - t.start) for f, t in zip(fixed, true)])
    assert err < 0.3 and a.score > 0.95 and a.significant


def test_alignment_leaves_good_subtitles_alone():
    true = _speech_cues()
    a = estimate(cue_mask(true, 36000), true)
    assert not a.significant and a.scale == 1.0 and a.offset == 0.0


def test_alignment_ignores_unusable_audio():
    assert not estimate(np.zeros(5000, bool), _speech_cues(100)).significant
    assert mask_from_db(np.full(1000, -30.0)).sum() == 0  # constant level has no speech structure


# ------------------------------------------------------------------ speech recognition helpers


def test_words_to_cues_breaks_on_greek_punctuation_and_pauses():
    words = [Word(0.0, 0.4, "Καλημέρα"), Word(0.5, 0.9, "κόσμε!"), Word(1.0, 1.4, "Πώς"), Word(1.5, 1.9, "είσαι;"),
             Word(4.0, 4.4, "Καλά")]
    cues = words_to_cues(words, "el")
    assert [c.text for c in cues] == ["Καλημέρα κόσμε!", "Πώς είσαι;", "Καλά"]
    assert cues[0].start == 0.0 and cues[1].end == 1.9


def test_words_to_cues_cjk_joins_without_spaces_and_limits_length():
    words = [Word(i * 0.2, i * 0.2 + 0.2, ch) for i, ch in enumerate("你好世界今天天气很好我们去公园玩吧" * 4)]
    cues = words_to_cues(words, "zh", max_chars=20)
    assert all(" " not in c.text and len(c.text) <= 20 for c in cues)


def test_hallucination_filters():
    assert is_hallucination("Thanks for watching!", 0.0, 0.0, 1.0)
    assert is_hallucination("Υποτιτλισμός", 0.0, 0.0, 1.0)
    assert is_hallucination("ok", 0.9, -2.0, 1.0)
    assert not is_hallucination(GREEK, 0.05, -0.3, 1.2)
    assert len(drop_repeats([Cue(i, i + 1, "same") for i in range(8)])) == 3


# ------------------------------------------------------------------ rendering


def test_wrap_text_greek_and_cjk():
    measure = lambda s: 10.0 * len(s)  # noqa: E731
    lines = wrap_text("Καλημέρα κόσμε σήμερα είναι μια όμορφη μέρα", measure, 200, "greek")
    assert len(lines) >= 2 and all(measure(l) <= 200 for l in lines)
    assert " ".join(lines) == "Καλημέρα κόσμε σήμερα είναι μια όμορφη μέρα"
    cjk = wrap_text("你好世界今天天气很好我们去公园玩吧", measure, 100, "cjk")
    assert "".join(cjk) == "你好世界今天天气很好我们去公园玩吧" and all(measure(l) <= 100 for l in cjk)
    assert wrap_text("a\n\nb", measure, 100) == ["a", "b"]


def test_renderer_draws_greek_text_in_the_bottom_centre():
    if find_font("greek") is None:
        pytest.skip("no Greek capable font on this machine")
    frame = np.full((360, 640, 3), 90, np.uint8)
    out = SubtitleRenderer(640, 360, 0.06, lang="el").draw(frame.copy(), GREEK)
    changed = np.abs(out.astype(int) - frame.astype(int)).sum(axis=2) > 30
    ys, xs = np.nonzero(changed)
    assert len(ys) > 200 and ys.min() > 360 * 0.6  # lower part only
    assert abs((xs.min() + xs.max()) / 2 - 320) < 40  # centred


# ------------------------------------------------------------------ track, with a fake recognizer

pytestmark_ff = needs_ffmpeg


def _bursts(path, bursts, dur, size="320x180"):
    """Video with an audio track that has tone bursts at the given (start, end) times (stands in for speech)."""
    sr = 16000
    t = np.arange(int(sr * dur)) / sr
    sig = np.zeros_like(t)
    for a, b in bursts:
        m = (t >= a) & (t < b)
        sig[m] = 0.5 * np.sin(2 * np.pi * 300 * t[m]) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t[m]))
    sig += 0.002 * np.random.default_rng(0).standard_normal(len(t))
    wav = str(path) + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=24:d={dur},format=gray",
                    "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)


class FakeAsr:
    """Pretends to recognize Greek: one cue at 1.0-2.0 s inside every window it receives."""

    def __init__(self):
        self.calls = 0
        self.languages = []

    def transcribe(self, samples, language="auto"):
        self.calls += 1
        self.languages.append(language)
        return AsrResult([Cue(1.0, 2.2, GREEK)], "el", 0.95)


@pytest.fixture()
def cfg(tmp_path):
    return config_from_dict({
        "paths": {"cache_dir": str(tmp_path / "cache"), "models_dir": str(tmp_path / "w")},
        "server": {"allow_local_files": True, "allowed_roots": [str(tmp_path)], "prebuffer_max_wait": 30},
        "processing": {"mode": "fast", "model": "demo", "output_height": 360, "buffer_seconds": 8},
        "subs": {"chunk_seconds": 5, "lookahead_seconds": 10, "background_full": False, "wait_seconds": 20},
    })


def _track(cfg, path, tmp_path, backend=None, **over):
    from dataclasses import replace
    s = replace(cfg.processing, subtitles="burn", **over)
    info = probe(cfg, str(path))
    cache = SegmentCache(cfg.paths.cache_dir)
    return SubtitleTrack(cfg, s, str(path), None, info, cache, source_key(str(path)), backend=backend)


@needs_ffmpeg
def test_asr_track_fills_ahead_of_the_playhead_and_locks_language(cfg, tmp_path):
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 20)
    fake = FakeAsr()
    track = _track(cfg, v, tmp_path, backend=fake, subtitle_source="asr", subtitle_lang="auto")
    assert track.ensure(0.0, 3.0, timeout=30)  # first chunk
    assert track.lang == "el"
    assert track.text_at_frame(int(round(1.5 * track.fps))) == GREEK
    assert track.text_at_frame(int(round(0.5 * track.fps))) is None
    track.focus(11.0, speed=4.0)  # viewer jumps ahead and plays at 4x: lookahead grows
    assert track.ensure(11.0, 14.0, timeout=30)
    assert fake.languages[0] == "auto" and set(fake.languages[1:]) <= {"el"}  # locked after detection
    assert track.text_at_frame(int(round(11.6 * track.fps))) == GREEK  # 10 s chunk start + pad + 1.0 s


@needs_ffmpeg
def test_asr_track_is_cached_and_not_recomputed(cfg, tmp_path):
    v = tmp_path / "film.mp4"
    _bursts(v, [(1, 2)], 10)
    fake = FakeAsr()
    t1 = _track(cfg, v, tmp_path, backend=fake, subtitle_source="asr")
    assert t1.wait_complete(timeout=60)
    calls = fake.calls
    t2 = _track(cfg, v, tmp_path, backend=fake, subtitle_source="asr")
    assert t2.complete and t2.cues() == t1.cues() and fake.calls == calls
    assert "Καλημέρα" in t2.export("vtt") and t2.export("srt").startswith("1\n")


@needs_ffmpeg
def test_existing_greek_sidecar_is_found_decoded_and_resynced_to_the_audio(cfg, tmp_path):
    speech = [(2, 4), (6, 8.5), (11, 12), (14, 17), (19, 21), (24, 26), (28, 29.5), (32, 35), (37, 38), (41, 44),
              (47, 49.5), (52, 53), (55, 58), (60, 61.5), (64, 67)]
    v = tmp_path / "ταινία.mp4"
    _bursts(v, speech, 70)
    delay = 3.0  # the subtitle file is 3 seconds late
    lines = [f"{i}\n{cu._fmt(a + delay, ',')} --> {cu._fmt(b + delay, ',')}\nΓραμμή {i} ΄ ά έ ή\n" for i, (a, b) in enumerate(speech, 1)]
    (tmp_path / "ταινία.el.srt").write_bytes("\n".join(lines).encode("cp1253", errors="replace"))
    track = _track(cfg, v, tmp_path, subtitle_source="auto", subtitle_lang="el")
    assert track.wait_complete(timeout=60)
    assert track.kind == "existing" and track.lang == "el"
    assert track.alignment is not None and track.alignment.offset == pytest.approx(-delay, abs=0.3)
    fps = track.fps
    # at the first burst (t = 3 s) the first line must be on screen, and not before it starts
    assert track.text_at_frame(int(round(3.0 * fps))) is not None and "Γραμμή 1" in track.text_at_frame(int(round(3.0 * fps)))
    assert track.text_at_frame(int(round(1.5 * fps))) is None
    assert track.text_at_frame(int(round(5.5 * fps))) is None


@needs_ffmpeg
def test_embedded_and_missing_sources_report_clearly(cfg, tmp_path):
    v = tmp_path / "plain.mp4"
    _bursts(v, [(1, 2)], 6)
    track = _track(cfg, v, tmp_path, subtitle_source="existing")
    assert not track.wait_complete(timeout=20) and "No existing subtitles" in (track.error or "")


# ------------------------------------------------------------------ playback speed 2x / 4x


def _session_stub(seg=4.0, buffer=40.0):
    from colorstream.config import ProcSettings
    from collections import deque
    s = Session.__new__(Session)
    s.settings = ProcSettings(buffer_seconds=buffer, segment_seconds=seg)
    s.seg, s.req_log, s.speed = seg, deque(maxlen=200), 1.0
    s._speed_since = time.time() - 100
    return s


@pytest.mark.parametrize("speed", [1.0, 2.0, 4.0])
def test_playback_speed_estimate_and_lookahead_scale(speed, monkeypatch):
    s = _session_stub()
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    s._speed_since = clock[0] - 100
    idx = 0
    for _ in range(40):  # steady fetching: one segment every (seg / speed) seconds
        s._update_speed(idx)
        idx += 1
        clock[0] += 4.0 / speed
    assert s.speed == pytest.approx(speed, rel=0.25)
    assert s.lookahead >= int(10 * speed * 0.75)  # 40 s buffer * speed / 4 s segments


def test_seek_back_resets_speed_measurement(monkeypatch):
    s = _session_stub()
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    s._speed_since = clock[0] - 100
    for i in range(30):
        s._update_speed(i)
        clock[0] += 1.0
    assert s.speed > 2
    s._update_speed(2)  # seek back
    assert len(s.req_log) == 1


# ------------------------------------------------------------------ burned in, frame exact (end to end)


def _frames(path, w=640, h=360):
    res = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray",
                          "-s", f"{w}x{h}", "pipe:1"], capture_output=True)
    return np.frombuffer(res.stdout, np.uint8).reshape(-1, h, w)


@needs_ffmpeg
def test_burned_greek_subtitles_appear_on_exactly_the_right_frames(cfg, tmp_path):
    if find_font("greek") is None:
        pytest.skip("no Greek capable font on this machine")
    v = tmp_path / "bw.mp4"
    _bursts(v, [(1, 2)], 5, size="640x360")
    (tmp_path / "bw.el.srt").write_text(f"1\n00:00:02,000 --> 00:00:03,000\n{GREEK}\n", encoding="utf-8")
    from dataclasses import replace
    plain = replace(cfg.processing, subtitles="off", crf=0)  # lossless: identical except where text is drawn
    burn = replace(cfg.processing, subtitles="burn", subtitle_lang="el", subtitle_align=False, subtitle_size=0.06, crf=0)
    convert_file(cfg, v, tmp_path / "plain.mkv", plain, ConvertOptions(chunk_seconds=10))
    convert_file(cfg, v, tmp_path / "burn.mkv", burn, ConvertOptions(chunk_seconds=10))
    a, b = _frames(tmp_path / "plain.mkv"), _frames(tmp_path / "burn.mkv")
    assert len(a) == len(b)
    diff = np.abs(a.astype(int) - b.astype(int))[:, 200:, :].mean(axis=(1, 2))  # lower part of the picture
    shown = np.nonzero(diff > 0.05)[0]
    # 24 fps, cue 2.0 s to 3.0 s -> frames 48 to 71 inclusive, nothing before or after
    assert shown.min() == 48 and shown.max() == 71 and len(shown) == 24


@needs_ffmpeg
def test_soft_subtitles_are_written_muxed_and_tagged(cfg, tmp_path):
    v = tmp_path / "bw.mp4"
    _bursts(v, [(1, 2)], 5)
    (tmp_path / "bw.el.srt").write_text(f"1\n00:00:02,000 --> 00:00:03,000\n{GREEK}\n", encoding="utf-8")
    from dataclasses import replace
    soft = replace(cfg.processing, subtitles="soft", subtitle_lang="el", subtitle_align=False)
    out = tmp_path / "o.mkv"
    convert_file(cfg, v, out, soft, ConvertOptions(chunk_seconds=10))
    assert GREEK in (tmp_path / "o.el.srt").read_text(encoding="utf-8")
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries", "stream=codec_name:stream_tags=language",
                           "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert "subrip" in info and "ell" in info


@needs_ffmpeg
def test_subtitle_settings_do_not_split_the_video_cache_unless_burned():
    from colorstream.config import ProcSettings
    base = ProcSettings()
    assert base.signature() == ProcSettings(subtitles="soft", subtitle_lang="el").signature()
    assert base.signature() != ProcSettings(subtitles="burn", subtitle_lang="el").signature()
    assert ProcSettings(subtitles="burn", subtitle_lang="el").signature() != ProcSettings(subtitles="burn", subtitle_lang="en").signature()
    assert ProcSettings(subtitles="burn", subtitle_lang="el").signature() == ProcSettings(subtitles="both", subtitle_lang="el").signature()


@needs_ffmpeg
def test_server_serves_greek_vtt_and_burned_streams(cfg, tmp_path):
    v = tmp_path / "bw.mp4"
    _bursts(v, [(1, 2)], 8, size="320x180")
    (tmp_path / "bw.el.srt").write_bytes(f"1\n00:00:02,000 --> 00:00:03,000\n{GREEK}\n".encode("cp1253"))
    mgr = SessionManager(cfg)
    client = TestClient(create_api(cfg, mgr))
    try:
        r = client.get("/subs.vtt", params={"src": str(v), "lang": "el", "subsrc": "existing"})
        assert r.status_code == 200 and r.text.startswith("WEBVTT") and GREEK in r.text
        assert r.headers["x-colorstream-subtitles"] == "complete"
        assert "text/vtt" in r.headers["content-type"]
        assert client.get("/subs.srt", params={"src": str(v), "lang": "el"}).status_code == 200
        assert client.get("/subs.txt", params={"src": str(v)}).status_code == 404
        assert client.get("/subs.vtt", params={"src": "magnet:?x"}).status_code == 400
        p = client.get("/stream", params={"src": str(v), "subs": "burn", "lang": "el"})
        assert p.status_code == 200
        st = client.get("/api/status").json()["sessions"][0]
        assert st["subtitles"]["language"] == "el" and st["playback_speed"] == 1.0
    finally:
        mgr.shutdown()
