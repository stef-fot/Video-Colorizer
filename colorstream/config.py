"""Configuration: one YAML file, sensible defaults, dataclasses for access."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("colorstream.config")


@dataclass
class PathsCfg:
    cache_dir: str = "cache"
    models_dir: str = "weights"
    cache_max_gb: float = 20.0


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    api_port: int = 7002
    dashboard_port: int = 7001
    max_sessions: int = 2
    session_idle_seconds: int = 300
    # Local files are refused by the HTTP server unless explicitly allowed.
    allow_local_files: bool = False
    allowed_roots: list = field(default_factory=list)
    # How long the playlist request may block while pre-buffering (seconds).
    prebuffer_max_wait: float = 25.0
    segment_wait_timeout: float = 120.0


@dataclass
class ProcSettings:
    """Everything that changes the pixels. Part of the cache key."""

    mode: str = "auto"  # auto | fast | quality
    model_fast: str = "ddcolor-tiny"
    model_quality: str = "ddcolor-large"
    model: str = ""  # explicit model name, overrides mode when set
    model_resolution: int = 256  # side of the square grayscale copy fed to the model
    output_height: int = 720  # 0 keeps source height
    batch_size: int = 4
    inference_stride: int = 1  # run the model every N frames, warp chroma in between
    saturation: float = 0.85
    smoothing: float = 0.88  # EMA weight of history, 0 disables temporal smoothing
    optical_flow: bool = False
    # --- restoration (any source, color or black and white; see README "Restoration") ---
    colorize: bool = True  # False = restore only, colors of a color source are kept untouched
    restore_sharpen: float = 0.0  # 0..1 halo-limited sharpening of the luminance
    restore_denoise: float = 0.0  # 0..1 motion-adaptive temporal plus light spatial denoise
    restore_deflicker: bool = False  # even out brightness flicker (old film, bad exposure)
    restore_stabilize: bool = False  # remove small camera shake (translation only)
    restore_text: str = ""  # remove hardcoded text: "" off | bottom | top | x,y,w,h as fractions of the picture
    # --- stable, natural color (see README "Stable colors") ---
    shot_memory: bool = True  # a returning shot (shot / reverse shot) starts from the colors it had before
    shot_consistency: float = 0.5  # 0..1 pull every shot toward the film's overall color level
    max_chroma_step: float = 5.0  # max change of a chroma value per frame, stops sudden color jumps (0 = off)
    chroma_limit: float = 80.0  # soft ceiling for chroma strength, avoids garish colors (0 = off)
    edge_aware: bool = True  # luma-guided chroma upsampling, colors stop leaking over edges
    # --- efficiency ---
    adaptive_stride: bool = True  # run the model only when the picture changed (or every max_stride frames)
    max_stride: int = 6
    motion_threshold: float = 3.0  # mean gray change (0..255, on a 16x16 thumbnail, so film grain is ignored) that triggers the model
    scene_cut_threshold: float = 0.5
    segment_seconds: float = 4.0
    buffer_seconds: float = 40.0
    warmup_frames: int = 8
    detect_color: bool = True
    color_threshold: float = 7.0
    force_colorize: bool = False
    crf: int = 21
    # Subtitles: off | soft (separate track, sidecar file) | burn (drawn on every frame) | both
    subtitles: str = "off"
    subtitle_lang: str = "auto"  # language the viewer wants: auto = as spoken; or el, en, es, zh, hi ... (translated)
    subtitle_audio_lang: str = "auto"  # language spoken in the film (hint for recognition), auto = detect
    subtitle_source: str = "auto"  # auto = use existing if found, else speech recognition | existing | asr
    subtitle_file: str = ""  # explicit .srt/.vtt/.ass to use
    subtitle_asr_model: str = "small"  # faster-whisper size: tiny, base, small, medium, large-v3
    subtitle_align: bool = True  # re-sync existing subtitles to the audio
    subtitle_size: float = 0.045  # burned text height as a fraction of the frame height

    def restoring(self) -> bool:
        return bool(self.restore_sharpen > 0 or self.restore_denoise > 0 or self.restore_deflicker
                    or self.restore_stabilize or self.restore_text)

    def burns_subtitles(self) -> bool:
        return self.subtitles in ("burn", "both")

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return self.model_quality if self.mode == "quality" else self.model_fast

    def signature(self) -> str:
        d = asdict(self)
        for k in ("buffer_seconds", "detect_color", "color_threshold", "force_colorize",
                  "mode", "model_fast", "model_quality", "model", "batch_size"):
            d.pop(k, None)
        if self.burns_subtitles():
            d["subtitles"] = "burn"  # burn and both produce the same pixels
        else:  # soft or off: the picture is the same, subtitle settings must not split the cache
            for k in [k for k in d if k.startswith("subtitle")]:
                d.pop(k)
        d["model"] = self.resolved_model()
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


@dataclass
class SubsCfg:
    device: str = "auto"  # speech recognition device: auto | cpu | cuda
    compute_type: str = "int8"  # int8 is the safest choice on GTX 16xx cards; "auto" lets the library decide
    prompt: str = ""  # optional vocabulary hint for the recognizer, e.g. names in the film
    chunk_seconds: float = 30.0  # recognition work unit
    lookahead_seconds: float = 90.0  # recognize this far ahead of the playhead (scaled by playback speed)
    background_full: bool = True  # after the look-ahead is done, keep transcribing the rest of the film
    wait_seconds: float = 180.0  # burn-in: max wait for subtitles of a frame range before giving up
    soft_wait_seconds: float = 30.0  # soft export endpoint: max wait for recognition
    font: str = ""  # path to a TTF/OTF covering your language, empty = automatic
    extract_remote: bool = False  # also read embedded subtitles / audio of remote URLs (downloads the file)
    # --- translation to the 21 target languages ---
    languages: list = field(default_factory=lambda: ["en", "zh", "hi", "es", "fr", "ar", "bn", "pt", "ru", "ur", "id",
                                                       "de", "ja", "sw", "mr", "te", "tr", "ta", "vi", "ko", "el"])
    translator: str = "nllb"  # nllb | argos
    nllb_dir: str = "weights/nllb-200-distilled-600M-ct2-int8"
    translate_device: str = "auto"
    translate_compute_type: str = "int8"
    pretranslate: list = field(default_factory=list)  # translate these right after recognition, e.g. ["el", "en"]
    min_duration: float = 0.7
    max_duration: float = 7.0
    chars_per_second: float = 20.0


@dataclass
class EncodeCfg:
    video_encoder: str = "auto"  # auto | libx264 | h264_nvenc | h264_videotoolbox | h264_qsv | h264_amf | h264_vaapi
    hwaccel_decode: str = "auto"  # auto | none | cuda | d3d11va | videotoolbox | vaapi
    preset: str = "veryfast"  # libx264 preset used for streaming
    vaapi_device: str = "/dev/dri/renderD128"


@dataclass
class OnnxCfg:
    providers: list = field(default_factory=lambda: ["auto"])
    device_id: int = 0
    fp16: bool = False  # GTX 16xx cards have flaky FP16, keep off unless you tested it
    intra_threads: int = 0
    allow_cpu_fallback: bool = True  # if the GPU provider cannot run a model, use the CPU (slow) instead of failing


@dataclass
class ModelSpec:
    name: str
    backend: str  # ddcolor | eccv16 | demo
    file: str = ""
    size: int = 0  # fixed input side, 0 means flexible
    url: str = ""
    sha256: str = ""
    description: str = ""


def default_models() -> list:
    return [
        ModelSpec("demo", "demo", description="No weights needed. Fake tint, only for testing the pipeline."),
        ModelSpec("eccv16", "eccv16", "eccv16.onnx", 256,
                  description="Zhang et al. 2016. Small and fast, muted colors. Export with scripts/export_onnx.py."),
        ModelSpec("siggraph17", "eccv16", "siggraph17.onnx", 256,
                  description="Zhang et al. 2017. Slightly better than eccv16. Export with scripts/export_onnx.py."),
        ModelSpec("ddcolor-tiny", "ddcolor", "ddcolor_tiny.onnx", 0,
                  url="https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper_tiny.pth",
                  description="DDColor tiny. Best speed/quality trade-off for real time."),
        ModelSpec("ddcolor-large", "ddcolor", "ddcolor_large.onnx", 0,
                  url="https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper.pth",
                  description="DDColor large (paper). Quality mode for file conversion."),
        ModelSpec("ddcolor-artistic", "ddcolor", "ddcolor_artistic.onnx", 0,
                  url="https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_artistic.pth",
                  description="DDColor artistic. More vivid colors."),
    ]


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    processing: ProcSettings = field(default_factory=ProcSettings)
    encode: EncodeCfg = field(default_factory=EncodeCfg)
    subs: SubsCfg = field(default_factory=SubsCfg)
    onnx: OnnxCfg = field(default_factory=OnnxCfg)
    models: list = field(default_factory=default_models)
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    # Where this config was loaded from, used when saving from the dashboard.
    source_path: str = ""

    def model_spec(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        known = ", ".join(m.name for m in self.models)
        raise KeyError(f"Unknown model '{name}'. Known models: {known}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source_path", None)
        return d


def _build(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys with a warning."""
    if data is None:
        return cls()
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for k, v in data.items():
        if k not in known:
            log.warning("Ignoring unknown config key '%s' in section %s", k, cls.__name__)
            continue
        kwargs[k] = v
    return cls(**kwargs)


def config_from_dict(data: dict) -> Config:
    data = copy.deepcopy(data or {})
    cfg = Config()
    cfg.paths = _build(PathsCfg, data.get("paths"))
    cfg.server = _build(ServerCfg, data.get("server"))
    cfg.processing = _build(ProcSettings, data.get("processing"))
    if cfg.processing.subtitles is False or cfg.processing.subtitles is None:
        cfg.processing.subtitles = "off"  # YAML reads an unquoted `off` as the boolean false
    cfg.encode = _build(EncodeCfg, data.get("encode"))
    cfg.subs = _build(SubsCfg, data.get("subs"))
    cfg.onnx = _build(OnnxCfg, data.get("onnx"))
    if "models" in data and data["models"]:
        # User entries override defaults of the same name and may add new ones.
        merged = {m.name: m for m in default_models()}
        for m in data["models"]:
            base = merged.get(m.get("name"))
            merged[m["name"]] = _build(ModelSpec, {**(asdict(base) if base else {}), **m})
        cfg.models = list(merged.values())
    cfg.ffmpeg = data.get("ffmpeg", cfg.ffmpeg)
    cfg.ffprobe = data.get("ffprobe", cfg.ffprobe)
    return cfg


def find_config_path(explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("COLORSTREAM_CONFIG")
    if env:
        return Path(env)
    for cand in ("config.local.yaml", "config.yaml"):
        if Path(cand).exists():
            return Path(cand)
    return None


def load_config(path: str | None = None) -> Config:
    p = find_config_path(path)
    data: dict = {}
    if p is not None:
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    cfg = config_from_dict(data)
    cfg.source_path = str(p) if p else ""
    return cfg


def save_config(cfg: Config, path: str | None = None) -> str:
    target = Path(path or cfg.source_path or "config.local.yaml")
    d = cfg.to_dict()
    # Only persist models that differ from defaults to keep the file readable.
    defaults = {m.name: asdict(m) for m in default_models()}
    d["models"] = [m for m in d["models"] if defaults.get(m["name"]) != m]
    if not d["models"]:
        d.pop("models")
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(d, fh, sort_keys=False, allow_unicode=True)
    cfg.source_path = str(target)
    return str(target)


# Settings that can be changed live from the dashboard, with their validators.
RUNTIME_SETTINGS = {
    "mode": lambda v: v if v in ("auto", "fast", "quality") else _bad("mode", v),
    "model": lambda v: str(v),
    "model_resolution": lambda v: _clamp_int(v, 128, 1024),
    "output_height": lambda v: _clamp_int(v, 0, 2160),
    "saturation": lambda v: _clamp_float(v, 0.0, 2.0),
    "smoothing": lambda v: _clamp_float(v, 0.0, 0.98),
    "buffer_seconds": lambda v: _clamp_float(v, 8.0, 600.0),
    "inference_stride": lambda v: _clamp_int(v, 1, 8),
    "optical_flow": lambda v: bool(v),
    "shot_memory": lambda v: bool(v),
    "shot_consistency": lambda v: _clamp_float(v, 0.0, 1.0),
    "max_chroma_step": lambda v: _clamp_float(v, 0.0, 60.0),
    "chroma_limit": lambda v: _clamp_float(v, 0.0, 128.0),
    "edge_aware": lambda v: bool(v),
    "adaptive_stride": lambda v: bool(v),
    "max_stride": lambda v: _clamp_int(v, 1, 30),
    "motion_threshold": lambda v: _clamp_float(v, 0.5, 30.0),
    "colorize": lambda v: bool(v),
    "restore_sharpen": lambda v: _clamp_float(v, 0.0, 1.0),
    "restore_denoise": lambda v: _clamp_float(v, 0.0, 1.0),
    "restore_deflicker": lambda v: bool(v),
    "restore_stabilize": lambda v: bool(v),
    "restore_text": lambda v: _text_region(v),
    "force_colorize": lambda v: bool(v),
    "batch_size": lambda v: _clamp_int(v, 1, 32),
    "subtitles": lambda v: v if v in ("off", "soft", "burn", "both") else _bad("subtitles", v),
    "subtitle_lang": lambda v: str(v or "auto"),
    "subtitle_audio_lang": lambda v: str(v or "auto"),
    "subtitle_source": lambda v: v if v in ("auto", "existing", "asr") else _bad("subtitle_source", v),
    "subtitle_asr_model": lambda v: str(v),
    "subtitle_align": lambda v: bool(v),
}


def _text_region(v) -> str:
    v = str(v or "").strip().lower()
    if v in ("", "off", "none"):
        return ""
    if v in ("bottom", "top"):
        return v
    try:
        parts = [float(x) for x in v.split(",")]
        if len(parts) == 4 and all(0 <= p <= 1 for p in parts) and parts[2] > 0 and parts[3] > 0:
            return ",".join(f"{p:g}" for p in parts)
    except ValueError:
        pass
    raise ValueError(f"Invalid restore_text {v!r}: use bottom, top, or x,y,w,h as fractions between 0 and 1")


def _bad(name, v):
    raise ValueError(f"Invalid value for {name}: {v!r}")


def _clamp_int(v, lo, hi):
    return max(lo, min(hi, int(v)))


def _clamp_float(v, lo, hi):
    return max(lo, min(hi, float(v)))


RESTORE_PRESETS = {
    "off": dict(restore_sharpen=0.0, restore_denoise=0.0),
    "light": dict(restore_sharpen=0.3, restore_denoise=0.2),
    "medium": dict(restore_sharpen=0.5, restore_denoise=0.4),
    "strong": dict(restore_sharpen=0.7, restore_denoise=0.6),
}


def apply_overrides(settings: ProcSettings, overrides: dict) -> ProcSettings:
    """Return a copy of settings with validated overrides applied."""
    overrides = dict(overrides or {})
    preset = overrides.pop("restore", None)
    if preset:
        if preset not in RESTORE_PRESETS:
            _bad("restore", preset)
        settings = replace(settings, **RESTORE_PRESETS[preset])
    clean = {}
    for k, v in (overrides or {}).items():
        if v is None or k not in RUNTIME_SETTINGS:
            continue
        clean[k] = RUNTIME_SETTINGS[k](v)
    return replace(settings, **clean)
