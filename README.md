# ColorStream

Colorize black and white films with AI, **in real time inside Stremio** or **as a standalone file converter**.
Both modes share one pipeline: FFmpeg decode, a small AI model that predicts only color (chroma) on a downscaled
copy, temporal smoothing against flicker, then a merge with your **original full resolution luminance**, and FFmpeg
encode.

> **Legal.** Use ColorStream only on video you own or that is in the public domain. It contains no code to find,
> download or unlock content. The Stremio addon is a wrapper: it only acts on direct `http(s)://` stream URLs that you
> give it (your own library) or that addons *you chose* return. Torrent and magnet links are rejected.

* [What is tested and what is not](#what-is-tested-and-what-is-not)
* [How it works](#how-it-works) · [Architecture details](docs/ARCHITECTURE.md)
* [Installation (Windows 11 + NVIDIA GTX 1660 Ti, also Linux/macOS)](#installation)
* [Run the server](#3-run-the-server) · [Install the addon in Stremio](#4-install-the-addon-in-stremio)
* [Use the CLI](#5-command-line-converter)
* [Configuration](#configuration) · [Dashboard](#dashboard) · [Docker](#docker)
* [Tests](#tests) · [Troubleshooting](#troubleshooting) · [Limitations](#limitations)

## What is tested and what is not

Be aware of this before you rely on it.

| Verified in development (Linux, CPU only, libx264, FFmpeg 6.1) | **Not verified, needs your machine** |
|---|---|
| Pipeline, HLS playlist/segments, seeking, session limits, caching | Real model weights (ONNX export scripts follow upstream code but were not run) |
| CLI convert with resume, chunking, audio copy, side-by-side | CUDA / DirectML speed, NVENC/NVDEC, your actual fps |
| Scene cut, EMA, flow, chroma/luma merge (39 Python tests) | Playback inside the real Stremio player |
| Addon manifest, catalog, stream wrapping (6 Node tests) | Windows paths, Docker image build |

Everything that does not need weights was exercised with a built-in fake `demo` model (a simple tint, **not** a
colorizer). Run `python -m colorstream benchmark` first to find out what your GPU really delivers.

## How it works

```
Stremio ─► addon :7000 ─► "Colorized (AI)" stream = http://127.0.0.1:7002/stream?src=<url>
                                                         │
                         HLS playlist + .ts segments ◄───┘  (FastAPI :7002, cache on disk, look-ahead, seek)
```

1. FFmpeg decodes (hardware decode when it works) to gray frames through a pipe. No temporary images.
2. A 256..512 px grayscale copy goes to the model, which predicts only Lab **a/b** chroma.
3. Chroma is stabilized: scene cut detection (histogram), optional optical flow warping, exponential moving average.
4. Chroma is upscaled and merged with the original lightness, so sharpness is untouched. Saturation default 0.85.
5. FFmpeg encodes (NVENC, VideoToolbox, QSV, AMF, VAAPI, else libx264). Audio is copied.

Playback starts after the first segments (4 s each) are ready. The server then works ahead of the playhead
(`buffer_seconds`, default 40 s), caches every segment (replays and seeks are instant), restarts processing on a far
seek, and pauses itself when the player stops asking. Sources that are already in color are passed through untouched.

## Installation

You need: **Python 3.10+**, **Node.js 18+** (addon only), **FFmpeg**, a GPU driver, and model weights.

### 0. Get the code

```bash
git clone https://github.com/stef-fot/video-colorizer.git
cd video-colorizer
```

### 1. Install FFmpeg

* **Windows 11:** `winget install Gyan.FFmpeg`, then open a new terminal and check `ffmpeg -version`.
  (Or download a "full" build from gyan.dev and add its `bin` folder to PATH.)
* **macOS:** `brew install ffmpeg` · **Debian/Ubuntu:** `sudo apt install ffmpeg`

Check that NVENC exists on NVIDIA: `ffmpeg -hide_banner -encoders | findstr nvenc` (Windows) or `| grep nvenc`.
ColorStream tests encoders itself at startup and silently falls back to `libx264` if one does not work.

### 2. GPU drivers and Python packages

Create a virtual environment first:

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
```

Pick **one** ONNX Runtime flavor:

| Your setup | Command | Notes |
|---|---|---|
| **Windows, any GPU (recommended start for GTX 1660 Ti)** | `pip install -r requirements-directml.txt` | Only needs a current GPU driver. A bit slower than CUDA. |
| NVIDIA, fastest | `pip install -r requirements-cuda.txt` | Needs the NVIDIA driver, **CUDA 12.x** and **cuDNN 9** installed and on PATH (see the ONNX Runtime CUDA requirements table for the exact versions of the release you install). |
| macOS (Apple Silicon) | `pip install -r requirements-cpu.txt` | The default macOS wheel includes CoreML. |
| No GPU | `pip install -r requirements-cpu.txt` | Works, but far below real time. A warning is printed. |

Update the NVIDIA driver from nvidia.com (or GeForce Experience). Then verify:

```bash
python -m colorstream check
```

It reports FFmpeg, the encoder that will be used, ONNX Runtime providers (you want `CUDAExecutionProvider` or
`DmlExecutionProvider`), the GPU and the installed models.

> **GTX 16xx note (1660 Ti):** these cards have no tensor cores and FP16 is unreliable on them. ColorStream defaults
> to FP32 (`onnx.fp16: false`). With 6 GB VRAM the tiny DDColor model at 256 px fits easily.

### Model weights

Weights are **not** included (licenses and size). Models available (`python -m colorstream models`):

| Name | Use | Notes |
|---|---|---|
| `ddcolor-tiny` | **fast / real time** (default) | DDColor tiny |
| `ddcolor-large` | **quality** mode, file conversion | DDColor paper model, much slower |
| `ddcolor-artistic` | more vivid colors | slower |
| `eccv16`, `siggraph17` | very light fallbacks | muted colors, input fixed at 256 px |
| `demo` | pipeline test only | fake tint, **not** a real colorizer |

Convert the PyTorch weights to ONNX once (needs `pip install -r requirements-export.txt`; CPU torch is enough):

```bash
git clone https://github.com/piddnad/DDColor external/DDColor
python scripts/download_models.py ddcolor-tiny ddcolor-large
python scripts/export_onnx.py ddcolor-tiny  --repo external/DDColor
python scripts/export_onnx.py ddcolor-large --repo external/DDColor

# optional light model
git clone https://github.com/richzhang/colorization external/colorization
python scripts/export_onnx.py eccv16 --repo external/colorization
```

The ONNX files land in `weights/` (`ddcolor_tiny.onnx` ...). The download URLs in `colorstream/config.py` point at
the authors' Hugging Face repository; if one is down or has moved, download the `.pth` by hand into `weights/`.
The export scripts were written against the upstream inference code and not run in the development environment, so
if an import fails compare with the upstream repository. Each model keeps its own license, check it before use.

If you already have ONNX files (for example a TensorRT-ready export), put them in `weights/` and declare them in
`config.yaml` under `models:`. DDColor-style models take `(N,3,H,W)` gray-as-RGB in 0..1 and return `(N,2,H,W)` ab;
Zhang-style models take `(N,1,H,W)` lightness 0..100 and return ab.

### 3. Run the server

```bash
python -m colorstream benchmark      # optional, measures your GPU and shows what "auto" will pick
python -m colorstream serve
```

* Stremio endpoint: `http://127.0.0.1:7002/stream?src=<url>` (HLS)
* Dashboard: `http://127.0.0.1:7001`

On the first stream, `mode: auto` runs a short benchmark (cached in `cache/benchmark.json`) and chooses model
resolution (192..512) and inference stride so that the estimated speed is at least 1.25x the video frame rate.
If your machine cannot keep up, you are told in the log and on the dashboard, and the player pre-buffers.

Quick test without Stremio (works with the `demo` model too), in VLC or mpv:

```bash
mpv "http://127.0.0.1:7002/stream?src=https://YOUR-OWN-FILE-URL.mp4"
```

### 4. Install the addon in Stremio

Use the **Stremio desktop app** (the browser version blocks `http://127.0.0.1` from an https page).

```bash
cd addon
npm install
cp addon.config.example.json addon.config.json     # Windows: copy addon.config.example.json addon.config.json
node index.js                                      # or scripts\start_windows.bat to start server + addon
```

Then in Stremio: open **Addons** (puzzle icon), type or paste this into the **addon URL** box at the top and press
**Install**:

```
http://127.0.0.1:7000/manifest.json
```

**What the addon can see.** A Stremio addon is only asked for *its own* streams, it cannot read the list another addon
returns. ColorStream therefore offers two sources:

1. **Your own library** (catalog "ColorStream Library" in Stremio's Discover). Copy `addon/library.example.json` to
   `addon/library.json` and list files/URLs you own or that are public domain. A local file path works if you set
   `server.allow_local_files: true` and add its folder to `server.allowed_roots` in `config.yaml`.
2. **Upstream addons you configure.** Put their manifest URLs in `upstreams` in `addon.config.json`
   (or `CS_UPSTREAMS=url1,url2`). For a title, ColorStream asks them for streams and, for every **direct http(s)**
   stream, adds a **"Colorized (AI)"** entry next to the original (`keepOriginals`). Torrent (`infoHash`), YouTube
   and external-link streams are ignored. If a stream needs request headers (`behaviorHints.proxyHeaders`), they are
   forwarded to the server. If you configure no upstream, the addon returns nothing for normal titles.

Pick "Colorized (AI)" in the stream list and press play. The first request waits (up to `prebuffer_max_wait`,
default 45 s) while the first segments are made. If the playlist request times out in Stremio, press play again,
the work done is cached.

### 5. Command line converter

```bash
python -m colorstream convert input.mkv -o output.mkv --quality high --model ddcolor-large --saturation 0.85 --resolution 720
```

| Option | Meaning |
|---|---|
| `--quality fast\|balanced\|high` | presets: model, model resolution, stride, flow, CRF (default `balanced`) |
| `--model NAME` | override the preset's model |
| `--saturation 0.85` | chroma multiplier |
| `--resolution 720` | output height (default: keep source) |
| `--model-res 512` · `--stride N` · `--flow/--no-flow` · `--smoothing 0.7` · `--batch 4` | fine control |
| `--encoder h264_nvenc` | force an encoder (default: auto) |
| `--no-audio` · `--no-subs` | by default every audio track and subtitle is copied without re-encoding |
| `--compare` | side by side output, original on the left |
| `--force` | colorize even if the source looks like color |
| `--chunk-seconds 120` · `--no-resume` · `--keep-temp` | resume control |

* A progress bar with ETA and live fps is shown. **Interrupted jobs resume**: finished chunks are kept in
  `<output>.colorstream-job/`; run the same command again. Changing settings or the input starts over.
* **Batch:** `python -m colorstream batch ./films -o ./films/colorized -r --quality balanced` converts every video
  in a folder, skips files that already exist or already look colored, and continues after a failure.
* MP4 output works, but MKV is safer for copying all audio/subtitle formats. If copying fails the tool retries with
  AAC audio and no subtitles and tells you.
* Other commands: `models`, `check`, `benchmark`, `clear-cache`.

## Configuration

One file: [`config.yaml`](config.yaml) (every option commented, all optional). Copy it to `config.local.yaml` for
personal changes, or pass `-c path`. The most useful settings:

| Setting | Default | Effect |
|---|---|---|
| `processing.mode` | `auto` | `auto` benchmarks and picks, `fast`, or `quality` |
| `processing.output_height` | `720` | streamed height; lower is faster. `0` keeps the source |
| `processing.model_resolution` | `256` | model input side (192..512), the main speed/quality knob |
| `processing.inference_stride` | `1` | run the model every N frames; with `optical_flow` color follows motion in between |
| `processing.saturation` | `0.85` | lower if colors look garish |
| `processing.smoothing` | `0.7` | temporal EMA; higher = steadier colors, slower to react |
| `processing.buffer_seconds` | `40` | look-ahead window; `server.max_sessions` limits concurrent sessions |
| `paths.cache_max_gb` | `20` | least recently used cache entries are deleted above this |

Suggested starting point for a **GTX 1660 Ti, 16 GB RAM, Windows 11** (an estimate, not a measurement; verify with
`benchmark`): `mode: auto`, `output_height: 720` (or 480 if it stutters), `ddcolor-tiny`, FP32, DirectML or CUDA, NVENC
encoding (Turing NVENC is fine). Use `--quality high` with `ddcolor-large` for offline conversion where speed does
not matter.

## Dashboard

`http://127.0.0.1:7001` shows active sessions (state, fps and speed versus real time, buffer ahead of the playhead,
cached segments, pause/resume/stop), GPU load and memory (NVIDIA via `nvidia-smi`), cache size, encoder in use, and
lets you change model, resolution, saturation, smoothing, buffer size, stride and flow. Changes apply to new sessions;
"Apply and save" writes them to your config file (comments in the file are not preserved).

## Docker

```bash
docker compose up --build
```

Needs the NVIDIA Container Toolkit on the host. Put weights in `./weights`, your own videos in `./media` (available
as `/media/...`). Ports are published on `127.0.0.1` only. On Windows a native install is simpler and faster than
Docker Desktop with WSL2 for a single user.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest            # 39 tests; integration tests need ffmpeg and use the demo model
cd addon && npm test        # 6 tests
```

Unit tests cover temporal smoothing (EMA, flow warp, engine state), scene cut detection (cuts, fades, thresholds),
and chroma/luma merging (luminance preservation, saturation, clipping, resolution independence), plus cache, playlist,
source validation, auto-tuning, CLI resume, and HTTP serving, seeking and session limits.

## Troubleshooting

Run `python -m colorstream check` first, and watch the dashboard.

**Stuttering / buffering.** Look at "Speed" on the dashboard: it must be above 1.0x.
Lower `output_height` (480), lower `model_resolution` (256 or 192), raise `inference_stride` to 2 or 3 with
`optical_flow: true`, make sure the encoder is hardware (`encoder` on the dashboard; `libx264` at 720p+ eats CPU),
and make sure ONNX Runtime uses the GPU (`check` must not show only `CPUExecutionProvider`). Close other GPU-heavy
apps. Raise `buffer_seconds` and `prebuffer_max_wait` so more is ready before play. A 4K source is decoded at full
size before scaling, which costs CPU/GPU decode time.

**Color flicker / colors changing every few seconds.** Increase `smoothing` (0.8 to 0.9). Enable `optical_flow`.
If you use `inference_stride > 1` without flow, enable flow. If hard cuts are not detected, lower
`scene_cut_threshold` (0.35); if resets happen inside a scene (flashes, fast lighting changes), raise it (0.65).
Raw model flicker is stronger at low `model_resolution`; try 384.

**Wrong or ugly colors.** The model guesses; it does not know the real colors. Try another model (`ddcolor-artistic`,
`ddcolor-large`), higher `model_resolution`, or lower `saturation` (0.7). Very contrasty, grainy or damaged film
confuses models: denoise/deflicker first (for example FFmpeg `hqdn3d`, or dedicated restoration tools). Green or
purple faces usually mean a too small model resolution or heavy grain. If the whole picture has one strange tint,
the source may be sepia/tinted: that is handled, but strong tint plus noise can mislead the model.
A color cast of exactly zero (stays gray) means the model returned NaN: use an FP32 ONNX file (the log warns).

**High GPU memory.** Use the tiny model, lower `model_resolution` (256), lower `batch_size` (1 or 2), set
`server.max_sessions: 1`. DDColor large at 512 px can need several GB. Stop idle sessions in the dashboard (idle
sessions free their GPU work after `session_idle_seconds`, the model itself stays loaded while the server runs).

**"Colorized (AI)" does not show up in Stremio.** Check in order: (1) addon process running and
`http://127.0.0.1:7000/manifest.json` opens in a browser; (2) installed in the **desktop** app, not the web app;
(3) you configured `upstreams` or a library item. Without them the addon has nothing to wrap; look at the addon console
("No upstream addons configured"); (4) the upstream returns a **direct http(s) `url`** stream for that title (torrent
streams are intentionally ignored); (5) the title id type: the addon answers ids starting with `tt` and `cs:`;
(6) after changing the config restart the addon and reinstall or refresh in Stremio.

**Playback fails right away.** Open the playlist URL in a browser or VLC to see the error text. Common messages:
*"Model weights ... not found"* (see Model weights), *"Cannot open source"* (URL unreachable or expired, signed URLs
expire), *"Too many active sessions"* (raise `server.max_sessions` or stop one), *"Local files are disabled"*.

**The first play takes long.** Probing a remote file, the color check (6 short seeks), loading the model and the
first benchmark run happen before the first segment. Later plays of the same title start instantly from the cache.

**Seeking is slow the first time.** A far seek restarts decoding at that point and needs the first segment
(plus a 1 s warm-up) before playback resumes. Seeking within already cached ranges is instant.

**Subtitles missing.** Subtitles are not put into the HLS stream; Stremio shows its own subtitles from its addons.
The CLI copies embedded subtitles into the output file.

## Limitations

* **Colors are AI estimates.** They are plausible guesses, not historically accurate. Clothes, cars, skin tones and
  skies are guessed from context and can differ between scenes.
* **Real-time speed depends on your GPU, your CPU and the source resolution.** Weak hardware means lower model
  resolution/stride, i.e. less stable color. Nothing is guaranteed to hit 24 fps; the benchmark and the dashboard tell
  you the truth. Estimated, not measured: a GTX 1660 Ti is expected to handle the tiny model at 256 px in real time at
  480p to 720p, large models are for offline use.
* **Very old or damaged footage** (heavy grain, scratches, flicker, unstable exposure) gives poor and unstable
  colors. Restore first: denoise, deflicker, stabilize.
* Colorization changes only color. It does not upscale, sharpen or repair the picture.
* Optical flow uses OpenCV Farneback on the small chroma image; a RAFT backend is not implemented.
* No live streams (a known duration and HTTP range support are required). Subtitles are not part of the HLS output.
* Each seek far outside the buffer costs a restart; temporal state does not carry across the jump.
* Each model's license applies to its weights. This project does not bundle weights.
