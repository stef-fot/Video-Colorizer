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
* [Restoration: sharper, steadier, text removal](#restoration-sharper-steadier-cleaner-picture)
* [Stable colors and speed](#stable-colors-and-speed)
* [Restoration: sharper, steadier, text removal](#restoration-sharper-steadier-cleaner-picture) and [which stream is used](#which-stream-is-used-720p-1080p-4k-or-whatever-exists)
* [Stable colors and speed](#stable-colors-and-speed)
* [Subtitles in 21 languages incl. Greek, any playback speed](#subtitles)
* [Configuration](#configuration) · [Dashboard](#dashboard) · [Docker](#docker)
* [Tests](#tests) · [Troubleshooting](#troubleshooting) · [Limitations](#limitations)

## What is tested and what is not

Be aware of this before you rely on it.

| Verified in development (Linux, CPU only, libx264, FFmpeg 6.1) | **Not verified, needs your machine** |
|---|---|
| Pipeline, HLS playlist/segments, seeking, session limits, caching | Real model weights (ONNX export scripts follow upstream code but were not run), the real Whisper recognizer |
| CLI convert with resume, chunking, audio copy, side-by-side | CUDA / DirectML speed, NVENC/NVDEC, your actual fps |
| Scene cut, EMA, flow, chroma/luma merge, subtitles (Greek decoding, re-sync, frame-exact burn-in, muxing) with a stand-in recognizer, restoration filters on synthetic frames (138 Python tests, 19 Node tests) | Playback inside the real Stremio player |
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

Convert the PyTorch weights to ONNX once (first run `pip install -r requirements-export.txt`, it installs torch, torchvision and the packages the DDColor code imports; CPU torch is enough. If the script still reports a missing package it names the `pip install` command to run):

```bash
git clone https://github.com/piddnad/DDColor external/DDColor
python scripts/download_models.py ddcolor-tiny ddcolor-large
python scripts/export_onnx.py ddcolor-tiny  --repo external/DDColor --size 256   # 256 px is the real-time size for a GTX 1660 Ti (default 512 is slower)
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

## Restoration (sharper, steadier, cleaner picture)

General options for **any** video you have the right to process (old films, your own scans and recordings, home video,
public domain). They work on the luminance only, on an ordinary CPU, and can be used with colorization or **without it**
(`--no-colorize`: a color file keeps its colors exactly).

| Option | What it does |
|---|---|
| `--restore light\|medium\|strong` | preset for sharpening plus denoising |
| `--sharpen 0..1` | sharpening with noise coring and halo limiting: the result may not overshoot the local min/max, so no white or dark outlines, and grain is not boosted |
| `--denoise 0..1` | per pixel motion-adaptive temporal denoise (still areas are averaged, moving areas are left alone, no ghosts) plus a light edge-preserving spatial pass from 0.25 up |
| `--deflicker` | evens out frame to frame brightness flicker, follows real fades |
| `--stabilize` | removes small camera shake (translation only, limited to 3% of the picture) |
| `--remove-text bottom\|top\|x,y,w,h` | removes **hardcoded text** such as burned-in subtitles or captions in the region you choose, by finding thin bright structures and filling them from the surroundings (inpainting) |
| `--no-colorize` | restoration only, no AI colorization |

```bash
python -m colorstream convert old.mkv -o old_better.mkv --no-colorize --restore medium --deflicker
python -m colorstream convert scan.mkv -o out.mkv --restore light --remove-text bottom    # restore and colorize
python -m colorstream convert film.mkv -o out.mkv --no-colorize --remove-text 0,0.82,1,0.14 --sharpen 0.4
```

Order of the stages: stabilize, deflicker, text removal, denoise, sharpen, then colorization on the cleaned picture (a
cleaner input also gives steadier colors). Temporal state resets at scene cuts. The same options exist in `config.yaml`
(`processing: restore_*`, `colorize`), in the dashboard, and per stream as `?restore=medium&deflicker=1&rmtext=bottom&colorize=0`.

**In Stremio** set the `restore` block in `addon.config.json` (see the example). When it is active, every stream also
appears as **"Enhanced (AI)"** (restoration only, colors untouched, color files included), next to "Colorized (AI)"
(colorized and restored). Color files are not passed through when you pick "Enhanced (AI)".

Be realistic: these are classic image filters, not a neural upscaler. They make a soft or noisy picture cleaner and
steadier, they do not invent detail that is not in the file, and heavy settings look artificial. Text removal works
best on light text over calm backgrounds, leaves traces on busy ones, and skips frames where the region is mostly bright.
Stabilization cannot fix rotation or rolling shutter and can fight fast pans. Measured cost per frame at 720p in the
build sandbox (a plain 4 core CPU, one frame at a time): sharpen 4 ms, temporal denoise 3 ms (about 5 ms with the
spatial pass), deflicker under 1 ms, stabilize 1 ms, text removal 1.5 ms while no text is found and about 18 ms while
text is being removed. Your machine will differ: watch the speed on the dashboard, and lower the options if playback
stalls.

### Which stream is used: 720p, 1080p, 4K, or whatever exists

The addon ranks the direct streams it finds: **720p first, then 1080p, then 4K** (change with `resolutionOrder`, for
example `[1080, 720, 2160]`), then any other known resolution (higher first), then streams whose size is unknown.
Nothing is dropped: if none of the preferred ones exists it simply uses what is available. A stream does not need a
720/1080/4K tag: the resolution is also read from the file name, and untagged streams are measured by the server
(`/api/probe`, 5 s limit, a widescreen 1920x800 counts as 1080p), so they are ranked by their real size. 720p is the
first choice because it is the size that colorizes in real time on modest hardware.

## Stable colors and speed

Colors "dancing" has two causes: the model answers a little differently on every frame, and every new shot used to
start from scratch. These are the fixes (all on by default, all in the `processing:` section of `config.yaml` and in
the dashboard):

| Setting | What it does |
|---|---|
| `smoothing: 0.88` | running average over about 8 frames instead of 3 (it was 0.7) |
| averaging at shot start | the first frames of a shot are a plain average, so one unlucky first frame cannot set the look of the whole shot |
| `max_chroma_step: 5` | a color value can only move so much per frame, so jumps are impossible |
| `shot_memory` | when a shot returns (dialogue shot / reverse shot, jump cuts) it starts from the colors it had before, so the same room keeps the same colors and the model is not asked again |
| `shot_consistency: 0.5` | each shot is gently pulled toward the film's overall color strength and tint, so one scene does not come out pale and the next neon |
| `chroma_limit: 80` | soft ceiling on color strength (no hard clipping) |
| `edge_aware` | the color is refined with the sharp black and white picture as a guide, so it stops leaking over edges |
| `optical_flow` | optional, follows moving objects (a little more CPU) |

Efficiency: with `adaptive_stride` the model runs only when the picture changed (a 16x16 thumbnail, so film grain is
ignored) or every `max_stride` frames, and always at a cut. Static dialogue and landscapes cost almost nothing. Shot
memory also skips the model for returning shots. Everything else (temporal ops, shot logic) runs on the small
model-size picture, so it costs a millisecond or two per frame; the only extra per-frame cost is `edge_aware` (about
8 ms at 720p, turn it off on a slow CPU).

Tuning: colors still flicker, raise `smoothing` to 0.92 and lower `max_chroma_step` to 3. Colors react too slowly to
what enters the picture, lower `smoothing` or turn on `optical_flow`. Shot colors feel forced, lower
`shot_consistency` (0 turns it off). Slow machine: lower `output_height` and `model_resolution`, raise `max_stride` and
`motion_threshold`, set `edge_aware: false`. CLI: `--consistency`, `--edge-aware/--no-edge-aware`,
`--adaptive/--no-adaptive`, `--chroma-limit`, and the presets (`--quality fast|balanced|high`) set sensible combinations.

What this cannot do: a model that is wrong about an object's color (a red car that should be blue) stays wrong, it is
just wrong steadily. Verified with synthetic tests only (shot memory, averaging, step limit, palette pull, adaptive
inference, edge snapping); judge the look on your own footage and tune from there.

## Subtitles

ColorStream creates subtitles for a film by itself, in **21 languages: the 20 most spoken languages plus Greek**
(English, Chinese, Hindi, Spanish, French, Arabic, Bengali, Portuguese, Russian, Urdu, Indonesian, German, Japanese,
Swahili, Marathi, Telugu, Turkish, Tamil, Vietnamese, Korean, **Greek**; ranking by speakers is approximate and the list
is yours to change in `subs.languages`). Eleven more (Italian, Dutch, Polish, Ukrainian, Hebrew, Thai ...) work too.

How it works, in three steps, and why it stays correct on every frame:

1. **Source subtitles, once per film.** It uses subtitles that already exist (a file next to the video such as
   `Film.en.srt`, or an embedded text track), re-synced to the audio, or it **listens** to the film (Whisper through
   `faster-whisper`, language auto-detected, word timestamps). Old Greek files in Windows-1253 / ISO-8859-7 are decoded
   correctly.
2. **Translation into the language you pick**, with Meta's NLLB-200 (one model for all 21 languages, no detour through
   English) running on your CPU or GPU. Only the text is translated, the **cue times are copied from the source**, so every
   translation appears and disappears on the same frames. Translation follows recognition chunk by chunk, ahead of
   the playhead, and every result is cached on disk.
3. **Frame exact output.** Cue times are snapped to whole frames. Soft tracks are written with edges half a frame early
   (`[(f0 - 0.5)/fps, (f1 - 0.5)/fps)`), so rounding or clock jitter in a player cannot move a cue onto a neighbouring
   frame. Streamed segments are encoded without B-frames, so presentation and decode time are identical and players
   that zero their clock on either one agree. Or the text is **burned into the frames**, which is exact by construction.

**Any playback speed (0.5x, 2x, 4x):** subtitles are tied to the film's own timeline (soft track) or are part of the
picture (burned), not to the wall clock, so they stay on the right frame when the player speeds up or slows down. The
server also measures how fast you actually watch and widens its look-ahead for colorization, recognition and
translation by that factor (dashboard: "Playback speed"). If your machine cannot colorize at 4x real time, fast-forward
is smooth only through what is already cached, so convert with the CLI first for effortless 2x/4x viewing.

### Setup (once)

```bash
pip install -r requirements-subtitles.txt          # faster-whisper, ctranslate2, sentencepiece, Pillow
python scripts/download_translation.py             # translation model, about 620 MB, into weights/
```

The translation model is a community CTranslate2 int8 build of NLLB-200 distilled 600M (the repository was checked on
Hugging Face: it contains `model.bin` and `sentencepiece.bpe.model`). **NLLB-200 is CC-BY-NC-4.0: non-commercial use
only.** For another option set `subs.translator: argos` (`pip install argostranslate`; smaller, installs packs on demand,
but pivots through English and has no good Swahili, Marathi, Telugu or Tamil).

### In Stremio

Open the addon config (`addon/addon.config.json`, see the example). By default (`"mode": "soft"`) each stream it
returns comes in two forms: **"Colorized (AI)"** and **"AI subtitles"** (the original stream, played directly by Stremio,
unchanged). Both carry the 21 subtitle tracks, which show up in Stremio's subtitle menu (Greek as "ell", Chinese
"zho" ...). A track is created when you pick it, from one recognition pass shared by all languages. When a title is opened
the addon also tells the server to start recognizing the first stream, and to pre-translate `prepareLangs`
(default Greek and English), so they are ready sooner. `"mode": "burn"` draws the first language of `langs` into the
colorized picture instead (works in every player, one language, in the cache like the colors).

Limits that come from Stremio, not from this project: a Stremio addon cannot see which file is being played when it is
asked for subtitles, so subtitles are attached to the **direct http(s) streams** the addon wraps (from your library or the
upstream addons you configured). Torrent streams (no URL) cannot be reached. The first time a film is watched,
recognition of a full film takes a while (minutes on a GPU, longer on a CPU) and a soft track that is fetched early is
**partial** (the server answers after `subs.soft_wait_seconds`, default 30 s, with what is ready, and says so in the
`X-ColorStream-Subtitles` header); pick the track again later in Stremio's menu to load the rest, or prepare the film
beforehand with the CLI below. A burned track needs no waiting trick: the picture simply waits for the text.

### CLI

```bash
# all 21 languages as .srt files next to the film (film.el.srt, film.en.srt, film.zh.srt ...)
python -m colorstream subtitles film.mkv --langs all
python -m colorstream subtitles ./films -r --langs el,en,es,fr     # a whole folder
python -m colorstream subtitles film.mkv --langs el --asr-model medium --audio-lang en

# colorize and burn Greek subtitles (translated from the spoken language)
python -m colorstream convert film.mkv -o film_color.mkv --subs burn --sub-lang el
# keep them switchable: film_color.el.srt plus a subtitle track tagged "ell"
python -m colorstream convert film.mkv -o film_color.mkv --subs soft --sub-lang el --asr-model medium
# use and re-sync your own file
python -m colorstream convert film.mkv -o out.mkv --subs burn --sub-file film.srt --sub-lang el
```

Options: `--sub-lang` (what you want to read), `--audio-lang` (what is spoken, default auto), `--sub-source
auto|existing|asr`, `--asr-model tiny|base|small|medium|large-v3`, `--no-sub-align`, `--sub-size`. In `config.yaml` the
same live under `processing:` (`subtitles`, `subtitle_lang`, `subtitle_audio_lang`, ...) and `subs:`; the dashboard shows
them too.

### Honest notes

* **Not run with the real models in development.** Model downloads were blocked in the build sandbox, so Whisper and NLLB
  never ran. Everything around them was tested with stand-ins: chunked recognition, language lock-in, translation chunk by
  chunk, caching, errors, the 21-language list, Greek decoding, re-sync, frame-exact burn-in (the cue showed on frames
  48 to 71 and no others), soft-track times, muxing, the server and CLI. Check one real clip in your language first.
* Translation quality varies: small models make mistakes with idioms, names and very short lines, and a machine
  translation of a machine transcription compounds errors. Colloquial speech, noise and music suffer most. For Greek
  recognition use `medium` or `large-v3` offline; `small` is the realistic real-time size on a GTX 1660 Ti.
* Translating on a CPU is slow-ish (a film in one language is minutes, not seconds). Each language is only translated when
  someone asks for it (or is listed in `pretranslate`), once.
* Re-sync needs audible speech and fixes delays up to about 2 minutes plus common frame rate conversions. Embedded tracks
  and audio of remote URLs are skipped unless `subs.extract_remote: true` (it downloads the file). Picture based subtitles
  (PGS, VobSub) cannot be read, there is no OCR.
* Burned text needs a font that covers the language (Windows has them; Linux/Docker: DejaVu and Noto, installed by the
  Dockerfile; or set `subs.font`). Arabic, Urdu and Hebrew need Pillow with libraqm (normal wheels have it).
* Not verified: how Stremio's mpv zeroes its clock for HLS plus external subtitles. The no-B-frames encoding and the half
  frame edges remove the known sources of a one or two frame offset; burned subtitles have no such question.

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
| `processing.smoothing` | `0.88` | temporal EMA; higher = steadier colors, slower to react |
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
python -m pytest            # 138 tests; integration tests need ffmpeg and use the demo model
cd addon && npm test        # 19 tests
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

**`The parameter is incorrect` / DirectML error when loading a model.** ColorStream now retries with safer settings
(basic and no graph optimization, DirectML's required options) and finally falls back to the CPU with a warning, so
it keeps running; `python -m colorstream check` shows which provider each model really uses. If it fell back to the
CPU: export the model smaller (`--size 256`), update the GPU driver, or switch to CUDA (`requirements-cuda.txt`).

**Stremio shows no streams at all.** The addon must be running (a second window with `node index.js`, the server window
is not enough) and needs something to show: a `library.json` (copy `library.example.json`) and/or `upstreams` in
`addon.config.json`. The addon prints its counts at startup and warns when both are empty. Then open
`http://127.0.0.1:7000/manifest.json` in a browser, restart the addon after any config change, and in Stremio
remove and reinstall the addon so it reloads the manifest.

**Playback fails right away.** Open the playlist URL in a browser or VLC to see the error text. Common messages:
*"Model weights ... not found"* (see Model weights), *"Cannot open source"* (URL unreachable or expired, signed URLs
expire), *"Too many active sessions"* (raise `server.max_sessions` or stop one), *"Local files are disabled"*.

**The first play takes long.** Probing a remote file, the color check (6 short seeks), loading the model and the
first benchmark run happen before the first segment. Later plays of the same title start instantly from the cache.

**Seeking is slow the first time.** A far seek restarts decoding at that point and needs the first segment
(plus a 1 s warm-up) before playback resumes. Seeking within already cached ranges is instant.

**Subtitles missing or wrong.** Colorization alone does not touch subtitles: Stremio shows its own from its addons. To get
subtitles from ColorStream turn them on (see Subtitles). Wrong timing: leave `subtitle_align` on. Squares instead of
Greek letters: no font found, set `subs.font`. Nothing appears for a generated track: check `python -m colorstream check`
and the dashboard ("Subtitles" line); `faster-whisper` must be installed. Garbled Greek letters from a file: the encoding
is detected from `subtitle_lang`, set it to `el`.

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
* Subtitle limits are listed in the Subtitles section.
* No live streams (a known duration and HTTP range support are required). Subtitles are not part of the HLS output.
* Each seek far outside the buffer costs a restart; temporal state does not carry across the jump.
* Each model's license applies to its weights. This project does not bundle weights.
