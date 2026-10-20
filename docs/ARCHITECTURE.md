# ColorStream architecture

## Components

```
Stremio ──► Node addon (7000) ──► adds "Colorized (AI)" stream pointing to ──┐
                                                                             ▼
Player (mpv inside Stremio) ◄── HLS playlist + .ts segments ◄── FastAPI server (7002)
                                                                   │  sessions, look-ahead, seek
Browser ──► Dashboard (7001) ──► JSON API on 7002                  ▼
                                                      Core pipeline (colorstream/pipeline.py)
                                                                   │
              ┌────────────────────────────────────────────────────┴─────────────────┐
              ▼                                                                      ▼
   FFmpeg decode (gray, pipe)                                         Segment cache (disk)
   → downscale → model (ONNX) → chroma                                key = source hash
   → EMA / flow / cut detection                                       + settings signature
   → upscale chroma + original luma                                   + segment index
   → FFmpeg encode (NVENC/…/x264) → HLS segments
```

The CLI (`colorstream convert`, `batch`) calls the same `run_pass()` function as the server.
The only difference is the output: an MPEG-TS part per chunk (resumable) instead of HLS segments.

## Per frame data flow

1. FFmpeg decodes (NVDEC/D3D11/VideoToolbox/VAAPI when it works, software otherwise), scales to the output size and
   emits **8-bit gray** frames over a pipe. Black and white sources only need luminance.
2. A 256..512 px square copy of the frame goes to the model, which predicts **Lab a/b** only.
3. `ChromaEngine` stabilizes the chroma at model resolution (cheap): histogram scene cut detection, optional
   Farneback flow warping of the previous chroma, EMA, and optional model stride (run the model every N frames).
4. `merge_luma_chroma` upscales the chroma to the frame size and merges it with the untouched original lightness
   (via a 256 entry gray to L lookup table), applies saturation and converts Lab to BGR with OpenCV.
5. FFmpeg encodes. Audio is copied (or converted to AAC if MPEG-TS cannot carry the codec).

Reader, processing and writer run in three threads with bounded queues.

## Streaming behaviour

* The playlist is a complete VOD playlist (all segments, `#EXT-X-ENDLIST`), so the player sees the full length and
  can seek anywhere. Segments are created on demand.
* One worker thread per session runs "passes". A pass starts at the first missing segment at or after the playhead,
  writes consecutive segments with a single FFmpeg encoder (continuous timestamps via `-output_ts_offset`) and
  stops when it reaches a cached segment, the end, or is cancelled.
* Look-ahead: the pass throttles itself when it is `buffer_seconds` ahead of the last requested segment, which also
  acts as pause (no requests means no GPU use).
* Seek: a request far from the running pass cancels it (FFmpeg processes are killed) and a new pass starts at the
  requested segment. The first `warmup_frames` frames are processed but not emitted so temporal state settles.
* Pre-buffering: the playlist request waits for the first segments. If the measured speed is below real time, it
  waits for a larger fraction of the look-ahead window (bounded by `prebuffer_max_wait`).
* Already-colored sources (median Lab chroma spread of 6 sampled frames above `color_threshold`) are not processed:
  the server redirects to the origin (or serves the file directly for local sources).

## Assumptions

* Python 3.10+, Node 18+, FFmpeg/ffprobe on PATH.
* The user's GPU is reachable through ONNX Runtime (CUDA, DirectML, CoreML or ROCm provider).
* A Stremio addon only receives requests for its own `stream` resource; it can not see the streams other addons
  return. The addon therefore (a) serves a personal library of files/URLs you own and (b) queries upstream addons
  that **you** list in its config, and wraps their direct HTTP(S) streams.
* Sources must be seekable over HTTP (Range requests) and have a known duration. Live streams are not supported.
* Subtitles are not muxed into HLS output (MPEG-TS cannot carry most subtitle formats). Stremio loads subtitles
  itself, so they keep working in the player.
* Model weights are not bundled. They are downloaded by the user from the upstream projects and converted to ONNX.

## Subtitles (colorstream/subtitles)

* `track.py` `SubtitleTrack`: one per video. Decides the source (explicit file, sidecar, embedded text track, else speech
  recognition), keeps cues as exact frame ranges, persists to `cache/<source>/subs_<hash>.json` so replays are instant.
* Speech recognition works in 30 s chunks (1 s overlap), nearest to the playhead first, looking ahead by
  `lookahead_seconds * playback speed`; the language is locked after the first confident detection.
* `align.py`: energy based speech mask from the audio vs the "a cue is visible" mask of the file; cross-correlation per
  part of the film gives a line `speech_time = scale * cue_time + offset` (candidate frame-rate scales are tried first).
  A change is only applied if it clearly improves the match.
* `render.py`: Pillow draws wrapped text (pixel width based, anywhere-breaks for CJK/Thai) with a font chosen per script;
  the bitmap is cached per cue text and alpha blended on the frames in `pipeline.py` right after the chroma merge.
* `session.py` estimates the playback speed from the player's segment requests and scales all look-ahead windows.
