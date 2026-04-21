# cosyvoice-open-tts

**English** · [中文](./README.zh.md)

OpenAI-compatible HTTP TTS service built on top of
[CosyVoice](https://github.com/FunAudioLLM/CosyVoice). Ships as a single CUDA
container image on GHCR.

Implements the [Open TTS spec](https://github.com/OpenTTSGroup/open-tts-spec):

- `POST /v1/audio/speech` — OpenAI-compatible synthesis
- `POST /v1/audio/clone` — one-shot zero-shot cloning (multipart upload)
- `POST /v1/audio/realtime` — chunked streaming synthesis
- `GET  /v1/audio/voices` — list file-based and built-in voices
- `GET  /v1/audio/voices/preview?id=...` — download a reference WAV
- `GET  /healthz` — engine status, capabilities, concurrency snapshot

Six output formats (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`); mono
`float32` encoded server-side. Voices live on disk as
`${VOICES_DIR}/<id>.{wav,txt,yml}` triples.

## Quick start

```bash
mkdir -p voices cache

# Drop a 5–15 s reference WAV plus its transcript:
cp ~/my-ref.wav voices/alice.wav
echo "This is the transcript of the reference clip." > voices/alice.txt

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/cosyvoice-open-tts:latest
```

First boot downloads the model weights (~5 GB) to `/root/.cache`. Mount the
cache directory to avoid repeat downloads. `/healthz` reports
`status="loading"` until the engine is ready.

```bash
curl -s localhost:8000/healthz | jq
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from CosyVoice.","voice":"file://alice","response_format":"mp3"}' \
  -o out.mp3
```

## Capabilities

| capability | value | notes |
|---|---|---|
| `clone` | `true` | zero-shot via `voice="file://..."` or `POST /v1/audio/clone` |
| `streaming` | `true` | chunked mp3/pcm/opus/aac via `POST /v1/audio/realtime` |
| `design` | `false` | CosyVoice needs a reference clip; `/v1/audio/design` is not exposed |
| `languages` | `false` | mixed Chinese/English/Japanese/Korean/Cantonese text works inline |
| `builtin_voices` | depends on the model | `true` for models that ship a populated `spk2info.pt` (e.g. `iic/CosyVoice2-0.5B`); `false` for pure zero-shot checkpoints (e.g. `Fun-CosyVoice3-0.5B-2512`) |

## Environment variables

### Engine (prefixed `COSYVOICE_`)

| variable | default | description |
|---|---|---|
| `COSYVOICE_MODEL` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ModelScope / HF repo id, or a local path |
| `COSYVOICE_VARIANT` | `v3` | `v2` or `v3` — selects the `CosyVoice2` / `CosyVoice3` inference class |
| `COSYVOICE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `COSYVOICE_CUDA_INDEX` | `0` | GPU index when multiple are visible |
| `COSYVOICE_DTYPE` | `float16` | `float16` / `bfloat16` / `float32`; `float16` ⇒ `fp16=True` on CUDA |
| `COSYVOICE_LOAD_JIT` | `false` | `CosyVoice2` only; ignored on v3 |
| `COSYVOICE_LOAD_TRT` | `false` | requires the `tensorrt-cu12` packages (not preinstalled) |
| `COSYVOICE_LOAD_VLLM` | `false` | requires `vllm` (not preinstalled) |
| `COSYVOICE_TRT_CONCURRENT` | `1` | TensorRT stream concurrency |
| `COSYVOICE_TEXT_FRONTEND` | `true` | pass through to `inference_*(text_frontend=...)` |
| `COSYVOICE_PROMPT_CACHE_SIZE` | `16` | LRU size for per-voice prompt embeddings |
| `COSYVOICE_MODEL_SOURCE` | `modelscope` | `modelscope` / `hf` / `local` |

### Service-level (no prefix)

| variable | default | description |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `VOICES_DIR` | `/voices` | scan root for file-based voices |
| `MAX_INPUT_CHARS` | `8000` | 413 above this |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `MAX_CONCURRENCY` | `1` | in-flight synthesis ceiling |
| `MAX_QUEUE_SIZE` | `0` | 0 = unbounded queue |
| `QUEUE_TIMEOUT` | `0` | seconds; 0 = unbounded wait |
| `MAX_AUDIO_BYTES` | `20971520` | upload limit for `/v1/audio/clone` |
| `CORS_ENABLED` | `false` | `true` mounts a `CORSMiddleware` that allows any origin / method / header on every endpoint (no credentials — see the [spec](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md#37-cors)). Keep `false` when the service is fronted by a reverse proxy or called same-origin. |

## Compose

See [`docker/docker-compose.example.yml`](docker/docker-compose.example.yml).

## API request parameters

GET endpoints (`/healthz`, `/v1/audio/voices`, `/v1/audio/voices/preview`)
take no body and at most a single `id` query parameter — see the
[Open TTS spec](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md)
for their response shape.

The tables below describe the POST endpoints that accept a request body. The
**Status** column uses a fixed vocabulary:

- **required** — rejected with 422 if missing.
- **supported** — accepted and consumed by CosyVoice.
- **ignored** — accepted for OpenAI compatibility; has no effect.
- **conditional** — behaviour depends on other fields or the loaded model;
  see the notes column.
- **extension** — CosyVoice-specific field, not part of the Open TTS spec.

### `POST /v1/audio/speech` (application/json)

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `model` | string | `null` | ignored | OpenAI compatibility only; any value is accepted and discarded. |
| `input` | string | — | required | 1..`MAX_INPUT_CHARS` chars. Empty ⇒ 422, over limit ⇒ 413. |
| `voice` | string | — | required | `file://<id>` loads `${VOICES_DIR}/<id>.wav` + `.txt`; a bare name selects a built-in SFT voice (only when `/healthz.capabilities.builtin_voices=true`). |
| `response_format` | enum | `mp3` | supported | One of `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm`. Global default overridden by `DEFAULT_RESPONSE_FORMAT`. |
| `speed` | float | `1.0` | supported | Range `[0.25, 4.0]`. Passed through to `inference_*(speed=…)`. |
| `instructions` | string \| null | `null` | conditional | With `voice="file://…"` and a non-empty value: the engine routes to `inference_instruct2` using this string as the style prompt; CosyVoice3 auto-appends `<\|endofprompt\|>`. With a built-in SFT voice this field is **ignored with a warning** — CosyVoice2/3 has no SFT-side instruct API. |

### `POST /v1/audio/clone` (multipart/form-data)

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `audio` | file | — | required | Extension must be one of `.wav/.mp3/.flac/.ogg/.opus/.m4a/.aac/.webm`. Over `MAX_AUDIO_BYTES` ⇒ 413. Resampled to 16 kHz internally; the upload is never persisted to `${VOICES_DIR}`. |
| `prompt_text` | string | — | required | Reference-clip transcript. Empty ⇒ 422. |
| `input` | string | — | required | Same semantics as `/speech.input`. |
| `response_format` | string | `mp3` | supported | Same as `/speech`. |
| `speed` | float | `1.0` | supported | Range `[0.25, 4.0]`. |
| `instructions` | string \| null | `null` | conditional | Same wrapping as `/speech.instructions`. This endpoint is always a clone path, so a non-empty value always routes to `inference_instruct2`. |
| `model` | string | `null` | ignored | OpenAI compatibility only. |

### `POST /v1/audio/realtime` (application/json)

Request body mirrors `/v1/audio/speech`. Only the divergences are listed here:

| Field | Status override | Notes |
|---|---|---|
| `response_format` | restricted | Only `mp3` / `pcm` / `opus` / `aac`. `flac` / `wav` return 422 before the stream starts. |
| `speed` | ignored by the engine | CosyVoice's `stream=True` code path drops `speed`. The service accepts the request and logs a warning; the audio is produced at `1.0x`. |
| `instructions`, `voice`, `input`, `model` | — | Same as `/speech`. |

## Known limitations

- CosyVoice silently drops `speed` when `stream=True`; the service logs a
  warning and still honours the request. Call `/v1/audio/speech` (non-stream)
  if you need precise speed control.
- Reference prompts are resampled to 16 kHz internally.
- `flac` and `wav` are available for `/v1/audio/speech` / `/v1/audio/clone`
  but are rejected with 422 on `/v1/audio/realtime` — they do not stream
  cleanly.
- First request for a new voice on the plain zero-shot path pays an
  embedding-extraction cost; subsequent requests hit the prompt LRU.
- CosyVoice3's LLM requires `<|endofprompt|>` to appear in the prompt. The
  service adds it automatically: for zero-shot, `voice.txt` is wrapped as
  `"You are a helpful assistant.<|endofprompt|>" + voice.txt` before being
  fed to the model; for instruct2, the `instructions` string gets
  `<|endofprompt|>` appended. If you already include the token yourself the
  service leaves your string untouched.
- The instruct2 path (`voice="file://..."` + non-empty `instructions`) does
  **not** share the prompt LRU cache: upstream's speaker-info store pins
  the prompt text to the reference transcript, so each instruct2 call
  re-extracts speaker/acoustic features. Plain zero-shot is still cached.
- CosyVoice's `deepspeed`, `tensorrt`, and `vllm` optional acceleration is
  **not preinstalled** to keep the image small. Rebuild the image with the
  extras you need if you enable `COSYVOICE_LOAD_TRT` / `COSYVOICE_LOAD_VLLM`
  / `COSYVOICE_LOAD_JIT`.
