# cosyvoice-open-tts

基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的 OpenAI 兼容
HTTP TTS 服务，单镜像发布到 GHCR。

遵循 [Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec)：

- `POST /v1/audio/speech` — OpenAI 兼容的文本合成
- `POST /v1/audio/clone` — 一次性上传音频做零样本克隆
- `POST /v1/audio/realtime` — 分块流式合成
- `GET  /v1/audio/voices` — 列出内置音色与文件克隆音色
- `GET  /v1/audio/voices/preview?id=...` — 下载参考音频
- `GET  /healthz` — 引擎状态、能力矩阵、并发快照

支持 `mp3`、`opus`、`aac`、`flac`、`wav`、`pcm` 六种输出格式（服务端编码单声道
float32）。音色目录通过 `${VOICES_DIR}/<id>.{wav,txt,yml}` 三件套提供。

## 快速开始

```bash
mkdir -p voices cache

# 准备一段 5-15 秒的参考音频和对应转录：
cp ~/my-ref.wav voices/alice.wav
echo "这是参考音频对应的转录文本。" > voices/alice.txt

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/cosyvoice-open-tts:latest
```

首次启动会从 ModelScope 下载约 5 GB 权重到 `/root/.cache`；挂载 cache 目录
避免重复下载。引擎加载期间 `/healthz` 返回 `status="loading"`。

```bash
curl -s localhost:8000/healthz | jq
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，来自 CosyVoice。","voice":"file://alice","response_format":"mp3"}' \
  -o out.mp3
```

## 能力矩阵

| capability | 取值 | 说明 |
|---|---|---|
| `clone` | `true` | 通过 `voice="file://..."` 或 `POST /v1/audio/clone` 做零样本克隆 |
| `streaming` | `true` | `POST /v1/audio/realtime` 流式输出 mp3/pcm/opus/aac |
| `design` | `false` | CosyVoice 必须有参考音频；不暴露 `/v1/audio/design` |
| `languages` | `false` | 文本中直接混排中英日韩粤，无需显式声明语种 |
| `builtin_voices` | 取决于模型 | 带非空 `spk2info.pt` 的模型（如 `iic/CosyVoice2-0.5B`）为 `true`；纯零样本模型（如 `Fun-CosyVoice3-0.5B-2512`）为 `false` |

`instructions` 字段在所有合成端点上都接受。

## 环境变量

### 引擎（带 `COSYVOICE_` 前缀）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `COSYVOICE_MODEL` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ModelScope / HF repo id 或本地路径 |
| `COSYVOICE_VARIANT` | `v3` | `v2` 或 `v3`，分派 `CosyVoice2` / `CosyVoice3` 推理类 |
| `COSYVOICE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `COSYVOICE_CUDA_INDEX` | `0` | 多卡场景指定 GPU 序号 |
| `COSYVOICE_DTYPE` | `float16` | `float16` / `bfloat16` / `float32`；CUDA + float16 时启用 `fp16` |
| `COSYVOICE_LOAD_JIT` | `false` | 仅 v2 生效；v3 忽略 |
| `COSYVOICE_LOAD_TRT` | `false` | 需额外安装 `tensorrt-cu12`（镜像未预装） |
| `COSYVOICE_LOAD_VLLM` | `false` | 需额外安装 `vllm`（镜像未预装） |
| `COSYVOICE_TRT_CONCURRENT` | `1` | TensorRT 并发流数 |
| `COSYVOICE_TEXT_FRONTEND` | `true` | 透传到 `inference_*(text_frontend=...)` |
| `COSYVOICE_PROMPT_CACHE_SIZE` | `16` | 每音色 prompt embedding LRU 大小 |
| `COSYVOICE_MODEL_SOURCE` | `modelscope` | `modelscope` / `hf` / `local` |

### 服务级（无前缀）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | uvicorn 日志级别 |
| `VOICES_DIR` | `/voices` | 文件克隆音色扫描根 |
| `MAX_INPUT_CHARS` | `8000` | 超出返回 413 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `MAX_CONCURRENCY` | `1` | 同时推理上限 |
| `MAX_QUEUE_SIZE` | `0` | 0 = 不限 |
| `QUEUE_TIMEOUT` | `0` | 秒；0 = 不限 |
| `MAX_AUDIO_BYTES` | `20971520` | `/v1/audio/clone` 上传大小限制 |

## Compose

参考 [`docker/docker-compose.example.yml`](docker/docker-compose.example.yml)。

## 请求参数

GET 端点（`/healthz`、`/v1/audio/voices`、`/v1/audio/voices/preview`）无请求体，
最多一个 `id` 查询参数；响应结构参见
[Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md)。

下列表格描述有请求体的 POST 端点。**状态**列使用固定词汇：

- **required** — 必填，缺失返回 422。
- **supported** — 可选字段，引擎实际消费。
- **ignored** — 为 OpenAI 兼容接受，但永远不生效。
- **conditional** — 行为取决于其他字段或加载的模型，详见"说明"列。
- **extension** — CosyVoice 特有扩展，规范未定义。

### `POST /v1/audio/speech`（application/json）

| 字段 | 类型 | 默认值 | 状态 | 说明 |
|---|---|---|---|---|
| `model` | string | `null` | ignored | 仅用于 OpenAI 兼容；任意值被接受后丢弃。 |
| `input` | string | — | required | 长度 1..`MAX_INPUT_CHARS`；空串 ⇒ 422，超长 ⇒ 413。 |
| `voice` | string | — | required | `file://<id>` 加载 `${VOICES_DIR}/<id>.wav` + `.txt`；普通字符串作为内置 SFT 音色名（仅当 `/healthz.capabilities.builtin_voices=true` 时可用）。 |
| `response_format` | enum | `mp3` | supported | `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm` 六选一；全局默认由 `DEFAULT_RESPONSE_FORMAT` 覆盖。 |
| `speed` | float | `1.0` | supported | 范围 `[0.25, 4.0]`；透传到 `inference_*(speed=…)`。 |
| `instructions` | string \| null | `null` | conditional | `voice="file://…"` 且非空时，引擎切到 `inference_instruct2` 把本字段作为风格指令使用，CosyVoice3 自动追加 `<\|endofprompt\|>`；内置 SFT 音色下**忽略并记 warning**（CosyVoice2/3 没有 SFT 侧的 instruct API）。 |

> 文本归一化开关不按请求暴露，由环境变量 `COSYVOICE_TEXT_FRONTEND`
> 全局配置（默认 `true`）。

### `POST /v1/audio/clone`（multipart/form-data）

| 字段 | 类型 | 默认值 | 状态 | 说明 |
|---|---|---|---|---|
| `audio` | file | — | required | 扩展名须属于 `.wav/.mp3/.flac/.ogg/.opus/.m4a/.aac/.webm`；超过 `MAX_AUDIO_BYTES` ⇒ 413。服务端内部重采到 16 kHz，**不会**持久化到 `${VOICES_DIR}`。 |
| `prompt_text` | string | — | required | 参考音频的转录文本；空串 ⇒ 422。 |
| `input` | string | — | required | 同 `/speech.input`。 |
| `response_format` | string | `mp3` | supported | 同 `/speech`。 |
| `speed` | float | `1.0` | supported | 范围 `[0.25, 4.0]`。 |
| `instructions` | string \| null | `null` | conditional | 包装规则同 `/speech.instructions`；本端点永远是克隆路径，所以非空时必走 `inference_instruct2`。 |
| `model` | string | `null` | ignored | 仅用于 OpenAI 兼容。 |

### `POST /v1/audio/realtime`（application/json）

请求体与 `/v1/audio/speech` 相同。下表只列与 `/speech` 的差异：

| 字段 | 状态覆盖 | 说明 |
|---|---|---|
| `response_format` | 受限 | 仅 `mp3` / `pcm` / `opus` / `aac`；`flac` / `wav` 在流开始前返回 422。 |
| `speed` | 被引擎忽略 | CosyVoice 的 `stream=True` 路径会丢弃 `speed`；服务端接受请求并记 warning，音频按 `1.0x` 生成。 |
| `instructions` / `voice` / `input` / `model` | — | 与 `/speech` 一致。 |

## 已知限制

- CosyVoice 在 `stream=True` 时会忽略 `speed`，服务只记录 warning 不拒绝请求。
  需要精确调速时请走 `/v1/audio/speech`（非流式）。
- 参考音频内部会重采样到 16 kHz。
- `flac` / `wav` 可用于 `/v1/audio/speech` 与 `/v1/audio/clone`，但
  `/v1/audio/realtime` 下会 422 拒绝——它们不适合流式封装。
- 普通零样本路径下新音色首次请求会支付一次 embedding 提取开销，后续命中
  LRU 缓存。
- CosyVoice3 的 LLM 要求 prompt 中必须包含 `<|endofprompt|>`。服务会自动
  处理：零样本路径下 `voice.txt` 会被包装成 `"You are a helpful
  assistant.<|endofprompt|>" + voice.txt` 再喂给模型；instruct2 路径下
  会在 `instructions` 末尾追加 `<|endofprompt|>`。如果你的字符串已经
  带了这个 token，服务会原样透传不再处理。
- instruct2 路径（`voice="file://..."` + 非空 `instructions`）**不共享**
  prompt LRU 缓存：上游的 speaker-info 缓存把 prompt_text 与参考音频转录
  绑死，复用会导致模型读到 `voice.txt` 而非用户指令，因此每次 instruct2
  都会重新抽取声学特征。普通零样本仍然命中缓存。
- 为压缩镜像体积，`deepspeed` / `tensorrt` / `vllm` 等可选加速**未预装**。
  若启用 `COSYVOICE_LOAD_TRT` / `COSYVOICE_LOAD_VLLM` / `COSYVOICE_LOAD_JIT`，
  请基于本镜像重建并加入对应依赖。
