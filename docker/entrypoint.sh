#!/usr/bin/env bash
set -euo pipefail

# Engine defaults
: "${COSYVOICE_MODEL:=FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
: "${COSYVOICE_VARIANT:=v3}"
: "${COSYVOICE_DEVICE:=auto}"
: "${COSYVOICE_DTYPE:=float16}"
: "${COSYVOICE_MODEL_SOURCE:=modelscope}"

# Service-level defaults
: "${VOICES_DIR:=/voices}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${PYTHONPATH:=/opt/api/engine:/opt/api/engine/third_party/Matcha-TTS}"
: "${COSYVOICE_ROOT:=/opt/api/engine}"

export COSYVOICE_MODEL COSYVOICE_VARIANT COSYVOICE_DEVICE COSYVOICE_DTYPE \
       COSYVOICE_MODEL_SOURCE VOICES_DIR HOST PORT LOG_LEVEL \
       PYTHONPATH COSYVOICE_ROOT

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
