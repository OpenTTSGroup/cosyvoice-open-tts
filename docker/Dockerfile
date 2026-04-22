# syntax=docker/dockerfile:1.7

# ============================================================================
# Build arguments
#   CUDA_VERSION    — base image CUDA version. CosyVoice pins torch==2.3.1,
#                     so the default matches the cu121 wheel (12.1.1).
#   CUDA_FLAVOR     — cuDNN tag segment. 12.1.x ships as ``cudnn8-runtime``;
#                     12.4+ uses ``cudnn-runtime``.
#   UBUNTU_VERSION  — base OS.
#   TORCH_VERSION   — upstream CosyVoice requirement.
#   TORCH_CUDA_TAG  — PyTorch wheel index tag matching CUDA_VERSION.
# ============================================================================
ARG CUDA_VERSION=12.1.1
ARG CUDA_FLAVOR=cudnn8-runtime
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.3.1
ARG TORCH_CUDA_TAG=cu121

# ============================================================================
# Builder: python venv with torch, CosyVoice runtime deps, and the API layer.
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION} AS builder

ARG TORCH_VERSION
ARG TORCH_CUDA_TAG

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        git \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip

# PyTorch — single layer, matched to CosyVoice's torch==2.3.1 constraint.
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}

# API layer dependencies (pydantic/fastapi/soundfile/av/pyyaml).
COPY docker/requirements.api.txt /tmp/requirements.api.txt
RUN pip install -r /tmp/requirements.api.txt

# CosyVoice engine dependencies (curated subset of upstream requirements.txt).
COPY docker/requirements.engine.txt /tmp/requirements.engine.txt
RUN pip install -r /tmp/requirements.engine.txt

# openai-whisper 20231117 is an sdist without pyproject.toml, and its
# setup.py does `import pkg_resources` at the top. Two fallout points:
#   1. PEP 517 build isolation spins up a clean env with the newest
#      setuptools (>=80), which has split pkg_resources out — wheel build
#      fails immediately. --no-build-isolation makes pip reuse the venv's
#      own setuptools (the Python 3.10 ensurepip bundled 59.6.0, still
#      shipping pkg_resources).
#   2. Ubuntu 22.04's python3.10 ensurepip bundles pip + setuptools but
#      NOT wheel, so setuptools' `bdist_wheel` command is unavailable.
#      Install wheel just long enough to build openai-whisper and then
#      uninstall it so nothing build-only leaks into the runtime image.
RUN pip install wheel && \
    pip install --no-build-isolation openai-whisper==20231117 && \
    pip uninstall -y wheel

# Strip bytecode caches the tests and builds left behind.
RUN find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# Runtime: minimal image carrying only the venv + CosyVoice source + app/.
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/api:/opt/api/engine:/opt/api/engine/third_party/Matcha-TTS \
    COSYVOICE_ROOT=/opt/api/engine

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/api

# The engine submodule (CosyVoice upstream) is consumed via PYTHONPATH —
# upstream has no pyproject.toml/setup.py, so we do not pip install it.
COPY engine /opt/api/engine
COPY app /opt/api/app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/voices", "/root/.cache"]
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
