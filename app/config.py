from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # --- Engine (COSYVOICE_* prefix) -----------------------------------------
    cosyvoice_model: str = Field(
        default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        description="ModelScope / HuggingFace repo id or local directory.",
    )
    cosyvoice_variant: Literal["v2", "v3"] = Field(
        default="v3",
        description="Dispatches CosyVoice2 (v2) vs CosyVoice3 (v3).",
    )
    cosyvoice_device: Literal["auto", "cuda", "cpu"] = "auto"
    cosyvoice_cuda_index: int = Field(default=0, ge=0)
    cosyvoice_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    cosyvoice_load_jit: bool = False
    cosyvoice_load_trt: bool = False
    cosyvoice_load_vllm: bool = False
    cosyvoice_trt_concurrent: int = Field(default=1, ge=1)
    cosyvoice_text_frontend: bool = True
    cosyvoice_prompt_cache_size: int = Field(default=16, ge=1)
    cosyvoice_model_source: Literal["modelscope", "hf", "local"] = "modelscope"

    # --- Service-level (no prefix) -------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    voices_dir: str = "/voices"
    max_input_chars: int = Field(default=8000, ge=1)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = "mp3"
    max_concurrency: int = Field(default=1, ge=1)
    max_queue_size: int = Field(default=0, ge=0)
    queue_timeout: float = Field(default=0.0, ge=0.0)
    max_audio_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    cors_enabled: bool = False

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir)

    @property
    def resolved_device(self) -> str:
        if self.cosyvoice_device == "cpu":
            return "cpu"
        if self.cosyvoice_device == "cuda":
            return f"cuda:{self.cosyvoice_cuda_index}"
        # auto
        import torch

        if torch.cuda.is_available():
            return f"cuda:{self.cosyvoice_cuda_index}"
        return "cpu"

    @property
    def use_fp16(self) -> bool:
        return (
            self.cosyvoice_dtype == "float16"
            and self.resolved_device.startswith("cuda")
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
