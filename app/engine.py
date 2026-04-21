from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from app.config import Settings


log = logging.getLogger(__name__)

# CosyVoice needs its repo root and the vendored Matcha-TTS on sys.path before
# ``cosyvoice`` can be imported. The Dockerfile sets PYTHONPATH, but for local
# runs we also honour ``COSYVOICE_ROOT`` as a fallback.
_cosy_root = os.environ.get("COSYVOICE_ROOT")
if _cosy_root:
    for _p in (_cosy_root, f"{_cosy_root}/third_party/Matcha-TTS"):
        if _p not in sys.path:
            sys.path.insert(0, _p)


# CosyVoice3's LLM asserts that ``<|endofprompt|>`` (token id 151646) appears
# somewhere in the LM input — see engine/cosyvoice/llm/llm.py:479. Both v2 and
# v3 training format suffix the instruct_text with this token (see
# engine/example.py). The wrappers below mirror the upstream examples so that
# v3 never hits the assert and v2 keeps the best-practice suffix.
ENDOFPROMPT = "<|endofprompt|>"
COSYVOICE3_SYSTEM_PREFIX = "You are a helpful assistant."


def _resolve_model_dir(repo_or_path: str, source: str) -> str:
    if os.path.isdir(repo_or_path):
        return repo_or_path
    if source == "local":
        raise FileNotFoundError(
            f"COSYVOICE_MODEL_SOURCE=local but {repo_or_path} is not a directory"
        )
    if source == "hf":
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=repo_or_path)
    # default: modelscope
    from modelscope import snapshot_download

    return snapshot_download(repo_or_path)


class TTSEngine:
    """Thin async wrapper around CosyVoice2 / CosyVoice3."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._device = settings.resolved_device
        self._dtype_str = settings.cosyvoice_dtype
        self._model, self._sample_rate, self._resolved_model_dir = self._load_model()
        try:
            self._builtin_spks: list[str] = list(self._model.list_available_spks())
        except Exception:  # pragma: no cover - defensive
            log.exception("list_available_spks() failed; assuming no built-in voices")
            self._builtin_spks = []

        self._prompt_cache: dict[tuple[str, float], str] = {}
        self._prompt_cache_order: list[tuple[str, float]] = []
        self._prompt_cache_max = settings.cosyvoice_prompt_cache_size
        self._prompt_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public attributes

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype_str(self) -> str:
        return self._dtype_str

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def builtin_voices_list(self) -> list[str]:
        return list(self._builtin_spks)

    @property
    def model_id(self) -> str:
        return self._settings.cosyvoice_model

    # ------------------------------------------------------------------
    # Model loading

    def _load_model(self) -> tuple[object, int, str]:
        model_dir = _resolve_model_dir(
            self._settings.cosyvoice_model, self._settings.cosyvoice_model_source
        )

        variant = self._settings.cosyvoice_variant
        fp16 = self._settings.use_fp16

        if variant == "v3":
            from cosyvoice.cli.cosyvoice import CosyVoice3

            model = CosyVoice3(
                model_dir,
                load_trt=self._settings.cosyvoice_load_trt,
                load_vllm=self._settings.cosyvoice_load_vllm,
                fp16=fp16,
                trt_concurrent=self._settings.cosyvoice_trt_concurrent,
            )
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice2

            model = CosyVoice2(
                model_dir,
                load_jit=self._settings.cosyvoice_load_jit,
                load_trt=self._settings.cosyvoice_load_trt,
                load_vllm=self._settings.cosyvoice_load_vllm,
                fp16=fp16,
                trt_concurrent=self._settings.cosyvoice_trt_concurrent,
            )

        return model, int(model.sample_rate), model_dir

    # ------------------------------------------------------------------
    # Prompt embedding cache

    def _load_prompt_wav_16k(self, path: str):
        from cosyvoice.utils.file_utils import load_wav

        return load_wav(path, 16000)

    def _wrap_prompt_text_for_zero_shot(self, ref_text: str) -> str:
        """CosyVoice3 requires <|endofprompt|> in the LM input; mirror the
        upstream example by prefixing the reference transcript with a
        standard system line. CosyVoice2 was trained without this prefix,
        so leave its prompt_text alone. Callers that already include the
        token are honoured verbatim.
        """
        if self._settings.cosyvoice_variant != "v3":
            return ref_text
        if ENDOFPROMPT in ref_text:
            return ref_text
        return f"{COSYVOICE3_SYSTEM_PREFIX}{ENDOFPROMPT}{ref_text}"

    def _wrap_instructions_for_instruct2(self, instructions: str) -> str:
        """Both CosyVoice2 and CosyVoice3 suffix the instruct string with
        <|endofprompt|> during training — v3 asserts on the token at
        inference time, v2 does not assert but loses quality without it.
        """
        if ENDOFPROMPT in instructions:
            return instructions
        return f"{instructions}{ENDOFPROMPT}"

    @staticmethod
    def _make_spk_id(ref_audio: str, ref_mtime: float) -> str:
        digest = hashlib.sha1(
            f"{ref_audio}:{ref_mtime}".encode("utf-8")
        ).hexdigest()[:16]
        return f"_fv_{digest}"

    def _ensure_zero_shot_spk(
        self,
        ref_audio: str,
        ref_mtime: Optional[float],
        ref_text: str,
    ):
        """Return (spk_id, prompt_wav).

        ``spk_id is None`` ⇒ caller must pass ``prompt_wav`` to ``inference_*``.
        Otherwise ``prompt_wav`` may be ``None`` and ``zero_shot_spk_id``
        carries the cached embedding inside CosyVoice's ``spk2info``.
        """
        if ref_mtime is None:
            return None, self._load_prompt_wav_16k(ref_audio)

        key = (ref_audio, ref_mtime)
        with self._prompt_lock:
            cached = self._prompt_cache.get(key)
            if cached is not None:
                # Move to MRU position.
                try:
                    self._prompt_cache_order.remove(key)
                except ValueError:
                    pass
                self._prompt_cache_order.append(key)
                return cached, None

        wav = self._load_prompt_wav_16k(ref_audio)
        spk_id = self._make_spk_id(ref_audio, ref_mtime)
        if not hasattr(self._model, "add_zero_shot_spk"):  # pragma: no cover
            # CosyVoice >= 2 has it; fall back to no caching if ever missing.
            return None, wav

        self._model.add_zero_shot_spk(ref_text, wav, spk_id)

        with self._prompt_lock:
            self._prompt_cache[key] = spk_id
            self._prompt_cache_order.append(key)
            while len(self._prompt_cache_order) > self._prompt_cache_max:
                old_key = self._prompt_cache_order.pop(0)
                old_spk = self._prompt_cache.pop(old_key, None)
                if old_spk is not None:
                    try:
                        self._model.frontend.spk2info.pop(old_spk, None)
                    except Exception:  # pragma: no cover
                        log.exception("failed to evict prompt cache entry %s", old_spk)

        return spk_id, None

    # ------------------------------------------------------------------
    # Tensor collection

    @staticmethod
    def _collect(generator) -> np.ndarray:
        import torch

        pieces = []
        for item in generator:
            pieces.append(item["tts_speech"])
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        audio = torch.cat(pieces, dim=1).squeeze(0).cpu().numpy()
        return audio.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Non-streaming synthesis paths

    async def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio: str,
        ref_text: str,
        ref_mtime: Optional[float] = None,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        **_: object,
    ) -> np.ndarray:
        tf = self._settings.cosyvoice_text_frontend

        def _run() -> np.ndarray:
            use_instruct = bool(instructions) and hasattr(
                self._model, "inference_instruct2"
            )
            if instructions and not use_instruct:
                log.warning(
                    "instructions ignored: inference_instruct2 not available on %s",
                    type(self._model).__name__,
                )
            if use_instruct:
                # instruct2 cannot share the zero_shot_spk cache: the upstream
                # spk2info entry pins prompt_text to the reference transcript,
                # but for instruct2 the model expects the instruction there.
                # Re-extract acoustic features every call.
                prompt_wav = self._load_prompt_wav_16k(ref_audio)
                gen = self._model.inference_instruct2(
                    text,
                    self._wrap_instructions_for_instruct2(instructions),
                    prompt_wav,
                    zero_shot_spk_id="",
                    stream=False,
                    speed=speed,
                    text_frontend=tf,
                )
            else:
                wrapped_ref_text = self._wrap_prompt_text_for_zero_shot(ref_text)
                spk_id, prompt_wav = self._ensure_zero_shot_spk(
                    ref_audio, ref_mtime, wrapped_ref_text
                )
                gen = self._model.inference_zero_shot(
                    text,
                    wrapped_ref_text,
                    prompt_wav,
                    zero_shot_spk_id=spk_id or "",
                    stream=False,
                    speed=speed,
                    text_frontend=tf,
                )
            return self._collect(gen)

        return await asyncio.to_thread(_run)

    async def synthesize_builtin(
        self,
        text: str,
        *,
        voice: str,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        **_: object,
    ) -> np.ndarray:
        tf = self._settings.cosyvoice_text_frontend
        if instructions:
            log.warning(
                "instructions ignored: CosyVoice2/3 has no instruct API for built-in SFT voices"
            )

        def _run() -> np.ndarray:
            gen = self._model.inference_sft(
                text,
                voice,
                stream=False,
                speed=speed,
                text_frontend=tf,
            )
            return self._collect(gen)

        return await asyncio.to_thread(_run)

    # ------------------------------------------------------------------
    # Streaming synthesis

    async def synthesize_realtime(
        self,
        text: str,
        *,
        kind: str,
        voice: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_mtime: Optional[float] = None,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        **_: object,
    ) -> AsyncIterator[np.ndarray]:
        tf = self._settings.cosyvoice_text_frontend
        if speed != 1.0:
            log.warning("CosyVoice ignores speed != 1.0 in streaming mode")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        sentinel = object()

        def _producer() -> None:
            try:
                if kind == "clone":
                    use_instruct = bool(instructions) and hasattr(
                        self._model, "inference_instruct2"
                    )
                    if instructions and not use_instruct:
                        log.warning(
                            "instructions ignored in stream: "
                            "inference_instruct2 not available on %s",
                            type(self._model).__name__,
                        )
                    if use_instruct:
                        prompt_wav = self._load_prompt_wav_16k(ref_audio)
                        gen = self._model.inference_instruct2(
                            text,
                            self._wrap_instructions_for_instruct2(instructions),
                            prompt_wav,
                            zero_shot_spk_id="",
                            stream=True,
                            speed=1.0,
                            text_frontend=tf,
                        )
                    else:
                        wrapped_ref_text = self._wrap_prompt_text_for_zero_shot(
                            ref_text or ""
                        )
                        spk_id, prompt_wav = self._ensure_zero_shot_spk(
                            ref_audio, ref_mtime, wrapped_ref_text
                        )
                        gen = self._model.inference_zero_shot(
                            text,
                            wrapped_ref_text,
                            prompt_wav,
                            zero_shot_spk_id=spk_id or "",
                            stream=True,
                            speed=1.0,
                            text_frontend=tf,
                        )
                else:
                    if instructions:
                        log.warning(
                            "instructions ignored for built-in voice in streaming"
                        )
                    gen = self._model.inference_sft(
                        text,
                        voice,
                        stream=True,
                        speed=1.0,
                        text_frontend=tf,
                    )

                for item in gen:
                    tensor = item["tts_speech"]
                    arr = tensor.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                    asyncio.run_coroutine_threadsafe(queue.put(arr), loop).result()
            except Exception as exc:  # pragma: no cover - surfaced via stream
                log.exception("streaming producer failed")
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
                except Exception:
                    pass
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(sentinel), loop
                    ).result()
                except Exception:
                    pass

        thread = threading.Thread(
            target=_producer, name="cosyvoice-stream", daemon=True
        )
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Drain any pending items so the producer thread can exit.
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
