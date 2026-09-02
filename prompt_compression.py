"""
prompt_compression.py — LLMLingua-2 wrapper + tiktoken counter.

Reads its defaults from the ecoGPT `Settings` singleton so the model name,
device, and target rate are all configured in one place (config.yaml).

NOTE: The historical filename `prompt_compresstion.py` is preserved as a
back-compat shim that re-exports from this module. New code should import
from `prompt_compression`.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import tiktoken
from llmlingua import PromptCompressor

from config import settings

# Module-level tiktoken encoder (no need to re-instantiate per call).
tokenizer = tiktoken.get_encoding("cl100k_base")

_MODULE_TAG = "prompt_compression"


def _resolve_device(device_setting: str) -> str:
    """
    Resolve the LLMLingua-2 device setting into the string the library
    expects ("cpu" or "cuda").

    Accepted values in config.yaml:
      - "auto"  -> use CUDA if torch sees a GPU, otherwise CPU. Logged.
      - "cpu"   -> always CPU.
      - "cuda"  -> require CUDA; raise at startup if it isn't available.
    """
    device_setting = device_setting.lower().strip()

    if device_setting == "cpu":
        return "cpu"

    # Lazy import: only pay the torch import cost if the user wants it.
    import torch

    if device_setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "config.yaml sets compression.device='cuda' but no CUDA device "
                "is visible to torch. Set device='auto' to fall back to CPU, "
                "or install a CUDA build of torch."
            )
        return "cuda"

    if device_setting == "auto":
        if torch.cuda.is_available():
            print(f"[{_MODULE_TAG}] device=auto -> using CUDA")
            return "cuda"
        print(f"[{_MODULE_TAG}] device=auto -> no GPU detected, using CPU")
        return "cpu"

    raise ValueError(
        f"Unknown compression.device '{device_setting}'. "
        "Expected one of: auto, cpu, cuda."
    )


class PromptCompressorClient:
    """
    Thin wrapper around Microsoft's LLMLingua-2.

    All defaults are read from `config.settings.compression`; explicit args
    to `compress_prompt` still win on a per-call basis.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name: str = model_name or settings.compression.model_name
        self.device_setting: str = device or settings.compression.device
        resolved: str = _resolve_device(self.device_setting)

        print(f"[{_MODULE_TAG}] loading {self.model_name} on device={resolved}")
        self.compressor = PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=True,
            device_map=resolved,
        )
        print(f"[{_MODULE_TAG}] compressor loaded")

    def compress_prompt(
        self,
        prompt: str,
        target_rate: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Compress `prompt` with LLMLingua-2 and measure wall-clock time.

        Returns (compressed_prompt, compression_time_ms).
        """
        if target_rate is None:
            target_rate = settings.compression.target_rate

        start = time.perf_counter()
        result = self.compressor.compress_prompt_llmlingua2(
            prompt,
            rate=target_rate,
            force_tokens=["\n", ".", "!", "?", ","],
            drop_consecutive=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result["compressed_prompt"], elapsed_ms

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens using the configured tiktoken encoder."""
        if not text:
            return 0
        return len(tokenizer.encode(text))


__all__ = ["PromptCompressorClient", "tokenizer"]
