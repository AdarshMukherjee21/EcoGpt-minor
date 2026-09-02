"""
raw_model_runner.py — Provider-agnostic OpenAI-compatible chat client.

Talks to any HTTP endpoint that implements OpenAI's `/v1/chat/completions`
shape. Works with OpenAI, Azure OpenAI, Together, Groq, OpenRouter, vLLM,
llama.cpp's server, LM Studio, Ollama (with the OpenAI compat layer), etc.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from config import settings


class RawModelRunner:
    """
    Direct interface to an OpenAI-compatible `/v1/chat/completions` endpoint.

    Per-call arguments are explicit (model, prompt, api_base, api_key) so
    the runner is fully provider-agnostic. When called from `app.py`, the
    api_base / api_key come from `config.settings.models`.
    """

    def __init__(
        self,
        timeout_seconds: Optional[int] = None,
        request_defaults: Optional[Dict[str, Any]] = None,
    ):
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.raw_runner.timeout_seconds
        )
        # Optional default fields (temperature, max_tokens, ...) merged into
        # every request body. Per-call kwargs in `run_model` take precedence.
        self.request_defaults: Dict[str, Any] = (
            dict(request_defaults)
            if request_defaults is not None
            else dict(settings.raw_runner.request_defaults)
        )

    def run_model(
        self,
        model: str,
        prompt: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **overrides: Any,
    ) -> str:
        """
        Send `prompt` to `model` via the OpenAI-compatible chat completions
        endpoint at `api_base`, authenticated with `api_key`.

        If api_base / api_key are omitted, they must be supplied via the
        caller (the wiring in `app.py` always passes them explicitly from
        config so this runner stays decoupled from any specific provider).
        """
        if not api_base:
            raise ValueError(
                "RawModelRunner.run_model requires `api_base`. "
                "Pass it explicitly from config (e.g. settings.models.small.api_base)."
            )
        if not api_key:
            # Fall back to OPENAI_API_KEY for convenience, matching the
            # common pattern where providers expose a single key env var.
            api_key = os.environ.get("OPENAI_API_KEY", "")

        url = api_base.rstrip("/") + "/chat/completions"

        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        # Apply defaults, then per-call overrides win.
        body.update(self.request_defaults)
        body.update(overrides)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.post(
                url,
                json=body,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"OpenAI-compatible request to {url} failed for model "
                f"'{model}': {exc}"
            ) from exc

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from {url} for model '{model}': "
                f"{data!r}"
            ) from exc
