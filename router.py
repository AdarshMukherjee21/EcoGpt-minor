"""
router.py — Microsoft RouteLLM wrapper, config-driven.

Drops the previous mock OpenAI base URL: RouteLLM's controller is now
configured with the real `api_base` / `api_key` for both strong and weak
models straight from `config.settings.models`.

For the MF router, embeddings are computed locally via sentence-transformers
(MiniLM) instead of OpenAI's embedding API — no real OpenAI key needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from config import settings

PROJECT_ROOT = Path(__file__).resolve().parent


def _patch_mf_embedding_client() -> None:
    """
    Monkey-patch RouteLLM's MF model to use a local sentence-transformers
    embedding model instead of OpenAI's embedding API.

    RouteLLM's `MFModel.forward()` calls:
        OPENAI_CLIENT.embeddings.create(input=[prompt], model=self.embedding_model)
    and accesses `.data[0].embedding`.

    We replace `OPENAI_CLIENT.embeddings.create` with a callable that
    returns an object with the same shape, backed by a local MiniLM model.
    """
    emb_cfg = settings.router.embedding
    if emb_cfg is None or emb_cfg.provider != "sentence-transformers":
        # No patch needed (e.g. BERT router, or user explicitly chose OpenAI).
        return

    from sentence_transformers import SentenceTransformer

    # Resolve device: "auto" -> CUDA if available, else CPU.
    device = emb_cfg.device.lower()
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"[router] loading local embedding model "
        f"{emb_cfg.model_name} on device={device}"
    )
    _local_model = SentenceTransformer(emb_cfg.model_name, device=device)

    # Patch the module-level OPENAI_CLIENT in RouteLLM's utils.
    from routellm.routers.similarity_weighted import utils as _routellm_utils

    class _LocalEmbeddingResponse:
        """Mimics the shape of OpenAI's EmbeddingCreateResponse."""

        def __init__(self, vectors):
            self.data = [type("Row", (), {"embedding": v})() for v in vectors]

    def _local_create(input, _model=None, **_kwargs):  # noqa: ARG001
        # `input` can be a str or list[str]; normalize to list.
        if isinstance(input, str):
            input = [input]
        vectors = _local_model.encode(input, convert_to_numpy=True)
        return _LocalEmbeddingResponse(vectors)

    _routellm_utils.OPENAI_CLIENT.embeddings.create = _local_create

    # Also patch the MF model module's reference (it imported OPENAI_CLIENT
    # by value, so we need to patch the same object it holds).
    from routellm.routers.matrix_factorization import model as _mf_model_mod

    _mf_model_mod.OPENAI_CLIENT.embeddings.create = _local_create
    # Tag the embedding model name so it's visible in logs / debugging.
    _mf_model_mod.MFModel.embedding_model = emb_cfg.model_name

    print(f"[router] embedding client patched -> {emb_cfg.model_name}")


class LLMRouterClient:
    def __init__(
        self,
        strong_model: Optional[str] = None,
        weak_model: Optional[str] = None,
        router_type: Optional[str] = None,
        threshold: Optional[float] = None,
        strong_api_base: Optional[str] = None,
        weak_api_base: Optional[str] = None,
        strong_api_key: Optional[str] = None,
        weak_api_key: Optional[str] = None,
    ):
        # Resolve all routing decisions from config, allowing explicit
        # overrides (handy for tests).
        strong_model = strong_model or settings.models.large.name
        weak_model = weak_model or settings.models.small.name
        router_type = router_type or settings.router.type
        threshold = threshold if threshold is not None else settings.router.threshold

        strong_api_base = strong_api_base or settings.models.large.api_base
        weak_api_base = weak_api_base or settings.models.small.api_base
        strong_api_key = (
            strong_api_key
            if strong_api_key is not None
            else settings.models.large.api_key
        )
        weak_api_key = (
            weak_api_key if weak_api_key is not None else settings.models.small.api_key
        )

        # RouteLLM's Controller speaks OpenAI and reads a single
        # OPENAI_API_BASE / OPENAI_API_KEY from the environment. If both
        # models share a base (typical: hosted OpenAI / Together / Groq /
        # a single vLLM), we just point at it. If they live on different
        # bases, the wiring in `app.py` falls back to a separate
        # `RawModelRunner` call for the strong model when needed.
        chosen_base = strong_api_base or weak_api_base
        chosen_key = strong_api_key or weak_api_key

        os.environ["OPENAI_API_KEY"] = chosen_key or "sk-dummy-key-for-local-routing"
        if chosen_base:
            os.environ["OPENAI_API_BASE"] = chosen_base

        # Make the local RouteLLM repo importable.
        repo_path = r"RouteLLM"
        if repo_path not in sys.path:
            sys.path.append(repo_path)

        # Patch the MF embedding client BEFORE importing Controller, so the
        # patch is in place when the MF model is instantiated.
        if router_type == "mf":
            _patch_mf_embedding_client()

        from routellm.controller import Controller  # local repo import

        # Patch Controller.completion so litellm gets an explicit provider
        # hint. litellm can't auto-detect the provider from a bare model
        # name like "gpt-oss:120b-cloud", and we always use the OpenAI-
        # compatible HTTP adapter regardless of the underlying provider.
        _patch_controller_for_litellm(Controller)

        self.strong_model = strong_model
        self.weak_model = weak_model
        self.strong_api_base = strong_api_base
        self.weak_api_base = weak_api_base
        self.strong_api_key = strong_api_key
        self.weak_api_key = weak_api_key
        self.router_model_name = f"router-{router_type}-{threshold}"

        # Build the per-router config. For the MF router, point at the
        # local MiniLM-trained checkpoint and tell the model to use the
        # sentence-transformers embedding provider (set up above via
        # _patch_mf_embedding_client). The router auto-derives text_dim
        # from the checkpoint's text_proj.0.weight shape, so 384-dim
        # MiniLM embeddings line up with the 384→128 projection layer.
        router_config = self._build_router_config(router_type)

        self.client = Controller(
            routers=[router_type],
            strong_model=strong_model,
            weak_model=weak_model,
            config=router_config,
        )

    def _build_router_config(self, router_type: str) -> dict:
        """
        Build the per-router config dict that gets passed to
        `routellm.controller.Controller(config=...)`. Each key in the
        returned dict is a router name; each value is the kwargs for
        that router's `__init__`.
        """
        if router_type != "mf":
            # BERT / causal_llm routers use the GPT-4 augmented defaults
            # from the controller. We don't override anything here yet.
            return {}

        # Local MiniLM-trained checkpoint (replaces the default
        # `routellm/mf_gpt4_augmented` hub model, which is 1536-dim).
        checkpoint_path = PROJECT_ROOT / "colab_outputs" / "checkpoint"

        emb_cfg = settings.router.embedding
        embedding_model = (
            emb_cfg.model_name
            if emb_cfg is not None
            else "sentence-transformers/all-MiniLM-L6-v2"
        )

        return {
            "mf": {
                "checkpoint_path": str(checkpoint_path),
                "embedding_provider": "sentence-transformers",
                "embedding_model": embedding_model,
            }
        }

    def generate_response(self, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.router_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "routed_to": response.model,
            "content": response.choices[0].message.content,
            "raw_response": response,
        }


def _patch_controller_for_litellm(Controller) -> None:
    """
    Wrap `Controller.completion` so the routed model name always carries
    an `openai/` provider prefix when handed to litellm.

    litellm can't auto-detect the provider from a bare model name like
    "gpt-oss:120b-cloud". ecoGPT is provider-agnostic but always uses the
    OpenAI-compatible HTTP adapter, so we inject the prefix here and
    let litellm's OpenAI adapter talk to whatever `api_base` / `api_key`
    the controller was configured with.
    """

    def completion(self, *, router=None, threshold=None, **kwargs):
        # The outer model name is of the form "router-{router_type}-{threshold}"
        # — parse both pieces out, regardless of whether router/threshold
        # were passed explicitly. The outer (None, None) defaults from
        # `chat.completions.create` must NOT win.
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])
        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        # Strip any existing provider prefix and re-add `openai/`. litellm
        # routes the request through the OpenAI adapter, which respects
        # `api_base` / `api_key` for whatever OpenAI-compatible endpoint
        # the user has configured.
        m = str(kwargs["model"])
        if "/" in m:
            m = m.split("/", 1)[1]
        kwargs["model"] = f"openai/{m}"
        # Inline the original litellm.completion() call so we don't recurse.
        from litellm import completion as _litellm_completion

        return _litellm_completion(
            api_base=self.api_base, api_key=self.api_key, **kwargs
        )

    Controller.completion = completion
    # Also patch acreate (the async sibling) to mirror the behavior.

    async def acompletion(self, *, router=None, threshold=None, **kwargs):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])
        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        m = str(kwargs["model"])
        if "/" in m:
            m = m.split("/", 1)[1]
        kwargs["model"] = f"openai/{m}"
        from litellm import acompletion as _litellm_acompletion

        return await _litellm_acompletion(
            api_base=self.api_base, api_key=self.api_key, **kwargs
        )

    Controller.acompletion = acompletion
