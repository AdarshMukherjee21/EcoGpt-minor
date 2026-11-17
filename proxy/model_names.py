"""Deterministic model-id -> display-name prettifier (shared by every listing).

Gateway metadata ``display_name`` always wins when present (app-level rule);
this module is the readable, deterministic fallback for raw ids like
``baidu/ernie-5.5-flash-preview:free`` -> ``Baidu Ernie 5.5 Flash Preview (free)``.

Rules (unit-tested in tests/test_model_names.py):
- provider prefix kept as leading words ("meta-llama/..." -> "Meta Llama ...");
- ``-`` / ``_`` / ``/`` / whitespace separate words; dots are kept so versions
  ("5.5", "1.7b") survive;
- every word capitalized; known acronyms (GPT, LLM, MoE, ...) and digit-suffixed
  units ("70b" -> "70B", "1m" -> "1M") keep their canonical casing;
- a ``:variant`` tail renders as a subtle "(variant)" suffix (":free" -> "(free)").
"""

from __future__ import annotations

import re

ACRONYMS = {
    "llm": "LLM", "llms": "LLMs", "gpt": "GPT", "moe": "MoE", "ai": "AI",
    "api": "API", "gpu": "GPU", "gpus": "GPUs", "tpu": "TPU", "bf16": "BF16",
    "fp16": "FP16", "fp8": "FP8", "hd": "HD", "tts": "TTS", "asr": "ASR",
    "iso": "ISO", "vlm": "VLM", "rag": "RAG", "cpu": "CPU", "nlp": "NLP",
}

# gpt-4o and friends: capital after a digit (4o -> 4O is wrong for gpt-4o, which
# canonically renders "GPT-4o" — handled by the acronym table for the model word
# and a keep-case rule for letter-after-digit tails).
_TAIL = re.compile(r"^(\d+(?:\.\d+)?)([a-z]{1,2})$")
_WORD_SPLIT = re.compile(r"[-_/\s]+")


def _word(raw: str) -> str:
    lowered = raw.lower()
    if lowered in ACRONYMS:
        return ACRONYMS[lowered]
    tail = _TAIL.match(lowered)
    if tail:  # version-ish token: 5.5 / 70b / 1m / 3.3b
        number, unit = tail.groups()
        return f"{number}{unit.upper()}" if unit else number
    # gpt-style mixed tokens (4o, 3.5-turbo pieces) keep as-is apart from casing
    if not lowered.isupper():
        return lowered[:1].upper() + lowered[1:]
    return raw.upper()


_SLUG_SPLIT = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Deterministic id-safe slug of a display name (or raw model id).

    Lowercase; runs of spaces/symbols/underscores become single dashes; only
    ``[a-z0-9-]`` survive; leading/trailing dashes collapse away.
    ``"GPT 6 Luna" -> "gpt-6-luna"``. Empty input yields "".
    """
    return "-".join(part for part in _SLUG_SPLIT.split((name or "").lower()) if part)


def pretty_name(model_id: str) -> str:
    """Human-readable display fallback for a raw model id; deterministic."""
    text = (model_id or "").strip()
    if not text:
        return ""
    base, colon, variant = text.partition(":")
    words = [_word(w) for w in _WORD_SPLIT.split(base) if w]
    named = " ".join(words)
    return f"{named} ({variant.strip().lower()})" if colon and variant.strip() else named
