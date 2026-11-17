"""Shared display-name prettifier + Desktop alias slugs (proxy/model_names.py)."""

from __future__ import annotations

from proxy.model_names import pretty_name, slugify


def test_pretty_name_splits_provider_model_and_labels_variants():
    assert pretty_name("baidu/ernie-5.5-flash-preview:free") == \
        "Baidu Ernie 5.5 Flash Preview (free)"
    assert pretty_name("gpt-6-luna") == "GPT 6 Luna"
    assert pretty_name("meta-llama/Llama-3.1-70B-Instruct") == "Meta Llama Llama 3.1 70B Instruct"


def test_pretty_name_is_deterministic_and_handles_degenerate_input():
    assert pretty_name("some-model:free") == pretty_name("some-model:free")
    assert pretty_name("") == ""
    assert pretty_name("   ") == ""
    assert pretty_name("model:") == "Model"  # empty variant -> no suffix


def test_slug_of_display_name():
    assert slugify("GPT 6 Luna") == "gpt-6-luna"
    assert slugify("Mystery Model (free)") == "mystery-model-free"
    assert slugify("  Trimmed  Dashes--here  ") == "trimmed-dashes-here"


def test_slug_collapses_symbols_underscores_to_single_dashes():
    assert slugify("Weird___name  with//symbols!!") == "weird-name-with-symbols"
    assert slugify("a_b c-d.e") == "a-b-c-d-e"


def test_slug_output_is_only_lowercase_alphanumeric_and_dashes():
    import re
    for raw in ("GPT 6 Luna", "!!!", "Élan café", "a" * 100, ""):
        slug = slugify(raw)
        assert re.fullmatch(r"[a-z0-9-]*", slug)
        assert slug == slug.strip("-")
        assert "--" not in slug
    assert slugify("") == ""
