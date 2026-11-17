import json

from proxy import claude_desktop
from proxy.claude_desktop import ECOGPT_ENTRY_ID, apply, restore, status

META = {
    "appliedId": "8f69f2f1-3275-4ad8-9317-4aa7e972f311",
    "entries": [
        {"id": "8f69f2f1-3275-4ad8-9317-4aa7e972f311", "name": "Claude Code Router"},
        {"id": ECOGPT_ENTRY_ID, "name": "ecoGPT"},
    ],
}


def _seed(app_dir):
    (app_dir / "configLibrary").mkdir(parents=True)
    (app_dir / "configLibrary" / "_meta.json").write_text(json.dumps({
        "appliedId": "8f69f2f1-3275-4ad8-9317-4aa7e972f311",
        "entries": [{"id": "8f69f2f1-3275-4ad8-9317-4aa7e972f311", "name": "Claude Code Router"}],
    }), encoding="utf-8")
    (app_dir / "configLibrary" / "8f69f2f1-3275-4ad8-9317-4aa7e972f311.json").write_text(
        json.dumps({"inferenceGatewayBaseUrl": "https://openrouter.ai/api/"}), encoding="utf-8")
    (app_dir / "developer_settings.json").write_text(json.dumps({"allowDevTools": False}), encoding="utf-8")
    (app_dir / "claude_desktop_config.json").write_text(json.dumps({"deploymentMode": "3p"}), encoding="utf-8")


def test_apply_creates_ecogpt_entry_and_backs_up(tmp_path):
    app_dir = tmp_path / "Claude-3p"
    backup_dir = tmp_path / "backup"
    _seed(app_dir)
    result = apply(app_dir, "http://127.0.0.1:18090", "sk-test", backup_dir=backup_dir)
    assert result["applied_id"] == ECOGPT_ENTRY_ID
    assert result["base_url"] == "http://127.0.0.1:18090"
    profile = json.loads((app_dir / "configLibrary" / f"{ECOGPT_ENTRY_ID}.json").read_text())
    assert profile["inferenceGatewayBaseUrl"] == "http://127.0.0.1:18090"
    assert profile["inferenceGatewayApiKey"] == "sk-test"
    assert profile["inferenceProvider"] == "gateway"
    assert json.loads((app_dir / "developer_settings.json").read_text()) == {"allowDevTools": True}
    meta = json.loads((app_dir / "configLibrary" / "_meta.json").read_text())
    assert len(meta["entries"]) == 2
    # byte-for-byte backups of the pre-existing config
    assert (backup_dir / "developer_settings.json").read_text() == json.dumps({"allowDevTools": False})
    assert json.loads((backup_dir / "configLibrary__meta.json").read_text())["entries"] == [
        {"id": "8f69f2f1-3275-4ad8-9317-4aa7e972f311", "name": "Claude Code Router"}
    ]


def test_apply_without_app_dir_fails_clean(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="Claude Desktop data not found"):
        apply(tmp_path / "missing", "http://x", "k", backup_dir=tmp_path / "b")


def test_apply_seeds_models_and_bearer_scheme(tmp_path):
    app_dir = tmp_path / "Claude-3p"
    backup_dir = tmp_path / "backup"
    _seed(app_dir)
    entries = claude_desktop.desktop_models(["openrouter/free", "nvidia/nemotron-3.5-lightning:free"])
    apply(app_dir, "http://127.0.0.1:18090", "sk-test", backup_dir=backup_dir, model_entries=entries)
    profile = json.loads((app_dir / "configLibrary" / f"{ECOGPT_ENTRY_ID}.json").read_text())
    # The app's own auth-scheme enum is ["bearer", "x-api-key"] with default
    # "bearer"; pinned explicitly so the static key is sent as a Bearer header.
    assert profile["inferenceGatewayAuthScheme"] == "bearer"
    assert profile["inferenceCredentialKind"] == "static"
    # Model picker seeded with validation-safe alias entries: the name must
    # contain an Anthropic allow-token, carry no deny token (ernie etc.) and
    # is what the app sends VERBATIM as the gateway model id.
    models = profile["inferenceModels"]
    assert [m["name"] for m in models] == ["openrouter-free", "nvidia-nemotron-3-5-lightning-free"]
    assert [m["labelOverride"] for m in models] == ["openrouter/free", "nvidia/nemotron-3.5-lightning:free"]
    assert [m["anthropicFamilyTier"] for m in models] == ["sonnet", "haiku"]
    assert models[0]["isFamilyDefault"] is True and models[1]["isFamilyDefault"] is True


def test_desktop_models_prefers_resolved_display_names():
    models = claude_desktop.desktop_models(
        ["gpt-6-luna", "mystery-model:free", "blank-name"],
        ["GPT-6 Luna", "", None],
    )
    # Caller-provided display names label the picker AND drive the alias slug;
    # missing/blank entries and a shorter list fall back to the raw model id
    # (module stays pure/resolved).
    assert [m["name"] for m in models] == ["gpt-6-luna", "mystery-model-free", "blank-name"]
    assert [m["labelOverride"] for m in models] == ["GPT-6 Luna", "mystery-model:free", "blank-name"]


def test_desktop_alias_name_is_display_name_slug():
    from proxy.claude_desktop import desktop_alias_name
    assert desktop_alias_name("GPT 6 Luna") == "gpt-6-luna"
    assert desktop_alias_name("  Dashy -- Name  ") == "dashy-name"
    # only [a-z0-9-], no leading/trailing dash; all-symbol names slug to ""
    import re
    for raw in ("Weird___Name!!", "A B C", "!!!"):
        slug = desktop_alias_name(raw)
        assert re.fullmatch(r"[a-z0-9-]*", slug)
        assert slug == slug.strip("-")


def test_alias_map_generates_unique_slug_aliases():
    from proxy.claude_desktop import desktop_alias_map
    mapping = desktop_alias_map(
        ["model-a", "model-b", "model-c"],
        ["Same Name", "SAME NAME", "other"],
    )
    names = sorted(mapping)
    assert sorted(names) == ["other", "same-name", "same-name-2"]
    assert mapping["same-name"] == "model-a"
    assert mapping["same-name-2"] == "model-b"
    assert mapping["other"] == "model-c"
    # aliases never shadow a real model id; self-colliding slugs need no alias
    assert not (set(mapping) & {"model-a", "model-b", "model-c"})


def test_alias_map_self_collision_needs_no_alias():
    from proxy.claude_desktop import desktop_alias_map
    mapping = desktop_alias_map(["gpt-6-luna"], ["GPT 6 Luna"])
    assert mapping == {}


def test_alias_map_is_idempotent_and_migrates_legacy_ids():
    from proxy.claude_desktop import desktop_alias_map, is_legacy_alias
    first = desktop_alias_map(["mystery-model:free"], ["Mystery Model (free)"])
    assert first == {"mystery-model-free": "mystery-model:free"}
    again = desktop_alias_map(["mystery-model:free"], ["Mystery Model (free)"], previous=first)
    assert again == first
    # a known alias is reused even when the display name changed
    third = desktop_alias_map(["mystery-model:free"], ["Renamed Model"], previous=first)
    assert third == first
    # legacy claude-eco-N registries/aliases are never reused
    legacy = desktop_alias_map(["mystery-model:free"], ["Mystery Model (free)"],
                               previous={"claude-eco-1": "mystery-model:free"})
    assert legacy == first
    assert not any(is_legacy_alias(alias) for alias in legacy)
    # aliases whose real model disappeared are dropped
    assert desktop_alias_map(["other-real"], ["Other"], previous=first) == {"other": "other-real"}


def test_apply_without_models_omits_inference_models(tmp_path):
    app_dir = tmp_path / "Claude-3p"
    _seed(app_dir)
    apply(app_dir, "http://127.0.0.1:18090", "sk-test", backup_dir=tmp_path / "b")
    profile = json.loads((app_dir / "configLibrary" / f"{ECOGPT_ENTRY_ID}.json").read_text())
    assert "inferenceModels" not in profile


def test_apply_migrates_legacy_claude_eco_entries(tmp_path):
    """Old claude-eco-N entries inside the Desktop config are replaced by slugs."""
    app_dir = tmp_path / "Claude-3p"
    _seed(app_dir)
    profile_path = app_dir / "configLibrary" / f"{ECOGPT_ENTRY_ID}.json"
    profile_path.write_text(json.dumps({
        "inferenceGatewayBaseUrl": "http://old", "inferenceGatewayApiKey": "k",
        "inferenceModels": [
            {"name": "claude-eco-1", "labelOverride": "Old A"},
            {"name": "claude-eco-2", "labelOverride": "Old B"},
        ]}), encoding="utf-8")
    entries = claude_desktop.desktop_models(["real-a", "real-b"], ["Alpha Model", "Beta Model"])
    apply(app_dir, "http://127.0.0.1:18090", "k", backup_dir=tmp_path / "b", model_entries=entries)
    profile = json.loads(profile_path.read_text())
    models = profile["inferenceModels"]
    assert [m["name"] for m in models] == ["alpha-model", "beta-model"]
    assert "claude-eco-1" not in json.dumps(profile) and "claude-eco-2" not in json.dumps(profile)
    # same fixed entry, still the single ecoGPT entry in meta (updated in place)
    meta = json.loads((app_dir / "configLibrary" / "_meta.json").read_text())
    assert meta["appliedId"] == ECOGPT_ENTRY_ID
    assert len([e for e in meta["entries"] if e["id"] == ECOGPT_ENTRY_ID]) == 1


def test_apply_idempotent_repeated_runs_update_in_place(tmp_path):
    app_dir = tmp_path / "Claude-3p"
    _seed(app_dir)
    entries = claude_desktop.desktop_models(["real-a", "real-b"], ["Alpha Model", "Beta Model"])
    first = apply(app_dir, "http://127.0.0.1:18090", "k1", backup_dir=tmp_path / "b", model_entries=entries)
    profile_path = app_dir / "configLibrary" / f"{ECOGPT_ENTRY_ID}.json"
    snapshot = profile_path.read_text()
    meta_snapshot = (app_dir / "configLibrary" / "_meta.json").read_text()
    second = apply(app_dir, "http://127.0.0.1:18090", "k1", backup_dir=tmp_path / "b", model_entries=entries)
    assert second["applied_id"] == first["applied_id"] == ECOGPT_ENTRY_ID
    assert profile_path.read_text() == snapshot
    assert (app_dir / "configLibrary" / "_meta.json").read_text() == meta_snapshot


def test_restore_swaps_back_and_clears_backup(tmp_path):
    app_dir = tmp_path / "Claude-3p"
    backup_dir = tmp_path / "backup"
    _seed(app_dir)
    apply(app_dir, "http://127.0.0.1:18090", "sk-test", backup_dir=backup_dir)
    result = restore(app_dir, backup_dir)
    assert result["applied_id"] == "8f69f2f1-3275-4ad8-9317-4aa7e972f311"
    assert result["base_url"] == "https://openrouter.ai/api/"
    assert not (app_dir / "configLibrary" / f"{ECOGPT_ENTRY_ID}.json").exists()
    assert json.loads((app_dir / "configLibrary" / "_meta.json").read_text())["entries"] == [
        {"id": "8f69f2f1-3275-4ad8-9317-4aa7e972f311", "name": "Claude Code Router"}
    ]
    assert not backup_dir.exists()


def test_restore_without_backup_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="No backup found"):
        restore(tmp_path / "Claude-3p", tmp_path / "b")


def test_status_reports_state(tmp_path):
    app_dir = tmp_path / "Claude-3p"
    _seed(app_dir)
    first = status(app_dir, tmp_path / "b")
    assert first["applied_id"] == "8f69f2f1-3275-4ad8-9317-4aa7e972f311"
    assert first["entry_name"] == "Claude Code Router"
    assert first["ecogpt_entry_id"] == ECOGPT_ENTRY_ID
    assert first["backup_exists"] is False
    apply(app_dir, "http://127.0.0.1:18090", "k", backup_dir=tmp_path / "b")
    assert status(app_dir, tmp_path / "b")["backup_exists"] is True
