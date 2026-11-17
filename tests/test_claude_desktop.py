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
    assert [m["name"] for m in models] == ["claude-eco-1", "claude-eco-2"]
    assert [m["labelOverride"] for m in models] == ["openrouter/free", "nvidia/nemotron-3.5-lightning:free"]
    assert [m["anthropicFamilyTier"] for m in models] == ["sonnet", "haiku"]
    assert models[0]["isFamilyDefault"] is True and models[1]["isFamilyDefault"] is True


def test_alias_names_are_validation_safe():
    from proxy.claude_desktop import desktop_alias_name
    allow = ("claude", "sonnet", "opus", "haiku", "fable", "mythos", "anthropic")
    deny = ("gpt", "gemini", "qwen", "ernie", "openai", "llama", "#", ":", "/")
    for index in range(30):
        name = desktop_alias_name(index)
        lowered = name.lower()
        assert any(token in lowered for token in allow)
        assert not any(token in lowered for token in deny)


def test_apply_without_models_omits_inference_models(tmp_path):
    app_dir = tmp_path / "Claude-3p"
    _seed(app_dir)
    apply(app_dir, "http://127.0.0.1:18090", "sk-test", backup_dir=tmp_path / "b")
    profile = json.loads((app_dir / "configLibrary" / f"{ECOGPT_ENTRY_ID}.json").read_text())
    assert "inferenceModels" not in profile


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
