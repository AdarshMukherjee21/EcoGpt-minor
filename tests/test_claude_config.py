import json

from proxy import claude_config


def test_apply_then_restore_returns_original_bytes(tmp_path):
    path = tmp_path / "settings.json"
    original = '{\n  "env": {\n    "ANTHROPIC_BASE_URL": "http://127.0.0.1:18080"\n  },\n  "theme": "dark"\n}\n'
    path.write_text(original, encoding="utf-8")

    status = claude_config.apply(path, "http://127.0.0.1:18090", ["claude-haiku-5-5"])
    assert status["base_url"] == "http://127.0.0.1:18090"
    assert status["backup_exists"] is True

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["availableModels"] == ["claude-haiku-5-5"]
    assert data["theme"] == "dark"

    claude_config.restore(path)
    assert path.read_text(encoding="utf-8") == original
    assert not claude_config.backup_path(path).exists()


def test_second_apply_keeps_original_backup(tmp_path):
    path = tmp_path / "settings.json"
    original = '{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:18080"}}'
    path.write_text(original, encoding="utf-8")

    claude_config.apply(path, "http://127.0.0.1:18090", [])
    claude_config.apply(path, "http://127.0.0.1:18090", ["claude-haiku-5-5"])
    claude_config.restore(path)

    assert path.read_text(encoding="utf-8") == original


def test_restore_removes_file_that_apply_created(tmp_path):
    path = tmp_path / "settings.json"

    claude_config.apply(path, "http://127.0.0.1:18090", [])
    assert path.exists()

    claude_config.restore(path)
    assert not path.exists()


def test_apply_writes_bearer_gateway_recipe(tmp_path):
    """Per the OpenRouter/DeepSeek cookbooks: bearer token, blanked API key,
    gateway model discovery, role model aliases."""
    path = tmp_path / "settings.json"
    path.write_text('{"env": {}}', encoding="utf-8")

    claude_config.apply(path, "http://127.0.0.1:18090",
                        ["claude-sonnet-5", "claude-haiku-4"],
                        gateway_key="sk-gateway")

    data = json.loads(path.read_text(encoding="utf-8"))
    env = data["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18090"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-gateway"
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    # Role aliases pin every model class to an exposed model
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-sonnet-5"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-5"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-haiku-4"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-haiku-4"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-haiku-4"
    assert data["availableModels"] == ["claude-sonnet-5", "claude-haiku-4"]

    state = claude_config.status(path)
    assert state["auth_token_set"] is True
    assert state["api_key_blank"] is True
    assert state["gateway_discovery_enabled"] is True


def test_role_env_without_second_model_falls_back_to_primary():
    values = claude_config.role_model_env(["only-model"])
    assert values["ANTHROPIC_MODEL"] == "only-model"
    assert values["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "only-model"
    assert claude_config.role_model_env([]) == {}
