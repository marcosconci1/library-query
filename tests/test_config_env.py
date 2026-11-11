import importlib
import os
from pathlib import Path

from library_query import config


def test_load_env_uses_the_explicit_module_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "library-query.env"
    env_file.write_text("OPENROUTER_API_KEY=test-key\nLIBRARY_ROOT=/tmp/test-library\n")
    monkeypatch.setenv("LIBRARY_QUERY_ENV_FILE", str(env_file))
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("LIBRARY_ROOT", raising=False)

    importlib.reload(config)

    assert config.OPENROUTER_API_KEY == "test-key"
    assert str(config.LIBRARY_ROOT) == "/tmp/test-library"

    monkeypatch.undo()
    importlib.reload(config)


def test_blank_env_values_do_not_override_defaults(monkeypatch, tmp_path):
    env_file = tmp_path / "library-query.env"
    env_file.write_text("LIBRARY_QUERY_STATE=\nLIBRARY_ROOT=   \nEMBEDDING_MODEL=custom-model\n")
    monkeypatch.setenv("LIBRARY_QUERY_ENV_FILE", str(env_file))
    for name in ("LIBRARY_QUERY_STATE", "LIBRARY_ROOT", "EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)

    importlib.reload(config)

    assert "LIBRARY_QUERY_STATE" not in os.environ
    assert "LIBRARY_ROOT" not in os.environ
    assert config.STATE_DIR == Path.home() / ".local/share/library-query"
    assert config.LIBRARY_ROOT == Path.home() / "library"
    assert config.LEDGER_PATH == config.STATE_DIR / "ledger.json"
    assert config.EMBEDDING_MODEL == "custom-model"

    monkeypatch.undo()
    importlib.reload(config)


def test_real_environment_still_wins_over_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "library-query.env"
    env_file.write_text("EMBEDDING_MODEL=from-file\nLIBRARY_QUERY_STATE=\n")
    monkeypatch.setenv("LIBRARY_QUERY_ENV_FILE", str(env_file))
    monkeypatch.setenv("EMBEDDING_MODEL", "from-environment")
    monkeypatch.setenv("LIBRARY_QUERY_STATE", str(tmp_path / "state"))

    config.load_env()

    assert os.environ["EMBEDDING_MODEL"] == "from-environment"
    assert os.environ["LIBRARY_QUERY_STATE"] == str(tmp_path / "state")
