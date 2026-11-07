import importlib

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
