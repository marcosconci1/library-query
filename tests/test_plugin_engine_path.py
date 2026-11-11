"""Engine resolver order for the Hermes plugin: env, setting, launcher, PATH."""
import importlib.util
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "integrations/hermes/library-query/__init__.py"


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("library_query_plugin_engine", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("LIBRARY_QUERY_BIN", raising=False)
    # Neutral defaults: no launcher, nothing on PATH.
    monkeypatch.setattr(module, "__file__", str(tmp_path / "a/b/c/plugin/__init__.py"))
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    return module


def _launcher(plugin, tmp_path):
    launcher = tmp_path / "a/library-query.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\n")
    return str(launcher.resolve())


def test_env_wins_over_setting(plugin, monkeypatch, tmp_path):
    _launcher(plugin, tmp_path)
    monkeypatch.setenv("LIBRARY_QUERY_BIN", "/env/bin")
    monkeypatch.setattr(plugin.shutil, "which", lambda name: "/path/bin")
    assert plugin.resolve_engine("/setting/bin") == "/env/bin"


def test_setting_wins_over_launcher(plugin, monkeypatch, tmp_path):
    _launcher(plugin, tmp_path)
    monkeypatch.setattr(plugin.shutil, "which", lambda name: "/path/bin")
    assert plugin.resolve_engine("/setting/bin") == "/setting/bin"


def test_launcher_wins_over_path(plugin, monkeypatch, tmp_path):
    expected = _launcher(plugin, tmp_path)
    monkeypatch.setattr(plugin.shutil, "which", lambda name: "/path/bin")
    assert plugin.resolve_engine(None) == expected


def test_path_is_last(plugin, monkeypatch):
    monkeypatch.setattr(plugin.shutil, "which", lambda name: "/path/bin" if name == "library-query" else None)
    assert plugin.resolve_engine(None) == "/path/bin"


def test_none_found_raises_bounded_error(plugin):
    with pytest.raises(plugin.RetrievalError) as exc:
        plugin.resolve_engine(None)
    message = str(exc.value)
    assert message.startswith("engine_not_found")
    assert "LIBRARY_QUERY_BIN" in message and "bin" in message
    assert "/ho" + "me/" not in message and str(Path.home()) not in message
