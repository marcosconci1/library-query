"""Release-checklist guard: pyproject, CHANGELOG and uv.lock versions agree."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def _changelog():
    m = re.search(r"^## (\d+\.\d+\.\d+)\b", (ROOT / "CHANGELOG.md").read_text(), re.M)
    assert m, "CHANGELOG.md has no '## X.Y.Z' heading"
    return m.group(1)


def _lock():
    pkgs = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    found = [p["version"] for p in pkgs if p["name"] == "library-query"]
    assert len(found) == 1, "uv.lock needs exactly one library-query entry"
    return found[0]


def test_versions_agree():
    py, log, lock = _pyproject(), _changelog(), _lock()
    assert re.fullmatch(r"\d+\.\d+\.\d+", py)
    assert py == log == lock


def test_hermes_plugin_version_matches_pyproject():
    text = (ROOT / "integrations/hermes/library-query/plugin.yaml").read_text()
    m = re.search(r"^version:\s*['\"]?(\d+\.\d+\.\d+)['\"]?\s*$", text, re.M)
    assert m, "plugin.yaml has no version: key"
    assert m.group(1) == _pyproject()
