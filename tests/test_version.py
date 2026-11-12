"""Release-checklist guard: pyproject, CHANGELOG, uv.lock, plugin.yaml, and installed metadata versions agree."""
import re
import tomllib
from pathlib import Path

import yaml

from library_query import doctor

ROOT = Path(__file__).resolve().parent.parent


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def _changelog_versions():
    versions = re.findall(r"^## (\d+\.\d+\.\d+)\b", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
    assert versions, "CHANGELOG.md has no '## X.Y.Z' heading"
    return versions


def _changelog():
    return _changelog_versions()[0]


def _lock():
    pkgs = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    found = [p["version"] for p in pkgs if p["name"] == "library-query"]
    assert len(found) == 1, "uv.lock needs exactly one library-query entry"
    return found[0]


def _plugin():
    text = (ROOT / "integrations/hermes/library-query/plugin.yaml").read_text(encoding="utf-8")
    m = re.search(r"^version:\s*['\"]?(\d+\.\d+\.\d+)['\"]?\s*$", text, re.M)
    assert m, "plugin.yaml has no version: key"
    manifest = yaml.safe_load(text)
    assert isinstance(manifest, dict) and str(manifest.get("version")) == m.group(1)
    return m.group(1)


def test_versions_agree():
    py, log, lock, plugin = _pyproject(), _changelog(), _lock(), _plugin()
    assert re.fullmatch(r"\d+\.\d+\.\d+", py)
    assert py == log == lock == plugin == doctor.tool_version()


def test_hermes_plugin_version_matches_pyproject():
    assert _plugin() == _pyproject()


def test_changelog_release_headings_are_unique_and_descending():
    versions = _changelog_versions()
    assert versions[0] == _pyproject()
    assert len(versions) == len(set(versions))
    parsed = [tuple(int(part) for part in v.split(".")) for v in versions]
    assert parsed == sorted(parsed, reverse=True)
