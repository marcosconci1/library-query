"""The CLI version matches the package metadata."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_version_prints_installed_package_version():
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project_version = tomllib.load(project_file)["project"]["version"]

    result = subprocess.run(
        [sys.executable, "-m", "library_query.cli", "--version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert re.fullmatch(r"library-query \d+\.\d+\.\d+\n", result.stdout)
    assert result.stdout.strip() == f"library-query {project_version}"
