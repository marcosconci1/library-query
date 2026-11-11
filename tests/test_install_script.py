"""Exercise install.sh with isolated command stubs and a temporary checkout."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TOOLS = ("bash", "env", "dirname", "readlink", "ln", "mkdir", "cat")


def _write_stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path: Path) -> dict[str, Path | dict[str, str]]:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    bin_dir = tmp_path / "bin"
    fake_path = tmp_path / "path"
    home.mkdir()
    repo.mkdir()
    bin_dir.mkdir()
    fake_path.mkdir()

    for tool in SYSTEM_TOOLS:
        source = shutil.which(tool)
        if source:
            (fake_path / tool).symlink_to(source)

    shutil.copy2(ROOT / "install.sh", repo / "install.sh")
    shutil.copy2(ROOT / "library-query.sh", repo / "library-query.sh")
    log = tmp_path / "calls.log"
    common = '''
printf '%s' "${TOOL_NAME}" >> "$CALL_LOG"
for arg in "$@"; do printf '\\t%s' "$arg" >> "$CALL_LOG"; done
printf '\\n' >> "$CALL_LOG"
'''
    for name in ("uv", "docker", "hermes"):
        body = common.replace("${TOOL_NAME}", name)
        if name == "uv":
            body += '''
if [ "${1:-}" = sync ]; then exit "${UV_SYNC_EXIT:-0}"; fi
for arg in "$@"; do
    if [ "$arg" = setup ]; then exit "${UV_SETUP_EXIT:-0}"; fi
    if [ "$arg" = --version ]; then printf '%s\\n' 'library-query 0.3.0'; exit 0; fi
done
exit 0
'''
        elif name == "docker":
            body += 'exit "${DOCKER_EXIT:-0}"\n'
        else:
            body += '''
profile=
expect_profile=
for arg in "$@"; do
    if [ "$expect_profile" = yes ]; then profile=$arg; expect_profile=; continue; fi
    if [ "$arg" = -p ]; then expect_profile=yes; fi
done
case " $* " in
    *' plugins install '*)
        if [ -n "${HERMES_FAIL_PROFILE:-}" ] && [ "$profile" = "$HERMES_FAIL_PROFILE" ]; then exit 1; fi
        ;;
esac
exit 0
'''
        _write_stub(fake_path / name, body)

    env = {
        "HOME": str(home),
        "PATH": str(fake_path),
        "CALL_LOG": str(log),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }
    return {"home": home, "repo": repo, "bin": bin_dir, "path": fake_path, "log": log, "env": env}


def _run(sandbox: dict[str, Path | dict[str, str]], *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
    repo = sandbox["repo"]
    env = dict(sandbox["env"])
    env.update(extra)
    return subprocess.run(
        [str(repo / "install.sh"), *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _calls(sandbox: dict[str, Path | dict[str, str]]) -> list[list[str]]:
    log = sandbox["log"]
    if not log.exists():
        return []
    return [line.split("\t") for line in log.read_text(encoding="utf-8").splitlines()]


def _calls_for(sandbox: dict[str, Path | dict[str, str]], name: str) -> list[list[str]]:
    return [call[1:] for call in _calls(sandbox) if call[0] == name]


def test_check_does_not_sync_link_or_install(sandbox):
    repo = sandbox["repo"]
    before = {path.relative_to(repo): path.read_bytes() for path in repo.iterdir() if path.is_file()}
    result = _run(sandbox, "--check", "--bin-dir", str(sandbox["bin"]))

    assert result.returncode == 0, result.stderr
    assert not (sandbox["bin"] / "library-query").exists()
    assert {path.relative_to(repo): path.read_bytes() for path in repo.iterdir() if path.is_file()} == before
    assert list(sandbox["home"].iterdir()) == []
    assert not any(call[:1] == ["sync"] for call in _calls_for(sandbox, "uv"))
    assert not any("setup" in call and "--check" in call for call in _calls_for(sandbox, "uv"))
    assert not any("run" in call for call in _calls_for(sandbox, "uv"))
    assert not (repo / ".venv").exists()
    assert not (repo / ".env").exists()
    assert not (sandbox["bin"] / "library-query").exists()
    assert not any("install" in call for call in _calls_for(sandbox, "hermes"))
    assert "[skip] setup check needs the installed engine" in result.stdout
    assert "would run: hermes plugins install" in result.stdout


def test_check_runs_setup_check_with_existing_engine(sandbox):
    (sandbox["repo"] / ".venv").mkdir()

    result = _run(sandbox, "--check", "--bin-dir", str(sandbox["bin"]))

    assert result.returncode == 0, result.stderr
    assert any(
        call[-2:] == ["setup", "--check"]
        for call in _calls_for(sandbox, "uv")
    )
    assert (sandbox["repo"] / ".venv").is_dir()


def test_missing_uv_fails_with_install_hint(sandbox):
    (sandbox["path"] / "uv").unlink()

    result = _run(sandbox)

    assert result.returncode == 1
    assert "install uv: https://docs.astral.sh/uv/" in result.stderr
    assert _calls(sandbox) == []


def test_full_run_syncs_links_and_installs_plugin(sandbox):
    result = _run(sandbox, "--bin-dir", str(sandbox["bin"]))
    link = sandbox["bin"] / "library-query"
    repo = sandbox["repo"]

    assert result.returncode == 0, result.stderr
    assert _calls_for(sandbox, "uv")[0] == ["sync", "--locked"]
    assert link.is_symlink()
    assert link.resolve() == repo / "library-query.sh"
    assert "[ok] engine version 0.3.0" in result.stdout
    assert f'export PATH="{sandbox["bin"]}:$PATH"' in result.stdout
    hermes_calls = _calls_for(sandbox, "hermes")
    assert hermes_calls == [
        ["plugins", "install", f"file://{repo}#integrations/hermes/library-query", "--force", "--enable", "--yes-deps"],
        ["plugins", "doctor", "library-query"],
    ]


def test_profiles_are_passed_and_each_is_installed(sandbox):
    result = _run(sandbox, "--profile", "a", "--profile", "b")

    assert result.returncode == 0, result.stderr
    calls = _calls_for(sandbox, "hermes")
    assert calls[0][:3] == ["-p", "a", "plugins"]
    assert calls[2][:3] == ["-p", "b", "plugins"]
    assert len([call for call in calls if "install" in call]) == 2
    assert len([call for call in calls if "doctor" in call]) == 2


def test_no_hermes_never_calls_hermes(sandbox):
    result = _run(sandbox, "--no-hermes")

    assert result.returncode == 0, result.stderr
    assert _calls_for(sandbox, "hermes") == []


def test_missing_hermes_is_skipped(sandbox):
    (sandbox["path"] / "hermes").unlink()

    result = _run(sandbox)

    assert result.returncode == 0, result.stderr
    assert "[skip] Hermes not found" in result.stdout
    assert _calls_for(sandbox, "hermes") == []


def test_missing_docker_is_optional(sandbox):
    (sandbox["path"] / "docker").unlink()

    result = _run(sandbox, "--no-hermes")

    assert result.returncode == 0, result.stderr
    assert "[warn] Docker Compose unavailable; start Qdrant by hand" in result.stdout


def test_regular_file_at_command_target_is_refused(sandbox):
    target = sandbox["bin"] / "library-query"
    target.write_bytes(b"keep this file byte for byte\n")
    before = target.read_bytes()

    result = _run(sandbox, "--bin-dir", str(sandbox["bin"]))

    assert result.returncode == 1
    assert "refusing to replace non-symlink" in result.stderr
    assert target.is_file() and not target.is_symlink()
    assert target.read_bytes() == before


def test_wrong_symlink_is_replaced(sandbox, tmp_path):
    target = sandbox["bin"] / "library-query"
    other = tmp_path / "other-launcher"
    other.write_text("old target", encoding="utf-8")
    target.symlink_to(other)

    result = _run(sandbox, "--bin-dir", str(sandbox["bin"]), "--no-hermes")

    assert result.returncode == 0, result.stderr
    assert target.is_symlink() and target.resolve() == sandbox["repo"] / "library-query.sh"
    assert "[ok] command link updated" in result.stdout


def test_second_run_keeps_the_same_symlink(sandbox):
    args = ("--bin-dir", str(sandbox["bin"]))
    first = _run(sandbox, *args)
    link = sandbox["bin"] / "library-query"
    target = link.resolve()
    inode = link.lstat().st_ino
    second = _run(sandbox, *args)

    assert first.returncode == second.returncode == 0
    assert link.is_symlink() and link.resolve() == target == sandbox["repo"] / "library-query.sh"
    assert link.lstat().st_ino == inode


def test_setup_failure_is_a_warning_not_installer_failure(sandbox):
    result = _run(sandbox, "--no-hermes", UV_SETUP_EXIT="7")

    assert result.returncode == 0
    assert "[warn] setup reported open items" in result.stdout


def test_failed_profile_does_not_stop_later_profile(sandbox):
    result = _run(sandbox, "--profile", "a", "--profile", "b", HERMES_FAIL_PROFILE="a")
    calls = _calls_for(sandbox, "hermes")

    assert result.returncode == 1
    assert "[fail] Hermes plugin failed for a" in result.stderr
    assert any(call[:3] == ["-p", "b", "plugins"] and "install" in call for call in calls)
    assert any(call[:3] == ["-p", "b", "plugins"] and "doctor" in call for call in calls)


def test_setup_flags_are_forwarded_and_key_prompt_is_not(sandbox):
    result = _run(sandbox, "--no-docker", "--library", "X", "--no-hermes")
    setup_calls = [call for call in _calls_for(sandbox, "uv") if "setup" in call]

    assert result.returncode == 0, result.stderr
    assert len(setup_calls) == 1
    assert "--no-docker" in setup_calls[0]
    assert setup_calls[0][setup_calls[0].index("--library") + 1] == "X"
    assert "--ask-key" not in setup_calls[0]


def test_unknown_option_returns_usage_status(sandbox):
    result = _run(sandbox, "--not-an-option")

    assert result.returncode == 2
    assert "Usage: ./install.sh" in result.stderr


def test_shell_syntax_and_optional_shellcheck():
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(ROOT / "install.sh")], check=True)

    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck is not installed; skipped shellcheck validation")
    subprocess.run([shellcheck, str(ROOT / "install.sh")], check=True)
