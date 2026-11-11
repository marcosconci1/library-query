"""`library-query setup`, fully offline: no network, no real docker, no real Qdrant."""
import json
import os
import stat
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from library_query import cli, config, doctor, journal, setup_cmd, store

TEMPLATE = (
    "# template\n"
    "LIBRARY_ROOT=/absolute/path/to/library\n"
    "QDRANT_URL=http://127.0.0.1:6333\n"
    "EMBEDDING_BASE_URL=https://openrouter.ai/api/v1\n"
    "EMBEDDING_MODEL=openai/text-embedding-3-small\n"
    "EMBEDDING_DIM=1536\n"
    "OPENROUTER_API_KEY=\n"
)
CANARY = "sk-or-v1-CANARYSETUPKEY0123456789abcdef"
ENV_KEYS = (
    "LIBRARY_ROOT", "QDRANT_URL", "OPENROUTER_API_KEY",
    "EMBEDDING_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_DIM",
)


class World:
    """Fake docker, fake Qdrant and a fake clock, all recorded for assertions."""

    def __init__(self, repo, env_file, state_dir):
        self.repo = repo
        self.env_file = env_file
        self.state_dir = state_dir
        self.qdrant_up = False
        self.docker_on_path = True
        self.compose_version_rc = 0
        self.up_rc = 0
        self.up_starts_qdrant = True
        self.answers_from_probe = None
        self.calls = []
        self.probes = 0
        self.sleeps = 0
        self.now = 0.0

    def docker_calls(self):
        return [c for c in self.calls if c[:2] == ["docker", "compose"] and c[-2:] == ["up", "-d"]]


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docker").mkdir(parents=True)
    (repo / ".env.example").write_text(TEMPLATE)
    (repo / "docker" / "docker-compose.yml").write_text("services: {}\n")
    env_file = tmp_path / "work" / ".env"
    env_file.parent.mkdir()
    state_dir = tmp_path / "state"
    w = World(repo, env_file, state_dir)

    with mock.patch.dict(os.environ):
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["LIBRARY_QUERY_ENV_FILE"] = str(env_file)
        monkeypatch.setattr(config, "repo_root", lambda: repo)
        for name in ("LIBRARY_ROOT", "QDRANT_URL", "OPENROUTER_API_KEY"):
            monkeypatch.setattr(config, name, getattr(config, name))
        monkeypatch.setattr(config, "STATE_DIR", state_dir)
        monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
        monkeypatch.setattr(journal, "_TRIMMED", set())

        def collection_exists(*a, **k):
            w.probes += 1
            if w.answers_from_probe is not None and w.probes >= w.answers_from_probe:
                w.qdrant_up = True
            if not w.qdrant_up:
                raise ConnectionError("down")
            return False

        monkeypatch.setattr(store, "collection_exists", collection_exists)

        def fake_run(cmd, **kwargs):
            w.calls.append(list(cmd))
            if cmd[-1] == "version":
                return SimpleNamespace(returncode=w.compose_version_rc)
            if cmd[-2:] == ["up", "-d"]:
                if w.up_rc == 0 and w.up_starts_qdrant:
                    w.qdrant_up = True
                return SimpleNamespace(returncode=w.up_rc)
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(
            setup_cmd, "subprocess", SimpleNamespace(run=fake_run, SubprocessError=subprocess.SubprocessError)
        )
        monkeypatch.setattr(
            setup_cmd, "shutil", SimpleNamespace(which=lambda name: "/fake/docker" if w.docker_on_path else None)
        )

        def sleep(seconds):
            w.sleeps += 1
            w.now += seconds

        monkeypatch.setattr(setup_cmd, "time", SimpleNamespace(monotonic=lambda: w.now, sleep=sleep))
        yield w


def run(argv, capsys):
    code = cli.main(["setup", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def journal_events(world):
    path = world.state_dir / journal.JOURNAL_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_creates_env_file_from_template_with_mode_0600(world, capsys):
    world.qdrant_up = True
    code, out, _ = run([], capsys)
    assert world.env_file.read_text() == TEMPLATE
    assert stat.S_IMODE(world.env_file.stat().st_mode) == 0o600
    assert "env_file: created" in out
    assert code == 1  # placeholder library, no collection yet


def test_second_run_leaves_modified_file_byte_identical(world, capsys, tmp_path):
    world.qdrant_up = True
    run([], capsys)
    custom = b"# mine\r\nLIBRARY_ROOT=/real/place\r\nOPENROUTER_API_KEY=abc\r\nZED=1\r\nAAA=2"
    world.env_file.write_bytes(custom)
    os.chmod(world.env_file, 0o640)
    library = tmp_path / "lib"
    library.mkdir()
    run(["--library", str(library)], capsys)
    assert world.env_file.read_bytes() == custom
    assert stat.S_IMODE(world.env_file.stat().st_mode) == 0o640


def test_library_replaces_placeholder_atomically_with_mode_0600_and_keeps_order(world, capsys, tmp_path):
    world.qdrant_up = True
    library = tmp_path / "docs" / ".." / "docs"
    (tmp_path / "docs").mkdir()
    world.env_file.write_text(TEMPLATE)
    os.chmod(world.env_file, 0o640)
    code, out, _ = run(["--library", str(library)], capsys)
    expected = TEMPLATE.replace("/absolute/path/to/library", str((tmp_path / "docs").resolve()))
    assert world.env_file.read_text() == expected
    assert stat.S_IMODE(world.env_file.stat().st_mode) == 0o600
    assert not [p for p in world.env_file.parent.iterdir() if p.name != ".env"]
    assert "[ok] library_root" in out


def test_library_replaces_empty_value(world, capsys, tmp_path):
    world.qdrant_up = True
    (tmp_path / "docs").mkdir()
    world.env_file.write_text("LIBRARY_ROOT=\nOTHER=1\n")
    run(["--library", str(tmp_path / "docs")], capsys)
    assert world.env_file.read_text() == f"LIBRARY_ROOT={(tmp_path / 'docs').resolve()}\nOTHER=1\n"


def test_library_never_replaces_a_real_value(world, capsys, tmp_path):
    world.qdrant_up = True
    (tmp_path / "docs").mkdir()
    original = "LIBRARY_ROOT=/srv/real-library\n"
    world.env_file.write_text(original)
    run(["--library", str(tmp_path / "docs")], capsys)
    assert world.env_file.read_text() == original
    assert journal_events(world)[-1]["library_set"] is False


def test_library_missing_directory_exits_2_and_changes_nothing(world, capsys, tmp_path):
    code, out, err = run(["--library", str(tmp_path / "nope")], capsys)
    assert code == 2
    assert len(err.strip().splitlines()) == 1
    assert out == ""
    assert not world.env_file.exists()
    assert world.calls == []
    assert not world.state_dir.exists()


def test_library_that_is_a_file_exits_2(world, capsys, tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x")
    code, _, _ = run(["--library", str(target)], capsys)
    assert code == 2 and not world.env_file.exists()


def test_check_creates_nothing_and_runs_no_subprocess(world, capsys, tmp_path):
    (tmp_path / "docs").mkdir()
    code, out, _ = run(["--check", "--library", str(tmp_path / "docs"), "--ask-key"], capsys)
    assert code == 1
    assert not world.env_file.exists()
    assert world.calls == []
    assert "[fail] qdrant" in out
    assert journal_events(world)[-1]["check_only"] is True


def test_check_does_not_edit_an_existing_file(world, capsys, tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    world.env_file.write_text(TEMPLATE)
    monkeypatch.setattr(setup_cmd.getpass, "getpass", lambda *a: pytest.fail("prompted under --check"))
    monkeypatch.setattr(setup_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    run(["--check", "--library", str(tmp_path / "docs"), "--ask-key"], capsys)
    assert world.env_file.read_text() == TEMPLATE
    assert world.calls == []


def test_qdrant_reachable_means_docker_is_never_called(world, capsys):
    world.qdrant_up = True
    _, out, _ = run([], capsys)
    assert world.calls == []
    assert "qdrant: already running" in out


def test_qdrant_down_with_docker_starts_compose_once_and_polls(world, capsys):
    world.up_starts_qdrant = False
    world.answers_from_probe = 4  # probe 1 is the pre-check; the container needs two more polls
    code, out, _ = run([], capsys)
    ups = world.docker_calls()
    assert len(ups) == 1
    assert ups[0][2] == "-f" and ups[0][3] == str(world.repo / "docker" / "docker-compose.yml")
    assert world.sleeps == 2
    assert "qdrant: started" in out
    assert journal_events(world)[-1]["qdrant_started"] is True


def test_qdrant_never_answering_gives_up_after_the_wait(world, capsys):
    world.up_starts_qdrant = False
    code, out, _ = run([], capsys)
    assert len(world.docker_calls()) == 1
    assert world.now >= setup_cmd.QDRANT_WAIT_SECONDS
    assert "qdrant: not running" in out
    assert journal_events(world)[-1]["qdrant_started"] is False


def test_qdrant_down_and_docker_missing_exits_with_hint(world, capsys):
    world.docker_on_path = False
    code, out, err = run([], capsys)
    assert code == 1
    assert world.calls == []
    assert "qdrant: not running (" in out and "docker" in out
    assert "Traceback" not in out + err
    assert "run `library-query index`" in out


def test_docker_without_compose_plugin_is_reported_not_run(world, capsys):
    world.compose_version_rc = 1
    code, out, _ = run([], capsys)
    assert world.docker_calls() == []
    assert "qdrant: not running" in out


def test_docker_up_failure_is_reported(world, capsys):
    world.up_rc = 1
    code, out, err = run([], capsys)
    assert code == 1
    assert "qdrant: not running (docker compose up failed" in out
    assert "Traceback" not in out + err


def test_no_docker_never_calls_docker(world, capsys):
    code, out, _ = run(["--no-docker"], capsys)
    assert world.calls == []
    assert "qdrant: not running" in out


def test_ask_key_writes_key_with_tty_and_empty_value_without_leaking(world, capsys, monkeypatch):
    world.qdrant_up = True
    monkeypatch.setattr(setup_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(setup_cmd.getpass, "getpass", lambda *a, **k: CANARY)
    code, out, err = run(["--ask-key", "--json"], capsys)
    assert f"OPENROUTER_API_KEY={CANARY}\n" in world.env_file.read_text()
    assert stat.S_IMODE(world.env_file.stat().st_mode) == 0o600
    for stream in (out, err, (world.state_dir / journal.JOURNAL_NAME).read_text()):
        assert CANARY not in stream


def test_ask_key_text_mode_does_not_leak(world, capsys, monkeypatch):
    world.qdrant_up = True
    monkeypatch.setattr(setup_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(setup_cmd.getpass, "getpass", lambda *a, **k: CANARY)
    _, out, err = run(["--ask-key"], capsys)
    assert "api_key: saved" in out
    assert CANARY not in out + err + (world.state_dir / journal.JOURNAL_NAME).read_text()
    assert "set OPENROUTER_API_KEY" not in out


def test_ask_key_skipped_without_tty(world, capsys, monkeypatch):
    world.qdrant_up = True
    monkeypatch.setattr(setup_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(setup_cmd.getpass, "getpass", lambda *a, **k: pytest.fail("prompted without a TTY"))
    _, out, _ = run(["--ask-key"], capsys)
    assert "OPENROUTER_API_KEY=\n" in world.env_file.read_text()
    assert "set OPENROUTER_API_KEY" in out


def test_ask_key_skipped_when_value_already_set(world, capsys, monkeypatch):
    world.qdrant_up = True
    world.env_file.write_text("OPENROUTER_API_KEY=existing-value\n")
    monkeypatch.setattr(setup_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(setup_cmd.getpass, "getpass", lambda *a, **k: pytest.fail("prompted with a key set"))
    run(["--ask-key"], capsys)
    assert world.env_file.read_text() == "OPENROUTER_API_KEY=existing-value\n"


def test_never_prompts_without_the_flag(world, capsys, monkeypatch):
    world.qdrant_up = True
    monkeypatch.setattr(setup_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(setup_cmd.getpass, "getpass", lambda *a, **k: pytest.fail("prompted without --ask-key"))
    run([], capsys)


def test_journal_event_has_booleans_only(world, capsys, tmp_path):
    (tmp_path / "docs").mkdir()
    world.docker_on_path = False
    run(["--library", str(tmp_path / "docs")], capsys)
    events = [e for e in journal_events(world) if e["kind"] == "setup_run"]
    assert len(events) == 1
    event = {k: v for k, v in events[0].items() if k not in ("ts", "kind")}
    assert event == {"env_created": True, "library_set": True, "qdrant_started": False, "check_only": False}
    assert all(isinstance(v, bool) for v in event.values())


def test_json_output_and_next_steps(world, capsys):
    world.qdrant_up = True
    code, out, _ = run(["--json"], capsys)
    payload = json.loads(out)
    assert payload["exit_code"] == code == 1
    assert payload["setup"]["env_created"] is True
    assert any("OPENROUTER_API_KEY" in step for step in payload["next"])
    assert "run `library-query index`" in payload["next"]


def test_all_passing_exits_0_without_next_block(world, capsys, tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    world.qdrant_up = True
    monkeypatch.setattr(store, "collection_exists", lambda *a, **k: True)
    world.env_file.write_text(TEMPLATE.replace("OPENROUTER_API_KEY=", "OPENROUTER_API_KEY=present"))
    code, out, _ = run(["--library", str(tmp_path / "docs")], capsys)
    assert code == 0
    assert "Next:" not in out
    assert all(line.startswith("[ok]") for line in out.splitlines() if line.startswith("["))


def test_ask_key_makes_a_group_readable_env_file_private(world, capsys, monkeypatch):
    world.qdrant_up = True
    world.env_file.write_text(TEMPLATE)
    os.chmod(world.env_file, 0o644)
    monkeypatch.setattr(setup_cmd.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(setup_cmd.getpass, "getpass", lambda *a, **k: CANARY)
    _, out, err = run(["--ask-key"], capsys)
    assert f"OPENROUTER_API_KEY={CANARY}\n" in world.env_file.read_text()
    assert stat.S_IMODE(world.env_file.stat().st_mode) == 0o600
    assert CANARY not in out + err


def test_library_write_makes_a_world_readable_env_file_private(world, capsys, tmp_path):
    world.qdrant_up = True
    (tmp_path / "docs").mkdir()
    world.env_file.write_text(TEMPLATE)
    os.chmod(world.env_file, 0o644)
    run(["--library", str(tmp_path / "docs")], capsys)
    assert stat.S_IMODE(world.env_file.stat().st_mode) == 0o600


def test_symlinked_env_file_is_replaced_at_its_target_with_mode_0600(world, capsys, tmp_path):
    world.qdrant_up = True
    (tmp_path / "docs").mkdir()
    target = tmp_path / "real.env"
    target.write_text(TEMPLATE)
    os.chmod(target, 0o644)
    world.env_file.symlink_to(target)
    run(["--library", str(tmp_path / "docs")], capsys)
    assert world.env_file.is_symlink()
    assert str((tmp_path / "docs").resolve()) in target.read_text()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.parametrize("separator", ["\u2028", "\x0b"])
def test_library_with_a_line_separator_in_its_name_is_refused(world, capsys, tmp_path, separator):
    name = f"lib{separator}EMBEDDING_BASE_URL=evil.example"
    bad = tmp_path / name
    try:
        bad.mkdir()
    except OSError as exc:
        pytest.skip(f"filesystem refuses a directory name with {separator!r}: {exc}")
    world.env_file.write_text(TEMPLATE)
    code, out, err = run(["--library", str(bad)], capsys)
    assert code == 2
    assert len(err.strip().splitlines()) == 1
    assert "evil.example" not in out + err and str(tmp_path) not in out + err
    assert world.env_file.read_text() == TEMPLATE
    assert world.calls == []


def test_library_with_a_normal_name_still_works_next_to_a_refused_one(world, capsys, tmp_path):
    world.qdrant_up = True
    good = tmp_path / "good library"
    good.mkdir()
    world.env_file.write_text(TEMPLATE)
    run(["--library", str(good)], capsys)
    assert f"LIBRARY_ROOT={good.resolve()}\n" in world.env_file.read_text()


@pytest.mark.parametrize("separator", ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029", "\x7f", "\t"])
def test_line_or_control_character_check_covers_every_splitlines_boundary(separator):
    assert setup_cmd._is_line_or_control_char(separator)
    assert not setup_cmd._is_line_or_control_char("a")
    assert not setup_cmd._is_line_or_control_char(" ")
    assert not setup_cmd._is_line_or_control_char("\u00e9")


def test_relative_library_resolves_against_the_caller_folder(world, capsys, tmp_path, monkeypatch):
    world.qdrant_up = True
    caller = tmp_path / "caller"
    (caller / "sub").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "sub").mkdir(parents=True)
    world.env_file.write_text(TEMPLATE)
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("LIBRARY_QUERY_CALLER_CWD", str(caller))
    run(["--library", "sub"], capsys)
    assert f"LIBRARY_ROOT={(caller / 'sub').resolve()}\n" in world.env_file.read_text()


def test_relative_library_uses_the_current_folder_without_the_caller_variable(world, capsys, tmp_path, monkeypatch):
    world.qdrant_up = True
    (tmp_path / "here" / "sub").mkdir(parents=True)
    world.env_file.write_text(TEMPLATE)
    monkeypatch.chdir(tmp_path / "here")
    monkeypatch.delenv("LIBRARY_QUERY_CALLER_CWD", raising=False)
    run(["--library", "sub"], capsys)
    assert f"LIBRARY_ROOT={(tmp_path / 'here' / 'sub').resolve()}\n" in world.env_file.read_text()
