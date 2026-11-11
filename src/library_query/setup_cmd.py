"""`library-query setup`: prepare the env file, Qdrant and a first health check.

Safe to repeat. An existing env file is never overwritten or reordered; the only edits are
the LIBRARY_ROOT line (when empty or still the template placeholder) and an empty
OPENROUTER_API_KEY line. The key is read with getpass and goes to the env file only: it is
never printed, journaled or put in an error message. `--check` changes nothing.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import config, doctor, journal

PLACEHOLDER_LIBRARY = "/absolute/path/to/library"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
#: How long to wait for Qdrant after `docker compose up -d`, and how often to probe.
QDRANT_WAIT_SECONDS = 60
QDRANT_POLL_INTERVAL = 2
DOCKER_PROBE_TIMEOUT = 30
DOCKER_UP_TIMEOUT = 300


def _is_line_or_control_char(ch: str) -> bool:
    """True for a character that `str.splitlines` could treat as a line boundary, or a control one."""
    return ord(ch) < 32 or ch == "\x7f" or len(ch.splitlines()) > 1 or ch in "\x85  "


def _lookup(lines: list[str], key: str) -> tuple[int | None, list[str]]:
    """Index of the first line that sets `key`, and every value set for it.

    Parsing mirrors `config.load_env`: stripped lines, `#` comments skipped, split on the first `=`.
    """
    first: int | None = None
    values: list[str] = []
    for index, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, _, value = text.partition("=")
        if name.strip() == key:
            if first is None:
                first = index
            values.append(value.strip())
    return first, values


def _replace_line(lines: list[str], index: int, key: str, value: str) -> None:
    carriage = "\r" if lines[index].endswith("\r") else ""
    lines[index] = f"{key}={value}{carriage}"


def _write_atomic(path: Path, data: bytes) -> None:
    """Replace `path` through a temp file in the same directory, with mode 0600."""
    real = Path(os.path.realpath(path))
    mode = 0o600
    fd, tmp_name = tempfile.mkstemp(dir=real.parent, prefix=f".{real.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, real)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _create_env_file(path: Path, template: Path) -> bool:
    """Copy the template to a new 0600 file. False when the file already exists."""
    data = template.read_bytes()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return True


def _read_key() -> str | None:
    """Ask for the API key without echo. None when nothing usable was entered."""
    try:
        value = getpass.getpass("OPENROUTER_API_KEY (input hidden, empty to skip): ").strip()
    except EOFError:
        return None
    if not value or any(ch.isspace() or ord(ch) < 32 for ch in value):
        return None
    return value


def _sync_process_env(path: Path, overrides: dict[str, str]) -> None:
    """Make this process see the env file, as `config.load_env` would at import time."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        text = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if not os.environ.get(name.strip()):
            os.environ[name.strip()] = value.strip()
    os.environ.update(overrides)
    if os.environ.get("LIBRARY_ROOT"):
        config.LIBRARY_ROOT = Path(os.environ["LIBRARY_ROOT"])
    config.QDRANT_URL = os.environ.get("QDRANT_URL") or DEFAULT_QDRANT_URL
    config.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def _docker_ready() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        probe = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, timeout=DOCKER_PROBE_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _start_qdrant() -> tuple[bool, str]:
    """Start Qdrant with docker compose and wait for it. Returns (started, reason when not)."""
    compose = config.repo_root() / "docker" / "docker-compose.yml"
    if not compose.is_file():
        return False, "docker/docker-compose.yml not found"
    if not _docker_ready():
        return False, "docker with the compose plugin not found; start Qdrant yourself or install Docker"
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose), "up", "-d"],
            capture_output=True,
            timeout=DOCKER_UP_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "docker compose could not be run"
    if result.returncode != 0:
        return False, "docker compose up failed; check that Docker is running"
    deadline = time.monotonic() + QDRANT_WAIT_SECONDS
    while True:
        if doctor.qdrant_reachable():
            return True, ""
        if time.monotonic() >= deadline:
            return False, f"container started but Qdrant did not answer within {QDRANT_WAIT_SECONDS}s"
        time.sleep(QDRANT_POLL_INTERVAL)


def _next_steps(results: list[doctor.Check], env_present: bool) -> list[str]:
    status = {name: ok for name, ok, _ in results}
    steps: list[str] = []
    if not env_present:
        steps.append("create the env file: copy .env.example and fill it in")
    if env_present and not os.environ.get("OPENROUTER_API_KEY"):
        steps.append("set OPENROUTER_API_KEY in the env file (or rerun setup with --ask-key)")
    if not status.get("library_root", True):
        steps.append("set LIBRARY_ROOT to your document folder (or rerun setup with --library PATH)")
    if not status.get("qdrant", True):
        steps.append("start Qdrant (rerun setup, or docker compose -f docker/docker-compose.yml up -d)")
    if not status.get("collection", True):
        steps.append("run `library-query index`")
    return steps


def run_setup(args: argparse.Namespace) -> int:
    check_only = bool(args.check)
    as_json = bool(getattr(args, "json", False))

    def say(line: str) -> None:
        if not as_json:
            print(line)

    library: str | None = None
    if args.library:
        candidate = Path(args.library).expanduser()
        if not candidate.is_absolute():
            # The launcher runs the engine from the repository, so the caller's folder
            # arrives in LIBRARY_QUERY_CALLER_CWD.
            candidate = Path(os.environ.get("LIBRARY_QUERY_CALLER_CWD") or os.getcwd()) / candidate
        if not candidate.is_dir():
            print("setup: --library must be an existing directory", file=sys.stderr)
            return 2
        library = str(candidate.resolve())
        if any(_is_line_or_control_char(ch) for ch in library):
            print("setup: --library path must not contain line breaks or control characters", file=sys.stderr)
            return 2

    env_path = config.env_file_path()
    template = config.repo_root() / ".env.example"
    env_created = library_set = qdrant_started = False
    overrides: dict[str, str] = {}

    if not check_only:
        if not env_path.exists():
            try:
                env_created = _create_env_file(env_path, template)
            except OSError as exc:
                say(f"env_file: could not create ({type(exc).__name__})")
        if env_created:
            say("env_file: created from .env.example")
        elif env_path.exists():
            say("env_file: exists, left as it is")

        if env_path.exists():
            try:
                lines = env_path.read_bytes().decode("utf-8").split("\n")
                changed = False
                if library is not None:
                    index, values = _lookup(lines, "LIBRARY_ROOT")
                    if index is not None and all(v in ("", PLACEHOLDER_LIBRARY) for v in values):
                        _replace_line(lines, index, "LIBRARY_ROOT", library)
                        overrides["LIBRARY_ROOT"] = library
                        changed = library_set = True
                    else:
                        say("library_root: not changed, the env file already sets it")
                if args.ask_key and sys.stdin.isatty():
                    index, values = _lookup(lines, "OPENROUTER_API_KEY")
                    if index is not None and not any(values):
                        secret = _read_key()
                        if secret:
                            _replace_line(lines, index, "OPENROUTER_API_KEY", secret)
                            overrides["OPENROUTER_API_KEY"] = secret
                            changed = True
                            say("api_key: saved to the env file")
                        else:
                            say("api_key: nothing saved")
                if changed:
                    _write_atomic(env_path, "\n".join(lines).encode("utf-8"))
                if library_set:
                    say("library_root: set")
            except (OSError, UnicodeDecodeError) as exc:
                overrides.clear()
                library_set = False
                say(f"env_file: could not update ({type(exc).__name__})")
    else:
        say("env_file: exists" if env_path.exists() else "env_file: missing (--check changes nothing)")

    _sync_process_env(env_path, overrides)

    if not check_only:
        if doctor.qdrant_reachable():
            say("qdrant: already running")
        elif args.no_docker:
            say("qdrant: not running (--no-docker; start Qdrant at the configured address)")
        else:
            qdrant_started, reason = _start_qdrant()
            say("qdrant: started" if qdrant_started else f"qdrant: not running ({reason})")

    results = doctor.run_checks()
    code = doctor.exit_code(results)
    steps = _next_steps(results, env_path.exists())
    journal.append_event(
        "setup_run",
        env_created=env_created,
        library_set=library_set,
        qdrant_started=qdrant_started,
        check_only=check_only,
    )

    if as_json:
        payload = doctor.to_json(results)
        payload["setup"] = {
            "env_created": env_created,
            "library_set": library_set,
            "qdrant_started": qdrant_started,
            "check_only": check_only,
        }
        payload["next"] = steps
        print(json.dumps(payload, ensure_ascii=False))
        return code
    for name, ok, detail in results:
        print(f"[{'ok' if ok else 'fail'}] {name}: {detail}")
    if steps:
        print("Next:")
        for step in steps:
            print(f"  - {step}")
    return code
