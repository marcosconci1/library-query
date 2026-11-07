"""Local health checks for library-query.

Each check returns `(name, ok, detail)`. Details never contain paths, URLs, or secret values,
so the results can be copied into a bug report as they are. Nothing here writes to user data,
and nothing raises: a check that errors becomes `ok=False` with the exception class name.
"""
from __future__ import annotations

import os
import platform
import time
from importlib import metadata
from collections.abc import Callable

from . import config, journal, store

REQUIRED_ENV = ("LIBRARY_ROOT", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_DIM")
#: Total wall-clock budget for the single Qdrant probe, in seconds.
QDRANT_DEADLINE_SECONDS = 3

Check = tuple[str, bool, str]


class _ProbeDeadline:
    """Minimal stand-in for the service deadline that `store` honours."""

    def __init__(self, budget: float):
        self._budget = budget
        self._start = time.monotonic()

    def remaining(self) -> float:
        return self._budget - (time.monotonic() - self._start)

    def service_timeout(self) -> int:
        return max(1, int(self._budget))

    def check(self) -> None:
        if self.remaining() <= 0:
            raise TimeoutError("probe deadline exhausted")


def _check_env() -> Check:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        return "env", False, "not set: " + ", ".join(missing)
    return "env", True, "all required variables are set"


def _check_library_root() -> Check:
    root = config.LIBRARY_ROOT
    if not root.exists():
        return "library_root", False, "LIBRARY_ROOT does not exist"
    if not root.is_dir():
        return "library_root", False, "LIBRARY_ROOT is not a directory"
    return "library_root", True, "LIBRARY_ROOT is a directory"


def _check_qdrant() -> list[Check]:
    """One probe answers both questions: reachable, and does the collection exist."""
    token = store.SERVICE_DEADLINE.set(_ProbeDeadline(QDRANT_DEADLINE_SECONDS))
    try:
        exists = store.collection_exists()
    except Exception as exc:  # noqa: BLE001 — a failing probe is a result, not a crash
        reason = type(exc).__name__
        return [
            ("qdrant", False, f"not reachable ({reason})"),
            ("collection", False, "not checked, Qdrant not reachable"),
        ]
    finally:
        store.SERVICE_DEADLINE.reset(token)
    return [
        ("qdrant", True, "reachable"),
        ("collection", exists, "collection exists" if exists else "collection missing, run index"),
    ]


def _check_ledger() -> Check:
    path = config.LEDGER_PATH
    if not path.exists():
        return "ledger", True, "no ledger yet (nothing indexed)"
    try:
        with open(path, "rb") as handle:
            handle.read(1)
    except OSError as exc:
        return "ledger", False, f"ledger not readable ({type(exc).__name__})"
    return "ledger", True, "ledger is readable"


def _guarded(name: str, check: Callable[[], Check]) -> Check:
    try:
        return check()
    except Exception as exc:  # noqa: BLE001
        return name, False, f"check failed ({type(exc).__name__})"


def run_checks() -> list[Check]:
    """Run every check, append a `doctor_snapshot` journal event, and never raise."""
    results: list[Check] = [
        _guarded("env", _check_env),
        _guarded("library_root", _check_library_root),
    ]
    try:
        results.extend(_check_qdrant())
    except Exception as exc:  # noqa: BLE001
        results.append(("qdrant", False, f"check failed ({type(exc).__name__})"))
    results.append(_guarded("ledger", _check_ledger))
    failed = [name for name, ok, _ in results if not ok]
    journal.append_event(
        "doctor_snapshot",
        outcome="ok" if not failed else "fail",
        checks={name: ok for name, ok, _ in results},
        failed=failed,
    )
    return results


def exit_code(results: list[Check]) -> int:
    """0 when every check passed, 1 when any failed (the doctor command's own contract)."""
    return 0 if all(ok for _, ok, _ in results) else 1


def to_json(results: list[Check]) -> dict:
    """Machine-readable form; mirrors the agent contract's `exit_code` field."""
    code = exit_code(results)
    return {
        "ok": code == 0,
        "exit_code": code,
        "checks": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in results],
    }


def tool_version() -> str:
    try:
        return metadata.version("library-query")
    except metadata.PackageNotFoundError:
        return "unknown"


def build_bundle(results: list[Check]) -> str:
    """Plain-text redacted report for bug reports.

    Deliberately contains no query text, no absolute paths, no environment values, and no
    command lines: only the version, interpreter and platform names, check outcomes, and the
    already-scrubbed journal tail.
    """
    lines = [
        "library-query diagnose bundle",
        f"tool version: {journal.scrub_text(tool_version(), 40)}",
        f"python: {platform.python_version()}",
        f"platform: {journal.scrub_text(platform.platform(), 120)}",
        "",
        "doctor checks:",
    ]
    for name, ok, detail in results:
        lines.append(f"  [{'ok' if ok else 'fail'}] {journal.scrub_text(name, 40)}: {journal.scrub_text(detail, 200)}")
    lines += ["", "journal tail (last 50):"]
    lines += [f"  {entry}" for entry in journal.tail(50)]
    report = "\n".join(lines) + "\n"
    journal.append_event("diagnose_built", checks=len(results), failed=sum(1 for _, ok, _ in results if not ok))
    return report
