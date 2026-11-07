"""Local diagnostic journal.

Everything here stays on this machine: no telemetry and no network calls. The journal is an
append-only JSON Lines file under the state directory with fixed retention (30 days, 1 MB).
Every text field is scrubbed before it is written, and writing never raises: diagnostics must
not break the tool.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import config

JOURNAL_NAME = "events.log.jsonl"
RETENTION_DAYS = 30
MAX_JOURNAL_BYTES = 1024 * 1024
TEXT_LIMIT = 1000

EVENT_KINDS = (
    "index_started",
    "index_finished",
    "query_finished",
    "doctor_snapshot",
    "diagnose_built",
)

_HEX_ID = re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{32}(?![0-9A-Fa-f])")
_ABSOLUTE_PATH = re.compile(r"(?<![\w.:])/(?:[\w.@+~-]+/)*[\w.@+~-]*")
# Common credential shapes. The journal never receives key values on purpose; this is a backstop.
_SECRET = re.compile(
    r"(?i)\b(?:sk|pk|key|token)[-_][A-Za-z0-9_-]{8,}"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"
    r"|\b(?:api[_-]?key|secret|password|token)\s*[=:]\s*\S+"
)

_LOCK = threading.Lock()
_TRIMMED: set[str] = set()


def scrub_text(value: Any, limit: int = TEXT_LIMIT) -> str:
    """Mask secrets, absolute paths and 32-hex ids, then cap the length."""
    text = value if isinstance(value, str) else str(value)
    text = _SECRET.sub("[redacted]", text)
    text = _ABSOLUTE_PATH.sub("[path]", text)
    text = _HEX_ID.sub("[id]", text)
    return text[:limit]


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {scrub_text(key, 80): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub_text(value)


def journal_path() -> Path:
    return Path(config.STATE_DIR) / JOURNAL_NAME


def _entry_time(line: str) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(json.loads(line)["ts"])
    except (ValueError, TypeError, KeyError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def trim_lines(
    lines: list[str],
    now: datetime | None = None,
    max_age_days: int = RETENTION_DAYS,
    max_bytes: int = MAX_JOURNAL_BYTES,
) -> list[str]:
    """Pure retention: keep entries newer than max_age_days, then drop oldest until within max_bytes.

    Lines without a parseable "ts" are dropped.
    """
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=max_age_days)
    kept = [line for line in lines if (stamp := _entry_time(line)) is not None and stamp >= cutoff]
    total = sum(len(line.encode("utf-8")) + 1 for line in kept)
    drop = 0
    while drop < len(kept) and total > max_bytes:
        total -= len(kept[drop].encode("utf-8")) + 1
        drop += 1
    return kept[drop:]


def _read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line for line in text.splitlines() if line.strip()]


def _trim_file(path: Path, now: datetime | None = None) -> None:
    try:
        lines = _read_lines(path)
    except FileNotFoundError:
        return
    kept = trim_lines(lines, now)
    if kept == lines:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("".join(line + "\n" for line in kept))
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def append_event(kind: str, **fields: Any) -> None:
    """Append one scrubbed JSON line. Never raises."""
    try:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **_sanitize(fields)}
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        path = journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            if str(path) not in _TRIMMED:
                _TRIMMED.add(str(path))
                _trim_file(path)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(line)
    except (OSError, ValueError, TypeError):
        return


def tail(count: int = 50) -> list[str]:
    """Last `count` journal lines, re-scrubbed. Never raises."""
    try:
        lines = _read_lines(journal_path())
    except OSError:
        return []
    return [scrub_text(line, 2000) for line in lines[-count:]]


def duration_bucket(seconds: float) -> str:
    """Coarse duration label so the journal never records exact timings."""
    if seconds < 0.5:
        return "<0.5s"
    if seconds < 2:
        return "0.5-2s"
    if seconds < 10:
        return "2-10s"
    return ">=10s"


def score_band(score: Any) -> str:
    """Coarse top-score label for a query's best hit."""
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return "none"
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"
