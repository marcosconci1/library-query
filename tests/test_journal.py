"""Local diagnostic journal: scrubbing, retention, never-raises, and the diagnose bundle."""
import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from library_query import config, doctor, journal

FAKE_KEY = "sk-or-v1-FAKEKEYFORLEAKSCAN0123456789"
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.fixture
def state(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    monkeypatch.setattr(journal, "_TRIMMED", set())
    return state_dir


def _entries(state_dir):
    path = state_dir / journal.JOURNAL_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def _line(ts, **fields):
    return json.dumps({"ts": ts.isoformat(), "kind": "query_finished", **fields})


def test_append_writes_valid_jsonl_and_creates_state_dir(state):
    assert not state.exists()
    journal.append_event("index_started", files_seen=3)
    journal.append_event("index_finished", outcome="ok", new=1)
    entries = _entries(state)
    assert [e["kind"] for e in entries] == ["index_started", "index_finished"]
    assert entries[0]["files_seen"] == 3
    stamp = datetime.fromisoformat(entries[0]["ts"])
    assert stamp.utcoffset() == timedelta(0)


def test_scrubs_paths_hex_ids_and_secrets(state):
    hex_id = "0123456789abcdef0123456789abcdef"
    journal.append_event(
        "query_finished",
        error=f"failed reading /home/someone/library/book.pdf for {hex_id} with {FAKE_KEY}",
        nested={"where": ["/etc/passwd"]},
    )
    (entry,) = _entries(state)
    assert "/home/" not in json.dumps(entry)
    assert "[path]" in entry["error"]
    assert "[id]" in entry["error"] and hex_id not in entry["error"]
    assert FAKE_KEY not in json.dumps(entry)
    assert entry["nested"]["where"] == ["[path]"]


def test_caps_text_length(state):
    journal.append_event("query_finished", error="x" * 5000)
    (entry,) = _entries(state)
    assert len(entry["error"]) == journal.TEXT_LIMIT


def test_scrub_text_leaves_plain_words_and_short_hex(state):
    assert journal.scrub_text("embedding api error 429") == "embedding api error 429"
    assert journal.scrub_text("abc123") == "abc123"


def test_trim_by_age():
    old = _line(NOW - timedelta(days=31))
    edge = _line(NOW - timedelta(days=29))
    fresh = _line(NOW)
    assert journal.trim_lines([old, edge, fresh], now=NOW) == [edge, fresh]


def test_trim_by_size_drops_oldest_first():
    lines = [_line(NOW - timedelta(minutes=10 - i), pad="p" * 100) for i in range(10)]
    per_line = len(lines[0].encode()) + 1
    kept = journal.trim_lines(lines, now=NOW, max_bytes=per_line * 4)
    assert kept == lines[-4:]
    assert journal.trim_lines(lines, now=NOW, max_bytes=per_line * 10) == lines


def test_trim_drops_unparseable_lines_and_handles_naive_timestamps():
    naive = json.dumps({"ts": "2026-05-31T00:00:00", "kind": "x"})
    assert journal.trim_lines(["not json", json.dumps({"kind": "x"}), naive], now=NOW) == [naive]


def test_first_open_in_process_trims_file_once(state):
    state.mkdir()
    path = state / journal.JOURNAL_NAME
    stale = _line(datetime.now(timezone.utc) - timedelta(days=60))
    path.write_text(stale + "\n")
    journal.append_event("index_started", files_seen=1)
    assert stale not in path.read_text()
    # Later appends in the same process do not re-read or rewrite the file.
    marker = os.stat(path).st_ino
    journal.append_event("index_started", files_seen=2)
    assert os.stat(path).st_ino == marker
    assert len(_entries(state)) == 2


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_append_never_raises_on_read_only_dir(tmp_path, monkeypatch):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setattr(config, "STATE_DIR", locked / "state")
    monkeypatch.setattr(journal, "_TRIMMED", set())
    try:
        journal.append_event("index_started", files_seen=1)  # must stay silent
    finally:
        locked.chmod(stat.S_IRWXU)
    assert not (locked / "state").exists()


def test_append_never_raises_when_state_dir_is_a_file(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setattr(config, "STATE_DIR", blocker / "state")
    monkeypatch.setattr(journal, "_TRIMMED", set())
    journal.append_event("index_started", files_seen=1)


def test_buckets_and_bands():
    assert journal.duration_bucket(0.1) == "<0.5s"
    assert journal.duration_bucket(1) == "0.5-2s"
    assert journal.duration_bucket(5) == "2-10s"
    assert journal.duration_bucket(60) == ">=10s"
    assert [journal.score_band(s) for s in (0.9, 0.5, 0.1, None)] == ["high", "medium", "low", "none"]


def test_diagnose_bundle_leaks_nothing(state, monkeypatch, tmp_path):
    secret_question = "what does my private diary say about Zebra"
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", FAKE_KEY)
    monkeypatch.setattr(config, "LIBRARY_ROOT", tmp_path / "library")
    journal.append_event("index_finished", outcome="error", error=f"boom at {tmp_path}/library with {FAKE_KEY}")
    journal.append_event("query_finished", outcome="error", error=f"/home/me/x {secret_question}"[:20])
    bundle = doctor.build_bundle([
        ("env", False, "not set: LIBRARY_ROOT"),
        ("library_root", False, f"missing {tmp_path}/library"),
    ])
    assert "OPENROUTER_API_KEY value" not in bundle
    assert FAKE_KEY not in bundle
    assert "/home/" not in bundle
    assert str(tmp_path) not in bundle
    assert secret_question not in bundle
    assert "library-query diagnose bundle" in bundle
    assert "python:" in bundle and "platform:" in bundle and "tool version:" in bundle
    assert "journal tail" in bundle
