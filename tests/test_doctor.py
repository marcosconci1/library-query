"""Doctor checks and the doctor/diagnose commands, fully offline."""
import json

import pytest

from library_query import cli, config, doctor, errors, journal, store

ENV = {
    "LIBRARY_ROOT": "x",
    "EMBEDDING_BASE_URL": "http://localhost/v1",
    "EMBEDDING_MODEL": "m",
    "EMBEDDING_DIM": "8",
}


@pytest.fixture
def healthy(tmp_path, monkeypatch):
    library = tmp_path / "library"
    library.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(config, "LIBRARY_ROOT", library)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    monkeypatch.setattr(journal, "_TRIMMED", set())
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(store, "collection_exists", lambda *a, **k: True)
    return state_dir


def _by_name(results):
    return {name: (ok, detail) for name, ok, detail in results}


def test_ok_path(healthy, capsys):
    results = doctor.run_checks()
    assert all(ok for _, ok, _ in results)
    assert [name for name, _, _ in results] == ["env", "library_root", "qdrant", "collection", "ledger"]
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 5 and all(line.startswith("[ok] ") for line in out)


def test_ledger_present_and_readable(healthy):
    config.LEDGER_PATH.write_text("{}")
    assert _by_name(doctor.run_checks())["ledger"][0] is True


def test_missing_env_and_library_root_fail_with_exit_1(healthy, monkeypatch, capsys):
    monkeypatch.delenv("EMBEDDING_MODEL")
    monkeypatch.setattr(config, "LIBRARY_ROOT", healthy / "nope")
    checks = _by_name(doctor.run_checks())
    assert checks["env"] == (False, "not set: EMBEDDING_MODEL")
    assert checks["library_root"][0] is False
    assert cli.main(["doctor"]) == 1
    assert "[fail] env" in capsys.readouterr().out


def test_library_root_that_is_a_file_fails(healthy, monkeypatch):
    target = healthy / "file"
    target.write_text("")
    monkeypatch.setattr(config, "LIBRARY_ROOT", target)
    assert _by_name(doctor.run_checks())["library_root"][0] is False


def test_qdrant_unreachable_fails_both_checks_without_leaking(healthy, monkeypatch):
    def boom(*a, **k):
        raise errors.QdrantUnreachableError("Could not reach Qdrant at http://secret-host:6333")

    monkeypatch.setattr(store, "collection_exists", boom)
    checks = _by_name(doctor.run_checks())
    assert checks["qdrant"] == (False, "not reachable (QdrantUnreachableError)")
    assert checks["collection"][0] is False
    assert "secret-host" not in json.dumps(checks)


def test_missing_collection_fails(healthy, monkeypatch):
    monkeypatch.setattr(store, "collection_exists", lambda *a, **k: False)
    checks = _by_name(doctor.run_checks())
    assert checks["qdrant"][0] is True
    assert checks["collection"][0] is False


def test_unreadable_ledger_fails(healthy, monkeypatch):
    config.LEDGER_PATH.mkdir()  # exists but cannot be read as a file
    ok, detail = _by_name(doctor.run_checks())["ledger"]
    assert ok is False and "IsADirectoryError" in detail


def test_check_that_raises_becomes_a_failed_result(healthy, monkeypatch):
    def broken():
        raise RuntimeError("secret detail")

    monkeypatch.setattr(doctor, "_check_env", broken)
    ok, detail = _by_name(doctor.run_checks())["env"]
    assert ok is False and detail == "check failed (RuntimeError)"


def test_json_output_and_exit_codes(healthy, monkeypatch, capsys):
    assert cli.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["exit_code"] == 0
    assert {c["name"] for c in payload["checks"]} == {"env", "library_root", "qdrant", "collection", "ledger"}

    monkeypatch.delenv("LIBRARY_ROOT")
    assert cli.main(["doctor", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["exit_code"] == 1


def test_doctor_snapshot_is_journaled(healthy):
    doctor.run_checks()
    lines = (healthy / journal.JOURNAL_NAME).read_text().splitlines()
    entry = json.loads(lines[-1])
    assert entry["kind"] == "doctor_snapshot" and entry["outcome"] == "ok"
    assert entry["failed"] == []


def test_diagnose_prints_redacted_bundle(healthy, monkeypatch, capsys):
    key = "sk-or-v1-FAKEKEYFORLEAKSCAN0123456789"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", key)
    monkeypatch.delenv("EMBEDDING_MODEL")
    journal.append_event("index_finished", outcome="error", error=f"{healthy}/library {key}")
    assert cli.main(["diagnose"]) == 0
    bundle = capsys.readouterr().out
    for forbidden in ("OPENROUTER_API_KEY value", "/home/", key, str(healthy)):
        assert forbidden not in bundle
    assert "[fail] env" in bundle and "journal tail" in bundle
    assert '"kind":"index_finished"' in bundle


def test_index_and_query_failures_journal_class_name_only(healthy, monkeypatch):
    question = "private question text about Zebra"

    def boom(*a, **k):
        raise errors.EmbeddingApiError(f"upstream said: {question}")

    monkeypatch.setattr(store, "query_places", boom)
    assert cli.main(["query", question, "--cache-mode", "off"]) == errors.EXIT_SERVICE_ERROR
    text = (healthy / journal.JOURNAL_NAME).read_text()
    entry = json.loads(text.splitlines()[-1])
    assert entry["kind"] == "query_finished" and entry["outcome"] == "error"
    assert entry["error"] == "EmbeddingApiError"
    assert question not in text
