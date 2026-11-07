"""Retrieval cache: configuration, exact keys, budget, storage, maintenance.

Covers tasks.md sections 1.2/1.3 and 3.x/5.x: defaults and validation, key
sensitivity/equivalence, the shared admission budget, private SQLite storage
with TTL/LRU/eviction and safety checks, read-only status with the
recovery-required contract, and scoped clear. Everything runs offline against
temporary cache directories with controllable clocks.
"""
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import FakeClock, make_settings

from library_query import cache, cli, config, errors, revisions


def run_cli(argv):
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def read_error(capsys):
    lines = capsys.readouterr().err.strip().splitlines()
    assert lines, "expected a structured error line on stderr"
    return json.loads(lines[-1])


def store_embedding(storage, settings, probe, vector=None, ttl=None, budget=None):
    vector = vector if vector is not None else [0.1] * config.EMBEDDING_DIM
    budget = budget or cache.AdmissionBudget()
    return storage.put_payload(
        "embedding", cache.embedding_key(settings, probe), revisions.embedding_fingerprint(),
        {"v": 1, "vector": vector}, ttl or settings.embedding_ttl, budget,
    )


# --- 1.2 configuration ----------------------------------------------------------------


def test_defaults_and_mode_precedence(cache_env, monkeypatch):
    settings = cache.resolve_settings(None)
    assert settings.mode == "off"
    assert settings.namespace == "local"
    assert settings.max_entries == 4096
    assert settings.max_bytes == 64 * 1024 * 1024
    assert settings.max_entry_bytes == 512 * 1024
    assert settings.embedding_ttl == 604800
    assert settings.evidence_ttl == 86400
    assert settings.empty_ttl == 300

    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MODE", "use")
    assert cache.resolve_settings(None).mode == "use"
    # The explicit flag wins over the environment value.
    assert cache.resolve_settings("refresh").mode == "refresh"
    assert cache.resolve_settings("off").mode == "off"


def test_off_mode_short_circuits_bound_validation(cache_env, monkeypatch):
    """`off` never validates bounds: junk env config cannot break uncached runs."""
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MAX_ENTRIES", "not-a-number")
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_NAMESPACE", "")
    settings = cache.resolve_settings(None, strict=False)
    assert settings.mode == "off"
    # Maintenance commands validate strictly.
    with pytest.raises(errors.CacheUnsafeError):
        cache.resolve_settings(None, strict=True)


@pytest.mark.parametrize("value", ["yes", "USE", "1"])
def test_invalid_cache_mode_rejected(cache_env, monkeypatch, value):
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MODE", value)
    with pytest.raises(errors.CacheUnsafeError):
        cache.resolve_settings(None)


@pytest.mark.parametrize("value", ["true", "false", "1.5", "-3", "0", "4096x", "4_096"])
def test_boolean_and_non_integer_limits_rejected(cache_env, monkeypatch, value):
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MODE", "use")
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MAX_ENTRIES", value)
    with pytest.raises(errors.CacheUnsafeError):
        cache.resolve_settings(None)


def test_namespace_validation(cache_env, monkeypatch):
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MODE", "use")
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_NAMESPACE", "x" * 200)
    with pytest.raises(errors.CacheUnsafeError):
        cache.resolve_settings(None)
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_NAMESPACE", "team-a")
    assert cache.resolve_settings(None).namespace == "team-a"


def test_embedding_revision_token_validation(cache_env, monkeypatch):
    monkeypatch.setenv("LIBRARY_QUERY_EMBEDDING_REVISION", "")
    with pytest.raises(errors.CacheUnsafeError):
        revisions.operator_embedding_revision()
    monkeypatch.setenv("LIBRARY_QUERY_EMBEDDING_REVISION", "2026-09")
    assert revisions.operator_embedding_revision() == "2026-09"
    monkeypatch.delenv("LIBRARY_QUERY_EMBEDDING_REVISION", raising=False)
    assert revisions.operator_embedding_revision() == "1"


def test_cache_dir_fallbacks(cache_env, monkeypatch):
    monkeypatch.delenv("LIBRARY_QUERY_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_env.state_dir / "xdg"))
    assert cache.resolve_cache_dir(os.environ) == cache_env.state_dir / "xdg" / "library-query"
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert cache.resolve_cache_dir(os.environ) == Path.home() / ".cache" / "library-query"


# --- 1.3 exact keys ---------------------------------------------------------------------


def test_probe_exact_text_sensitivity(cache_env):
    settings = make_settings(cache_env.cache_dir)
    base = cache.embedding_key(settings, "how does C++ move semantics work?")
    assert cache.embedding_key(settings, "how does C++ move semantics work?") == base
    # Case, punctuation, and negation all change the exact key.
    assert cache.embedding_key(settings, "How does C++ move semantics work?") != base
    assert cache.embedding_key(settings, "how does c++ move semantics work?") != base
    assert cache.embedding_key(settings, "how does C++ move semantics not work?") != base
    assert cache.embedding_key(settings, "how does C++  move semantics work?") != base


def test_normalized_tag_equivalence(cache_env):
    a = cache.flat_filters("catalogued", ["Machine Learning", "review"])
    b = cache.flat_filters("catalogued", ["review", "machine  learning", "Machine Learning"])
    assert a == b == {"scope": "catalogued", "tags_all": ["machine learning", "review"]}
    assert cache.flat_filters("all", None) == {"scope": "all", "tags_all": []}


def test_filter_and_window_differences_change_keys(cache_env):
    settings = make_settings(cache_env.cache_dir)
    verified = revisions.VerifiedRevision(
        corpus_id="corpus", incarnation="inc", revision=3,
        representation_fingerprint="repr", embedding_fingerprint="emb",
    )
    base = cache.candidate_key(
        settings, probe_text="q", verified=verified, method=cache.METHOD_FLAT,
        filters={"scope": "all", "tags_all": []}, window={"limit": 5},
    )
    same = cache.candidate_key(
        settings, probe_text="q", verified=verified, method=cache.METHOD_FLAT,
        filters={"scope": "all", "tags_all": []}, window={"limit": 5},
    )
    assert same == base
    variants = [
        {"filters": {"scope": "catalogued", "tags_all": []}},
        {"filters": {"scope": "all", "tags_all": ["x"]}},
        {"window": {"limit": 10}},
        {"method": cache.METHOD_GROUPED},
        {"probe_text": "other"},
    ]
    for override in variants:
        kwargs = {"probe_text": "q", "verified": verified, "method": cache.METHOD_FLAT,
                      "filters": {"scope": "all", "tags_all": []}, "window": {"limit": 5}}
        kwargs.update(override)
        assert cache.candidate_key(settings, **kwargs) != base, override


def test_namespace_and_model_identity_change_keys(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir)
    key = cache.embedding_key(settings, "probe")
    assert cache.embedding_key(make_settings(cache_env.cache_dir, namespace="other"), "probe") != key

    monkeypatch.setenv("LIBRARY_QUERY_EMBEDDING_REVISION", "2")
    assert cache.embedding_key(settings, "probe") != key
    monkeypatch.delenv("LIBRARY_QUERY_EMBEDDING_REVISION", raising=False)

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "vendor/other-model")
    assert cache.embedding_key(settings, "probe") != key
    monkeypatch.undo()


def test_endpoint_identity_is_credential_free(cache_env, monkeypatch):
    from urllib.parse import urlsplit, urlunsplit

    settings = make_settings(cache_env.cache_dir)
    clean_url = config.EMBEDDING_BASE_URL
    clean = cache.embedding_key(settings, "probe")
    identity = revisions.embedding_identity()
    assert "@" not in identity["endpoint"]
    assert "api-key" not in json.dumps(identity)

    parts = urlsplit(clean_url)
    dirty = urlunsplit((parts.scheme, f"user:secret-api-key@{parts.netloc}", parts.path, "api-key=secret", ""))
    monkeypatch.setattr(config, "EMBEDDING_BASE_URL", dirty)
    assert revisions.credential_free_url(dirty) == revisions.credential_free_url(clean_url)
    assert cache.embedding_key(settings, "probe") == clean
    assert revisions.embedding_identity()["endpoint"] == revisions.credential_free_url(clean_url)


def test_probe_text_is_never_persisted(cache_env, tmp_path):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    probe = "my private diagnosis question about rumination"
    store_embedding(storage, settings, probe)
    db_bytes = storage.db_path.read_bytes()
    assert probe.encode("utf-8") not in db_bytes
    assert cache.probe_digest(probe).encode("ascii") not in db_bytes


# --- 3.4 admission budget -----------------------------------------------------------------


def test_budget_shares_remainder_across_operations():
    clock = FakeClock()
    budget = cache.AdmissionBudget(clock=clock)
    assert budget.remaining_ms() == pytest.approx(100.0)
    budget.begin()
    clock.advance(0.06)
    budget.end()
    assert budget.remaining_ms() == pytest.approx(40.0)
    assert budget.busy_timeout_ms() == 40
    budget.begin()
    clock.advance(0.05)
    budget.end()
    assert budget.exhausted()
    assert budget.busy_timeout_ms() == 0
    # Elapsed is measured actual time, not clipped to the budget.
    clock.advance(5.0)
    assert budget.elapsed_ms() == 110


def test_budget_does_not_charge_provider_latency_between_operations():
    clock = FakeClock()
    budget = cache.AdmissionBudget(clock=clock)
    budget.begin()
    clock.advance(0.02)
    budget.end()
    clock.advance(30.0)  # provider/network latency between cache operations
    assert budget.remaining_ms() == pytest.approx(80.0)


def test_session_reports_overrun_honestly(cache_env):
    """A filesystem overrun is reported with actual elapsed time, not clipped."""
    settings = make_settings(cache_env.cache_dir)
    clock = FakeClock()
    session = cache.CacheSession(settings, "library", clock=clock)
    storage = session._store()
    store_embedding(storage, settings, "warm-probe")

    real_end = cache.AdmissionBudget.end

    def slow_end(self):
        clock.advance(0.35)  # simulated filesystem stall inside the operation
        real_end(self)

    from unittest import mock
    with mock.patch.object(cache.AdmissionBudget, "end", slow_end):
        session2 = cache.CacheSession(settings, "library", clock=clock)
        vector = session2._lookup_embedding("warm-probe")
        assert vector is None  # initialization overran; no further optional work starts
        assert session2._lookup_embedding("other-probe") is None
        session2.abort()
        diag = session2.diag
        assert diag.cache_elapsed_ms >= 350  # measured, not clipped to 100
        assert "cache_budget_exhausted" in diag.to_dict()["bypass_reasons"]


# --- 3.1 private storage safety -------------------------------------------------------------


def test_created_storage_modes_and_no_parent_chmod(cache_env, tmp_path):
    base = tmp_path / "preexisting"
    base.mkdir()
    os.chmod(base, 0o755)
    cache_dir = base / "nested" / "cache"
    settings = make_settings(cache_dir)
    storage = cache._Storage(settings)
    storage.check_safe()
    storage._connect()
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(storage.db_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(base.stat().st_mode) == 0o755, "shared parents must not be chmod'ed"


def test_rejects_location_inside_library(cache_env):
    cache_dir = cache_env.lib_root / "cache"
    cache_dir.mkdir()
    settings = make_settings(cache_dir)
    storage = cache._Storage(settings)
    assert storage is not None
    with pytest.raises(errors.CacheUnsafeError):
        storage.check_safe()


def test_rejects_location_inside_git_worktree(cache_env, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    cache_dir = repo / "cache"
    cache_dir.mkdir()
    with pytest.raises(errors.CacheUnsafeError):
        cache._Storage(make_settings(cache_dir)).check_safe()


def test_home_dotfiles_git_root_does_not_refuse(cache_env, monkeypatch):
    """A .git entry at the home root is a dotfiles repo, not a project worktree."""
    fake_home = Path(tempfile.mkdtemp(prefix="lq-home-", dir="/tmp"))
    try:
        (fake_home / ".git").mkdir()
        cache_dir = fake_home / ".local" / "share" / "library-query" / "cache"
        cache_dir.mkdir(parents=True)
        cache_dir.chmod(0o700)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        storage = cache._Storage(make_settings(cache_dir))
        storage.check_safe()
        storage._connect()
    finally:
        shutil.rmtree(fake_home, ignore_errors=True)


def test_project_git_below_home_still_refused(cache_env, monkeypatch):
    fake_home = Path(tempfile.mkdtemp(prefix="lq-home-", dir="/tmp"))
    try:
        repo = fake_home / "some-project"
        (repo / ".git").mkdir(parents=True)
        cache_dir = repo / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        with pytest.raises(errors.CacheUnsafeError):
            cache._Storage(make_settings(cache_dir)).check_safe()
    finally:
        shutil.rmtree(fake_home, ignore_errors=True)


def test_rejects_symlinked_and_writable_and_insecure_db(cache_env, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "cache").mkdir()
    link = tmp_path / "link"
    link.symlink_to(real / "cache")
    with pytest.raises(errors.CacheUnsafeError):
        cache._Storage(make_settings(link)).check_safe()

    writable = tmp_path / "writable"
    writable.mkdir()
    os.chmod(writable, 0o777)
    with pytest.raises(errors.CacheUnsafeError):
        cache._Storage(make_settings(writable)).check_safe()

    insecure = tmp_path / "insecure"
    settings = make_settings(insecure)
    storage = cache._Storage(settings)
    storage.check_safe()
    storage._connect()
    os.chmod(storage.db_path, 0o644)
    with pytest.raises(errors.CacheUnsafeError):
        cache._Storage(make_settings(insecure)).check_safe()


def test_session_bypasses_unsafe_storage_without_writing(cache_env, monkeypatch):
    cache_env.cache_dir.mkdir()
    os.chmod(cache_env.cache_dir, 0o777)
    session = cache.CacheSession(make_settings(cache_env.cache_dir), "library")
    assert session._store() is None
    assert "cache_unsafe" in session.diag.to_dict()["bypass_reasons"]
    assert not (cache_env.cache_dir / cache.CACHE_FILENAME).exists()


def test_session_bypasses_corrupt_database_and_never_deletes_it(cache_env):
    cache_env.cache_dir.mkdir()
    os.chmod(cache_env.cache_dir, 0o700)
    db = cache_env.cache_dir / cache.CACHE_FILENAME
    db.write_bytes(b"this is not a sqlite database at all")
    os.chmod(db, 0o600)
    session = cache.CacheSession(make_settings(cache_env.cache_dir), "library")
    assert session._store() is None
    diag = session.diag.to_dict()
    assert "cache_unavailable" in diag["bypass_reasons"]
    assert db.exists(), "an unfamiliar/corrupt database is never auto-deleted"


# --- 3.2 rows, TTL, LRU, eviction ------------------------------------------------------------


def _row(storage, settings, key, tier="embedding"):
    conn = sqlite3.connect(str(storage.db_path))
    try:
        return conn.execute(
            "SELECT expires_at, created_at, accessed_at, payload FROM cache_entries "
            "WHERE tier = ? AND namespace = ? AND entry_key = ?",
            (tier, settings.namespace, key),
        ).fetchone()
    finally:
        conn.close()


def test_ttl_expiry_and_hit_does_not_extend(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir)
    wall = FakeClock(start=1_000_000)
    monkeypatch.setattr(cache, "WALL_CLOCK", wall)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "probe", ttl=100)

    assert storage.get_payload("embedding", cache.embedding_key(settings, "probe"), cache.AdmissionBudget()) is not None
    wall.advance(60)
    assert storage.get_payload("embedding", cache.embedding_key(settings, "probe"), cache.AdmissionBudget()) is not None
    wall.advance(60)  # past the original expiry despite the hit
    assert storage.get_payload("embedding", cache.embedding_key(settings, "probe"), cache.AdmissionBudget()) is None


def test_rejects_future_created_timestamps(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir)
    wall = FakeClock(start=1_000_000)
    monkeypatch.setattr(cache, "WALL_CLOCK", wall)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "probe")
    key = cache.embedding_key(settings, "probe")
    conn = sqlite3.connect(str(storage.db_path))
    far_future = int(wall.now) + 3600
    conn.execute("UPDATE cache_entries SET created_at = ? WHERE entry_key = ?", (far_future, key))
    conn.commit()
    conn.close()
    assert storage.get_payload("embedding", key, cache.AdmissionBudget()) is None


def test_malformed_cached_vector_is_a_miss_and_removed(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    session = cache.CacheSession(settings, "library", clock=FakeClock())
    store_embedding(storage, settings, "bad-vector", vector=[0.1, 0.2])
    assert session._lookup_embedding("bad-vector") is None  # dim mismatch → invalid
    assert session.diag.embedding_misses == 1
    key = cache.embedding_key(settings, "bad-vector")
    assert _row(storage, settings, key) is None, "invalid rows are cleaned opportunistically"

    store_embedding(storage, settings, "nan-vector")
    key = cache.embedding_key(settings, "nan-vector")
    conn = sqlite3.connect(str(storage.db_path))
    conn.execute(
        "UPDATE cache_entries SET payload = ? WHERE entry_key = ?",
        (json.dumps({"v": 1, "vector": [float("nan")] * config.EMBEDDING_DIM}).encode(), key),
    )
    conn.commit()
    conn.close()
    assert session._lookup_embedding("nan-vector") is None
    assert "entry_invalid" in session.diag.to_dict()["bypass_reasons"]


def test_oversized_entries_are_skipped_not_truncated(cache_env):
    settings = make_settings(cache_env.cache_dir, max_entry_bytes=64)
    storage = cache._Storage(settings)
    reason = store_embedding(storage, settings, "big")
    assert reason == "entry_oversized"
    assert not storage.db_path.exists(), "oversized rows are skipped before any write"


def test_lru_eviction_is_deterministic_with_key_tiebreak(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir, max_entries=3)
    wall = FakeClock(start=1_000_000)
    monkeypatch.setattr(cache, "WALL_CLOCK", wall)
    storage = cache._Storage(settings)
    for probe in ("a", "b", "c"):
        wall.advance(1)
        store_embedding(storage, settings, probe)
    wall.advance(1)
    # Touch "a" so "b" is the least-recently used.
    assert storage.get_payload("embedding", cache.embedding_key(settings, "a"), cache.AdmissionBudget()) is not None
    wall.advance(1)
    store_embedding(storage, settings, "d")
    remaining = {
        row[0]
        for row in sqlite3.connect(str(storage.db_path)).execute("SELECT entry_key FROM cache_entries").fetchall()
    }
    keys = {p: cache.embedding_key(settings, p) for p in ("a", "b", "c", "d")}
    assert keys["b"] not in remaining
    assert {keys["a"], keys["c"], keys["d"]} <= remaining


def test_byte_pressure_evicts_lru(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir, max_entries=100, max_bytes=300)
    wall = FakeClock(start=1_000_000)
    monkeypatch.setattr(cache, "WALL_CLOCK", wall)
    storage = cache._Storage(settings)
    vector = [0.5] * 16  # small vector: ~110 bytes serialized per entry
    monkeypatch.setattr(config, "EMBEDDING_DIM", 16)
    for probe in ("a", "b", "c", "d"):
        wall.advance(1)
        store_embedding(storage, settings, probe, vector=vector)
    count = sqlite3.connect(str(storage.db_path)).execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
    assert count < 4, "byte pressure must evict even when entry count is under max_entries"


def test_empty_candidate_entries_use_shorter_ttl(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir)
    wall = FakeClock(start=1_000_000)
    monkeypatch.setattr(cache, "WALL_CLOCK", wall)
    session = cache.CacheSession(settings, "library", clock=FakeClock())
    session._verified = revisions.VerifiedRevision(
        corpus_id="c", incarnation="i", revision=1,
        representation_fingerprint="r", embedding_fingerprint="e",
    )
    # Publication requires the held reader guard; stand one in directly.
    ctx = revisions.managed_context("library")
    session._guard = revisions.acquire_writer_guard(ctx)
    payload = {"v": 1, "method": cache.METHOD_FLAT, "filters": {"scope": "all", "tags_all": []},
               "window": {"limit": 5}, "hits": []}
    session.stage_candidates(probe_text="empty-q", method=cache.METHOD_FLAT,
                             filters=payload["filters"], window=payload["window"], payload=payload)
    session.finalize()
    storage = cache._Storage(settings)
    key = cache.candidate_key(
        settings, probe_text="empty-q", verified=session._verified,
        method=cache.METHOD_FLAT, filters=payload["filters"], window=payload["window"],
    )
    expires_at, created_at, _, _ = _row(storage, settings, key, tier="evidence")
    assert expires_at - created_at == settings.empty_ttl


# --- 5.1 cache status ---------------------------------------------------------------------


def test_status_absent_storage_creates_nothing(cache_env, capsys):
    code = run_cli(["cache", "status", "--json"])
    out = capsys.readouterr()
    assert code == 0
    report = json.loads(out.out)
    assert report["storage"] == "absent"
    assert report["entries"]["embedding"]["count"] == 0
    assert report["entries"]["evidence"]["count"] == 0
    assert report["namespace"] == "local"
    assert report["limits"]["max_entries"] == 4096
    assert not cache_env.cache_dir.exists(), "absent storage remains absent"
    assert out.err.strip() == ""


def test_status_reports_scoped_counts_and_unverified_index(cache_env, capsys, monkeypatch):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "mine")
    other = make_settings(cache_env.cache_dir, namespace="someone-else")
    other_storage = cache._Storage(other)
    store_embedding(other_storage, other, "theirs")
    publish_stable_record(cache_env, revision=4)

    assert run_cli(["cache", "status", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["storage"] == "present"
    assert report["entries"]["embedding"]["count"] == 1, "other namespaces are out of scope"
    assert report["entries"]["embedding"]["bytes"] > 0
    assert report["database_bytes"] > 0
    assert report["index"]["revision"] == 4
    assert report["index"]["state"] == "stable"
    assert report["index"]["verified"] is False, "local revision state is explicitly unverified"


def test_status_never_contacts_services(cache_env, monkeypatch, capsys):
    from library_query import store
    monkeypatch.setattr(store, "qdrant", lambda: (_ for _ in ()).throw(AssertionError("no qdrant")))
    monkeypatch.setattr(store, "embed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no embed")))
    assert run_cli(["cache", "status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["storage"] == "absent"


def test_status_hot_journal_recovery_required_exit_three(cache_env, capsys, monkeypatch):
    """A real crashed writer leaves a hot journal; status must not recover it."""
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "probe")
    storage._conn.close()
    db = storage.db_path
    code = r"""
import sqlite3, os, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE padding (value BLOB)")
conn.executemany("INSERT INTO padding VALUES (?)", [(b"a" * 4096,)] * 128)
conn.commit()
conn.execute("PRAGMA cache_size=5")
conn.execute("BEGIN IMMEDIATE")
conn.execute("UPDATE padding SET value = ?", (b"b" * 4096,))
os._exit(0)
"""
    proc = subprocess.run([sys.executable, "-c", code, str(db)], timeout=10, check=False)
    assert proc.returncode == 0
    journal = db.parent / (db.name + "-journal")
    assert journal.exists() and journal.read_bytes()[:8] != bytes(8)
    before_db = (db.read_bytes(), db.stat().st_mtime_ns)
    before_journal = (journal.read_bytes(), journal.stat().st_mtime_ns)
    code = run_cli(["cache", "status", "--json"])
    captured = capsys.readouterr()
    assert code == 3
    assert captured.out == "", "no success bundle on stdout"
    error_lines = captured.err.strip().splitlines()
    assert len(error_lines) == 1, "exactly one structured error line"
    payload = json.loads(error_lines[0])
    assert payload["error"] == "cache_recovery_required"
    assert payload["exit_code"] == 3
    assert str(cache_env.cache_dir) not in payload["message"]
    assert (db.read_bytes(), db.stat().st_mtime_ns) == before_db
    assert (journal.read_bytes(), journal.stat().st_mtime_ns) == before_journal


def test_status_unsafe_storage_is_a_classified_error(cache_env, capsys):
    cache_env.cache_dir.mkdir()
    os.chmod(cache_env.cache_dir, 0o777)
    code = run_cli(["cache", "status", "--json"])
    payload = read_error(capsys)
    assert code == 3
    assert payload["error"] == "cache_unsafe"


def test_status_corrupt_database_reports_operator_recovery(cache_env, capsys):
    cache_env.cache_dir.mkdir()
    os.chmod(cache_env.cache_dir, 0o700)
    db = cache_env.cache_dir / cache.CACHE_FILENAME
    db.write_bytes(b"garbage")
    os.chmod(db, 0o600)
    code = run_cli(["cache", "status", "--json"])
    payload = read_error(capsys)
    assert code == 5
    assert payload["error"] == "internal_error"
    assert db.exists(), "status never auto-deletes an unfamiliar database"


def test_status_uses_readonly_connection_never_immutable(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "probe")
    seen = {}

    real_connect = cache._Storage.connect_readonly

    def spy_connect(self):
        conn = real_connect(self)
        seen["uri_readonly"] = True
        # A ro connection cannot write: prove it and prove no recovery is attempted.
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM cache_entries")
        return conn

    from unittest import mock
    with mock.patch.object(cache._Storage, "connect_readonly", spy_connect):
        report = cache.status_report(settings)
    assert report["storage"] == "present"
    assert seen["uri_readonly"]


# --- 5.2 cache clear ---------------------------------------------------------------------


def publish_stable_record(cache_env, revision=1, incarnation="inc-1", collection=None, state="stable"):
    ctx = revisions.managed_context(collection)
    record = {
        "schema_version": 1,
        "context_digest": ctx.context_digest,
        "corpus_id": ctx.corpus_id,
        "collection": ctx.collection,
        "incarnation": incarnation,
        "revision": revision,
        "state": state,
        "embedding_fingerprint": ctx.embedding_fingerprint,
        "representation_fingerprint": ctx.representation_fingerprint,
        "published_at": "2026-10-01T00:00:00+00:00",
    }
    revisions.publish_record(ctx, record)
    return ctx


def test_clear_embeddings_scoped_to_namespace_and_model(cache_env, monkeypatch):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "mine")
    other_ns = make_settings(cache_env.cache_dir, namespace="other")
    store_embedding(cache._Storage(other_ns), other_ns, "theirs")
    publish_stable_record(cache_env, revision=1)

    result = cache.clear_storage(settings, "embeddings")
    assert result["removed"]["embeddings"] == 1
    assert result["removed"]["evidence"] == 0
    assert _row(storage, settings, cache.embedding_key(settings, "mine")) is None
    assert _row(cache._Storage(other_ns), other_ns, cache.embedding_key(other_ns, "theirs")) is not None


def test_clear_all_never_touches_index_records_or_directories(cache_env):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "mine")
    ctx = publish_stable_record(cache_env, revision=2)
    revisions.acquire_writer_guard(ctx).release()  # materialize the guard file
    guard_bytes = ctx.guard_path.read_bytes()

    result = cache.clear_storage(settings, "all")
    assert result["removed"]["embeddings"] == 1
    assert ctx.record_path.exists(), "revision records are never deleted by clear"
    assert ctx.guard_path.read_bytes() == guard_bytes
    assert settings.cache_dir.exists()
    assert result["evidence_targeting"] == "ok"


def test_clear_evidence_without_corpus_identity_never_broadens(cache_env):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "mine")
    result = cache.clear_storage(settings, "all")
    assert result["evidence_targeting"] == "unavailable"
    assert result["removed"]["evidence"] == 0
    assert result["removed"]["embeddings"] == 1, "explicit embedding targeting still proceeds"


def test_clear_absent_storage_stays_absent(cache_env, capsys):
    code = run_cli(["cache", "clear", "--kind", "all", "--json"])
    out = capsys.readouterr()
    assert code == 0
    report = json.loads(out.out)
    assert report["storage"] == "absent"
    assert report["removed"] == {"embeddings": 0, "evidence": 0}
    assert not cache_env.cache_dir.exists()


def test_clear_invalid_kind_exits_two(cache_env, capsys):
    code = run_cli(["cache", "clear", "--kind", "everything", "--json"])
    assert code == 2


def test_clear_corrupt_database_reports_failure_and_keeps_file(cache_env):
    cache_env.cache_dir.mkdir()
    os.chmod(cache_env.cache_dir, 0o700)
    db = cache_env.cache_dir / cache.CACHE_FILENAME
    db.write_bytes(b"garbage")
    os.chmod(db, 0o600)
    with pytest.raises(errors.LibraryQueryError) as excinfo:
        cache.clear_storage(make_settings(cache_env.cache_dir), "all")
    assert excinfo.value.category == "internal_error"
    assert excinfo.value.exit_code == 5
    assert db.exists()


# --- 5.3 discovery -------------------------------------------------------------------------


def test_describe_exposes_cache_contract_from_runtime_definitions(capsys):
    assert run_cli(["describe", "--json"]) == 0
    schema = json.loads(capsys.readouterr().out)
    contract = schema["cache"]
    assert contract["modes"] == ["off", "use", "refresh"]
    assert contract["default_mode"] == "off"
    assert contract["storage"]["journal"] == "rollback"
    assert contract["admission_budget_ms"] == 100
    assert contract["recovery_error"]["category"] == "cache_recovery_required"
    assert contract["recovery_error"]["exit_code"] == 3
    assert "cache_unsafe" in schema["error_categories"]
    assert "index_revision_unavailable" in schema["error_categories"]
    assert set(contract["bypass_reasons"]) == set(cache.BYPASS_REASONS)

    fields = contract["diagnostics_schema"]
    for name in ("embedding_hits", "candidate_hits", "embedding_request_count",
                 "search_request_count", "validation_request_count", "cache_elapsed_ms"):
        assert fields[name]["type"] == "integer"

    cache_cmd = schema["commands"]["cache"]
    assert cache_cmd["name"] == "cache"
    for query in (schema["commands"]["query"], schema["commands"]["research"]):
        names = [arg["name"] for arg in query["arguments"]]
        assert "cache_mode" in names


def test_describe_cache_contract_matches_module_constants():
    from library_query import errors
    contract = cache.describe_schema()
    assert contract["modes"] == list(cache.CACHE_MODES)
    assert contract["bypass_reasons"] == sorted(cache.BYPASS_REASONS)
    assert contract["settings"][0]["default"] == cache.DEFAULT_MAX_ENTRIES
    assert errors.ERROR_EXIT_CODES[contract["recovery_error"]["category"]] == 3


# --- off mode: no storage interaction --------------------------------------------------------


def test_off_mode_never_touches_cache_storage(cache_env, monkeypatch, capsys):
    """Default-off query: no cache directory creation, no reads, no eviction."""
    from library_query import store
    monkeypatch.setattr(store, "embed", lambda texts, timeout=None: [[0.0] * config.EMBEDDING_DIM for _ in texts])
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[])
    monkeypatch.setattr(store, "qdrant", lambda: client)

    assert run_cli(["query", "anything", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    diag = output["cache"]
    assert diag["mode"] == "off"
    assert diag["embedding_hits"] == 0 and diag["candidate_hits"] == 0
    assert diag["index"] is None
    assert diag["bypass_reasons"] == []
    assert not cache_env.cache_dir.exists()
    assert list(cache_env.state_dir.iterdir()) == [], "off mode publishes no revision state either"


def test_off_flag_overrides_invalid_env_and_touches_nothing(cache_env, monkeypatch, capsys):
    from library_query import store
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MODE", "use")
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MAX_ENTRIES", "junk")
    monkeypatch.setattr(store, "embed", lambda texts, timeout=None: [[0.0] * config.EMBEDDING_DIM for _ in texts])
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[])
    monkeypatch.setattr(store, "qdrant", lambda: client)

    assert run_cli(["query", "anything", "--json", "--cache-mode", "off"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["cache"]["mode"] == "off"
    assert not cache_env.cache_dir.exists()


def test_invalid_cache_config_fails_closed_at_cli(cache_env, monkeypatch, capsys):
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_MODE", "sideways")
    code = run_cli(["query", "anything", "--json"])
    payload = read_error(capsys)
    assert code == 3
    assert payload["error"] == "cache_unsafe"


def test_contended_storage_wait_uses_only_remaining_allowance(cache_env):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    store_embedding(storage, settings, "locked")
    blocker = sqlite3.connect(storage.db_path)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        budget = cache.AdmissionBudget(total_ms=20, clock=time.monotonic)
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError):
            storage.get_payload("embedding", cache.embedding_key(settings, "locked"), budget)
        assert time.monotonic() - started < 1
        assert budget.busy_timeout_ms() <= 1
    finally:
        blocker.rollback()
        blocker.close()


def test_batch_publication_is_atomic_across_processes_and_journal_private(cache_env):
    settings = make_settings(cache_env.cache_dir)
    storage = cache._Storage(settings)
    conn = storage._connect()
    code = r"""
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("BEGIN IMMEDIATE")
for key in ("first", "second"):
    conn.execute("INSERT INTO cache_entries VALUES ('embedding','local',?,'ctx',1,1,2,1,?,2)", (key, b'{}'))
print("READY", flush=True)
sys.stdin.readline()
conn.commit()
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(storage.db_path)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "READY"
        assert conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0] == 0
        journal = Path(str(storage.db_path) + "-journal")
        assert stat.S_IMODE(journal.stat().st_mode) == 0o600
        _stdout, stderr = child.communicate("commit\n", timeout=10)
        assert child.returncode == 0, stderr
        assert conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0] == 2
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        conn.close()


@pytest.mark.parametrize("name", ["LIBRARY_QUERY_CACHE_NAMESPACE", "LIBRARY_QUERY_CACHE_MAX_ENTRIES"])
def test_explicit_empty_cache_settings_are_rejected(cache_env, monkeypatch, name):
    monkeypatch.setenv(name, "")
    with pytest.raises(errors.CacheUnsafeError):
        cache.resolve_settings("use")


@pytest.mark.parametrize("value", [True, False, 1.5, 0, -1])
def test_typed_settings_reject_non_positive_integer_bounds(cache_env, value):
    with pytest.raises(errors.CacheUnsafeError):
        make_settings(cache_env.cache_dir, max_entries=value)


def test_untrusted_oversized_numbers_are_invalid_vectors(cache_env):
    vector = [0.0] * config.EMBEDDING_DIM
    vector[0] = 10 ** 1000
    assert cache._validated_vector({"v": 1, "vector": vector}) is None
