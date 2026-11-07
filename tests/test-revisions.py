"""Managed indexed-corpus revisions: publication protocol and guard races.

Covers tasks.md section 2: per-context records beside the manifest, durable
dirty publication before any corpus write, stable publication only after
verified completion, idle-pass revision preservation, crash/interrupt dirty
recovery, and multiprocess reader/writer guard behavior. Children run the real
interpreter against temporary state directories passed through the environment.
"""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import configure_readiness_client

from library_query import cache, cli, config, errors, revisions, store

STATE_ENV_VARS = [
    "LIBRARY_QUERY_CACHE_MODE",
    "LIBRARY_QUERY_CACHE_NAMESPACE",
    "LIBRARY_QUERY_CACHE_DIR",
    "LIBRARY_QUERY_EMBEDDING_REVISION",
    "XDG_CACHE_HOME",
]


@pytest.fixture
def rev_env(tmp_path, monkeypatch):
    lib_root = tmp_path / "library"
    lib_root.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(config, "LIBRARY_ROOT", lib_root)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    for name in STATE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_DIR", str(tmp_path / "cache"))
    return SimpleNamespace(lib_root=lib_root, state_dir=state_dir, tmp_path=tmp_path)


def run_cli(argv):
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def record_files(state_dir):
    return sorted(p.name for p in state_dir.glob("index-*-revision.json"))


def guard_files(state_dir):
    return sorted(p.name for p in state_dir.glob("index-*-guard.lock"))


def stable_record(ctx, revision=1, incarnation="inc-1"):
    return {
        "schema_version": 1,
        "context_digest": ctx.context_digest,
        "corpus_id": ctx.corpus_id,
        "collection": ctx.collection,
        "incarnation": incarnation,
        "revision": revision,
        "state": "stable",
        "embedding_fingerprint": ctx.embedding_fingerprint,
        "representation_fingerprint": ctx.representation_fingerprint,
        "published_at": "2026-10-01T00:00:00+00:00",
    }


# --- 2.1 per-context records ------------------------------------------------------------


def test_contexts_have_separate_records_and_guards(rev_env):
    ctx_a = revisions.managed_context("collection-a")
    ctx_b = revisions.managed_context("collection-b")
    assert ctx_a.record_path != ctx_b.record_path
    assert ctx_a.guard_path != ctx_b.guard_path
    revisions.publish_record(ctx_a, stable_record(ctx_a, revision=1))
    revisions.publish_record(ctx_b, stable_record(ctx_b, revision=9, incarnation="inc-b"))
    assert revisions.validate_stable_record(revisions.read_record(ctx_a), ctx_a).revision == 1
    assert revisions.validate_stable_record(revisions.read_record(ctx_b), ctx_b).revision == 9


def test_explicit_preparation_initializes_state(rev_env):
    assert record_files(rev_env.state_dir) == []
    ctx = revisions.managed_context()
    revisions.publish_record(ctx, stable_record(ctx))
    assert record_files(rev_env.state_dir) == [f"index-{ctx.context_digest}-revision.json"]


def test_read_only_commands_never_create_revision_records(rev_env, monkeypatch, capsys):
    """query/research/browse/describe/cache status never initialize or repair state."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(vectors_config=SimpleNamespace(params=SimpleNamespace(size=config.EMBEDDING_DIM)))
    client.query_points.return_value = MagicMock(points=[])
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "QdrantClient", lambda **kw: client)
    monkeypatch.setattr(store, "embed", lambda texts, timeout=None: [[0.0] * config.EMBEDDING_DIM for _ in texts])
    monkeypatch.setattr(store, "check_research_ready", lambda collection_name=None: 0)

    assert run_cli(["query", "anything", "--json", "--cache-mode", "use"]) == 0
    capsys.readouterr()
    assert run_cli(["describe", "--json"]) == 0
    capsys.readouterr()
    assert run_cli(["cache", "status", "--json"]) == 0
    capsys.readouterr()
    assert record_files(rev_env.state_dir) == []
    assert guard_files(rev_env.state_dir) == []


def test_browse_never_creates_revision_records(rev_env, capsys):
    assert run_cli(["browse", "--json"]) == 0
    capsys.readouterr()
    assert record_files(rev_env.state_dir) == []
    assert guard_files(rev_env.state_dir) == []


def test_research_never_creates_revision_records(rev_env, monkeypatch, capsys, tmp_path):

    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(vectors_config=SimpleNamespace(params=SimpleNamespace(size=config.EMBEDDING_DIM)))
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "QdrantClient", lambda **kw: client)
    fake = SimpleNamespace(embed_calls=[])
    monkeypatch.setattr(store, "check_research_ready", lambda collection_name=None: 0)
    monkeypatch.setattr(
        store, "embed",
        lambda texts, timeout=None: fake.embed_calls.append(texts)
        or [[0.1] * config.EMBEDDING_DIM for _ in texts],
    )
    monkeypatch.setattr(store, "query_points_grouped", lambda *a, **k: [])
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"queries": [{"text": "q0"}]}))
    assert run_cli(["research", "--input", str(request), "--json", "--cache-mode", "use"]) == 0
    capsys.readouterr()
    assert record_files(rev_env.state_dir) == []


# --- 2.2 durable dirty publication --------------------------------------------------------


def test_guard_acquisition_then_dirty_then_stable(rev_env):
    ctx = revisions.managed_context()
    with revisions.writer_session(ctx) as session:
        session.mark_dirty(recreated=False)
        record = revisions.read_record(ctx)
        assert record["state"] == "dirty"
        assert record["revision"] == 1
        published = session.publish_stable()
        assert published["state"] == "stable"
        assert published["revision"] == 1
    # Guard released; record stable and durable.
    verified = revisions.validate_stable_record(revisions.read_record(ctx), ctx)
    assert verified is not None and verified.revision == 1


def test_dirty_publication_failure_aborts_before_any_corpus_write(rev_env, monkeypatch):
    """Guard/record-sync/directory-sync failure prevents every corpus mutation."""
    ctx = revisions.managed_context()
    client = MagicMock()
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "QdrantClient", lambda **kw: client)
    calls = {"mutations": 0}
    for name in ("upsert", "set_payload", "delete", "create_collection", "create_payload_index"):
        monkeypatch.setattr(client, name, lambda *a, **k: calls.__setitem__("mutations", calls["mutations"] + 1))

    monkeypatch.setattr(revisions, "_sync_dir", lambda path: (_ for _ in ()).throw(OSError("dir sync failed")))

    with revisions.writer_session(ctx) as session, pytest.raises(errors.IndexRevisionUnavailableError):
        session.mark_dirty(recreated=False)
    assert calls["mutations"] == 0, "no corpus write may happen when dirty publication fails"
    # The abort-before-mutation contract: whatever the filesystem did with the
    # replaced file, no *stable* record may remain eligible for cache reuse.
    assert revisions.validate_stable_record(revisions.read_record(ctx), ctx) is None


def test_guard_open_failure_aborts_indexing(rev_env, monkeypatch):
    ctx = revisions.managed_context()
    monkeypatch.setattr(revisions, "_open_guard_fd", lambda *a, **k: (_ for _ in ()).throw(OSError("permission denied")))
    with pytest.raises(errors.IndexRevisionUnavailableError):
        revisions.writer_session(ctx)


def test_symlinked_guard_refused(rev_env):
    ctx = revisions.managed_context()
    ctx.guard_path.symlink_to(os.devnull)
    with pytest.raises(errors.IndexRevisionUnavailableError):
        revisions.acquire_writer_guard(ctx)


def test_guard_file_is_stable_across_publication(rev_env):
    ctx = revisions.managed_context()
    revisions.acquire_writer_guard(ctx).release()
    stat_before = ctx.guard_path.stat()
    for revision in (1, 2, 3):
        record = stable_record(ctx, revision=revision)
        record["state"] = "dirty"
        revisions.publish_record(ctx, record)
        revisions.publish_record(ctx, stable_record(ctx, revision=revision))
    stat_after = ctx.guard_path.stat()
    assert (stat_before.st_ino, stat_before.st_mtime_ns) == (stat_after.st_ino, stat_after.st_mtime_ns)


def test_interrupted_mutation_leaves_durable_dirty_state(rev_env):
    ctx = revisions.managed_context()
    with pytest.raises(RuntimeError, match="simulated crash"), revisions.writer_session(ctx) as session:
        session.mark_dirty(recreated=False)
        raise RuntimeError("simulated crash mid-mutation")
    record = revisions.read_record(ctx)
    assert record["state"] == "dirty"
    assert revisions.validate_stable_record(record, ctx) is None, "dirty never verifies"


def test_explicit_index_pass_recovers_dirty_state(rev_env):
    ctx = revisions.managed_context()
    with revisions.writer_session(ctx) as session:
        session.mark_dirty(recreated=False)  # crash: no stable publication
    with revisions.writer_session(ctx) as session:
        _incarnation, revision = session.planned_mutation(recreated=False)
        assert revision == 1, "a dirty pending revision is retried, not skipped"
        session.mark_dirty(recreated=False)
        published = session.publish_stable()
        assert published["state"] == "stable"
        assert published["revision"] == 1


# --- 2.3 stable publication semantics -------------------------------------------------------


def test_planned_mutation_advances_monotonically(rev_env):
    ctx = revisions.managed_context()
    session = revisions.writer_session(ctx)
    try:
        assert session.planned_mutation(recreated=True)[1] == 1
    finally:
        session.close()
    revisions.publish_record(ctx, stable_record(ctx, revision=7))
    session = revisions.writer_session(ctx)
    try:
        incarnation, revision = session.planned_mutation(recreated=False)
        assert revision == 8
        assert incarnation == "inc-1"
    finally:
        session.close()
    # Recreation never reuses the previous incarnation.
    session = revisions.writer_session(ctx)
    try:
        incarnation2, revision2 = session.planned_mutation(recreated=True)
        assert incarnation2 != "inc-1"
        assert revision2 == 1
    finally:
        session.close()


def test_idle_pass_preserves_stable_revision(rev_env):
    ctx = revisions.managed_context()
    revisions.publish_record(ctx, stable_record(ctx, revision=5))
    # An idle verification session performs no publication at all.
    with revisions.writer_session(ctx) as session:
        assert session.current_stable()["revision"] == 5
    assert revisions.read_record(ctx)["revision"] == 5
    assert revisions.read_record(ctx)["state"] == "stable"


def test_corrupt_record_never_verifies(rev_env):
    ctx = revisions.managed_context()
    ctx.record_path.write_text("{ not json")
    record = revisions.read_record(ctx)
    assert record["state"] == "dirty" and record["corrupt"]
    assert revisions.validate_stable_record(record, ctx) is None


def test_incompatible_fingerprint_never_verifies(rev_env):
    ctx = revisions.managed_context()
    record = stable_record(ctx)
    record["embedding_fingerprint"] = "0" * 64
    revisions.publish_record(ctx, record)
    assert revisions.validate_stable_record(revisions.read_record(ctx), ctx) is None


# --- 2.4 guard races (real multiprocess) -----------------------------------------------------


def _child_env(rev_env):
    env = dict(os.environ)
    env["LIBRARY_ROOT"] = str(rev_env.lib_root)
    env["LIBRARY_QUERY_STATE"] = str(rev_env.state_dir)
    env["LIBRARY_QUERY_CACHE_DIR"] = str(rev_env.tmp_path / "cache")
    env["LIBRARY_QUERY_COLLECTION"] = "library"
    for name in STATE_ENV_VARS:
        env.pop(name, None)
    env["LIBRARY_QUERY_CACHE_DIR"] = str(rev_env.tmp_path / "cache")
    return env


def _run_child(code, rev_env, timeout=60):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout, check=False, env=_child_env(rev_env),
        cwd=str(rev_env.tmp_path),
    )


CHILD_TRY_EXCLUSIVE = r"""
import os, sys
sys.path.insert(0, {src!r})
from library_query import revisions
ctx = revisions.managed_context()
import fcntl
fd = os.open(ctx.guard_path, os.O_RDWR | os.O_CREAT, 0o600)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print("WOULD_BLOCK"); sys.exit(0)
print("ACQUIRED"); sys.exit(1)
"""


def test_reader_shared_guard_blocks_writer_exclusive(rev_env):
    """A reader holding the shared guard prevents any writer mutation until release."""
    ctx = revisions.managed_context()
    revisions.acquire_writer_guard(ctx).release()  # materialize the guard file
    guard = revisions.try_acquire_reader_guard(ctx, cache.AdmissionBudget())
    assert guard is not None
    try:
        proc = _run_child(CHILD_TRY_EXCLUSIVE.format(src=str(revisions.__file__).rsplit("/", 1)[0]), rev_env=rev_env)
        assert "WOULD_BLOCK" in proc.stdout, proc.stderr
    finally:
        guard.release()
    proc = _run_child(CHILD_TRY_EXCLUSIVE.format(src=str(revisions.__file__).rsplit("/", 1)[0]), rev_env=rev_env)
    assert "ACQUIRED" in proc.stdout, proc.stderr


def test_busy_guard_bypasses_with_bounded_wait(rev_env):
    """A writer holding the exclusive guard: a reader with a small budget bypasses."""
    ctx = revisions.managed_context()
    writer = revisions.acquire_writer_guard(ctx)
    try:
        budget = cache.AdmissionBudget(total_ms=40)
        start = time.monotonic()
        guard = revisions.try_acquire_reader_guard(ctx, budget)
        elapsed = time.monotonic() - start
        assert guard is None
        assert elapsed < 2.0, "bounded wait, never unbounded"
        assert revisions.reader_bypass_reason(ctx) == "index_busy"
    finally:
        writer.release()


def test_guard_released_when_holder_process_dies(rev_env):
    ctx = revisions.managed_context()
    code = r"""
import fcntl, os, sys
sys.path.insert(0, {src!r})
from library_query import revisions
ctx = revisions.managed_context()
fd = os.open(ctx.guard_path, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
os._exit(0)
""".format(src=str(revisions.__file__).rsplit("/", 1)[0])
    proc = _run_child(code, rev_env)
    assert proc.returncode == 0
    # The lock died with the process: a fresh writer acquires immediately.
    writer = revisions.acquire_writer_guard(ctx)
    writer.release()


CHILD_PUBLISH = r"""
import json, sys, time
sys.path.insert(0, @SRC@)
from library_query import revisions
collection, marker = @COLLECTION@, @MARKER@
ctx = revisions.managed_context(collection)
with revisions.writer_session(ctx) as session:
    session.mark_dirty(recreated=False)
    open(marker, "w").write("dirty")
    # Wait for the sibling marker before publishing stable.
    for _ in range(600):
        if len(__import__("os").listdir(__import__("pathlib").Path(marker).parent)) >= 2:
            break
        time.sleep(0.01)
    published = session.publish_stable()
print(json.dumps({"collection": collection, "revision": published["revision"]}))
"""


def test_concurrent_writers_for_different_contexts_stay_independent(rev_env):
    """Overlapping dirty/stable publications cannot clobber another context."""
    src = str(revisions.__file__).rsplit("/", 1)[0]
    markers = rev_env.tmp_path / "markers"
    markers.mkdir()
    procs = []
    for collection in ("collection-a", "collection-b"):
        code = (CHILD_PUBLISH
                .replace("@SRC@", repr(src))
                .replace("@COLLECTION@", repr(collection))
                .replace("@MARKER@", repr(str(markers / f"{collection}.ready"))))
        procs.append(_run_child(code, rev_env))
    results = {}
    for proc in procs:
        assert proc.returncode == 0, proc.stderr
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        results[row["collection"]] = row
    ctx_a = revisions.managed_context("collection-a")
    ctx_b = revisions.managed_context("collection-b")
    rec_a = revisions.read_record(ctx_a)
    rec_b = revisions.read_record(ctx_b)
    assert rec_a["state"] == "stable" and rec_a["revision"] == 1
    assert rec_b["state"] == "stable" and rec_b["revision"] == 1
    assert rec_a["incarnation"] != rec_b["incarnation"]
    # Re-publishing one context cannot disturb the other.
    with revisions.writer_session(ctx_a) as session:
        session.mark_dirty(recreated=False)
        session.publish_stable()
    assert revisions.read_record(ctx_b)["revision"] == 1


def test_unsupported_platform_reports_guard_unavailable(rev_env, monkeypatch):
    monkeypatch.setattr(revisions, "HAVE_FCNTL", False)
    ctx = revisions.managed_context()
    assert revisions.try_acquire_reader_guard(ctx, cache.AdmissionBudget()) is None
    assert revisions.reader_bypass_reason(ctx) == "guard_unavailable"
    with pytest.raises(errors.IndexRevisionUnavailableError):
        revisions.writer_session(ctx)


def test_missing_guard_and_record_bypass_as_unmanaged(rev_env):
    ctx = revisions.managed_context()
    assert revisions.try_acquire_reader_guard(ctx, cache.AdmissionBudget()) is None
    assert revisions.reader_bypass_reason(ctx) == "unmanaged_index"


def test_model_and_representation_changes_share_guard_and_rotate_incarnation(rev_env, monkeypatch):
    old = revisions.managed_context()
    revisions.publish_record(old, stable_record(old, revision=5))
    monkeypatch.setenv("LIBRARY_QUERY_EMBEDDING_REVISION", "2")
    changed = revisions.managed_context()
    assert changed.guard_path == old.guard_path
    assert changed.record_path == old.record_path
    assert revisions.validate_stable_record(revisions.read_record(changed), changed) is None
    with revisions.writer_session(changed) as writer:
        writer.mark_dirty(recreated=False)
        published = writer.publish_stable()
    assert published["incarnation"] != "inc-1"
    assert published["embedding_fingerprint"] == changed.embedding_fingerprint


def test_revision_locations_follow_configured_manifest(rev_env, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_PATH", rev_env.tmp_path / "other-state" / "ledger.json")
    assert revisions.managed_context().record_path.parent == config.LEDGER_PATH.parent


@pytest.fixture
def index_env(rev_env, monkeypatch):
    client = MagicMock()
    configure_readiness_client(client)
    client.collection_exists.return_value = True
    client.upsert.return_value = SimpleNamespace(status="completed")
    client.set_payload.return_value = SimpleNamespace(status="completed")
    client.delete.return_value = SimpleNamespace(status="completed")
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "QdrantClient", lambda **kw: client)
    monkeypatch.setattr(store, "embed", MagicMock(side_effect=lambda texts: [[0.0] * config.EMBEDDING_DIM for _ in texts]))
    monkeypatch.setattr(store, "ensure_collection", lambda: None)
    monkeypatch.setattr(store, "verify_collection_configured", lambda: None)
    monkeypatch.setattr(cli, "_research_repairs_needed", lambda: False)
    monkeypatch.setattr(cli, "backfill_legacy_payloads", lambda state: 0)
    monkeypatch.setattr(cli, "_prepare_research_index", lambda state: {"ready": True, "reindexed": 0})
    rev_env.client = client
    return rev_env


def test_index_initializes_idle_state_and_recovers_dirty_without_actions(index_env):
    import argparse
    ctx = revisions.managed_context()
    assert cli.cmd_index(argparse.Namespace()) == 0
    first = revisions.read_record(ctx)
    assert first["state"] == "stable"
    assert cli.cmd_index(argparse.Namespace()) == 0
    assert revisions.read_record(ctx) == first
    with revisions.writer_session(ctx) as writer:
        writer.mark_dirty(recreated=False)
    assert cli.cmd_index(argparse.Namespace()) == 0
    recovered = revisions.read_record(ctx)
    assert recovered["state"] == "stable" and recovered["revision"] == first["revision"] + 1


@pytest.mark.parametrize("sync", ["_sync_file", "_sync_dir"])
def test_index_sync_failure_prevents_service_mutation(index_env, monkeypatch, sync):
    import argparse
    inbox = index_env.lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "document.txt").write_text("Searchable library content. " * 20)
    monkeypatch.setattr(revisions, sync, lambda *args: (_ for _ in ()).throw(OSError("sync failed")))
    with pytest.raises(errors.IndexRevisionUnavailableError):
        cli.cmd_index(argparse.Namespace())
    assert not index_env.client.upsert.called
    assert not index_env.client.set_payload.called
    assert not index_env.client.delete.called


def test_queued_index_write_retains_dirty_and_retry_recovers(index_env):
    import argparse
    inbox = index_env.lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "document.txt").write_text("Searchable library content. " * 20)
    index_env.client.upsert.return_value = SimpleNamespace(status="acknowledged")
    with pytest.raises(errors.IndexRevisionUnavailableError):
        cli.cmd_index(argparse.Namespace())
    ctx = revisions.managed_context()
    assert revisions.read_record(ctx)["state"] == "dirty"
    assert index_env.client.upsert.call_args.kwargs["wait"] is True
    index_env.client.upsert.return_value = SimpleNamespace(status="completed")
    assert cli.cmd_index(argparse.Namespace()) == 0
    assert revisions.read_record(ctx)["state"] == "stable"


def test_new_source_and_path_move_invalidate_without_move_embedding(index_env):
    import argparse
    inbox = index_env.lib_root / "_Inbox"
    inbox.mkdir()
    source = inbox / "document.txt"
    source.write_text("Searchable library content. " * 20)
    cli.cmd_index(argparse.Namespace())
    ctx = revisions.managed_context()
    first = revisions.read_record(ctx)
    store.embed.reset_mock()
    source.rename(inbox / "moved.txt")
    cli.cmd_index(argparse.Namespace())
    moved = revisions.read_record(ctx)
    assert moved["revision"] > first["revision"]
    store.embed.assert_not_called()
    assert index_env.client.set_payload.call_args.kwargs["wait"] is True
    (inbox / "competing.txt").write_text("New competing knowledge. " * 20)
    cli.cmd_index(argparse.Namespace())
    assert revisions.read_record(ctx)["revision"] > moved["revision"]


def test_index_rebuilds_existing_documents_for_changed_representation(index_env, monkeypatch):
    import argparse
    inbox = index_env.lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "document.txt").write_text("Searchable library content. " * 20)
    cli.cmd_index(argparse.Namespace())
    ctx = revisions.managed_context()
    first = revisions.read_record(ctx)
    store.embed.reset_mock()
    monkeypatch.setattr(config, "CHUNK_SIZE", config.CHUNK_SIZE + 1)
    cli.cmd_index(argparse.Namespace())
    changed = revisions.read_record(revisions.managed_context())
    assert changed["incarnation"] != first["incarnation"]
    assert store.embed.call_count == 1


def test_failed_idle_audit_revokes_old_proof_and_returns_failure(index_env, monkeypatch, capsys):
    import argparse
    from library_query import readiness
    assert cli.cmd_index(argparse.Namespace()) == 0
    capsys.readouterr()
    monkeypatch.setattr(cli, "_prepare_research_index", lambda state: {
        "ready": False, "reindexed": [], "reason": "invalid passage found"})
    assert cli.cmd_index(argparse.Namespace()) == errors.EXIT_CONFIG_ERROR
    assert json.loads(capsys.readouterr().out)["research_ready"] is False
    ctx = revisions.managed_context()
    record = revisions.read_record(ctx)
    remote = readiness.remote_validation(index_env.client.get_collection())
    assert isinstance(record, dict) and record["state"] == "dirty"
    assert isinstance(remote, dict) and remote["state"] == "dirty"
