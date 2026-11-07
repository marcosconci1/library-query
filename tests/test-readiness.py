"""Audited publication, reader races, corruption, and cache-independent readiness."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from conftest import FakeClock, configure_readiness_client, make_settings
from library_query import cache, cli, config, errors, readiness, research, revisions, store


@pytest.fixture
def managed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LIBRARY_ROOT", tmp_path / "library")
    client = MagicMock()
    configure_readiness_client(client)
    client.scroll.return_value = ([SimpleNamespace(id=1, payload={
        "document_id": "doc_x", "chunk_index": 0, "text": "A valid passage."})], None)
    monkeypatch.setattr(store, "qdrant", lambda: client)
    def info(collection, deadline=None):
        if deadline:
            deadline.check()
        return client.get_collection(collection_name=collection)
    monkeypatch.setattr(readiness, "collection_info", info)
    ctx = revisions.managed_context()
    with revisions.writer_session(ctx) as writer:
        readiness.invalidate(writer, recreated=False)
        record = readiness.publish(writer)
    client.scroll.reset_mock()
    client.count.reset_mock()
    return SimpleNamespace(ctx=ctx, client=client, record=record)


def deadline(clock=None):
    import time
    return research._Deadline(25, clock or time.monotonic)


def test_publication_matches_remote_and_audits_before_stable(managed):
    proof = managed.record["research_validation"]
    assert proof == readiness.remote_validation(managed.client.get_collection())
    assert readiness.valid_local(managed.record, managed.ctx).revision == 1


@pytest.mark.parametrize("failure", ["remote", "readback", "local"])
def test_failed_publication_leaves_dirty_state(managed, monkeypatch, failure):
    with revisions.writer_session(managed.ctx) as writer:
        readiness.invalidate(writer, recreated=False)
        def fail(*a, **kw):
            raise errors.IndexRevisionUnavailableError("interrupted publication")
        if failure == "remote":
            monkeypatch.setattr(readiness, "update_remote", fail)
        elif failure == "readback":
            monkeypatch.setattr(readiness, "collection_info", fail)
        else:
            monkeypatch.setattr(revisions, "publish_record", fail)
        with pytest.raises(errors.IndexRevisionUnavailableError):
            readiness.publish(writer)
    assert revisions.read_record(managed.ctx)["state"] == "dirty"
    with pytest.raises(errors.ResearchIndexNotReadyError):
        with readiness.reader(config.COLLECTION, deadline()):
            pytest.fail("dirty state admitted")


@pytest.mark.parametrize("field,value", [
    ("generation", "different"), ("audit_version", 2), ("state", "dirty"),
    ("revision", 99), ("embedding_fingerprint", "different"),
    ("representation_fingerprint", "different"), ("incarnation", "restored"),
])
def test_remote_record_mismatch_rejects_before_cache_or_embedding(managed, monkeypatch, field, value):
    remote = dict(managed.record["research_validation"], **{field: value})
    readiness.update_remote(config.COLLECTION, remote)
    monkeypatch.setattr(store, "embed", lambda *a, **kw: pytest.fail("must not embed"))
    request = research.parse_request(json.dumps({"queries": [{"text": "focused question"}]}).encode())
    with pytest.raises(errors.ResearchIndexNotReadyError):
        research.run(request)


def test_missing_remote_metadata_and_wrong_schema_reject(managed):
    info = managed.client.get_collection()
    info.config.metadata.clear()
    with pytest.raises(errors.ResearchIndexNotReadyError):
        readiness.verify_info(info, managed.record["research_validation"])
    info.config.metadata[readiness.METADATA_KEY] = managed.record["research_validation"]
    info.payload_schema.pop("scope")
    with pytest.raises(errors.ResearchIndexNotReadyError):
        readiness.verify_info(info, managed.record["research_validation"])


def test_exclusive_writer_blocks_reader_without_scanning(managed):
    with revisions.writer_session(managed.ctx):
        with pytest.raises(errors.ResearchIndexNotReadyError):
            with readiness.reader(config.COLLECTION, deadline()):
                pytest.fail("writer contention admitted")
    managed.client.scroll.assert_not_called()


def test_reader_holds_guard_through_evidence_assembly(managed, monkeypatch):
    import fcntl
    import os
    def embed(texts, **kwargs):
        fd = os.open(managed.ctx.guard_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        return [[0.0] * config.EMBEDDING_DIM]
    monkeypatch.setattr(store, "embed", embed)
    monkeypatch.setattr(store, "query_points_grouped", lambda *a, **kw: [])
    request = research.parse_request(b'{"queries":[{"text":"focused question"}]}')
    research.run(request)
    managed.client.scroll.assert_not_called()


@pytest.mark.parametrize("mode", ["off", "use", "refresh", "bypassed"])
def test_all_cache_modes_use_live_readiness_without_corpus_scan(managed, monkeypatch, tmp_path, mode):
    embed = MagicMock(return_value=[[0.0] * config.EMBEDDING_DIM])
    search = MagicMock(return_value=[])
    monkeypatch.setattr(store, "embed", embed)
    monkeypatch.setattr(store, "query_points_grouped", search)
    request = research.parse_request(b'{"queries":[{"text":"focused question"}]}')
    for _ in range(2):
        if mode == "off":
            research.run(request)
        else:
            session = cache.CacheSession(make_settings(tmp_path / "cache", mode="refresh" if mode == "refresh" else "use"), config.COLLECTION)
            if mode == "bypassed":
                monkeypatch.setattr(session, "_store", lambda: None)
            with session:
                research.run(request, cache=session)
    managed.client.scroll.assert_not_called()
    managed.client.count.assert_not_called()
    if mode == "use":
        assert embed.call_count == search.call_count == 1
    else:
        assert search.call_count == 2


@pytest.mark.parametrize("text", [None, "", 123, ["passage"]])
def test_invalid_text_prevents_stable_publication(managed, text):
    managed.client.scroll.return_value = ([SimpleNamespace(id=1, payload={
        "document_id": "doc_x", "chunk_index": 0, "text": text})], None)
    with revisions.writer_session(managed.ctx) as writer:
        readiness.invalidate(writer, recreated=False)
        with pytest.raises(errors.ResearchIndexNotReadyError):
            readiness.publish(writer)
    assert revisions.read_record(managed.ctx)["state"] == "dirty"


def test_expired_readiness_never_embeds(managed, monkeypatch):
    clock = FakeClock()
    def slow(collection, deadline=None):
        clock.advance(30)
        return managed.client.get_collection()
    monkeypatch.setattr(readiness, "collection_info", slow)
    monkeypatch.setattr(store, "embed", lambda *a, **kw: pytest.fail("expired request embedded"))
    request = research.parse_request(b'{"queries":[{"text":"focused question"}]}')
    with pytest.raises(errors.ResearchTimeoutError):
        research.run(request, clock=clock)


def test_missing_local_record_with_existing_guard_requires_reconciliation(managed):
    managed.ctx.record_path.unlink()
    with pytest.raises(errors.ResearchIndexNotReadyError):
        with readiness.reader(config.COLLECTION, deadline()):
            pytest.fail("missing managed state admitted")


def test_unreachable_service_fails_before_warm_cache_use(managed, monkeypatch, tmp_path):
    monkeypatch.setattr(store, "embed", lambda texts, **kw: [[0.0] * config.EMBEDDING_DIM])
    monkeypatch.setattr(store, "query_points_grouped", lambda *a, **kw: [])
    request = research.parse_request(b'{"queries":[{"text":"focused question"}]}')
    with cache.CacheSession(make_settings(tmp_path / "cache"), config.COLLECTION) as session:
        research.run(request, cache=session)
    def down(*a, **kw):
        raise errors.QdrantUnreachableError("service unavailable")
    monkeypatch.setattr(readiness, "collection_info", down)
    monkeypatch.setattr(store, "embed", lambda *a, **kw: pytest.fail("must not embed"))
    with cache.CacheSession(make_settings(tmp_path / "cache"), config.COLLECTION) as session:
        with pytest.raises(errors.QdrantUnreachableError):
            research.run(request, cache=session)


def test_legacy_scan_stops_at_deadline_before_next_page(monkeypatch):
    clock = FakeClock()
    client = MagicMock()
    def page(**kwargs):
        assert 0 < kwargs["timeout"] <= 25
        clock.advance(30)
        return [], 2
    client.scroll.side_effect = page
    monkeypatch.setattr(store, "QdrantClient", lambda **kw: client)
    token = store.SERVICE_DEADLINE.set(deadline(clock))
    try:
        with pytest.raises(errors.ResearchTimeoutError):
            store.scan_identity_payloads(include_text=False)
        assert client.scroll.call_count == 1
    finally:
        store.SERVICE_DEADLINE.reset(token)


def test_readiness_command_checks_live_proof_without_audit_or_writes(managed, capsys):
    managed.client.update_collection.reset_mock()
    before = managed.ctx.record_path.read_bytes()
    assert cli.main(["readiness", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["research_ready"] is True
    assert result["repaired"] is False
    assert result["index_revision"] == managed.record["revision"]
    managed.client.scroll.assert_not_called()
    managed.client.count.assert_not_called()
    managed.client.update_collection.assert_not_called()
    assert managed.ctx.record_path.read_bytes() == before


@pytest.mark.parametrize("missing", ["local_proof", "remote_proof"])
def test_explicit_marker_repair_audits_without_embedding_then_rechecks(managed, monkeypatch, capsys, missing):
    if missing == "local_proof":
        legacy = dict(managed.record)
        legacy.pop("research_validation")
        revisions.publish_record(managed.ctx, legacy)
    else:
        managed.client.get_collection().config.metadata.clear()
    assert cli.main(["readiness", "--json"]) == 3
    assert "research_index_not_ready" in capsys.readouterr().err
    monkeypatch.setattr(store, "embed", lambda *a, **kw: pytest.fail("marker repair must not embed"))
    assert cli.main(["readiness", "--repair", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["repaired"] is True
    assert result["index_revision"] == managed.record["revision"] + 1
    assert managed.client.scroll.call_count == 1
    record = revisions.read_record(managed.ctx)
    assert isinstance(record, dict)
    assert record["research_validation"] == readiness.remote_validation(managed.client.get_collection())
    assert cli.main(["readiness", "--repair", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["repaired"] is False
    assert managed.client.scroll.call_count == 1


@pytest.mark.parametrize("fault,reason", [
    ("dirty", "local_revision_dirty"),
    ("fingerprint", "local_revision_incompatible"),
    ("missing_proof", "local_validation_missing_or_incompatible"),
    ("remote_missing", "remote_validation_missing"),
    ("remote_conflict", "remote_validation_mismatch"),
    ("missing_record", "local_revision_missing"),
])
def test_readiness_reports_specific_failure_without_mutating(managed, capsys, fault, reason):
    record = dict(managed.record)
    if fault == "dirty":
        record["state"] = "dirty"
    elif fault == "fingerprint":
        record["embedding_fingerprint"] = "different"
    elif fault == "missing_proof":
        record.pop("research_validation")
    elif fault == "remote_missing":
        managed.client.get_collection().config.metadata.clear()
    elif fault == "remote_conflict":
        readiness.update_remote(config.COLLECTION, dict(record["research_validation"], revision=999))
    if fault == "missing_record":
        managed.ctx.record_path.unlink()
    else:
        revisions.publish_record(managed.ctx, record)
    managed.client.update_collection.reset_mock()
    assert cli.main(["readiness", "--json"]) == 3
    assert reason in json.loads(capsys.readouterr().err)["message"]
    managed.client.update_collection.assert_not_called()
    managed.client.scroll.assert_not_called()


def test_marker_repair_refuses_conflicting_remote_generation(managed, capsys):
    proof = dict(managed.record["research_validation"], generation="external-restore")
    readiness.update_remote(config.COLLECTION, proof)
    managed.client.update_collection.reset_mock()
    before = managed.ctx.record_path.read_bytes()
    assert cli.main(["readiness", "--repair", "--json"]) == 3
    assert "remote_validation_conflict" in capsys.readouterr().err
    assert managed.ctx.record_path.read_bytes() == before
    managed.client.scroll.assert_not_called()
    managed.client.update_collection.assert_not_called()


@pytest.mark.parametrize("fault", ["dirty", "fingerprint", "missing_record", "remote_revision"])
def test_marker_repair_refuses_unproven_corpus(managed, capsys, fault):
    record = dict(managed.record)
    if fault == "dirty":
        record["state"] = "dirty"
    elif fault == "fingerprint":
        record["embedding_fingerprint"] = "unknown-model"
    elif fault == "remote_revision":
        readiness.update_remote(config.COLLECTION, dict(record["research_validation"], revision=999))
    if fault == "missing_record":
        managed.ctx.record_path.unlink()
    else:
        revisions.publish_record(managed.ctx, record)
    managed.client.update_collection.reset_mock()
    assert cli.main(["readiness", "--repair", "--json"]) == 3
    assert "research_index_not_ready" in capsys.readouterr().err
    managed.client.scroll.assert_not_called()
    managed.client.update_collection.assert_not_called()


def test_marker_repair_is_nonblocking_while_writer_owns_guard(managed, capsys):
    with revisions.writer_session(managed.ctx):
        assert cli.main(["readiness", "--repair", "--json"]) == 3
    assert "index_busy" in capsys.readouterr().err
    managed.client.scroll.assert_not_called()


def test_marker_repair_failed_audit_leaves_dirty_not_ready(managed, capsys):
    managed.client.get_collection().config.metadata.clear()
    managed.client.scroll.return_value = ([SimpleNamespace(id=1, payload={
        "document_id": "doc_x", "chunk_index": -1, "text": "Untrusted."})], None)
    assert cli.main(["readiness", "--repair", "--json"]) == 3
    capsys.readouterr()
    record = revisions.read_record(managed.ctx)
    assert isinstance(record, dict) and record["state"] == "dirty"
    assert cli.main(["readiness", "--json"]) == 3
    assert "local_revision_dirty" in capsys.readouterr().err


def test_marker_repair_checks_later_audit_pages_before_publishing(managed, capsys):
    managed.client.get_collection().config.metadata.clear()
    managed.client.scroll.side_effect = [
        ([SimpleNamespace(id=1, payload={"document_id": "doc_x", "chunk_index": 0,
                                        "text": "Valid first-page passage."})], 2),
        ([SimpleNamespace(id=2, payload={"document_id": "doc_x", "chunk_index": 1,
                                        "text": None})], None),
    ]
    assert cli.main(["readiness", "--repair", "--json"]) == 3
    assert "research_index_not_ready" in capsys.readouterr().err
    assert managed.client.scroll.call_count == 2
    assert managed.client.scroll.call_args.kwargs["offset"] == 2
    record = revisions.read_record(managed.ctx)
    assert isinstance(record, dict) and record["state"] == "dirty"
    remote = readiness.remote_validation(managed.client.get_collection())
    assert isinstance(remote, dict) and remote["state"] == "dirty"
