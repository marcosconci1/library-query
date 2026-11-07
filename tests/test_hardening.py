"""Reconciliation hardening and legacy payload backfill.

Covers the findings from the implementation review (tasks 7.1-7.5) and the
one-time backfill of structural payloads onto legacy pre-manifest points.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from conftest import configure_readiness_client

from library_query import cli, config, ledger, store


@pytest.fixture
def mock_env(tmp_path: Path, monkeypatch):
    lib_root = tmp_path / "library"
    lib_root.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    monkeypatch.setattr(config, "LIBRARY_ROOT", lib_root)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")

    mock_client = MagicMock()
    configure_readiness_client(mock_client)
    mock_client.collection_exists.return_value = True
    monkeypatch.setattr(store, "qdrant", lambda: mock_client)
    monkeypatch.setattr(store, "embed", MagicMock(side_effect=lambda texts: [[0.05] * 1536 for _ in texts]))

    return {"lib_root": lib_root, "ledger_file": state_dir / "ledger.json", "mock_client": mock_client}


def test_in_place_overwrite_classifies_changed_exactly_once(mock_env):
    """A document overwritten in place is CHANGED once, never also DELETED."""
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    doc_path = inbox / "a.txt"
    doc_path.write_text("Original content that is about to be replaced entirely. " * 20)

    cli.cmd_index(argparse.Namespace())
    old_doc_id = next(iter(ledger._load(lib_root)["documents"].keys()))

    doc_path.write_text("Brand new unrelated content occupying the same path. " * 20)
    plan = ledger.reconcile(lib_root)

    assert len(plan["changed"]) == 1
    assert plan["changed"][0]["old_doc_id"] == old_doc_id
    assert len(plan["deleted"]) == 0
    assert len(plan["moved"]) == 0
    # The superseded document must not also surface as UNCHANGED.
    assert old_doc_id not in [item["doc_id"] for item in plan["unchanged"]]


def test_migration_is_pure_offline_with_deterministic_point_ids(mock_env, monkeypatch):
    """Task 7.2: v1->v2 migration never consults Qdrant and always recomputes IDs."""
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "legacy.txt").write_text("Legacy indexed content from the v1 era. " * 20)

    lookup_spy = MagicMock(return_value=[999, 998, 997])
    monkeypatch.setattr(store, "get_point_ids_for_doc", lookup_spy)

    mock_env["ledger_file"].write_text(
        json.dumps(
            {
                "files": {
                    "_Inbox/legacy.txt": "legacyhash",
                    "_meta:_Inbox/legacy.txt": {"chunks": 3},
                }
            }
        )
    )

    state = ledger._load(lib_root)

    lookup_spy.assert_not_called()
    doc = state["documents"]["doc_legacyhash"]
    assert doc["point_ids"] == [store.make_point_id("doc_legacyhash", i) for i in range(3)]


def test_get_point_ids_stops_on_none_cursor_not_falsy(mock_env):
    """Task 7.3: a falsy-but-not-None cursor must not be read as exhaustion."""
    mock_client = mock_env["mock_client"]
    page_one = ([MagicMock(id=1), MagicMock(id=2)], 0)  # cursor 0 is falsy but valid
    page_two = ([MagicMock(id=3)], None)
    mock_client.scroll.side_effect = [page_one, page_two]

    ids = store.get_point_ids_for_doc("_Inbox/some.txt")

    assert ids == [1, 2, 3]
    assert mock_client.scroll.call_count == 2


def test_get_point_ids_breaks_on_repeated_cursor(mock_env):
    """A non-advancing cursor must terminate rather than loop forever."""
    mock_client = mock_env["mock_client"]
    mock_client.scroll.return_value = ([MagicMock(id=7)], "stuck")

    ids = store.get_point_ids_for_doc("_Inbox/some.txt")

    assert ids == [7, 7]
    assert mock_client.scroll.call_count == 2


def test_delete_and_update_use_point_ids_only(mock_env):
    """Task 7.4: no payload-filter fallbacks; empty point_ids is a strict noop."""
    mock_client = mock_env["mock_client"]

    store.delete_points([])
    store.update_payloads([], {"scope": "inbox"})
    assert not mock_client.delete.called
    assert not mock_client.set_payload.called

    store.delete_points([11, 22])
    selector = mock_client.delete.call_args[1]["points_selector"]
    assert selector.points == [11, 22]

    store.update_payloads([33], {"scope": "catalogued"})
    assert mock_client.set_payload.call_args[1]["points"] == [33]


def test_scope_filter_matches_scope_field_only(mock_env):
    """`--scope` must filter on `scope`, not on `status` (which is read-state)."""
    mock_client = mock_env["mock_client"]
    mock_client.query_points.return_value = MagicMock(points=[])

    store.query_places("q", scope="catalogued")
    query_filter = mock_client.query_points.call_args[1]["query_filter"]

    assert query_filter.should is None
    assert len(query_filter.must) == 1
    condition = query_filter.must[0]
    assert condition.key == "scope"
    assert condition.match.value == "catalogued"


def test_zotero_tag_filters_intersect_with_scope(mock_env):
    """Each requested tag is a required Qdrant payload condition."""
    mock_client = mock_env["mock_client"]
    mock_client.query_points.return_value = MagicMock(points=[])

    store.query_places("q", scope="catalogued", tags=["machine learning", "review"])
    query_filter = mock_client.query_points.call_args[1]["query_filter"]

    assert [(condition.key, condition.match.value) for condition in query_filter.must] == [
        ("scope", "catalogued"),
        ("tags", "machine learning"),
        ("tags", "review"),
    ]


def test_backfill_sets_scope_and_type_on_legacy_points(mock_env):
    """Legacy points lacking scope are backfilled from the manifest, without re-embedding."""
    lib_root = mock_env["lib_root"]
    mock_client = mock_env["mock_client"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    doc_path = inbox / "legacy.txt"
    doc_path.write_text("A document indexed before the manifest existed. " * 20)

    content_hash = ledger.file_hash(doc_path)
    doc_id = f"doc_{content_hash}"
    deterministic = [store.make_point_id(doc_id, 0)]
    ledger._save(
        {
            "version": 2,
            "documents": {
                doc_id: {
                    "content_hash": content_hash,
                    "path": "_Inbox/legacy.txt",
                    "scope": "inbox",
                    "type": "inbox",
                    "metadata": {"title": "Legacy Doc", "authors": None, "year": None},
                    "chunks": 1,
                    "point_ids": deterministic,
                    "indexed_at": "2026-01-01T00:00:00Z",
                }
            },
        }
    )

    # Qdrant reports points still missing `scope`, including a pre-deterministic ID.
    mock_client.count.return_value = MagicMock(count=2)
    mock_client.scroll.return_value = ([MagicMock(id=424242)], None)

    cli.cmd_index(argparse.Namespace())

    assert not store.embed.called
    backfill_calls = [
        c[1] for c in mock_client.set_payload.call_args_list if "document_id" in c[1]["payload"]
    ]
    assert len(backfill_calls) == 1
    call = backfill_calls[0]
    assert call["payload"]["scope"] == "inbox"
    assert call["payload"]["type"] == "inbox"
    assert call["payload"]["document_id"] == doc_id
    assert call["payload"]["path"] == "_Inbox/legacy.txt"
    # Both the deterministic and the legacy point ID are targeted.
    assert set(call["points"]) == set(deterministic) | {424242}


def test_navigation_files_inherited_from_v1_are_purged(mock_env):
    """A v1 ledger that indexed navigation files must not leave them as ghost vectors."""
    lib_root = mock_env["lib_root"]
    (lib_root / "README.md").write_text("Library overview, a navigation document.")
    collections = lib_root / "Collections"
    collections.mkdir()
    (collections / "README.md").write_text("Collections index, also navigation.")

    mock_env["ledger_file"].write_text(
        json.dumps(
            {
                "files": {
                    "README.md": "roothash",
                    "_meta:README.md": {"chunks": 1},
                    "Collections/README.md": "collhash",
                    "_meta:Collections/README.md": {"chunks": 1},
                }
            }
        )
    )

    plan = ledger.reconcile(lib_root)
    deleted_paths = sorted(item["path"] for item in plan["deleted"])
    assert deleted_paths == ["Collections/README.md", "README.md"]
    assert plan["unchanged"] == []

    cli.cmd_index(argparse.Namespace())
    assert ledger._load(lib_root)["documents"] == {}


def test_backfill_skipped_when_no_points_missing_scope(mock_env):
    """A fully-backfilled collection costs zero extra Qdrant writes."""
    lib_root = mock_env["lib_root"]
    mock_client = mock_env["mock_client"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "doc.txt").write_text("Content already carrying full payloads. " * 20)

    cli.cmd_index(argparse.Namespace())
    mock_client.reset_mock()
    mock_client.count.return_value = MagicMock(count=0)

    cli.cmd_index(argparse.Namespace())

    assert not mock_client.set_payload.called
