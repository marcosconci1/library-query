"""Comprehensive validation suite covering the 14 reconciliation scenarios.

1. New document
2. Unchanged document
3. Content modification
4. Move from inbox -> catalogued
5. Move between catalogue paths
6. Metadata change without content change
7. Document deletion (including ghost papers/retrieval-primer.pdf)
8. Same filename with different content
9. Same content at a new path
10. Scope filtering
11. Collections directory files strictly ignored
12. catalog.bib strictly ignored as vector content
13. Metadata enrichment from README/catalog sources
14. Fallback folder-name parsing

Key invariant: metadata-only or path-only change MUST NOT trigger re-embedding.
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
    ledger_file = state_dir / "ledger.json"

    monkeypatch.setattr(config, "LIBRARY_ROOT", lib_root)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", ledger_file)

    mock_client = MagicMock()
    configure_readiness_client(mock_client)
    mock_client.collection_exists.return_value = True
    monkeypatch.setattr(store, "qdrant", lambda: mock_client)

    embed_spy = MagicMock(side_effect=lambda texts: [[0.05] * 1536 for _ in texts])
    monkeypatch.setattr(store, "embed", embed_spy)

    return {
        "lib_root": lib_root,
        "state_dir": state_dir,
        "ledger_file": ledger_file,
        "mock_client": mock_client,
        "embed_spy": embed_spy,
    }


def test_scenario_01_new_document(mock_env):
    """Scenario 1: Indexing a brand new document extracts, chunks, and embeds it."""
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "new-doc.txt").write_text("A fresh piece of knowledge about algorithms and data structures." * 5)

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1
    state = ledger._load(lib_root)
    assert len(state["documents"]) == 1
    doc = next(iter(state["documents"].values()))
    assert doc["scope"] == "inbox"
    assert doc["path"] == "_Inbox/new-doc.txt"


def test_failed_upsert_batch_leaves_document_pending_for_retry(mock_env, monkeypatch):
    from library_query import revisions

    source = mock_env["lib_root"] / "Videos" / "Example" / "source.md"
    source.parent.mkdir(parents=True)
    source.write_text("\n\n".join(
        f"## [00:0{i}:00] — Part {i}\n\n" + "Source material about reliable indexing. " * 25
        for i in range(4)
    ))
    monkeypatch.setattr(config, "QDRANT_MAX_BATCH_BYTES", 18000)
    client = mock_env["mock_client"]
    client.upsert.side_effect = [MagicMock(), RuntimeError("batch failure")]
    with pytest.raises(RuntimeError, match="batch failure"):
        cli.cmd_index(argparse.Namespace())
    assert ledger._load()["documents"] == {}
    assert ledger.stats()["pending"] == 1
    ctx = revisions.managed_context()
    assert revisions.read_record(ctx)["state"] == "dirty"
    first_ids = [p.id for p in client.upsert.call_args_list[0].kwargs["points"]]
    client.upsert.side_effect = None
    client.upsert.reset_mock()
    cli.cmd_index(argparse.Namespace())
    assert [p.id for p in client.upsert.call_args_list[0].kwargs["points"]] == first_ids
    assert ledger.stats()["pending"] == 0


@pytest.mark.parametrize("fail_repair", [False, True])
def test_idle_index_repairs_unproven_legacy_chunk_order(mock_env, monkeypatch, fail_repair):
    from library_query import revisions

    source = mock_env["lib_root"] / "Books" / "Example" / "source.txt"
    source.parent.mkdir(parents=True)
    source.write_text("Source material with a stable first passage. " * 20)
    monkeypatch.setattr(store, "ensure_grouping_index", lambda: False)
    monkeypatch.setattr(store, "check_research_ready", lambda: 1)
    monkeypatch.setattr(store, "verify_collection_configured", lambda: None)
    monkeypatch.setattr(store, "scan_identity_payloads", lambda *a, **kw: {})
    cli.cmd_index(argparse.Namespace())
    state = ledger._load()
    doc_id, doc = next(iter(state["documents"].items()))
    old_ids = [123]
    doc["point_ids"] = old_ids
    ledger._save(state)
    # A formerly backfilled legacy index may look valid while its chunk order
    # has no source-order proof. It must be rebuilt, even on an idle pass.
    facts = {123: {"document_id": doc_id, "chunk_index": 0, "text_present": True, "doc": doc["path"]}}
    assert store.invalid_identity_reason(facts[123]) is None
    monkeypatch.setattr(store, "scan_identity_payloads", lambda *a, **kw: facts)
    client = mock_env["mock_client"]
    client.reset_mock()
    assert cli._research_repairs_needed()
    if fail_repair:
        client.upsert.side_effect = RuntimeError("replacement failed")
    cli.cmd_index(argparse.Namespace())
    record = revisions.read_record(revisions.managed_context())
    repaired = ledger._load()["documents"][doc_id]
    if fail_repair:
        assert repaired["point_ids"] == old_ids
        client.delete.assert_not_called()
        assert record["state"] == "dirty"
        assert cli._documents_needing_point_id_repair(ledger._load())
    else:
        assert repaired["point_ids"] == [store.make_point_id(doc_id, 0)]
        assert client.upsert.called
        assert client.delete.call_args.kwargs["points_selector"].points == old_ids
        assert record["state"] == "stable"
        assert record["revision"] == 2
        assert not cli._documents_needing_point_id_repair(ledger._load())


def test_scenario_02_unchanged_document(mock_env):
    """Scenario 2: An unchanged document triggers 0 embedding calls and is a noop."""
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    doc_path = inbox / "new-doc.txt"
    doc_path.write_text("Consistent document content that never changes during this session." * 5)

    # Initial index
    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1

    # Second index run with no changes
    mock_env["embed_spy"].reset_mock()
    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 0


def test_scenario_03_content_modification(mock_env):
    """Scenario 3: Content modification triggers re-indexing (CHANGED) with new embeddings."""
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    doc_path = inbox / "doc.txt"
    doc_path.write_text("Version 1 content before any edits happen in the text document." * 5)

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1
    old_state = ledger._load(lib_root)
    old_hash = next(iter(old_state["documents"].values()))["content_hash"]

    # Modify content in place
    doc_path.write_text("Version 2 updated completely with new insights and different sentences." * 5)
    mock_env["embed_spy"].reset_mock()

    plan = ledger.reconcile(lib_root)
    assert len(plan["changed"]) == 1
    assert plan["changed"][0]["old_doc_id"] == f"doc_{old_hash}"

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1
    new_state = ledger._load(lib_root)
    assert len(new_state["documents"]) == 1
    new_doc = next(iter(new_state["documents"].values()))
    assert new_doc["content_hash"] != old_hash


def test_scenario_04_inbox_to_catalogued_move(mock_env):
    """Scenario 4: Move from inbox to catalogued preserves vectors with ZERO embedding calls."""
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    content = "Substantive book content about clean code architecture and refactoring." * 5
    (inbox / "temp-drop.txt").write_text(content)

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1
    state_before = ledger._load(lib_root)
    doc_id = next(iter(state_before["documents"].keys()))

    # Move file from _Inbox to Books/Martin_2008_Clean-Code/book.txt
    (inbox / "temp-drop.txt").unlink()
    book_dir = lib_root / "Books" / "Martin_2008_Clean-Code"
    book_dir.mkdir(parents=True)
    (book_dir / "book.txt").write_text(content)

    mock_env["embed_spy"].reset_mock()
    plan = ledger.reconcile(lib_root)
    assert len(plan["moved"]) == 1
    assert plan["moved"][0]["doc_id"] == doc_id
    assert plan["moved"][0]["scope"] == "catalogued"
    assert plan["moved"][0]["type"] == "book"

    cli.cmd_index(argparse.Namespace())
    # KEY INVARIANT: 0 embedding calls!
    assert mock_env["embed_spy"].call_count == 0
    assert mock_env["mock_client"].set_payload.called

    state_after = ledger._load(lib_root)
    assert doc_id in state_after["documents"]
    updated_doc = state_after["documents"][doc_id]
    assert updated_doc["path"] == "Books/Martin_2008_Clean-Code/book.txt"
    assert updated_doc["scope"] == "catalogued"
    assert updated_doc["type"] == "book"
    assert updated_doc["metadata"]["title"] == "Clean Code"


def test_scenario_05_move_between_catalogue_paths(mock_env):
    """Scenario 5: Move between catalogue paths updates path/metadata with ZERO embedding calls."""
    lib_root = mock_env["lib_root"]
    content = "Original paper on distributed systems consensus and state machines." * 5
    old_dir = lib_root / "Papers" / "Lamport_1998_Part-Time-Parliament"
    old_dir.mkdir(parents=True)
    (old_dir / "paper.txt").write_text(content)

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1
    doc_id = next(iter(ledger._load(lib_root)["documents"].keys()))

    # Move to new directory with updated title
    (old_dir / "paper.txt").unlink()
    old_dir.rmdir()
    new_dir = lib_root / "Papers" / "Lamport_1998_Paxos-Consensus"
    new_dir.mkdir(parents=True)
    (new_dir / "paper.txt").write_text(content)

    mock_env["embed_spy"].reset_mock()
    cli.cmd_index(argparse.Namespace())
    # KEY INVARIANT: 0 embedding calls!
    assert mock_env["embed_spy"].call_count == 0

    state = ledger._load(lib_root)
    assert state["documents"][doc_id]["path"] == "Papers/Lamport_1998_Paxos-Consensus/paper.txt"
    assert state["documents"][doc_id]["metadata"]["title"] == "Paxos Consensus"


def test_scenario_06_metadata_change_without_content_change(mock_env):
    """Scenario 6: Metadata change (e.g. README frontmatter edit) updates payloads with ZERO embeddings."""
    lib_root = mock_env["lib_root"]
    content = "Deep learning transformers and attention mechanisms research." * 5
    paper_dir = lib_root / "Papers" / "Vaswani_2017_Attention"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper.txt").write_text(content)
    readme = paper_dir / "README.md"
    readme.write_text("---\ntitle: Initial Title\nstatus: unread\n---")

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1

    # Edit metadata frontmatter only, content unchanged
    readme.write_text("""---
title: "Attention Is All You Need: Groundbreaking Architecture"
status: read
subjects:
  - transformers
  - attention
---""")

    mock_env["embed_spy"].reset_mock()
    plan = ledger.reconcile(lib_root)
    assert len(plan["unchanged"]) == 1
    assert plan["unchanged"][0]["metadata_changed"] is True

    cli.cmd_index(argparse.Namespace())
    # KEY INVARIANT: 0 embedding calls!
    assert mock_env["embed_spy"].call_count == 0
    assert mock_env["mock_client"].set_payload.called

    state = ledger._load(lib_root)
    doc = next(iter(state["documents"].values()))
    assert doc["metadata"]["title"] == "Attention Is All You Need: Groundbreaking Architecture"
    assert doc["metadata"]["subjects"] == ["transformers", "attention"]


def test_scenario_07_deletion_purges_vectors_and_ghosts(mock_env):
    """Scenario 7: Deletion purges vectors and manifest record, including ghost papers/retrieval-primer.pdf."""
    lib_root = mock_env["lib_root"]
    ledger_file = mock_env["ledger_file"]

    # Preload v1 ledger with real file + stale ghost
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "active.txt").write_text("Active file content currently stored in the library repository." * 5)

    v1_state = {
        "files": {
            "_Inbox/active.txt": "activehash123",
            "_meta:_Inbox/active.txt": {"chunks": 2},
            "papers/retrieval-primer.pdf": "ghosthash999",
            "_meta:papers/retrieval-primer.pdf": {"chunks": 1},
        }
    }
    ledger_file.write_text(json.dumps(v1_state))

    plan = ledger.reconcile(lib_root)
    deleted_paths = [d["path"] for d in plan["deleted"]]
    assert "papers/retrieval-primer.pdf" in deleted_paths

    mock_env["embed_spy"].reset_mock()
    cli.cmd_index(argparse.Namespace())
    assert mock_env["mock_client"].delete.called

    state = ledger._load(lib_root)
    assert "doc_ghosthash999" not in state["documents"]


def test_scenario_08_same_filename_different_content(mock_env):
    """Scenario 8: Same filename with different content is classified as CHANGED."""
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "paper.txt").write_text("Alpha version of document text that will be replaced." * 5)

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1

    # Overwrite file with new content
    (inbox / "paper.txt").write_text("Beta completely different content replacing previous paper." * 5)
    mock_env["embed_spy"].reset_mock()

    plan = ledger.reconcile(lib_root)
    assert len(plan["changed"]) == 1
    assert len(plan["new"]) == 0
    assert len(plan["deleted"]) == 0

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1


def test_scenario_09_same_content_new_path(mock_env):
    """Scenario 9: Same content at a new path is detected as move and preserves vectors."""
    lib_root = mock_env["lib_root"]
    content = "Unique content text describing database index topologies and B-trees." * 5
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "draft.txt").write_text(content)

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1

    (inbox / "draft.txt").unlink()
    books = lib_root / "Books" / "Database_2021_Internals"
    books.mkdir(parents=True)
    (books / "book.txt").write_text(content)

    mock_env["embed_spy"].reset_mock()
    plan = ledger.reconcile(lib_root)
    assert len(plan["moved"]) == 1

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 0


def test_scenario_10_scope_filtering(mock_env, monkeypatch):
    """Scenario 10: Scope filtering applies Qdrant payload filters for inbox and catalogued."""
    mock_client = mock_env["mock_client"]
    mock_client.query_points.return_value = MagicMock(points=[])

    # Query with default scope (all)
    store.query_places("test question", scope="all")
    args_all = mock_client.query_points.call_args[1]
    assert args_all["query_filter"] is None

    # Query with inbox scope
    store.query_places("test question", scope="inbox")
    args_inbox = mock_client.query_points.call_args[1]
    assert args_inbox["query_filter"] is not None

    # Query with catalogued scope
    store.query_places("test question", scope="catalogued")
    args_cat = mock_client.query_points.call_args[1]
    assert args_cat["query_filter"] is not None


def test_scenario_11_collections_ignored(mock_env):
    """Scenario 11: Collections directory files are strictly ignored and never embedded."""
    lib_root = mock_env["lib_root"]
    collections = lib_root / "Collections"
    collections.mkdir()
    (collections / "README.md").write_text("Collections Overview Guide.")
    (collections / "Machine-Learning.md").write_text("List of recommended ML literature.")

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 0
    state = ledger._load(lib_root)
    assert len(state["documents"]) == 0


def test_scenario_12_catalog_bib_ignored_as_content(mock_env):
    """Scenario 12: catalog.bib is strictly ignored as vector content."""
    lib_root = mock_env["lib_root"]
    (lib_root / "catalog.bib").write_text("""@book{Knuth_1997_TAOCP,
  title = {The Art of Computer Programming},
  author = {Donald E. Knuth},
  year = {1997}
}""")

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 0
    state = ledger._load(lib_root)
    assert len(state["documents"]) == 0


def test_scenario_13_metadata_enrichment_sources(mock_env):
    """Scenario 13: catalog facts override README bibliography; local fields remain."""
    lib_root = mock_env["lib_root"]
    book_dir = lib_root / "Books" / "Knuth_1997_TAOCP"
    book_dir.mkdir(parents=True)
    (book_dir / "book.txt").write_text("Fundamental algorithms analysis and data structures." * 5)

    (book_dir / "README.md").write_text("""---
title: "The Art of Computer Programming, Vol. 1"
authors:
  - Donald E. Knuth
year: 1997
subjects:
  - algorithms
  - computer-science
identifier:
  isbn: "9780201896831"
status: read
---""")

    (lib_root / "catalog.bib").write_text("""@book{Knuth_1997_TAOCP,
  title = {Catalog Title},
  doi = {10.1000/182}
}""")

    cli.cmd_index(argparse.Namespace())
    state = ledger._load(lib_root)
    doc = next(iter(state["documents"].values()))
    assert doc["metadata"]["title"] == "Catalog Title"
    assert doc["metadata"]["authors"] == ["Donald E. Knuth"]
    assert doc["metadata"]["year"] == 1997
    assert doc["metadata"]["subjects"] == ["algorithms", "computer-science"]
    # Identifier subfields resolve individually: the catalog DOI wins its own
    # key while the independent local ISBN survives the fallback merge.
    assert doc["metadata"]["identifier"] == {"isbn": "9780201896831", "doi": "10.1000/182"}


def test_scenario_14_fallback_folder_parsing(mock_env):
    """Scenario 14: Fallback folder-name parsing for complex and malformed names."""
    lib_root = mock_env["lib_root"]

    # Complex name: García_Márquez_1967_Cien-Años-de-Soledad
    dir1 = lib_root / "Books" / "García_Márquez_1967_Cien-Años-de-Soledad"
    dir1.mkdir(parents=True)
    (dir1 / "book.txt").write_text("Muchos años después, frente al pelotón de fusilamiento..." * 5)

    # Malformed name: Algorithms-Handbook
    dir2 = lib_root / "Books" / "Algorithms-Handbook"
    dir2.mkdir(parents=True)
    (dir2 / "book.txt").write_text("Sorting and graph algorithms handbook reference text." * 5)

    cli.cmd_index(argparse.Namespace())
    state = ledger._load(lib_root)
    docs_by_path = {d["path"]: d for d in state["documents"].values()}

    doc1 = docs_by_path["Books/García_Márquez_1967_Cien-Años-de-Soledad/book.txt"]
    assert doc1["metadata"]["authors"] == ["García Márquez"]
    assert doc1["metadata"]["year"] == 1967
    assert doc1["metadata"]["title"] == "Cien Años de Soledad"

    doc2 = docs_by_path["Books/Algorithms-Handbook/book.txt"]
    assert doc2["metadata"]["authors"] is None
    assert doc2["metadata"]["year"] is None
    assert doc2["metadata"]["title"] == "Algorithms Handbook"


def test_scenario_15_move_and_replace_same_path(mock_env):
    """Scenario 15: A document vacates path P while unrelated new content lands at P.

    The relocated document must keep its vectors (MOVED) and the arriving content must
    classify as NEW, never CHANGED — a CHANGED classification would delete the very
    point IDs the move just preserved.
    """
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    original = "Relocating document about compiler construction and parsing theory." * 5
    (inbox / "slot.txt").write_text(original)

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1
    moved_doc_id = next(iter(ledger._load(lib_root)["documents"].keys()))
    preserved_point_ids = ledger._load(lib_root)["documents"][moved_doc_id]["point_ids"]

    # In one pass: the original moves out of _Inbox/slot.txt, and unrelated new
    # content takes over that exact path.
    book_dir = lib_root / "Books" / "Aho_1986_Dragon-Book"
    book_dir.mkdir(parents=True)
    (book_dir / "book.txt").write_text(original)
    (inbox / "slot.txt").write_text("Totally unrelated arrival about marine biology and coral reefs." * 5)

    mock_env["embed_spy"].reset_mock()
    plan = ledger.reconcile(lib_root)

    assert len(plan["moved"]) == 1
    assert plan["moved"][0]["doc_id"] == moved_doc_id
    assert plan["moved"][0]["new_path"] == "Books/Aho_1986_Dragon-Book/book.txt"
    # The arrival is NEW, not CHANGED: nothing is scheduled for deletion.
    assert len(plan["new"]) == 1
    assert plan["new"][0]["path"] == "_Inbox/slot.txt"
    assert len(plan["changed"]) == 0
    assert len(plan["deleted"]) == 0

    cli.cmd_index(argparse.Namespace())
    # Only the arriving document is embedded; the moved one is not re-embedded.
    assert mock_env["embed_spy"].call_count == 1

    state = ledger._load(lib_root)
    assert state["documents"][moved_doc_id]["path"] == "Books/Aho_1986_Dragon-Book/book.txt"
    # The moved document's vectors survived untouched.
    assert state["documents"][moved_doc_id]["point_ids"] == preserved_point_ids
    deleted_ids = [
        pid
        for call in mock_env["mock_client"].delete.call_args_list
        for pid in getattr(call[1].get("points_selector"), "points", [])
    ]
    assert not set(preserved_point_ids) & set(deleted_ids)


def test_scenario_16_epub_indexing(mock_env, make_epub):
    """Scenario 16: Indexing an EPUB extracts, chunks, and preserves section anchor in payload."""
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    make_epub(
        inbox / "book.epub",
        title="The Art of Computer Programming",
        author="Donald E. Knuth",
        sections=[
            "Section 1: Fundamental algorithms and foundational mathematical concepts for computer science." * 3,
            "Section 2: Information structures, linear lists, trees, and multilinked structures." * 3,
        ],
    )

    cli.cmd_index(argparse.Namespace())
    assert mock_env["embed_spy"].call_count == 1
    state = ledger._load(lib_root)
    assert len(state["documents"]) == 1
    doc = next(iter(state["documents"].values()))
    assert doc["scope"] == "inbox"
    assert doc["type"] == "inbox"
    assert doc["path"] == "_Inbox/book.epub"
    assert doc["metadata"]["title"] == "The Art of Computer Programming"
    assert doc["metadata"]["authors"] == ["Donald E. Knuth"]

    # Verify upserted points payload contains the 1-based section anchor
    upsert_calls = mock_env["mock_client"].upsert.call_args_list
    assert len(upsert_calls) == 1
    upserted_points = upsert_calls[0][1]["points"]
    assert len(upserted_points) == 1  # 2 sections merged into 1 chunk or depending on chunk size
    assert upserted_points[0].payload["page"] == 1


def test_scenario_17_unsupported_configured_extension_never_produces_ledger_entry(mock_env, monkeypatch):
    """Scenario 17: An unsupported extension added to SUPPORTED_EXTENSIONS fails fast before ledger or Qdrant mutation."""
    from library_query import errors

    monkeypatch.setattr(config, "SUPPORTED_EXTENSIONS", {".pdf", ".epub", ".md", ".txt", ".odt"})
    lib_root = mock_env["lib_root"]
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "document.odt").write_bytes(b"PK\x03\x04 fake odt binary content")

    with pytest.raises(errors.UnsupportedFormatError) as exc_info:
        cli.cmd_index(argparse.Namespace())

    assert ".odt" in exc_info.value.message
    # Verify no embeddings and no ledger entries were created
    assert mock_env["embed_spy"].call_count == 0
    assert not mock_env["ledger_file"].exists()
