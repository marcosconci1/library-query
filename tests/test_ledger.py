"""Tests for ledger v2 schema, auto-migration, stats, conflicts, and durability."""
import json
from pathlib import Path

import pytest

from library_query import config, ledger


def test_v1_to_v2_migration(tmp_path: Path, monkeypatch):
    ledger_file = tmp_path / "ledger.json"
    monkeypatch.setattr(config, "LEDGER_PATH", ledger_file)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "LIBRARY_ROOT", tmp_path / "library")

    lib_root = tmp_path / "library"
    lib_root.mkdir()
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "sample.pdf").write_text("dummy sample pdf")

    # Write a v1 ledger fixture with a valid file and a stale ghost file
    v1_data = {
        "files": {
            "_Inbox/sample.pdf": "abc123hash",
            "_meta:_Inbox/sample.pdf": {"chunks": 5},
            "papers/retrieval-primer.pdf": "deadbeefhash",
            "_meta:papers/retrieval-primer.pdf": {"chunks": 1},
        }
    }
    ledger_file.write_text(json.dumps(v1_data))

    # Loading triggers auto-migration
    state = ledger._load(root=lib_root)
    assert state["version"] == 2
    assert "documents" in state

    # Check sample.pdf migrated entry
    doc_sample = state["documents"]["doc_abc123hash"]
    assert doc_sample["content_hash"] == "abc123hash"
    assert doc_sample["path"] == "_Inbox/sample.pdf"
    assert doc_sample["scope"] == "inbox"
    assert doc_sample["type"] == "inbox"
    assert doc_sample["chunks"] == 5
    assert len(doc_sample["point_ids"]) == 5

    # Check stale ghost migrated entry
    doc_ghost = state["documents"]["doc_deadbeefhash"]
    assert doc_ghost["content_hash"] == "deadbeefhash"
    assert doc_ghost["path"] == "papers/retrieval-primer.pdf"
    assert len(doc_ghost["point_ids"]) == 1

    # On reconciliation, ghost is flagged as DELETED because it's not on disk
    plan = ledger.reconcile(root=lib_root)
    deleted_paths = [d["path"] for d in plan["deleted"]]
    assert "papers/retrieval-primer.pdf" in deleted_paths


def test_ledger_stats_breakdown(tmp_path: Path, monkeypatch):
    ledger_file = tmp_path / "ledger.json"
    monkeypatch.setattr(config, "LEDGER_PATH", ledger_file)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "LIBRARY_ROOT", tmp_path / "library")

    lib_root = tmp_path / "library"
    lib_root.mkdir()
    inbox = lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "file1.pdf").write_text("inbox 1")
    (inbox / "file2.pdf").write_text("inbox 2")

    book_dir = lib_root / "Books" / "Author_2020_Book"
    book_dir.mkdir(parents=True)
    (book_dir / "book.pdf").write_text("book content")

    # Populate ledger with 1 inbox doc and 1 catalogued doc
    v2_state = {
        "version": 2,
        "documents": {
            "doc_1": {
                "content_hash": "hash1",
                "path": "_Inbox/file1.pdf",
                "scope": "inbox",
                "type": "inbox",
                "metadata": {},
                "chunks": 2,
                "point_ids": [1, 2],
            },
            "doc_2": {
                "content_hash": "hash2",
                "path": "Books/Author_2020_Book/book.pdf",
                "scope": "catalogued",
                "type": "book",
                "metadata": {},
                "chunks": 3,
                "point_ids": [3, 4, 5],
            },
        },
    }
    ledger_file.write_text(json.dumps(v2_state))

    st = ledger.stats(root=lib_root)
    assert st["files_in_library"] == 3
    assert st["indexed"] == 2
    assert st["inbox"] == 1
    assert st["catalogued"] == 1
    # file2.pdf is pending
    assert st["pending"] >= 1


# --- Duplicate-content conflicts (2026-09-23) ---------------------------------------
#
# Content hash is document identity, so the same bytes at two paths makes "where is
# this document?" unanswerable. Reconciliation used to answer it anyway by rebinding
# the manifest to whichever path sorted last, which orphaned the original file (still
# on disk, no longer tracked, its Qdrant payload now describing the other folder) and
# made the pair oscillate between the two locations on every subsequent pass.


def _temp_library(tmp_path: Path, monkeypatch) -> Path:
    """A library root with the ledger redirected into the same tmp dir."""
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    lib_root = tmp_path / "library"
    (lib_root / "Books").mkdir(parents=True)
    monkeypatch.setattr(config, "LIBRARY_ROOT", lib_root)
    return lib_root


def _write_book(lib_root: Path, folder: str, name: str, body: str) -> Path:
    book_dir = lib_root / "Books" / folder
    book_dir.mkdir(parents=True, exist_ok=True)
    path = book_dir / name
    path.write_text(body)
    return path


def test_duplicate_content_is_reported_as_a_conflict(tmp_path: Path, monkeypatch):
    lib_root = _temp_library(tmp_path, monkeypatch)
    first = _write_book(lib_root, "C_2020_Dup", "one.txt", "identical bytes")

    # Index the first copy.
    plan = ledger.reconcile(root=lib_root)
    assert len(plan["new"]) == 1
    assert plan["conflicts"] == []
    ledger.mark_done(first, chunk_count=1, root=lib_root)

    # A byte-identical second copy appears at a different path.
    _write_book(lib_root, "D_2020_Dup", "two.txt", "identical bytes")
    plan = ledger.reconcile(root=lib_root)

    assert len(plan["conflicts"]) == 1, "duplicate content was not reported"
    conflict = plan["conflicts"][0]
    assert conflict["reason"] == "duplicate_content"
    assert sorted(conflict["paths"]) == [
        "Books/C_2020_Dup/one.txt",
        "Books/D_2020_Dup/two.txt",
    ]
    assert conflict["recorded_path"] == "Books/C_2020_Dup/one.txt"

    # No move is planned, and the original binding is untouched.
    assert plan["moved"] == [], "a duplicate was silently rebound as a move"
    assert plan["deleted"] == [], "the original was dropped from the manifest"


def test_duplicate_conflict_does_not_rebind_the_manifest(tmp_path: Path, monkeypatch):
    """The recorded path survives the pass, so nothing oscillates."""
    lib_root = _temp_library(tmp_path, monkeypatch)
    first = _write_book(lib_root, "C_2020_Dup", "one.txt", "identical bytes")
    ledger.reconcile(root=lib_root)
    ledger.mark_done(first, chunk_count=1, root=lib_root)

    _write_book(lib_root, "D_2020_Dup", "two.txt", "identical bytes")

    recorded_before = ledger._load(root=lib_root)["documents"]
    for _ in range(3):
        plan = ledger.reconcile(root=lib_root)
        assert len(plan["conflicts"]) == 1
        assert plan["moved"] == []
        assert plan["deleted"] == []

    assert ledger._load(root=lib_root)["documents"] == recorded_before, (
        "reconciliation mutated the manifest for a conflicted hash"
    )


def test_resolving_the_duplicate_restores_normal_move_detection(tmp_path: Path, monkeypatch):
    """Deleting one copy makes the hash unambiguous again; MOVED resumes."""
    lib_root = _temp_library(tmp_path, monkeypatch)
    first = _write_book(lib_root, "C_2020_Dup", "one.txt", "identical bytes")
    ledger.reconcile(root=lib_root)
    ledger.mark_done(first, chunk_count=1, root=lib_root)

    second = _write_book(lib_root, "D_2020_Dup", "two.txt", "identical bytes")
    assert len(ledger.reconcile(root=lib_root)["conflicts"]) == 1

    # Remove the original: the content now lives only at D. That is a real move.
    first.unlink()
    plan = ledger.reconcile(root=lib_root)

    assert plan["conflicts"] == []
    assert len(plan["moved"]) == 1
    assert plan["moved"][0]["new_path"] == "Books/D_2020_Dup/two.txt"
    assert second.exists()


def test_scan_content_paths_keeps_every_path_for_a_hash(tmp_path: Path, monkeypatch):
    lib_root = _temp_library(tmp_path, monkeypatch)
    _write_book(lib_root, "C_2020_Dup", "one.txt", "identical bytes")
    _write_book(lib_root, "D_2020_Dup", "two.txt", "identical bytes")
    _write_book(lib_root, "E_2020_Uniq", "three.txt", "different bytes")

    by_hash = ledger.scan_content_paths(lib_root)

    assert sorted(len(paths) for paths in by_hash.values()) == [1, 2]
    # The lossy form still answers one path per hash, deterministically the first.
    collapsed = ledger.scan_content_files(lib_root)
    assert set(collapsed) == set(by_hash)
    for digest, path in collapsed.items():
        assert path == by_hash[digest][0]


# --- (size, mtime) hash cache -------------------------------------------------------


def _count_hashes(monkeypatch) -> list[Path]:
    """Record every path `scan_content_paths` actually reads bytes from."""
    hashed: list[Path] = []
    real = ledger.file_hash

    def spy(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(ledger, "file_hash", spy)
    return hashed


def test_second_scan_does_not_rehash_unchanged_files(tmp_path: Path, monkeypatch):
    lib_root = _temp_library(tmp_path, monkeypatch)
    _write_book(lib_root, "A_2020_One", "one.txt", "first body")
    _write_book(lib_root, "B_2020_Two", "two.txt", "second body")

    hashed = _count_hashes(monkeypatch)

    first = ledger.scan_content_paths(lib_root)
    assert len(hashed) == 2, "cold scan must hash every content file"

    second = ledger.scan_content_paths(lib_root)
    assert len(hashed) == 2, "warm scan must not read any unchanged file"
    assert second == first


def test_hash_cache_is_persisted_in_the_ledger(tmp_path: Path, monkeypatch):
    lib_root = _temp_library(tmp_path, monkeypatch)
    path = _write_book(lib_root, "A_2020_One", "one.txt", "first body")

    ledger.scan_content_paths(lib_root)

    cache = json.loads(config.LEDGER_PATH.read_text())["hash_cache"]
    entry = cache["Books/A_2020_One/one.txt"]
    info = path.stat()
    assert entry["size"] == info.st_size
    assert entry["mtime_ns"] == info.st_mtime_ns
    assert entry["content_hash"] == ledger.file_hash(path)


def test_changed_file_is_rehashed_and_the_cache_updated(tmp_path: Path, monkeypatch):
    lib_root = _temp_library(tmp_path, monkeypatch)
    path = _write_book(lib_root, "A_2020_One", "one.txt", "first body")
    ledger.scan_content_paths(lib_root)

    hashed = _count_hashes(monkeypatch)
    path.write_text("a different body entirely")
    after = ledger.scan_content_paths(lib_root)

    assert hashed == [path], "a file whose (size, mtime) moved must be re-read"
    assert list(after) == [ledger.file_hash(path)]
    cache = json.loads(config.LEDGER_PATH.read_text())["hash_cache"]
    assert cache["Books/A_2020_One/one.txt"]["content_hash"] == ledger.file_hash(path)


def test_new_file_is_hashed_while_known_files_are_not(tmp_path: Path, monkeypatch):
    lib_root = _temp_library(tmp_path, monkeypatch)
    _write_book(lib_root, "A_2020_One", "one.txt", "first body")
    ledger.scan_content_paths(lib_root)

    hashed = _count_hashes(monkeypatch)
    fresh = _write_book(lib_root, "B_2020_Two", "two.txt", "second body")
    ledger.scan_content_paths(lib_root)

    assert hashed == [fresh]


def test_deleted_files_fall_out_of_the_hash_cache(tmp_path: Path, monkeypatch):
    lib_root = _temp_library(tmp_path, monkeypatch)
    _write_book(lib_root, "A_2020_One", "one.txt", "first body")
    doomed = _write_book(lib_root, "B_2020_Two", "two.txt", "second body")
    ledger.scan_content_paths(lib_root)

    doomed.unlink()
    ledger.scan_content_paths(lib_root)

    cache = json.loads(config.LEDGER_PATH.read_text())["hash_cache"]
    assert set(cache) == {"Books/A_2020_One/one.txt"}


def test_hash_cache_survives_mark_done(tmp_path: Path, monkeypatch):
    """`mark_done` rewrites the manifest; it must not drop the cache alongside it."""
    lib_root = _temp_library(tmp_path, monkeypatch)
    path = _write_book(lib_root, "A_2020_One", "one.txt", "first body")
    ledger.scan_content_paths(lib_root)

    ledger.mark_done(path, chunk_count=1, root=lib_root)

    hashed = _count_hashes(monkeypatch)
    ledger.scan_content_paths(lib_root)
    assert hashed == []


# --- Ledger durability --------------------------------------------------------------


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path: Path, monkeypatch):
    lib_root = _temp_library(tmp_path, monkeypatch)
    state = {"version": 2, "documents": {"doc_a": {"content_hash": "a", "path": "x"}}}

    ledger._save(state)

    assert json.loads(config.LEDGER_PATH.read_text()) == state
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert leftovers == [], f"temp files left behind: {leftovers}"
    assert lib_root.exists()


def test_save_does_not_truncate_the_old_ledger_when_serialization_fails(
    tmp_path: Path, monkeypatch
):
    """A failed write must leave the previous manifest intact, not a truncated file."""
    _temp_library(tmp_path, monkeypatch)
    good = {"version": 2, "documents": {"doc_a": {"content_hash": "a", "path": "x"}}}
    ledger._save(good)

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        ledger._save({"version": 2, "documents": {"doc_b": Unserializable()}})

    assert json.loads(config.LEDGER_PATH.read_text()) == good
    assert [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name] == []


def test_corrupt_ledger_is_backed_up_before_falling_back_to_empty(
    tmp_path: Path, monkeypatch
):
    """An empty manifest re-embeds the whole corpus, so the evidence must survive."""
    lib_root = _temp_library(tmp_path, monkeypatch)
    corrupt_payload = '{"version": 2, "documents": {"doc_a": {"content_ha'
    config.LEDGER_PATH.write_text(corrupt_payload)

    state = ledger._load(root=lib_root)

    assert state == {"version": 2, "documents": {}}, "corrupt ledger did not fail safe"

    backups = sorted(tmp_path.glob("ledger.json.corrupt-*"))
    assert len(backups) == 1, f"corrupt ledger not preserved: {list(tmp_path.iterdir())}"
    assert backups[0].read_text() == corrupt_payload, "backup lost the original bytes"
    assert not config.LEDGER_PATH.exists(), "corrupt file was left in place"


def test_corrupt_backup_always_leaves_a_backup_and_clears_the_ledger(
    tmp_path: Path, monkeypatch
):
    """Two corruptions in a row must each fail safe."""
    lib_root = _temp_library(tmp_path, monkeypatch)

    config.LEDGER_PATH.write_text("{bad one")
    ledger._load(root=lib_root)
    assert sorted(tmp_path.glob("ledger.json.corrupt-*"))
    assert not config.LEDGER_PATH.exists()

    config.LEDGER_PATH.write_text("{bad two")
    ledger._load(root=lib_root)

    assert sorted(tmp_path.glob("ledger.json.corrupt-*")), "no backup after second pass"
    assert not config.LEDGER_PATH.exists()
