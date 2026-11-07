"""Tests for explicit note synchronization through the shared inventory.

Covers the modified zotero-integration capability: mapped items use
``Notes/<item_id>-zotero.md``, legacy item directories keep their existing
``zotero-notes.md`` location, unmigrated legacy copies block duplicate
creation, manual files are protected, and dry runs never write.
"""
import json
from pathlib import Path

import pytest

from library_query import items, ledger, structure, zotero


def _mapped_item(root: Path, item_id: str = "item-0123456789abcdef0123456789abcdef", citekey: str = "Mapped2024") -> Path:
    record = root / "Metadata" / f"{item_id}.md"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        "---\n"
        "schema_version: 1\n"
        f"item_id: {item_id}\n"
        "type: book\n"
        f"citekey: {citekey}\n"
        "attachments:\n"
        "  - path: Books/flat-book.pdf\n"
        "    role: source\n"
        "---\n"
    )
    (root / "Books").mkdir(exist_ok=True)
    (root / "Books" / "flat-book.pdf").write_bytes(b"flat source")
    return record


def _legacy_item(root: Path, folder: str = "Legacy_2024_Item") -> Path:
    item_dir = root / "Books" / folder
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "book.pdf").write_bytes(b"legacy source")
    (root / "catalog.bib").write_text(
        f"""@book{{{folder.replace('_', '')},
  title = {{Legacy Item}},
  year = {{2024}},
  file = {{Books/{folder}/book.pdf}}
}}
"""
    )
    return item_dir


def test_mapped_item_syncs_to_generated_note_path(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    _mapped_item(root)
    monkeypatch.setattr(
        zotero, "fetch_notes",
        lambda citekeys: {"Mapped2024": [{"note": "<p>Mapped note text</p>"}]},
    )

    summary = zotero.sync_notes(root)

    target = root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md"
    assert summary["updated"] == 1
    assert target.is_file()
    assert target.read_text(encoding="utf-8").startswith(zotero.GENERATED_MARKER)
    assert "Mapped note text" in target.read_text(encoding="utf-8")
    # The generated note resolves catalogued scope and the item's shelf type.
    inventory = items.build_inventory(root)
    assert structure.get_scope_and_type(target, root, inventory) == ("catalogued", "book")


def test_multi_format_item_creates_one_note(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    record = root / "Metadata" / "item-0123456789abcdef0123456789abcdef.md"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        "---\nschema_version: 1\nitem_id: item-0123456789abcdef0123456789abcdef\n"
        "type: book\ncitekey: Dual2024\nattachments:\n"
        "  - path: Books/dual.pdf\n    role: source\n"
        "  - path: Books/dual.epub\n    role: source\n---\n"
    )
    (root / "Books").mkdir(exist_ok=True)
    (root / "Books" / "dual.pdf").write_bytes(b"pdf")
    (root / "Books" / "dual.epub").write_bytes(b"epub")
    fetched = {}

    def fake_fetch(citekeys):
        fetched["keys"] = citekeys
        return {"Dual2024": [{"note": "<p>one</p>"}]}

    monkeypatch.setattr(zotero, "fetch_notes", fake_fetch)

    summary = zotero.sync_notes(root)

    assert fetched["keys"] == ["Dual2024"]  # citekeys deduplicated for the fetch
    assert summary["items"] == 1
    assert len(list((root / "Notes").glob("*.md"))) == 1


def test_legacy_item_keeps_existing_note_location(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    item_dir = _legacy_item(root)
    monkeypatch.setattr(
        zotero, "fetch_notes",
        lambda citekeys: {"Legacy2024Item": [{"note": "<p>legacy note</p>"}]},
    )

    summary = zotero.sync_notes(root)

    target = item_dir / zotero.GENERATED_FILENAME
    assert summary["updated"] == 1
    assert target.is_file()
    assert not (root / "Notes").exists()


def test_manual_file_protected_at_mapped_and_legacy_targets(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    _mapped_item(root)
    mapped_target = root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md"
    mapped_target.parent.mkdir(parents=True)
    mapped_target.write_text("manual mapped notes")
    item_dir = _legacy_item(root)
    legacy_target = item_dir / zotero.GENERATED_FILENAME
    legacy_target.write_text("manual legacy notes")
    monkeypatch.setattr(
        zotero, "fetch_notes", lambda citekeys: {k: [{"note": "<p>x</p>"}] for k in citekeys}
    )

    summary = zotero.sync_notes(root)

    assert sorted(summary["conflicts"]) == sorted(
        ["Notes/item-0123456789abcdef0123456789abcdef-zotero.md",
         "Books/Legacy_2024_Item/zotero-notes.md"]
    )
    assert mapped_target.read_text(encoding="utf-8") == "manual mapped notes"
    assert legacy_target.read_text(encoding="utf-8") == "manual legacy notes"


def test_unmigrated_legacy_copy_blocks_duplicate_creation(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    # Mapped item whose flat source bytes equal the legacy copy's item: the
    # legacy directory still exists with its generated note.
    item_dir = root / "Books" / "Mapped_2024_Item"
    item_dir.mkdir(parents=True)
    (item_dir / "book.pdf").write_bytes(b"legacy source")
    (item_dir / zotero.GENERATED_FILENAME).write_text(
        f"{zotero.GENERATED_MARKER}\n<!-- citekey: Mapped2024 -->\n\nold note\n"
    )
    (root / "catalog.bib").write_text(
        """@book{Mapped2024,
  title = {Mapped},
  year = {2024},
  file = {Books/Mapped_2024_Item/book.pdf}
}
"""
    )
    record = root / "Metadata" / "item-0123456789abcdef0123456789abcdef.md"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        "---\nschema_version: 1\nitem_id: item-0123456789abcdef0123456789abcdef\n"
        "type: book\ncitekey: Mapped2024\nattachments:\n"
        "  - path: Books/mapped-flat.pdf\n    role: source\n---\n"
    )
    (root / "Books" / "mapped-flat.pdf").write_bytes(b"flat source")
    monkeypatch.setattr(
        zotero, "fetch_notes",
        lambda citekeys: {"Mapped2024": [{"note": "<p>new text</p>"}]},
    )

    summary = zotero.sync_notes(root)

    assert any("unmigrated" in c for c in summary["conflicts"])
    assert not (root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md").exists()
    assert (item_dir / zotero.GENERATED_FILENAME).is_file()
    assert "old note" in (item_dir / zotero.GENERATED_FILENAME).read_text(encoding="utf-8")


def test_unmigrated_copy_detected_when_record_claims_legacy_paths(tmp_path: Path, monkeypatch):
    """A record may keep pointing at legacy paths during transition; the stray
    generated note in that directory must still block duplicate creation."""
    root = tmp_path / "library"
    root.mkdir()
    item_dir = root / "Books" / "Mapped_2024_Item"
    item_dir.mkdir(parents=True)
    (item_dir / "book.pdf").write_bytes(b"legacy source")
    (item_dir / zotero.GENERATED_FILENAME).write_text(
        f"{zotero.GENERATED_MARKER}\n<!-- citekey: Mapped2024 -->\n\nold note\n"
    )
    (root / "catalog.bib").write_text(
        "@book{Mapped2024,\n title = {Mapped},\n year = {2024},\n file = {Books/Mapped_2024_Item/book.pdf}\n}\n"
    )
    record = root / "Metadata" / "item-0123456789abcdef0123456789abcdef.md"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        "---\nschema_version: 1\nitem_id: item-0123456789abcdef0123456789abcdef\n"
        "type: book\nattachments:\n  - path: Books/Mapped_2024_Item/book.pdf\n    role: source\n---\n"
    )
    monkeypatch.setattr(
        zotero, "fetch_notes",
        lambda citekeys: {"Mapped2024": [{"note": "<p>new text</p>"}]},
    )

    summary = zotero.sync_notes(root)

    assert any("unmigrated" in c for c in summary["conflicts"])
    assert not (root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md").exists()
    assert "old note" in (item_dir / zotero.GENERATED_FILENAME).read_text(encoding="utf-8")


def test_unmigrated_conflict_also_when_target_already_exists(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    item_dir = root / "Books" / "Mapped_2024_Item"
    item_dir.mkdir(parents=True)
    (item_dir / "book.pdf").write_bytes(b"legacy source")
    (item_dir / zotero.GENERATED_FILENAME).write_text(
        f"{zotero.GENERATED_MARKER}\n<!-- citekey: Mapped2024 -->\n\nold note\n"
    )
    (root / "catalog.bib").write_text(
        "@book{Mapped2024,\n title = {Mapped},\n year = {2024},\n file = {Books/Mapped_2024_Item/book.pdf}\n}\n"
    )
    _mapped_item(root)
    existing_target = root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md"
    existing_target.parent.mkdir(parents=True, exist_ok=True)
    existing_target.write_text(f"{zotero.GENERATED_MARKER}\n<!-- citekey: Mapped2024 -->\n\nnew note\n")
    before = existing_target.read_text(encoding="utf-8")
    monkeypatch.setattr(
        zotero, "fetch_notes",
        lambda citekeys: {"Mapped2024": [{"note": "<p>changed</p>"}]},
    )

    summary = zotero.sync_notes(root)

    assert any("unmigrated" in c for c in summary["conflicts"])
    # Neither location was modified: no silent removal, no overwrite.
    assert existing_target.read_text(encoding="utf-8") == before
    assert "old note" in (item_dir / zotero.GENERATED_FILENAME).read_text(encoding="utf-8")


def test_missing_citekey_reported_without_guessing(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    record = root / "Metadata" / "item-0123456789abcdef0123456789abcdef.md"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        "---\nschema_version: 1\nitem_id: item-0123456789abcdef0123456789abcdef\n"
        "type: book\nattachments:\n  - path: Books/none.pdf\n    role: source\n---\n"
    )
    monkeypatch.setattr(zotero, "fetch_notes", lambda citekeys: {})

    summary = zotero.sync_notes(root)

    assert summary["skipped_without_citekey"] == 1
    assert summary["updated"] == 0
    assert not (root / "Notes").exists()


def test_empty_result_removes_only_owned_generated_files(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    _mapped_item(root)
    target = root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{zotero.GENERATED_MARKER}\n<!-- citekey: Mapped2024 -->\n\nstale\n")
    monkeypatch.setattr(zotero, "fetch_notes", lambda citekeys: {"Mapped2024": []})

    summary = zotero.sync_notes(root)

    assert summary["removed"] == 1
    assert not target.exists()


def test_endpoint_failure_preserves_all_note_files(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    _mapped_item(root)
    legacy_dir = _legacy_item(root)
    target = root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{zotero.GENERATED_MARKER}\n<!-- citekey: Mapped2024 -->\n\nkeep\n")
    legacy_target = legacy_dir / zotero.GENERATED_FILENAME
    legacy_target.write_text("manual keep me")

    def fail(citekeys):
        raise zotero.errors.ZoteroUnreachableError("endpoint down")

    monkeypatch.setattr(zotero, "fetch_notes", fail)

    with pytest.raises(zotero.errors.ZoteroUnreachableError):
        zotero.sync_notes(root)
    assert target.read_text(encoding="utf-8").endswith("keep\n")
    assert legacy_target.read_text(encoding="utf-8") == "manual keep me"


def test_dry_run_creates_nothing(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    _mapped_item(root)
    monkeypatch.setattr(
        zotero, "fetch_notes",
        lambda citekeys: {"Mapped2024": [{"note": "<p>would write</p>"}]},
    )

    summary = zotero.sync_notes(root, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["updated"] == 1
    assert not (root / "Notes").exists()


def test_sync_never_touches_ledger_state(tmp_path: Path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    _mapped_item(root)
    monkeypatch.setattr(
        zotero, "fetch_notes",
        lambda citekeys: {"Mapped2024": [{"note": "<p>x</p>"}]},
    )
    for forbidden in ("scan_content_paths", "scan_content_files", "_load"):
        monkeypatch.setattr(
            ledger, forbidden,
            lambda *a, **k: pytest.fail(f"sync_notes must not call ledger.{forbidden}"),
        )

    zotero.sync_notes(root)
    assert (root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md").is_file()
