"""Tests for dual-layout classification, mapped-note types, and metadata precedence."""
from pathlib import Path

from library_query import items, structure


def _record(path: Path, item_id: str, record_type: str, attachments: str, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "schema_version: 1\n"
        f"item_id: {item_id}\n"
        f"type: {record_type}\n"
        f"{extra}"
        "attachments:\n"
        f"{attachments}"
        "---\n"
        "Local writing.\n"
    )


def test_role_separation_for_metadata_notes_and_shelf_navigation(tmp_path: Path):
    (tmp_path / "Books").mkdir()
    (tmp_path / "Papers").mkdir()
    (tmp_path / "Videos").mkdir()
    (tmp_path / "Notes").mkdir()
    (tmp_path / "Metadata").mkdir()
    (tmp_path / "Operations").mkdir()

    assert structure.classify_file(tmp_path / "Metadata" / "item-x.md", tmp_path) == "metadata"
    assert structure.classify_file(tmp_path / "Books" / "README.md", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / "Papers" / "README.md", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / "Videos" / "README.md", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / "Notes" / "README.md", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / "Notes" / "some-note.md", tmp_path) == "content"
    assert structure.classify_file(tmp_path / "Books" / "item" / "README.md", tmp_path) == "metadata"
    assert structure.classify_file(tmp_path / "Operations" / "README.md", tmp_path) == "metadata"
    assert structure.classify_file(tmp_path / "Operations" / "ZOTERO.md", tmp_path) == "content"


def test_flat_and_legacy_scope_type_derivation(tmp_path: Path):
    flat = tmp_path / "Papers" / "retrieval-study-2025.pdf"
    legacy = tmp_path / "Books" / "Author_2020_Title" / "book.pdf"
    assert structure.get_scope_and_type(flat, tmp_path) == ("catalogued", "paper")
    assert structure.get_scope_and_type(legacy, tmp_path) == ("catalogued", "book")


def test_mapped_note_inherits_item_type_and_unmapped_note_stays_other(tmp_path: Path):
    (tmp_path / "Papers").mkdir()
    (tmp_path / "Papers" / "study.pdf").write_bytes(b"study")
    (tmp_path / "Notes").mkdir()
    (tmp_path / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md").write_text("note")
    (tmp_path / "Notes" / "loose.md").write_text("loose")
    _record(
        tmp_path / "Metadata" / "item-0123456789abcdef0123456789abcdef.md",
        "item-0123456789abcdef0123456789abcdef",
        "paper",
        "  - path: Papers/study.pdf\n    role: source\n",
    )
    inventory = items.build_inventory(tmp_path)

    mapped = tmp_path / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md"
    loose = tmp_path / "Notes" / "loose.md"
    assert structure.get_scope_and_type(mapped, tmp_path, inventory) == ("catalogued", "paper")
    assert structure.get_scope_and_type(loose, tmp_path, inventory) == ("catalogued", "other")
    # Without the shared inventory the deterministic naming still resolves.
    assert structure.get_scope_and_type(mapped, tmp_path) == ("catalogued", "paper")
    assert any(w["reason"] == "unmapped_note" for w in inventory["warnings"])


def test_item_record_does_not_override_inbox_or_operations_scope(tmp_path: Path):
    (tmp_path / "_Inbox").mkdir()
    inbox_file = tmp_path / "_Inbox" / "draft.pdf"
    inbox_file.write_bytes(b"draft")
    _record(
        tmp_path / "Metadata" / "item-0123456789abcdef0123456789abcdef.md",
        "item-0123456789abcdef0123456789abcdef",
        "book",
        "  - path: _Inbox/draft.pdf\n    role: source\n".replace("_Inbox", "Books")
        + "",
    )
    # A record cannot even be written for an inbox path (shelf validation), and
    # scope derivation never consults records for inbox files.
    assert structure.get_scope_and_type(inbox_file, tmp_path) == ("inbox", "inbox")
    assert structure.get_scope_and_type(tmp_path / "Operations" / "guide.md", tmp_path) == (
        "operations",
        "operation",
    )


def test_record_metadata_precedence_and_empty_local_values(tmp_path: Path):
    (tmp_path / "Books").mkdir()
    (tmp_path / "Books" / "algorithms-handbook-2024.pdf").write_bytes(b"pdf")
    _record(
        tmp_path / "Metadata" / "item-0123456789abcdef0123456789abcdef.md",
        "item-0123456789abcdef0123456789abcdef",
        "book",
        "  - path: Books/algorithms-handbook-2024.pdf\n    role: source\n",
        extra="citekey: Algo2024\nsubjects: []\nidentifier:\n  url: https://example.local/video\n",
    )
    (tmp_path / "catalog.bib").write_text(
        """@book{Algo2024,
  title = {Corrected Catalog Title},
  author = {Catalog Author},
  year = {2024},
  keywords = {review, algorithms},
  doi = {10.1000/xyz}
}
"""
    )
    inventory = items.build_inventory(tmp_path)

    meta = structure.resolve_metadata(
        tmp_path / "Books" / "algorithms-handbook-2024.pdf", tmp_path, inventory=inventory
    )

    # Catalog wins bibliographic facts; the item ID attaches additively.
    assert meta["title"] == "Corrected Catalog Title"
    assert meta["authors"] == ["Catalog Author"]
    assert meta["item_id"] == "item-0123456789abcdef0123456789abcdef"
    # Explicit empty local subjects stay empty while Zotero tags survive.
    assert meta["subjects"] == []
    assert meta["tags"] == ["review", "algorithms"]
    # Identifier subfields merge individually.
    assert meta["identifier"] == {"url": "https://example.local/video", "doi": "10.1000/xyz"}
    assert meta["citekey"] == "Algo2024"


def test_shelf_readme_does_not_enrich_flat_sources(tmp_path: Path):
    (tmp_path / "Books").mkdir()
    flat = tmp_path / "Books" / "lonely.pdf"
    flat.write_bytes(b"lonely")
    (tmp_path / "Books" / "README.md").write_text(
        "---\ntitle: Shelf Navigation Title\nsubjects: [navigation]\n---\n"
    )

    meta = structure.resolve_metadata(flat, tmp_path)

    assert meta["title"] == "lonely"
    assert meta["subjects"] is None


def test_flat_filenames_are_not_parsed_as_item_directories(tmp_path: Path):
    (tmp_path / "Books").mkdir()
    flat = tmp_path / "Books" / "Author_2020_Something.pdf"
    flat.write_bytes(b"x")

    meta = structure.resolve_metadata(flat, tmp_path)

    # No folder parsing for a flat filename: title comes from the stem as-is.
    assert meta["title"] == "Author_2020_Something"
    assert meta["authors"] is None
    assert meta["year"] is None


def test_record_overrides_legacy_sidecar_for_local_metadata(tmp_path: Path):
    (tmp_path / "Books" / "Legacy_2020_Item").mkdir(parents=True)
    source = tmp_path / "Books" / "Legacy_2020_Item" / "book.pdf"
    source.write_bytes(b"legacy")
    (tmp_path / "Books" / "Legacy_2020_Item" / "README.md").write_text(
        "---\ntitle: Sidecar Title\nsubjects: [from-sidecar]\nstatus: unread\n---\n"
    )
    _record(
        tmp_path / "Metadata" / "item-0123456789abcdef0123456789abcdef.md",
        "item-0123456789abcdef0123456789abcdef",
        "book",
        "  - path: Books/Legacy_2020_Item/book.pdf\n    role: source\n",
        extra="subjects: [from-record]\n",
    )
    inventory = items.build_inventory(tmp_path)

    meta = structure.resolve_metadata(source, tmp_path, inventory=inventory)

    assert meta["subjects"] == ["from-record"]
    assert meta["item_id"] == "item-0123456789abcdef0123456789abcdef"


def test_unmapped_flat_source_never_inherits_neighbor_metadata(tmp_path: Path):
    (tmp_path / "Papers").mkdir()
    mapped = tmp_path / "Papers" / "mapped.pdf"
    mapped.write_bytes(b"mapped")
    orphan = tmp_path / "Papers" / "orphan.pdf"
    orphan.write_bytes(b"orphan")
    _record(
        tmp_path / "Metadata" / "item-0123456789abcdef0123456789abcdef.md",
        "item-0123456789abcdef0123456789abcdef",
        "paper",
        "  - path: Papers/mapped.pdf\n    role: source\n",
        extra="title: Mapped Paper\n",
    )
    inventory = items.build_inventory(tmp_path)

    meta = structure.resolve_metadata(orphan, tmp_path, inventory=inventory)

    assert meta["item_id"] is None
    assert meta["title"] == "orphan"
    assert "Papers/orphan.pdf" in inventory["unmapped"]
