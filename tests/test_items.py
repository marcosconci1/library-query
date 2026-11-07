"""Tests for portable item records, the shared inventory, and validation."""
import unicodedata
from pathlib import Path

import pytest

from library_query import items, structure


def _record(
    path: Path,
    item_id: str = "item-0123456789abcdef0123456789abcdef",
    record_type: str = "book",
    attachments: str = "  - path: Books/algorithms-handbook-2024.pdf\n    role: source\n",
    extra: str = "",
    body: str = "Human notes.\n",
) -> Path:
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
        f"{body}"
    )
    return path


def test_parse_item_record_roundtrip_preserves_body_and_extras(tmp_path: Path):
    record_path = _record(
        tmp_path / "Metadata" / "item-0123456789abcdef0123456789abcdef.md",
        extra="citekey: Algo2024\ncustom_field: keep-me\n",
        body="\nLocal writing stays.\n",
    )
    record, conflict = items.parse_item_record(record_path, tmp_path)

    assert conflict is None
    assert record["item_id"] == "item-0123456789abcdef0123456789abcdef"
    assert record["frontmatter"]["custom_field"] == "keep-me"
    assert record["body"].strip() == "Local writing stays."
    assert record["attachments"] == [
        {"path": "Books/algorithms-handbook-2024.pdf", "role": "source"}
    ]


@pytest.mark.parametrize(
    "bad_id,snippet",
    [
        ("not-an-id", "attachments:\n  - path: Books/x.pdf\n    role: source\n"),
        ("item-0123456789abcdef0123456789abcdef", "schema_version: 2\nitem_id: item-0123456789abcdef0123456789abcdef\nattachments:\n  - path: Books/x.pdf\n    role: source\n"),
    ],
)
def test_parse_item_record_rejects_invalid_records(tmp_path: Path, bad_id, snippet):
    path = tmp_path / "Metadata" / f"{bad_id}.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"---\n{snippet}---\n")
    record, conflict = items.parse_item_record(path, tmp_path)
    assert record is None and conflict


def test_record_requires_source_role_and_matching_shelf(tmp_path: Path):
    base = tmp_path / "Metadata"
    no_source, conflict = items.parse_item_record(
        _record(
            base / "item-11111111111111111111111111111111.md",
            item_id="item-11111111111111111111111111111111",
            attachments="  - path: Notes/item-11111111111111111111111111111111-zotero.md\n    role: note\n",
        ),
        tmp_path,
    )
    assert no_source is None and "no source" in conflict

    wrong_shelf, conflict = items.parse_item_record(
        _record(
            base / "item-22222222222222222222222222222222.md",
            item_id="item-22222222222222222222222222222222",
            record_type="paper",
            attachments="  - path: Books/some-book.pdf\n    role: source\n",
        ),
        tmp_path,
    )
    assert wrong_shelf is None and "Papers" in conflict


def test_record_rejects_escaping_and_absolute_paths(tmp_path: Path):
    record, conflict = items.parse_item_record(
        _record(
            tmp_path / "Metadata" / "item-33333333333333333333333333333333.md",
            item_id="item-33333333333333333333333333333333",
            attachments="  - path: ../outside.pdf\n    role: source\n",
        ),
        tmp_path,
    )
    assert record is None and "unsafe" in conflict


def test_inventory_reports_duplicate_ownership_citekeys_and_collisions(tmp_path: Path):
    (tmp_path / "Books").mkdir()
    (tmp_path / "Books" / "one.pdf").write_bytes(b"one")
    (tmp_path / "Books" / "two.pdf").write_bytes(b"two")
    _record(
        tmp_path / "Metadata" / "item-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.md",
        item_id="item-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        attachments="  - path: Books/one.pdf\n    role: source\n",
        extra="citekey: Shared2024\n",
    )
    _record(
        tmp_path / "Metadata" / "item-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.md",
        item_id="item-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        attachments="  - path: Books/one.pdf\n    role: source\n",
    )
    _record(
        tmp_path / "Metadata" / "item-cccccccccccccccccccccccccccccccc.md",
        item_id="item-cccccccccccccccccccccccccccccccc",
        attachments="  - path: Books/two.pdf\n    role: source\n",
        extra="citekey: Shared2024\n",
    )
    # NFC/casefold portability collision between two same-shelf declarations.
    _record(
        tmp_path / "Metadata" / "item-dddddddddddddddddddddddddddddddd.md",
        item_id="item-dddddddddddddddddddddddddddddddd",
        attachments="  - path: Books/Café-2024.pdf\n    role: source\n",
    )
    _record(
        tmp_path / "Metadata" / "item-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.md",
        item_id="item-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        attachments="  - path: Books/café-2024.pdf\n    role: source\n",
    )

    inventory = items.build_inventory(tmp_path)
    reasons = [c["reason"] for c in inventory["conflicts"]]

    assert "ambiguous_attachment_ownership" in reasons
    assert "duplicate_citekey" in reasons
    assert "path_portability_collision" in reasons


def test_inventory_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "library"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.pdf").write_bytes(b"external bytes")
    (root / "Books").mkdir()
    link = root / "Books" / "linked.pdf"
    link.symlink_to(outside / "secret.pdf")
    _record(
        root / "Metadata" / "item-ffffffffffffffffffffffffffffffff.md",
        item_id="item-ffffffffffffffffffffffffffffffff",
        attachments="  - path: Books/linked.pdf\n    role: source\n",
    )

    inventory = items.build_inventory(root)

    assert any(
        c["reason"] == "attachment_escapes_library" for c in inventory["conflicts"]
    )
    # The rejected record attaches metadata to no file.
    assert inventory["by_path"].get("Books/linked.pdf") is None


def test_inventory_mixed_flat_legacy_and_missing_declared_attachment(tmp_path: Path):
    (tmp_path / "Books" / "Legacy_2020_Old").mkdir(parents=True)
    (tmp_path / "Books" / "Legacy_2020_Old" / "book.pdf").write_bytes(b"old")
    (tmp_path / "Papers").mkdir()
    (tmp_path / "Papers" / "flat-study.pdf").write_bytes(b"flat")
    _record(
        tmp_path / "Metadata" / "item-0123456789abcdef0123456789abcdef.md",
        record_type="paper",
        attachments="  - path: Papers/flat-study.pdf\n    role: source\n"
                    "  - path: Papers/missing.pdf\n    role: source\n",
    )

    inventory = items.build_inventory(tmp_path)

    assert list(inventory["legacy_items"]) == ["Books/Legacy_2020_Old"]
    assert inventory["unmapped"] == []
    assert inventory["by_path"]["Papers/flat-study.pdf"] == "item-0123456789abcdef0123456789abcdef"
    assert any(
        w["reason"] == "missing_declared_attachment" and w["path"] == "Papers/missing.pdf"
        for w in inventory["warnings"]
    )


def test_legacy_preview_item_id_is_stable_and_normalization_canonical(tmp_path: Path):
    nfc = items.legacy_preview_item_id("Books/Café_2020_Mixed")
    nfd = items.legacy_preview_item_id("Books/Café_2020_Mixed")  # decomposed é
    assert nfc == nfd
    assert nfc.startswith("item-") and len(nfc) == len("item-") + 32

    # Case is preserved: no case folding in the UUID name input.
    assert items.legacy_preview_item_id("Books/café_2020_mixed") != nfc

    # Canonically equivalent directory names on disk derive the same preview ID
    # without any .git directory involved.
    assert not (tmp_path / ".git").exists()


def test_catalog_multi_attachment_and_absolute_base_correlation(tmp_path: Path):
    (tmp_path / "Books" / "Algo_2024_Handbook").mkdir(parents=True)
    (tmp_path / "Books" / "Algo_2024_Handbook" / "book.pdf").write_bytes(b"pdf")
    (tmp_path / "Books" / "Algo_2024_Handbook" / "book.epub").write_bytes(b"epub")
    (tmp_path / "catalog.bib").write_text(
        """@book{Algo2024Handbook,
  title = {Algorithms Handbook},
  author = {Example Author},
  year = {2024},
  file = {Main:/data/library-host/Books/Algo_2024_Handbook/book.pdf:application/pdf;
          Ebook:/data/library-host/Books/Algo_2024_Handbook/book.epub:application/epub+zip}
}
""",
        encoding="utf-8",
    )
    bib = structure.parse_catalog_bib(tmp_path / "catalog.bib")

    for rel in (
        "Books/Algo_2024_Handbook/book.pdf",
        "Books/Algo_2024_Handbook/book.epub",
    ):
        meta = structure.resolve_metadata(tmp_path / rel, tmp_path, bib)
        assert meta["title"] == "Algorithms Handbook"
        assert meta["citekey"] == "Algo2024Handbook"


def test_catalog_path_and_citekey_disagreement_is_a_conflict(tmp_path: Path):
    (tmp_path / "Papers" / "Study_2025_One").mkdir(parents=True)
    (tmp_path / "Papers" / "Study_2025_One" / "paper.pdf").write_bytes(b"paper")
    (tmp_path / "catalog.bib").write_text(
        """@article{PathParent2025,
  title = {Path Parent},
  year = {2025},
  file = {Papers/Study_2025_One/paper.pdf}
}
@article{CiteParent2025,
  title = {Cite Parent},
  year = {2024}
}
""",
        encoding="utf-8",
    )
    diagnostics: list[dict] = []
    _record(
        tmp_path / "Metadata" / "item-0123456789abcdef0123456789abcdef.md",
        record_type="paper",
        attachments="  - path: Papers/Study_2025_One/paper.pdf\n    role: source\n",
        extra="citekey: CiteParent2025\n",
    )
    inventory = items.build_inventory(tmp_path)

    meta = structure.resolve_metadata(
        tmp_path / "Papers" / "Study_2025_One" / "paper.pdf",
        tmp_path,
        inventory=inventory,
        diagnostics=diagnostics,
    )

    assert any(d["reason"] == "association_conflict" for d in diagnostics)
    # No mixing: contested catalog facts are dropped, local/fallback facts stay.
    assert meta["title"] == "One"


def test_effective_citekey_from_record_or_catalog_path(tmp_path: Path):
    (tmp_path / "Books").mkdir()
    (tmp_path / "Books" / "x.pdf").write_bytes(b"x")
    (tmp_path / "catalog.bib").write_text(
        "@book{CatalogKey2024,\n title = {X},\n year = {2024},\n file = {Books/x.pdf}\n}\n"
        "@book{MissingKey2024,\n title = {Y},\n year = {2024},\n file = {Books/nope.pdf}\n}\n"
    )
    explicit = _record(
        tmp_path / "Metadata" / "item-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.md",
        item_id="item-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        attachments="  - path: Books/x.pdf\n    role: source\n",
        extra="citekey: ExplicitKey\n",
    )
    _record(
        tmp_path / "Metadata" / "item-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.md",
        item_id="item-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        attachments="  - path: Books/nope.pdf\n    role: source\n",
    )
    inventory = items.build_inventory(tmp_path)

    assert items.effective_citekey(inventory, inventory["records"]["item-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]) == "ExplicitKey"
    # A missing declared attachment still correlates by exported path.
    assert items.effective_citekey(inventory, inventory["records"]["item-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]) == "MissingKey2024"
    explicit.unlink()
    no_citekey = _record(
        tmp_path / "Metadata" / "item-cccccccccccccccccccccccccccccccc.md",
        item_id="item-cccccccccccccccccccccccccccccccc",
        attachments="  - path: Books/unknown.pdf\n    role: source\n",
    )
    inventory = items.build_inventory(tmp_path)
    assert items.effective_citekey(inventory, inventory["records"]["item-cccccccccccccccccccccccccccccccc"]) is None
