"""Tests for offline browsing and explicitly generated navigation."""
import json
import os
from pathlib import Path
from urllib.parse import unquote

from library_query import browse, config, ledger


def _library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    (root / "Books").mkdir(parents=True)
    (root / "Papers").mkdir()
    (root / "Videos").mkdir()
    (root / "Collections").mkdir()
    (root / "_Inbox").mkdir()
    return root


def _record(root: Path, item_id: str, record_type: str, attachments: str, extra: str = "") -> None:
    record = root / "Metadata" / f"{item_id}.md"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
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


def test_browse_json_inventory_grouping_ordering_and_warnings(tmp_path: Path):
    root = _library(tmp_path)
    (root / "Books" / "algorithms-handbook-2024.pdf").write_bytes(b"pdf")
    (root / "Books" / "algorithms-handbook-2024.epub").write_bytes(b"epub")
    (root / "Books" / "notes and (extras) #1.pdf").write_bytes(b"odd")
    _record(
        root,
        "item-0123456789abcdef0123456789abcdef",
        "book",
        "  - path: Books/algorithms-handbook-2024.pdf\n    role: source\n"
        "  - path: Books/algorithms-handbook-2024.epub\n    role: source\n",
        extra="citekey: Algo2024\nsubjects: [computer science, algorithms]\nstatus: unread\n",
    )
    (root / "Books" / "Zeta_2019_Last").mkdir()
    (root / "Books" / "Zeta_2019_Last" / "book.pdf").write_bytes(b"legacy zeta")
    (root / "Books" / "alpha-untitled.pdf").write_bytes(b"untitled flat")

    payload = browse.browse_json(root)

    assert payload["version"] == 1
    titles = [i["title"] for i in payload["items"]]
    # Case-folded ordering; unmapped files are shown, not omitted.
    assert titles == ["algorithms-handbook-2024", "alpha-untitled", "Last", "notes and (extras) #1"]
    mapped = payload["items"][0]
    assert mapped["item_id"] == "item-0123456789abcdef0123456789abcdef"
    assert mapped["sources"] == [
        "Books/algorithms-handbook-2024.pdf",
        "Books/algorithms-handbook-2024.epub",
    ]
    assert mapped["subjects"] == ["computer science", "algorithms"]
    # The odd flat file is unmapped and warns instead of inheriting metadata.
    unmapped = payload["items"][1]
    assert unmapped["item_id"] is None
    assert any("unmapped" in w for w in unmapped["warnings"])
    assert payload["items"][2]["legacy"] is True


def test_browse_year_unknown_sorts_after_known(tmp_path: Path):
    root = _library(tmp_path)
    (root / "Papers" / "same-title.pdf").write_bytes(b"a")
    (root / "Papers" / "other.pdf").write_bytes(b"b")
    (root / "catalog.bib").write_text(
        "@article{Known2020,\n title = {Same Title},\n year = {2020},\n file = {Papers/same-title.pdf}\n}\n"
    )
    _record(
        root,
        "item-0123456789abcdef0123456789abcdef",
        "paper",
        "  - path: Papers/same-title.pdf\n    role: source\n",
    )
    # Unknown year: no catalog entry, filename carries no parseable year.
    payload = browse.browse_json(root)
    by_path = {(i["sources"] or [""])[0]: i for i in payload["items"]}
    assert by_path["Papers/same-title.pdf"]["year"] == 2020


def test_browse_is_side_effect_free_for_absent_legacy_and_corrupt_manifests(
    tmp_path: Path, monkeypatch
):
    root = _library(tmp_path)
    (root / "Books" / "flat.pdf").write_bytes(b"flat")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    monkeypatch.setattr(config, "LIBRARY_ROOT", root)

    for forbidden in ("scan_content_paths", "scan_content_files", "_load"):
        monkeypatch.setattr(
            ledger, forbidden,
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"browse called ledger.{forbidden}")),
        )

    snapshots = {}
    manifests = {
        "absent": None,
        "legacy": json.dumps({"files": {"Books/ghost.pdf": "hash1"}}),
        "corrupt": '{"version": 2, "documents": {"doc_a": {"content_ha',
    }
    for label, content in manifests.items():
        ledger_file = state_dir / "ledger.json"
        ledger_file.unlink(missing_ok=True)
        if content is not None:
            ledger_file.write_text(content)
        info = ledger_file.stat() if ledger_file.exists() else None
        snapshots[label] = (ledger_file.exists(), ledger_file.read_bytes() if info else None, info)

        payload = browse.browse_json(root)
        assert payload["items"][0]["sources"] == ["Books/flat.pdf"]
        assert "flat" in browse.format_human(root)
        result = browse.write_navigation(root)
        assert result["conflicts"] == []

        assert ledger_file.exists() == snapshots[label][0]
        if info:
            assert ledger_file.read_bytes() == snapshots[label][1], f"{label} manifest bytes changed"
            assert ledger_file.stat().st_mtime_ns == info.st_mtime_ns, f"{label} manifest mtime changed"
        ledger_file.unlink(missing_ok=True)


def test_browse_human_output_lists_shelves_and_files(tmp_path: Path):
    root = _library(tmp_path)
    (root / "Books" / "book-one.pdf").write_bytes(b"one")
    out = browse.format_human(root)
    assert "Books (1 items)" in out
    assert "book-one.pdf" in out
    assert "[file: pdf]" in out


def test_write_generates_blocks_preserving_prose_and_grouping_subjects(tmp_path: Path):
    root = _library(tmp_path)
    (root / "Books" / "algorithms-handbook-2024.pdf").write_bytes(b"pdf")
    _record(
        root,
        "item-0123456789abcdef0123456789abcdef",
        "book",
        "  - path: Books/algorithms-handbook-2024.pdf\n    role: source\n",
        extra="subjects: [computer science, algorithms]\n",
    )
    (root / "Books" / "README.md").write_text("# Books\n\nCurated introduction prose.\n")
    (root / "Collections" / "course-page.md").write_text("Curated course page.\n")
    (root / "Operations").mkdir()
    (root / "Operations" / "guide.md").write_text("Operator guide body.\n")

    result = browse.write_navigation(root)

    assert result["conflicts"] == []
    shelf = (root / "Books" / "README.md").read_text(encoding="utf-8")
    assert "Curated introduction prose." in shelf  # prose preserved byte-for-byte
    assert shelf.index("Curated introduction prose.") < shelf.index(browse.MANAGED_START)
    assert "| algorithms-handbook-2024 |" in shelf
    assert "[pdf](algorithms-handbook-2024.pdf)" in shelf
    assert "[metadata](../Metadata/item-0123456789abcdef0123456789abcdef.md)" in shelf

    root_readme = (root / "README.md").read_text(encoding="utf-8")
    assert "[Books](Books/README.md) — 1 items" in root_readme
    assert "[course-page](Collections/course-page.md)" in root_readme

    subjects = (root / "Collections" / "subjects.md").read_text(encoding="utf-8")
    assert "### computer science" in subjects
    assert "### algorithms" in subjects
    # One physical attachment referenced from several subject groups.
    assert subjects.count("[pdf](../Books/algorithms-handbook-2024.pdf)") == 2
    # Operational guides and metadata records are never counted or listed.
    assert "guide.md" not in shelf
    payload = browse.browse_json(root)
    assert payload["items"][0]["type"] == "book"
    assert all("Operations" not in (i["sources"] or [""])[0] for i in payload["items"])


def test_write_is_byte_and_mtime_idempotent(tmp_path: Path):
    root = _library(tmp_path)
    (root / "Books" / "flat.pdf").write_bytes(b"flat")
    first = browse.write_navigation(root)
    assert first["conflicts"] == []
    targets = ["README.md", "Books/README.md", "Papers/README.md",
               "Videos/README.md", "Collections/subjects.md"]
    mtimes = {rel: (root / rel).stat().st_mtime_ns for rel in targets}
    bytes_before = {rel: (root / rel).read_bytes() for rel in targets}

    second = browse.write_navigation(root)

    assert second["written"] == []
    assert sorted(second["unchanged"]) == sorted(targets)
    for rel in targets:
        assert (root / rel).stat().st_mtime_ns == mtimes[rel], f"{rel} mtime moved"
        assert (root / rel).read_bytes() == bytes_before[rel], f"{rel} bytes moved"


def test_write_refuses_unmarked_subjects_page_and_malformed_markers(tmp_path: Path):
    root = _library(tmp_path)
    (root / "Books" / "flat.pdf").write_bytes(b"flat")
    (root / "Collections" / "subjects.md").write_text("Hand-written subjects page.\n")

    result = browse.write_navigation(root)

    assert result["written"] == []
    assert any("no managed markers" in c for c in result["conflicts"])
    assert (root / "Collections" / "subjects.md").read_text() == "Hand-written subjects page.\n"
    assert not (root / "README.md").exists()  # preflight aborted before any write

    (root / "Collections" / "subjects.md").write_text(
        f"{browse.MANAGED_START}\n{browse.MANAGED_START}\nx\n{browse.MANAGED_END}\n"
    )
    result = browse.write_navigation(root)
    assert result["written"] == []
    assert any("duplicate" in c for c in result["conflicts"])


def test_write_preflights_permission_errors_before_writing(tmp_path: Path, monkeypatch):
    root = _library(tmp_path)
    (root / "Books" / "flat.pdf").write_bytes(b"flat")
    readme = root / "Books" / "README.md"
    readme.write_text("# Books\n")
    monkeypatch.setattr(os, "access", lambda path, mode: False)

    result = browse.write_navigation(root)

    assert result["written"] == []
    assert result["conflicts"]
    assert readme.read_text(encoding="utf-8") == "# Books\n"
    assert not (root / "README.md").exists()


def test_missing_optional_note_produces_no_broken_links(tmp_path: Path):
    root = _library(tmp_path)
    (root / "Books" / "flat.pdf").write_bytes(b"flat")
    _record(
        root,
        "item-0123456789abcdef0123456789abcdef",
        "book",
        "  - path: Books/flat.pdf\n    role: source\n",
    )
    result = browse.write_navigation(root)
    assert result["conflicts"] == []
    shelf = (root / "Books" / "README.md").read_text(encoding="utf-8")
    assert "zotero" not in shelf
    # With the note present the link appears.
    (root / "Notes").mkdir(exist_ok=True)
    (root / "Notes" / "item-0123456789abcdef0123456789abcdef-zotero.md").write_text("note")
    result = browse.write_navigation(root)
    assert result["conflicts"] == []
    shelf = (root / "Books" / "README.md").read_text(encoding="utf-8")
    assert "[notes](../Notes/item-0123456789abcdef0123456789abcdef-zotero.md)" in shelf


LINK_RE = __import__("re").compile(r"\[[^\]]*\]\(([^)]+)\)")


def _extract_links(content: str) -> list[str]:
    return LINK_RE.findall(content)


def test_links_resolve_to_exact_files_and_encode_once(tmp_path: Path):
    root = _library(tmp_path)
    odd_name = "study%20results (draft) #1 — caf\u00e9.pdf"
    (root / "Notes").mkdir(exist_ok=True)
    (root / "Notes" / odd_name).write_bytes(b"odd note file")
    (root / "Books" / "flat.pdf").write_bytes(b"flat")
    _record(
        root,
        "item-0123456789abcdef0123456789abcdef",
        "book",
        "  - path: Books/flat.pdf\n    role: source\n"
        f'  - path: "Notes/{odd_name}"\n    role: note\n',
    )
    result = browse.write_navigation(root)
    assert result["conflicts"] == []

    shelf = (root / "Books" / "README.md").read_text(encoding="utf-8")
    links = _extract_links(shelf)
    assert links
    for link in links:
        decoded = unquote(link)
        assert not decoded.startswith("/"), "absolute host path leaked into a link"
        # Encode exactly once: the decoded form must be the actual on-disk path.
        target = (root / "Books" / decoded).resolve()
        assert target.is_file(), f"link {link} does not resolve to an existing file"
    # The literal percent escape survives as %25.
    assert "study%2520results%20%28draft%29%20%231%20%E2%80%94%20caf%C3%A9.pdf" in shelf


def test_browse_write_only_touches_managed_navigation(tmp_path: Path, monkeypatch):
    root = _library(tmp_path)
    (root / "Books" / "flat.pdf").write_bytes(b"flat")
    _record(
        root,
        "item-0123456789abcdef0123456789abcdef",
        "book",
        "  - path: Books/flat.pdf\n    role: source\n",
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    monkeypatch.setattr(config, "LIBRARY_ROOT", root)
    for forbidden in ("scan_content_paths", "scan_content_files", "_load"):
        monkeypatch.setattr(
            ledger, forbidden,
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"write called ledger.{forbidden}")),
        )

    result = browse.write_navigation(root)

    assert result["conflicts"] == []
    assert not (state_dir / "ledger.json").exists()
    assert (root / "Books" / "flat.pdf").read_bytes() == b"flat"


def _item(title, authors=None, subjects=None, citekey=None):
    return {"title": title, "authors": authors or [], "subjects": subjects or [], "tags": None, "citekey": citekey}


def test_search_matches_author_subject_and_citekey():
    item = _item("Maps of Meaning", ["Jordan B. Peterson"], ["psychology-popular"], "Peterson1999")
    assert browse.matches_search(item, "peterson")
    assert browse.matches_search(item, "Jordan Peterson")
    assert browse.matches_search(item, "psychology")
    assert browse.matches_search(item, "peterson1999")
    assert not browse.matches_search(item, "manson")


def test_search_ignores_case_accents_and_punctuation():
    item = _item("The Subtle Art of Not Giving a F*ck", ["Mark Manson"])
    assert browse.matches_search(item, "f ck")
    assert browse.matches_search(item, "FCK")
    assert browse.matches_search(item, "subtle art")
    assert browse.matches_search(_item("The $100 Startup"), "100 startup")
    assert browse.matches_search(_item("Introdução à Álgebra"), "introducao algebra")


def test_search_needs_every_word_and_word_starts_only():
    item = _item("Running Lean", ["Ash Maurya"])
    assert not browse.matches_search(item, "running agile")
    assert not browse.matches_search(item, "unning")
    assert not browse.matches_search(item, "   ")
