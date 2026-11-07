"""Tests for read-only flat-layout migration planning (tasks 5.1-5.4)."""
import json
import os
import unicodedata
from pathlib import Path

from library_query import browse, cli, config, items, layout_plan, ledger


def _legacy_library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    book = root / "Books" / "Martin_2008_Clean-Code"
    book.mkdir(parents=True)
    (book / "book.pdf").write_bytes(b"clean code bytes")
    (book / "Solutions-Manual.pdf").write_bytes(b"solutions bytes")
    (book / "README.md").write_text(
        "---\n"
        "title: Clean Code\n"
        "year: 2008\n"
        "subjects: [software-craftsmanship]\n"
        "status: read\n"
        "rating: 5\n"
        "---\n"
        "\nMy reading notes.\n"
    )
    inbox = root / "_Inbox"
    inbox.mkdir()
    (inbox / "new-paper.pdf").write_bytes(b"inbound")
    return root


def _fixed_probe(value):
    return lambda directory: value


# --- naming (5.1) ----------------------------------------------------------------


def test_clean_title_first_targets_with_supplement_label(tmp_path: Path):
    root = _legacy_library(tmp_path)
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    moves = {a["path"]: a["target"] for a in plan["file_actions"] if a["action"] == "move"}

    assert moves["Books/Martin_2008_Clean-Code/book.pdf"] == "Books/clean-code-2008.pdf"
    assert moves["Books/Martin_2008_Clean-Code/Solutions-Manual.pdf"] == (
        "Books/clean-code-2008-solutions-manual.pdf"
    )


def test_acquisition_noise_and_device_names_are_stripped(tmp_path: Path):
    assert layout_plan.clean_label("Algorithms Handbook (z-lib.org)") == "algorithms-handbook"
    assert layout_plan.clean_label("Study (1)") == "study"
    assert layout_plan.slugify("Title\x01 With Control") == "title-with-control"
    assert layout_plan.slugify("trailing dot.") == "trailing-dot"
    # An exact Windows device stem is escaped; "con-2020" is a legal name.
    assert layout_plan.source_stem("con", None, "book") == "con-doc"
    assert layout_plan.source_stem("con", 2020, "book") == "con-2020"
    assert layout_plan.source_stem("com1", None, "book") == "com1-doc"
    # A distinguishing label makes the name legal again.
    assert layout_plan.source_stem("com1", None, "x.pdf".rsplit(".", 1)[0]) == "com1-x"
    assert layout_plan.video_stem("nul", "Nul.md") == "nul-doc"
    assert layout_plan.video_stem("nul", "Nul [abc].en.md") == "nul-abc-en"  # legal: not the exact reserved name


def test_component_limit_143_bytes_keeps_complete_code_points(tmp_path: Path):
    root = _legacy_library(tmp_path)
    long_title = "Ünïcodé " + "very-long-title-" * 6 + "end"
    book = root / "Books" / "Author_2021_Long"
    book.mkdir()
    (book / "book.pdf").write_bytes(b"long")
    (book / "README.md").write_text(f"---\ntitle: {long_title}\nyear: 2021\n---\n")

    plan = layout_plan.plan_layout(root, probe=_fixed_probe(143))
    assert plan["conflicts"] == []
    for action in plan["file_actions"]:
        if action["action"] == "move":
            component = action["target"].split("/", 1)[1]
            encoded = unicodedata.normalize("NFC", component).encode("utf-8")
            assert len(encoded) <= 143, f"{component} exceeds the 143-byte limit"
            # Truncation only at complete code points.
            assert encoded.decode("utf-8") == component
            assert component.endswith(".pdf")


def test_collision_escape_extends_hash_suffix(tmp_path: Path):
    root = _legacy_library(tmp_path)
    # Two distinct papers that would both become retrieval-study-2025.pdf.
    (root / "Papers").mkdir(exist_ok=True)
    for author in ("A", "B"):
        paper = root / "Papers" / f"{author}_2025_Retrieval-Study"
        paper.mkdir()
        (paper / "paper.pdf").write_bytes(f"distinct bytes {author}".encode())
        (paper / "README.md").write_text("---\ntitle: Retrieval Study\nyear: 2025\n---\n")

    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    targets = [
        a["target"] for a in plan["file_actions"]
        if a["action"] == "move" and a["target"].startswith("Papers/")
    ]
    assert len(targets) == 2
    assert len(set(targets)) == 2
    assert all(t.startswith("Papers/retrieval-study-2025") and t.endswith(".pdf") for t in targets)
    # The loser of the collision gets a hash suffix that is a hex prefix of its
    # source SHA-256; the winner keeps the clean name.
    by_path = {a["path"]: a for a in plan["file_actions"] if a["action"] == "move"}
    suffixed = [
        a for a in by_path.values()
        if a["target"].rsplit("/", 1)[1].startswith("retrieval-study-2025-")
    ]
    assert len(suffixed) == 1
    stem = suffixed[0]["target"].rsplit("/", 1)[1][:-len(".pdf")]
    suffix = stem.removeprefix("retrieval-study-2025-")
    assert len(suffix) >= 8
    assert suffixed[0]["sha256"].startswith(suffix)


def test_duplicate_bytes_are_reported_never_removed(tmp_path: Path):
    root = _legacy_library(tmp_path)
    book = root / "Books" / "Dup_2020_Twin"
    book.mkdir()
    (book / "one.pdf").write_bytes(b"same bytes")
    (book / "two.pdf").write_bytes(b"same bytes")

    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    reasons = [c["reason"] for c in plan["conflicts"]]
    assert "duplicate_bytes" in reasons
    sources = sorted(p.name for p in (root / "Books" / "Dup_2020_Twin").iterdir())
    assert sources == ["one.pdf", "two.pdf"]  # nothing was removed


def test_multibyte_truncation_never_splits_code_points():
    base = "研究-" * 40 + "2024"
    fitted = layout_plan.fit_component(base, "", "pdf", 60)
    assert fitted is not None
    component = f"{fitted}.pdf"
    assert len(component.encode("utf-8")) <= 60
    component.encode("utf-8").decode("utf-8")  # no split code points
    assert component.endswith("-2024.pdf")


def test_required_parts_that_cannot_fit_are_conflicts(tmp_path: Path):
    root = _legacy_library(tmp_path)
    # A 20-byte limit cannot hold any stem plus ".pdf".
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(4))
    assert any(c["reason"] == "component_limit" for c in plan["conflicts"])


def test_unknown_probe_is_recorded_not_assumed_255(tmp_path: Path):
    root = _legacy_library(tmp_path)
    plan = layout_plan.plan_layout(root, probe=lambda directory: None)
    for entry in plan["naming"]["component_limits"].values():
        assert entry == {"value": None, "unit": None, "provenance": "unknown"}
    assert plan["naming"]["applying_host_verified"] is False


def test_naming_contract_records_rules_and_host_unverified(tmp_path: Path):
    root = _legacy_library(tmp_path)
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(143))
    naming = plan["naming"]
    assert naming["algorithm_version"] == 1
    assert naming["emitted_normalization"] == "NFC"
    assert naming["collision_comparison"] == "NFC then full casefold"
    assert naming["base_stem_ceiling_bytes"] == 180
    assert naming["component_limits"]["Books"] == {
        "value": 143, "unit": "bytes", "provenance": "planning-host-probe",
    }
    assert naming["applying_host_verified"] is False


def test_plan_is_reproducible_for_identical_inputs(tmp_path: Path):
    root = _legacy_library(tmp_path)
    first = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    second = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --- plan content (5.2) -----------------------------------------------------------


def test_plan_v1_shape_and_complete_record_content(tmp_path: Path):
    root = _legacy_library(tmp_path)
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))

    assert plan["version"] == 1
    assert set(plan) >= {
        "items", "file_actions", "navigation_rewrites",
        "conflicts", "prerequisites", "naming",
    }
    item = plan["items"][0]
    assert item["item_id"].startswith("item-")
    record = item["record"]
    assert "schema_version: 1" in record
    assert f"item_id: {item['item_id']}" in record
    # Preserved body and extra frontmatter field from the legacy README.
    assert "My reading notes." in record
    assert "rating: 5" in record
    assert "Books/clean-code-2008.pdf" in record
    assert "role: source" in record
    # Every affected file carries hash, size, action, and reverse mapping.
    for action in plan["file_actions"]:
        assert action["sha256"] and action["size"] >= 0
        assert action["action"] in {"move", "retain", "convert-metadata"}
        if action["action"] == "move":
            assert action["reverse"] == {"from": action["target"], "to": action["path"]}
    readme_action = next(
        a for a in plan["file_actions"] if a["action"] == "convert-metadata"
    )
    assert readme_action["path"] == "Books/Martin_2008_Clean-Code/README.md"
    assert readme_action["target"] == f"Metadata/{item['item_id']}.md"


def test_inbox_is_retained_as_intake_boundary(tmp_path: Path):
    root = _legacy_library(tmp_path)
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    inbox = next(a for a in plan["file_actions"] if a["path"] == "_Inbox/new-paper.pdf")
    assert inbox["action"] == "retain"
    assert (root / "_Inbox" / "new-paper.pdf").read_bytes() == b"inbound"


def test_preview_ids_are_stable_without_git_and_canonical(tmp_path: Path):
    root = _legacy_library(tmp_path)
    assert not (root / ".git").exists()
    first = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    second = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    assert [i["item_id"] for i in first["items"]] == [i["item_id"] for i in second["items"]]

    # Canonically equivalent scanned directory names derive the same ID.
    nfc_id = items.legacy_preview_item_id("Books/Café_2020_X")
    nfd_dir = "Books/Café_2020_X"
    assert items.legacy_preview_item_id(nfd_dir) == nfc_id


def test_plan_never_moves_writes_or_contacts_runtime_state(tmp_path: Path, monkeypatch):
    root = _legacy_library(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    for forbidden in ("scan_content_paths", "scan_content_files", "_load"):
        monkeypatch.setattr(
            ledger, forbidden,
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"plan called ledger.{forbidden}")),
        )
    calls: list[str] = []
    monkeypatch.setattr(os, "system", lambda cmd: calls.append(cmd))

    ledger_file = state_dir / "ledger.json"
    for label, content in (
        ("absent", None),
        ("legacy", json.dumps({"files": {"Books/ghost.pdf": "abc"}})),
        ("corrupt", '{"version": 2, "documents": {"doc_a": {"content_ha'),
    ):
        ledger_file.unlink(missing_ok=True)
        if content is not None:
            ledger_file.write_text(content)
            info = ledger_file.stat()
            before = (ledger_file.read_bytes(), info.st_mtime_ns)
        plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
        assert plan["conflicts"] == []
        if content is not None:
            assert (ledger_file.read_bytes(), ledger_file.stat().st_mtime_ns) == before, (
                f"{label} manifest was mutated"
            )
        assert ledger_file.exists() == (content is not None)
    assert calls == []
    # The library itself is byte-identical after planning.
    assert (root / "Books" / "Martin_2008_Clean-Code" / "book.pdf").read_bytes() == b"clean code bytes"


def test_plan_contains_no_absolute_host_paths(tmp_path: Path):
    root = _legacy_library(tmp_path)
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    serialized = json.dumps(plan)
    assert str(tmp_path) not in serialized
    assert str(root) not in serialized


# --- conflicts and prerequisites (5.3, 5.4) ---------------------------------------


def test_unknown_file_blocks_retiring_directory(tmp_path: Path):
    root = _legacy_library(tmp_path)
    (root / "Books" / "Martin_2008_Clean-Code" / "notes.todo").write_text("mystery")
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    assert any(
        c["reason"] == "unknown_file" and c["path"] == "Books/Martin_2008_Clean-Code/notes.todo"
        for c in plan["conflicts"]
    )
    # The mystery file still exists; the plan never removes anything.
    assert (root / "Books" / "Martin_2008_Clean-Code" / "notes.todo").is_file()


def test_manual_note_target_and_existing_target_conflict(tmp_path: Path):
    root = _legacy_library(tmp_path)
    note = root / "Books" / "Martin_2008_Clean-Code" / "zotero-notes.md"
    note.write_text("hand-written notes, not generated")
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    assert any(c["reason"] == "manual_note_target" for c in plan["conflicts"])
    assert note.read_text(encoding="utf-8") == "hand-written notes, not generated"


def test_generated_note_move_preserves_bytes(tmp_path: Path):
    root = _legacy_library(tmp_path)
    note = root / "Books" / "Martin_2008_Clean-Code" / "zotero-notes.md"
    note.write_text("<!-- library-query:zotero-notes -->\n<!-- citekey: X -->\n\nbody\n")
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    item = plan["items"][0]
    note_action = next(
        a for a in plan["file_actions"]
        if a["path"] == "Books/Martin_2008_Clean-Code/zotero-notes.md"
    )
    assert note_action["target"] == f"Notes/{item['item_id']}-zotero.md"
    assert "preserve rendered note bytes" in note_action["note"]
    assert note.read_bytes() == b"<!-- library-query:zotero-notes -->\n<!-- citekey: X -->\n\nbody\n"


def test_plan_cli_blocked_conflicts_exit_nonzero_with_json(tmp_path: Path, monkeypatch, capsys):
    root = _legacy_library(tmp_path)
    (root / "Books" / "Martin_2008_Clean-Code" / "mystery.bin").write_bytes(b"?")
    monkeypatch.setattr(config, "LIBRARY_ROOT", root)

    code = cli.main(["plan-layout", "--json"])

    captured = capsys.readouterr()
    assert code == 2
    payload = json.loads(captured.out)  # machine-readable plan still on stdout
    assert payload["version"] == 1
    assert any(c["reason"] == "unknown_file" for c in payload["conflicts"])
    error = json.loads(captured.err.strip().splitlines()[-1])
    assert error["error"] == "layout_plan_blocked"
    # The source tree is untouched.
    assert (root / "Books" / "Martin_2008_Clean-Code" / "mystery.bin").is_file()


def test_plan_cli_clean_exits_zero(tmp_path: Path, monkeypatch, capsys):
    root = _legacy_library(tmp_path)
    monkeypatch.setattr(config, "LIBRARY_ROOT", root)
    assert cli.main(["plan-layout", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == 1


def test_curated_link_conversion_proposals_and_ambiguous_conflicts(tmp_path: Path):
    root = _legacy_library(tmp_path)
    collections = root / "Collections"
    collections.mkdir()
    (collections / "course.md").write_text(
        "See [the clean code book](Books/Martin_2008_Clean-Code) and "
        "[[Books/Martin_2008_Clean-Code/Solutions-Manual.pdf|the solutions]] "
        "and [a vanished place](Collections/never-was.md).\n",
        encoding="utf-8",
    )
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    item = plan["items"][0]
    proposals = {(r["page"], r["link"]): r["proposed_target"] for r in plan["navigation_rewrites"]}

    directory_link = proposals[("Collections/course.md", "Books/Martin_2008_Clean-Code")]
    assert directory_link == f"Metadata/{item['item_id']}.md"
    file_link = proposals[
        ("Collections/course.md", "Books/Martin_2008_Clean-Code/Solutions-Manual.pdf")
    ]
    assert file_link == "Books/clean-code-2008-solutions-manual.pdf"
    assert any(
        c["reason"] == "ambiguous_curated_link" and "never-was.md" in c["detail"]
        for c in plan["conflicts"]
    )


def test_generated_navigation_links_are_not_curated_conflicts(tmp_path: Path):
    root = _legacy_library(tmp_path)
    (root / "Collections").mkdir()
    browse.write_navigation(root)  # creates managed blocks with generated links
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    assert not any(c["reason"] == "ambiguous_curated_link" for c in plan["conflicts"])


def test_prerequisites_and_unverified_relinking_status(tmp_path: Path):
    root = _legacy_library(tmp_path)
    plan = layout_plan.plan_layout(root, probe=_fixed_probe(255))
    ids = {p["id"] for p in plan["prerequisites"]}
    assert "applying-host-naming-fullpath" in ids
    assert "zotero-relinking-verified" in ids
    assert "inverse-map-and-journal" in ids
    assert "pause-ingestion-and-note-sync" in ids
    relink = next(p for p in plan["prerequisites"] if p["id"] == "zotero-relinking-verified")
    assert relink["status"] == "unverified"
    assert "relinkAttachmentFile()" in relink["description"] or "relinkAttachmentFile" in relink["description"]
    assert "delete/recreate" in relink["description"] or "never" in relink["description"]
    host = next(p for p in plan["prerequisites"] if p["id"] == "applying-host-naming-fullpath")
    assert host["applying_host_verified"] is False
    # A clean local preview never claims live attachments are safe to move.
    serialized = json.dumps(plan).lower()
    assert "safe to move" not in serialized.replace("never claims", "")
