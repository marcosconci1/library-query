"""Offline end-to-end fixture: approved legacy-to-flat moves, reindex, rollback.

Simulates the owner-approved migration on a temporary library: the plan's
moves are applied by the test itself (never by module commands), sources are
re-indexed with zero embedding calls for moved bytes, browsing links resolve,
tag filters and metadata exclusions hold, and the saved inverse map rolls the
batch back with identities still intact.
"""
import argparse
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote
from unittest.mock import MagicMock

import pytest
from conftest import configure_readiness_client

from library_query import browse, cli, config, layout_plan, ledger, store

LONG = (
    "Substantive document text about retrieval systems, indexing pipelines, "
    "and vector identity invariants that is long enough to survive chunking. "
)

CATALOG = """@book{Algo2020,
  title = {Algorithms Handbook},
  author = {Example Author},
  year = {2020},
  keywords = {algorithms, textbook},
  file = {Books/Legacy_2020_Algorithms/book.txt}
}
@article{Study2025,
  title = {Retrieval Study},
  author = {Researcher One},
  year = {2025},
  keywords = {retrieval, study},
  file = {Papers/Researcher_2025_Retrieval-Study/paper.txt}
}
"""


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

    embed_spy = MagicMock(side_effect=lambda texts: [[0.05] * 1536 for _ in texts])
    monkeypatch.setattr(store, "embed", embed_spy)

    return {
        "lib_root": lib_root,
        "state_dir": state_dir,
        "mock_client": mock_client,
        "embed_spy": embed_spy,
    }


def _build_legacy_library(root: Path) -> dict:
    book = root / "Books" / "Legacy_2020_Algorithms"
    book.mkdir(parents=True)
    (book / "book.txt").write_text(LONG + "book body")
    (book / "Solutions-Manual.txt").write_text(LONG + "solutions body")
    (book / "zotero-notes.md").write_text(
        "<!-- library-query:zotero-notes -->\n<!-- citekey: Algo2020 -->\n\n"
        + LONG + "generated note body\n"
    )
    (book / "README.md").write_text(
        "---\ntitle: Algorithms Handbook\nyear: 2020\n"
        "subjects: [algorithms]\nstatus: reading\nrating: 4\n---\n\nReading notes.\n"
    )

    paper = root / "Papers" / "Researcher_2025_Retrieval-Study"
    paper.mkdir(parents=True)
    (paper / "paper.txt").write_text(LONG + "paper body")
    (paper / "README.md").write_text("---\ntitle: Retrieval Study\nyear: 2025\n---\n")

    video = root / "Videos" / "Channel_2026_Retrieval"
    video.mkdir(parents=True)
    (video / "Retrieval [abc123def45].transcript.en.md").write_text(
        "# Retrieval Design\n\n## [00:00:00] — Opening\n\n" + LONG + "transcript body\n"
    )

    operations = root / "Operations"
    operations.mkdir()
    (operations / "guide.md").write_text(LONG + "operator guide body")

    inbox = root / "_Inbox"
    inbox.mkdir()
    (inbox / "pending.txt").write_text(LONG + "inbound body")

    (root / "catalog.bib").write_text(CATALOG)
    (root / "README.md").write_text("# Library\n\nHuman overview prose.\n")

    return {
        "book_dir": book,
        "note_rel": "Books/Legacy_2020_Algorithms/zotero-notes.md",
    }


def _apply_plan(root: Path, plan: dict) -> None:
    """Apply approved moves/conversions exactly as the plan describes."""
    for action in plan["file_actions"]:
        if action["action"] == "move":
            source = root / action["path"]
            target = root / action["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        elif action["action"] == "convert-metadata":
            item = next(i for i in plan["items"] if i["record_path"] == action["target"])
            target = root / action["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item["record"], encoding="utf-8")


def _reexport_catalog(root: Path, plan: dict) -> None:
    """Simulate the owner re-exporting catalog.bib after relinking (plan step 6)."""
    catalog = (root / "catalog.bib").read_text(encoding="utf-8")
    for action in plan["file_actions"]:
        if action["action"] == "move" and action["path"].startswith(("Books/", "Papers/", "Videos/")):
            catalog = catalog.replace(action["path"], action["target"])
    (root / "catalog.bib").write_text(catalog, encoding="utf-8")


def test_end_to_end_migration_index_browse_and_rollback(mock_env):
    lib_root = mock_env["lib_root"]
    fixtures = _build_legacy_library(lib_root)

    # --- 1. Index the legacy layout -------------------------------------------
    assert cli.cmd_index(argparse.Namespace()) == 0
    # 2 book sources + generated note + paper + video + operations + inbox.
    assert mock_env["embed_spy"].call_count == 7
    state = ledger._load(lib_root)
    assert len(state["documents"]) == 7
    identities = {doc_id: doc["point_ids"] for doc_id, doc in state["documents"].items()}
    upserted_paths = [
        point.payload["path"]
        for c in mock_env["mock_client"].upsert.call_args_list
        for point in c[1]["points"]
    ]
    # Metadata/navigation exclusions: only real content is embedded.
    assert not any(p.startswith("Metadata/") or p.endswith("README.md") for p in upserted_paths)
    assert "catalog.bib" not in upserted_paths
    scopes = {doc["path"]: (doc["scope"], doc["type"]) for doc in state["documents"].values()}
    assert scopes["_Inbox/pending.txt"] == ("inbox", "inbox")
    assert scopes["Operations/guide.md"] == ("operations", "operation")
    assert scopes["Books/Legacy_2020_Algorithms/book.txt"][1] == "book"

    # --- 2. Plan and apply the approved migration ------------------------------
    plan = layout_plan.plan_layout(lib_root, probe=lambda directory: 255)
    assert plan["conflicts"] == [], json.dumps(plan["conflicts"], indent=1)
    assert len(plan["items"]) == 3
    _apply_plan(lib_root, plan)
    _reexport_catalog(lib_root, plan)

    # --- 3. Reconcile moved bytes with zero embeddings --------------------------
    mock_env["embed_spy"].reset_mock()
    mock_env["mock_client"].reset_mock()
    summary = cli.cmd_index(argparse.Namespace())
    assert summary == 0
    assert mock_env["embed_spy"].call_count == 0, "moved bytes must not be re-embedded"

    state = ledger._load(lib_root)
    assert set(state["documents"]) == set(identities)
    for doc_id, point_ids in identities.items():
        assert state["documents"][doc_id]["point_ids"] == point_ids
    by_path = {doc["path"]: doc for doc in state["documents"].values()}
    assert "Books/algorithms-handbook-2020.txt" in by_path
    assert "Notes/" in next(
        doc["path"] for doc in state["documents"].values() if "zotero" in doc["path"]
    )
    note_doc = next(doc for doc in state["documents"].values() if "zotero" in doc["path"])
    assert note_doc["type"] == "book"  # inherited from the mapped item
    assert note_doc["scope"] == "catalogued"
    # The moved book carries the immutable item ID in its payload update.
    payload_updates = [
        c[1]["payload"] for c in mock_env["mock_client"].set_payload.call_args_list
    ]
    book_update = next(
        p for p in payload_updates
        if p.get("path") == "Books/algorithms-handbook-2020.txt"
    )
    item_id = plan["items"][0]["item_id"]
    assert book_update["item_id"] == item_id
    assert book_update["subjects"] == ["algorithms"]

    # --- 4. Changed note text follows normal indexing ---------------------------
    note_path = lib_root / note_doc["path"]
    note_path.write_text(note_path.read_text(encoding="utf-8") + "New annotation paragraph.\n", encoding="utf-8")
    mock_env["embed_spy"].reset_mock()
    assert cli.cmd_index(argparse.Namespace()) == 0
    assert mock_env["embed_spy"].call_count == 1  # only the changed note
    # Changed note text is a new content identity; the rollback invariant
    # applies to the identities that exist after that normal reindex.
    identities = {d: doc["point_ids"] for d, doc in ledger._load(lib_root)["documents"].items()}

    # --- 5. Browsing: links resolve and navigation regenerates ------------------
    payload = browse.browse_json(lib_root)
    assert payload["version"] == 1
    for item in payload["items"]:
        for path in item["sources"] + item["notes"]:
            assert (lib_root / path).is_file(), f"browse lists missing file {path}"
        if item["item_id"] == item_id:
            assert item["tags"] == ["algorithms", "textbook"]  # re-exported catalog

    result = browse.write_navigation(lib_root)
    assert result["conflicts"] == []
    shelf = (lib_root / "Books" / "README.md").read_text(encoding="utf-8")
    assert "algorithms-handbook-2020.txt" in shelf
    link_targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", shelf)
    assert link_targets
    for link in link_targets:
        decoded = unquote(link)
        assert not decoded.startswith("/")
        assert (lib_root / "Books" / decoded).resolve().is_file(), f"broken link {link}"

    # Tag filters still intersect on the re-exported catalog tags.
    mock_env["mock_client"].query_points.return_value = MagicMock(points=[])
    store.query_places("q", scope="catalogued", tags=["algorithms"])
    conditions = mock_env["mock_client"].query_points.call_args[1]["query_filter"].must
    assert ("tags", "algorithms") in [(c.key, c.match.value) for c in conditions]

    # --- 6. Rollback through the saved inverse map -------------------------------
    for action in plan["file_actions"]:
        if action["action"] == "move":
            current = lib_root / action["reverse"]["from"]
            original = lib_root / action["reverse"]["to"]
            if current.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current), str(original))
    for item in plan["items"]:
        (lib_root / item["record_path"]).unlink(missing_ok=True)
    # Owner re-exports the catalog again after relinking back (rollback step).
    catalog = (lib_root / "catalog.bib").read_text(encoding="utf-8")
    for action in plan["file_actions"]:
        if action["action"] == "move" and action["path"].startswith(("Books/", "Papers/", "Videos/")):
            catalog = catalog.replace(action["target"], action["path"])
    (lib_root / "catalog.bib").write_text(catalog, encoding="utf-8")

    mock_env["embed_spy"].reset_mock()
    assert cli.cmd_index(argparse.Namespace()) == 0
    assert mock_env["embed_spy"].call_count == 0, "rollback must not re-embed moved bytes"
    state = ledger._load(lib_root)
    assert set(state["documents"]) == set(identities)
    for doc_id, point_ids in identities.items():
        assert state["documents"][doc_id]["point_ids"] == point_ids
    rolled_back = {doc["path"] for doc in state["documents"].values()}
    assert "Books/Legacy_2020_Algorithms/book.txt" in rolled_back
    assert "Books/Legacy_2020_Algorithms/zotero-notes.md" in rolled_back


def test_cli_browse_json_and_human(mock_env, capsys):
    lib_root = mock_env["lib_root"]
    _build_legacy_library(lib_root)

    assert cli.main(["browse", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == 1
    assert len(payload["items"]) == 3

    assert cli.main(["browse"]) == 0
    out = capsys.readouterr().out
    assert "Books (" in out and "Legacy_2020_Algorithms" in out


def test_cli_browse_write_conflict_exit_code(mock_env, monkeypatch, capsys):
    lib_root = mock_env["lib_root"]
    _build_legacy_library(lib_root)
    (lib_root / "Collections").mkdir()
    (lib_root / "Collections" / "subjects.md").write_text("manual page\n")

    code = cli.main(["browse", "--write"])

    captured = capsys.readouterr()
    assert code == 2
    error = json.loads(captured.err.strip().splitlines()[-1])
    assert error["error"] == "navigation_write_blocked"
    # stdout still carries the machine-readable conflict report.
    report = json.loads(captured.out)
    assert report["written"] == [] and report["conflicts"]


def test_cli_plan_layout_requires_library_root(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "LIBRARY_ROOT", tmp_path / "missing")
    code = cli.main(["plan-layout", "--json"])
    assert code == 3
    error = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert error["error"] == "invalid_library_root"


def test_plan_layout_ignores_zotero_and_git(monkeypatch, tmp_path):
    """Planning works with Zotero/Git endpoints sabotaged (offline guarantee)."""
    lib_root = tmp_path / "library"
    _build_legacy_library(lib_root)

    def explode(*a, **k):
        raise AssertionError("plan-layout must stay offline")

    monkeypatch.setattr(store, "embed", explode)
    monkeypatch.setattr(store, "qdrant", explode)
    monkeypatch.setattr(cli.zotero, "fetch_notes", explode)
    monkeypatch.setattr("library_query.zotero.httpx.post", explode)
    monkeypatch.setattr("subprocess.run", explode)

    plan = layout_plan.plan_layout(lib_root, probe=lambda directory: 255)
    assert plan["conflicts"] == []
