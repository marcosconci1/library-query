"""Tests for safe Better BibTeX child-note synchronization."""
from pathlib import Path

import httpx
import pytest

from library_query import errors, structure, zotero


def _item_with_catalog(tmp_path: Path, content_root: str = "Books") -> tuple[Path, Path]:
    root = tmp_path / "library"
    item_dir = root / content_root / "Example_2024_Notes"
    item_dir.mkdir(parents=True)
    source = item_dir / "book.txt"
    source.write_text("Original source content.", encoding="utf-8")
    (root / "catalog.bib").write_text(
        """@book{Example2024Notes,
  title = {Example Notes},
  author = {Example Author},
  year = {2024},
  file = {%s/Example_2024_Notes/book.txt}
}
""" % content_root,
        encoding="utf-8",
    )
    return root, item_dir


def test_notes_to_markdown_preserves_text_boundaries():
    rendered = zotero.notes_to_markdown(
        "<p>Highlighted <b>idea</b></p><blockquote>Quoted passage</blockquote><ul><li>Follow up</li></ul>"
    )

    assert rendered == "Highlighted idea\n> Quoted passage\n- Follow up"


def test_sync_notes_creates_indexable_generated_markdown(tmp_path: Path, monkeypatch):
    root, item_dir = _item_with_catalog(tmp_path)
    monkeypatch.setattr(
        zotero,
        "fetch_notes",
        lambda citekeys: {
            "Example2024Notes": [{"note": "<p>Highlighted <b>idea</b></p><blockquote>Quoted passage</blockquote>"}]
        },
    )

    summary = zotero.sync_notes(root)
    generated = item_dir / zotero.GENERATED_FILENAME

    assert summary == {
        "items": 1,
        "updated": 1,
        "unchanged": 0,
        "removed": 0,
        "conflicts": [],
        "skipped_without_citekey": 0,
        "dry_run": False,
    }
    assert generated.read_text(encoding="utf-8") == (
        "<!-- library-query:zotero-notes -->\n"
        "<!-- citekey: Example2024Notes -->\n\n"
        "# Zotero notes and annotations\n\n"
        "Highlighted idea\n> Quoted passage\n"
    )
    assert structure.classify_file(generated, root) == "content"

    repeat = zotero.sync_notes(root)
    assert repeat["updated"] == 0
    assert repeat["unchanged"] == 1


def test_sync_notes_never_overwrites_manual_file(tmp_path: Path, monkeypatch):
    root, item_dir = _item_with_catalog(tmp_path)
    target = item_dir / zotero.GENERATED_FILENAME
    target.write_text("Human-authored notes stay put.", encoding="utf-8")
    monkeypatch.setattr(zotero, "fetch_notes", lambda citekeys: {"Example2024Notes": [{"note": "<p>Synced</p>"}]})

    summary = zotero.sync_notes(root)

    assert summary["conflicts"] == ["Books/Example_2024_Notes/zotero-notes.md"]
    assert target.read_text(encoding="utf-8") == "Human-authored notes stay put."


def test_sync_notes_dry_run_does_not_write(tmp_path: Path, monkeypatch):
    root, item_dir = _item_with_catalog(tmp_path)
    monkeypatch.setattr(zotero, "fetch_notes", lambda citekeys: {"Example2024Notes": [{"note": "<p>Synced</p>"}]})

    summary = zotero.sync_notes(root, dry_run=True)

    assert summary["updated"] == 1
    assert summary["dry_run"] is True
    assert not (item_dir / zotero.GENERATED_FILENAME).exists()


def test_sync_notes_supports_video_item_folders(tmp_path: Path, monkeypatch):
    root, item_dir = _item_with_catalog(tmp_path, content_root="Videos")
    monkeypatch.setattr(zotero, "fetch_notes", lambda citekeys: {"Example2024Notes": [{"note": "<p>Video note</p>"}]})

    summary = zotero.sync_notes(root)

    assert summary["items"] == 1
    assert summary["updated"] == 1
    assert (item_dir / zotero.GENERATED_FILENAME).is_file()


def test_fetch_notes_uses_better_bibtex_json_rpc(monkeypatch):
    sent = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"jsonrpc": "2.0", "id": 1, "result": {"Example2024Notes": [{"note": "<p>x</p>"}]}}

    def fake_post(url, **kwargs):
        sent["url"] = url
        sent.update(kwargs)
        return Response()

    monkeypatch.setattr(zotero.httpx, "post", fake_post)

    assert zotero.fetch_notes(["Example2024Notes"]) == {"Example2024Notes": [{"note": "<p>x</p>"}]}
    assert sent["json"] == {
        "jsonrpc": "2.0",
        "method": "item.notes",
        "params": [["Example2024Notes"]],
        "id": 1,
    }


def test_fetch_notes_classifies_unreachable_service(monkeypatch):
    def fail(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(zotero.httpx, "post", fail)

    with pytest.raises(errors.ZoteroUnreachableError):
        zotero.fetch_notes(["Example2024Notes"])
