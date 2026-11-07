"""Explicit, local Better BibTeX note synchronization for library items.

Targets are discovered through the shared side-effect-free inventory:
explicit mapped items use ``Notes/<item_id>-zotero.md``; legacy book, paper,
and video item directories without an explicit mapping keep their existing
``zotero-notes.md`` location.  Only files bearing the generated-file marker
are updated or removed; manual files at either target are conflicts and are
left untouched.  An unmigrated legacy generated copy corresponding to a
mapped item is reported as a conflict before any new target is created, so
synchronization never maintains two searchable copies of one item's notes.
"""
from html.parser import HTMLParser
from pathlib import Path
import re

import httpx

from . import config, errors, items, structure

GENERATED_MARKER = "<!-- library-query:zotero-notes -->"
GENERATED_FILENAME = "zotero-notes.md"

CITEKEY_MARKER = re.compile(r"^<!--\s*citekey:\s*(?P<citekey>.*?)\s*-->", re.IGNORECASE)


class _NoteTextParser(HTMLParser):
    """Convert Zotero's HTML note fragments into plain Markdown-like text."""

    _BLOCK_TAGS = {"blockquote", "div", "h1", "h2", "h3", "h4", "li", "p", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001 - HTMLParser callback
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "br":
            self._newline()
        elif tag in self._BLOCK_TAGS:
            self._newline()
            if tag == "li":
                self.parts.append("- ")
            elif tag == "blockquote":
                self.parts.append("> ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if not self._ignored_depth and tag in self._BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = []
        for line in "".join(self.parts).splitlines():
            cleaned = " ".join(line.split())
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines)


def notes_to_markdown(note_html: str) -> str:
    """Strip Zotero note HTML while preserving paragraph and quote boundaries."""
    parser = _NoteTextParser()
    parser.feed(note_html)
    parser.close()
    return parser.text()


def fetch_notes(citekeys: list[str]) -> dict[str, list[dict]]:
    """Fetch child notes through Better BibTeX's local JSON-RPC API."""
    if not citekeys:
        return {}
    request = {
        "jsonrpc": "2.0",
        "method": "item.notes",
        "params": [citekeys],
        "id": 1,
    }
    try:
        response = httpx.post(config.ZOTERO_BBT_URL, json=request, timeout=config.ZOTERO_BBT_TIMEOUT)
    except httpx.HTTPError as exc:
        raise errors.ZoteroUnreachableError(
            f"Could not reach Better BibTeX at {config.ZOTERO_BBT_URL}: {exc}"
        ) from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise errors.ZoteroUnreachableError(
            f"Better BibTeX returned HTTP {response.status_code} from {config.ZOTERO_BBT_URL}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise errors.ZoteroInvalidResponseError("Better BibTeX returned non-JSON note data") from exc
    if not isinstance(payload, dict):
        raise errors.ZoteroInvalidResponseError("Better BibTeX returned an invalid JSON-RPC response")
    if payload.get("error"):
        error = payload["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise errors.ZoteroInvalidResponseError(f"Better BibTeX rejected item.notes: {message}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise errors.ZoteroInvalidResponseError("Better BibTeX item.notes result is not an object")
    return {str(key): value for key, value in result.items() if isinstance(value, list)}


def _generated_citekey(path: Path) -> str | None:
    """Read the citekey marker of an existing generated note file, if present."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:400]
    except OSError:
        return None
    for line in head.splitlines():
        match = CITEKEY_MARKER.match(line.strip())
        if match:
            return match.group("citekey")
    return None


def _note_targets(root: Path, inventory: dict, bib_cache: dict) -> tuple[list[dict], list[str], int]:
    """Resolve note-sync targets from the shared inventory.

    Returns ``(targets, conflicts, skipped_without_citekey)``.  Each target is
    ``{"owner", "citekey", "target" (absolute Path), "target_rel"}``.  Citekey
    ownership stays unambiguous: duplicate effective citekeys across distinct
    items and legacy directories whose citekey belongs to a mapped item are
    reported as conflicts, never synced.
    """
    conflicts: list[str] = []
    targets: list[dict] = []
    skipped = 0

    citekey_owner: dict[str, str] = {}

    def claim(citekey: str, owner: str) -> bool:
        existing = citekey_owner.setdefault(citekey, owner)
        if existing != owner:
            conflicts.append(
                f"citekey {citekey} has ambiguous ownership ({existing} vs {owner}); skipped"
            )
            return False
        return True

    mapped_by_citekey: dict[str, str] = {}
    for item_id, record in sorted(inventory["records"].items()):
        citekey = items.effective_citekey(inventory, record, bib_cache)
        if not citekey:
            skipped += 1
            continue
        if not claim(citekey, f"item {item_id}"):
            continue
        mapped_by_citekey[citekey] = item_id
        targets.append({
            "owner": f"item {item_id}",
            "citekey": citekey,
            "target": root / items.generated_note_relpath(item_id),
            "target_rel": items.generated_note_relpath(item_id),
        })

    handled_note_rels: set[str] = set()
    for dir_rel, entry in sorted(inventory["legacy_items"].items()):
        dir_path = root / dir_rel
        # A legacy generated copy that corresponds to a mapped item — by
        # explicit attachment mapping or its citekey marker — is an unmigrated
        # copy: report it and never create a second note for that item.
        legacy_note = dir_path / GENERATED_FILENAME
        if legacy_note.is_file():
            handled_note_rels.add(f"{dir_rel}/{GENERATED_FILENAME}")
            marker_citekey = _generated_citekey(legacy_note)
            owner = mapped_by_citekey.get(marker_citekey or "")
            if owner is not None:
                conflicts.append(
                    f"unmigrated generated note {dir_rel}/{GENERATED_FILENAME} belongs to "
                    f"{owner}; migrate it before synchronizing"
                )
                # Drop the mapped target so no duplicate is created and no
                # copy is silently removed.
                targets[:] = [t for t in targets if t["owner"] != f"item {owner}"]
                continue

        citekey = None
        ambiguous = False
        for rel in sorted(entry["files"]):
            metadata = structure.resolve_metadata(root / rel, root, bib_cache, inventory=inventory)
            file_citekey = metadata.get("citekey")
            if file_citekey and citekey and file_citekey != citekey:
                ambiguous = True
            citekey = citekey or file_citekey
        if ambiguous or not citekey:
            if ambiguous:
                conflicts.append(f"conflicting citekeys in legacy item {dir_rel}; skipped")
            else:
                skipped += 1
            continue
        if citekey in mapped_by_citekey:
            # The whole legacy directory correlates to a mapped item; without
            # a generated marker the correlation is ambiguous, which is a
            # conflict rather than a guess.
            conflicts.append(
                f"legacy item {dir_rel} correlates to {mapped_by_citekey[citekey]} by citekey "
                f"{citekey} but has no explicit mapping; review before synchronizing"
            )
            targets[:] = [t for t in targets if t["owner"] != f"item {mapped_by_citekey[citekey]}"]
            continue
        if not claim(citekey, f"legacy item {dir_rel}"):
            continue
        targets.append({
            "owner": f"legacy item {dir_rel}",
            "citekey": citekey,
            "target": dir_path / GENERATED_FILENAME,
            "target_rel": f"{dir_rel}/{GENERATED_FILENAME}",
        })

    # Generated notes in directories fully claimed by a record never surface
    # as legacy items; scan the shelves directly so an unmigrated copy is
    # still reported instead of silently duplicated.
    for note_path in sorted(root.glob("*/*/zotero-notes.md")):
        note_rel = str(note_path.relative_to(root)).replace("\\", "/")
        if note_rel in handled_note_rels or not note_path.is_file():
            continue
        try:
            head = note_path.read_text(encoding="utf-8", errors="replace")[:400]
        except OSError:
            continue
        if not head.startswith(GENERATED_MARKER):
            continue
        marker_citekey = _generated_citekey(note_path)
        owner = mapped_by_citekey.get(marker_citekey or "")
        if owner is not None:
            conflicts.append(
                f"unmigrated generated note {note_rel} belongs to {owner}; "
                "migrate it before synchronizing"
            )
            targets[:] = [t for t in targets if t["owner"] != f"item {owner}"]

    return targets, conflicts, skipped


def _render_note_file(citekey: str, notes: list[dict]) -> str:
    rendered = []
    for note in notes:
        if not isinstance(note, dict) or not isinstance(note.get("note"), str):
            continue
        text = notes_to_markdown(note["note"])
        if text:
            rendered.append(text)
    if not rendered:
        return ""
    sections = "\n\n---\n\n".join(rendered)
    return (
        f"{GENERATED_MARKER}\n"
        f"<!-- citekey: {citekey} -->\n\n"
        "# Zotero notes and annotations\n\n"
        f"{sections}\n"
    )


def sync_notes(root: Path, dry_run: bool = False) -> dict:
    """Sync Zotero child notes into generated, searchable item Markdown files.

    Only files bearing ``GENERATED_MARKER`` are updated or removed.  A manually
    authored file at either the mapped (``Notes/<item_id>-zotero.md``) or
    legacy (``<item-dir>/zotero-notes.md``) target is reported as a conflict
    and left untouched.  Endpoint failures raise before any write, so existing
    note bytes always survive a failed sync.
    """
    inventory = items.build_inventory(root)
    bib_file = root / "catalog.bib"
    bib_cache = structure.parse_catalog_bib(bib_file) if bib_file.is_file() else {}
    targets, conflicts, skipped = _note_targets(root, inventory, bib_cache)

    # Deduplicate citekeys for the single fetch while keeping per-item
    # ownership resolved above.
    notes_by_key = fetch_notes(sorted({t["citekey"] for t in targets}))
    summary = {
        "items": len(targets),
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
        "conflicts": conflicts,
        "skipped_without_citekey": skipped,
        "dry_run": dry_run,
    }
    for target in targets:
        destination = target["target"]
        rendered = _render_note_file(target["citekey"], notes_by_key.get(target["citekey"], []))
        existing = destination.read_text(encoding="utf-8", errors="replace") if destination.exists() else ""
        if destination.exists() and not existing.startswith(GENERATED_MARKER):
            conflicts.append(str(target["target_rel"]))
            continue
        if rendered:
            if existing == rendered:
                summary["unchanged"] += 1
            else:
                summary["updated"] += 1
                if not dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(rendered, encoding="utf-8")
        elif destination.exists():
            summary["removed"] += 1
            if not dry_run:
                destination.unlink()
    return summary
