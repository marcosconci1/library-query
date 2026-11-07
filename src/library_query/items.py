"""Portable item records and the shared, side-effect-free library inventory.

Enforces:
- One Markdown record per bibliographic item at ``Metadata/<item_id>.md``.
- Explicit attachment mapping (path + role) instead of folder-name assumptions.
- A single validated path-to-item index built once per scan and reused by
  indexing, browsing, planning, and note synchronization.

The inventory is deliberately separate from persistence: it only reads library
files and never touches the ledger scanner/loader, whose hash-cache and
manifest-migration writes are an indexing-side responsibility.  Read-only
commands (``browse``, ``plan-layout``) can therefore run against a library copy
whose runtime manifest is absent, legacy, or corrupt without changing a byte.
"""
from pathlib import Path
import re
import unicodedata
import uuid

import yaml

from . import config, structure

#: Shelves and the validated item type each one carries.
SHELF_TYPES = {"Books": "book", "Papers": "paper", "Videos": "video"}

#: Valid attachment roles for item-record frontmatter.
ATTACHMENT_ROLES = {"source", "supplement", "note"}

#: Deterministic generated-note suffix for mapped items.
GENERATED_NOTE_SUFFIX = "-zotero.md"

#: Version 1 record schema.
SCHEMA_VERSION = 1

ITEM_ID_PATTERN = re.compile(r"^item-[0-9a-f]{32}$")

#: UUIDv5 name prefix for stable legacy preview IDs (no Git, no case folding).
LEGACY_ID_URN_PREFIX = "urn:library-query:legacy-item:v1:"


def generated_note_relpath(item_id: str) -> str:
    """Return the deterministic generated-note repository path for an item ID."""
    return f"Notes/{item_id}{GENERATED_NOTE_SUFFIX}"


def legacy_preview_item_id(rel_dir: str) -> str:
    """Derive the stable preview item ID for a legacy item directory.

    The name input is the NFC-normalized scanned POSIX relative directory, so
    canonically equivalent inputs (NFC vs NFD) derive the same ID even in a
    library copy without ``.git``.  Case is preserved: normalization or case
    collisions remain conflicts rather than authorizing an item merge.
    """
    normalized = unicodedata.normalize("NFC", rel_dir.replace("\\", "/"))
    digest = uuid.uuid5(uuid.NAMESPACE_URL, f"{LEGACY_ID_URN_PREFIX}{normalized}")
    return f"item-{digest.hex}"


def parse_item_record(path: Path, root: Path) -> tuple[dict | None, str | None]:
    """Parse and validate one ``Metadata/<item_id>.md`` record.

    Returns ``(record, conflict)``: exactly one is None.  The record keeps the
    raw frontmatter (unknown fields are preserved for lossless migration) plus
    the Markdown body, and adds ``item_id``, ``record_path``, and parsed
    ``attachments`` with repository-relative POSIX paths.
    """
    rel = str(path.relative_to(root)).replace("\\", "/")
    stem = path.stem
    if not ITEM_ID_PATTERN.match(stem):
        return None, f"invalid item_id in filename: {rel}"
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, f"unreadable item record {rel}: {exc}"

    if not content.startswith("---"):
        return None, f"item record {rel} has no frontmatter"
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, f"item record {rel} has malformed frontmatter"
    try:
        data = yaml.safe_load(parts[1])
    except Exception:
        data = None
    if not isinstance(data, dict):
        return None, f"item record {rel} has non-mapping frontmatter"

    if data.get("schema_version") != SCHEMA_VERSION:
        return None, f"item record {rel} has unsupported schema_version"
    item_id = str(data.get("item_id") or "")
    if item_id != stem or not ITEM_ID_PATTERN.match(item_id):
        return None, f"item record {rel} item_id does not match its filename"

    record_type = str(data.get("type") or "")
    if record_type not in set(SHELF_TYPES.values()):
        return None, f"item record {rel} has unsupported type {record_type!r}"

    raw_attachments = data.get("attachments")
    if not isinstance(raw_attachments, list) or not raw_attachments:
        return None, f"item record {rel} must declare a non-empty attachments list"

    attachments = []
    for entry in raw_attachments:
        if not isinstance(entry, dict):
            return None, f"item record {rel} has a non-object attachment entry"
        rel_path = str(entry.get("path") or "").replace("\\", "/")
        role = str(entry.get("role") or "")
        if not rel_path or rel_path.startswith("/") or ".." in rel_path.split("/"):
            return None, f"item record {rel} declares an unsafe attachment path {rel_path!r}"
        if role not in ATTACHMENT_ROLES:
            return None, f"item record {rel} declares unsupported attachment role {role!r}"
        attachments.append({"path": rel_path, "role": role})

    if not any(a["role"] == "source" for a in attachments):
        return None, f"item record {rel} declares no source attachment"

    for attachment in attachments:
        top = attachment["path"].split("/", 1)[0]
        if attachment["role"] == "note":
            if top != "Notes":
                return None, (
                    f"item record {rel} places note attachment {attachment['path']} outside Notes/"
                )
        elif top not in SHELF_TYPES:
            return None, (
                f"item record {rel} places attachment {attachment['path']} outside a source shelf"
            )
        expected_shelf = next(s for s, t in SHELF_TYPES.items() if t == record_type)
        if attachment["role"] != "note" and top != expected_shelf:
            return None, (
                f"item record {rel} has {record_type} type but attachment "
                f"{attachment['path']} is not in {expected_shelf}/"
            )

    return (
        {
            "item_id": item_id,
            "record_path": rel,
            "type": record_type,
            "frontmatter": data,
            "body": parts[2],
            "attachments": attachments,
        },
        None,
    )


def _contained(root: Path, rel_path: str) -> tuple[Path | None, str | None]:
    """Resolve a declared attachment path, rejecting escapes outside the root.

    Symlinks are resolved before the containment check so a record cannot
    address external bytes through a link inside the library.  The external
    target itself is never read here.
    """
    candidate = root / rel_path
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None, f"attachment path {rel_path} escapes the library root"
    return resolved, None


def record_facts(record: dict) -> dict:
    """Project the repository-local metadata fields of a validated record.

    Presence-aware: an explicit empty ``subjects`` list stays empty instead of
    falling back to catalog values.  Identifier subfields stay independent so a
    catalog DOI correction cannot erase a local video URL.
    """
    data = record["frontmatter"]
    facts: dict = {}
    if data.get("title"):
        facts["title"] = str(data["title"]).strip()
    authors = data.get("authors") or data.get("author")
    if authors:
        if isinstance(authors, list):
            facts["authors"] = [str(a).strip() for a in authors if str(a).strip()]
        else:
            facts["authors"] = [str(authors).strip()]
    year = data.get("year")
    if year is not None:
        try:
            facts["year"] = int(year)
        except (ValueError, TypeError):
            pass
    if "subjects" in data and data["subjects"] is not None:
        subjects = data["subjects"]
        if isinstance(subjects, list):
            facts["subjects"] = [str(s).strip() for s in subjects if str(s).strip()]
        else:
            facts["subjects"] = [s.strip() for s in str(subjects).split(",") if s.strip()]
    if data.get("citekey"):
        facts["citekey"] = str(data["citekey"]).strip()
    if isinstance(data.get("identifier"), dict):
        facts["identifier"] = {
            str(k): ("" if v is None else str(v)) for k, v in data["identifier"].items()
        }
    return facts


def load_records(root: Path) -> tuple[dict[str, dict], list[dict]]:
    """Load and validate every item record under ``Metadata/``.

    Returns ``(records_by_id, conflicts)``; conflicting records are excluded
    from the index so a malformed record can never attach metadata to a file.
    """
    records: dict[str, dict] = {}
    conflicts: list[dict] = []
    metadata_dir = root / "Metadata"
    if not metadata_dir.is_dir():
        return records, conflicts

    for path in sorted(metadata_dir.rglob("*.md")):
        record, conflict = parse_item_record(path, root)
        rel = str(path.relative_to(root)).replace("\\", "/")
        if conflict:
            conflicts.append({"reason": "invalid_item_record", "path": rel, "detail": conflict})
            continue
        if record["item_id"] in records:
            conflicts.append({
                "reason": "duplicate_item_id",
                "path": rel,
                "detail": f"item_id {record['item_id']} is also declared by "
                          f"{records[record['item_id']]['record_path']}",
            })
            continue
        records[record["item_id"]] = record

    # Duplicate non-empty citekeys across distinct records are ambiguous ownership.
    by_citekey: dict[str, str] = {}
    for item_id, record in records.items():
        citekey = record["frontmatter"].get("citekey")
        if not citekey:
            continue
        citekey = str(citekey).strip()
        if citekey in by_citekey:
            conflicts.append({
                "reason": "duplicate_citekey",
                "path": record["record_path"],
                "detail": f"citekey {citekey} is shared by {by_citekey[citekey]} and {item_id}",
            })
        else:
            by_citekey[citekey] = item_id

    # Portability collisions: NFC-then-casefold equivalence is the conservative
    # comparison policy; collisions stay conflicts rather than merging items.
    seen_paths: dict[str, str] = {}
    for record in records.values():
        for rel in [record["record_path"]] + [a["path"] for a in record["attachments"]]:
            key = unicodedata.normalize("NFC", rel).casefold()
            if key in seen_paths and seen_paths[key] != rel:
                conflicts.append({
                    "reason": "path_portability_collision",
                    "path": rel,
                    "detail": f"{rel} collides with {seen_paths[key]} under NFC+casefold",
                })
            else:
                seen_paths[key] = rel

    # Containment and duplicate attachment ownership.
    by_path: dict[str, str] = {}
    for item_id, record in records.items():
        contained_paths = []
        for attachment in record["attachments"]:
            resolved, conflict = _contained(root, attachment["path"])
            if conflict:
                conflicts.append({
                    "reason": "attachment_escapes_library",
                    "path": record["record_path"],
                    "detail": f"{attachment['path']}: {conflict}",
                })
                continue
            contained_paths.append(attachment)
            owner = by_path.get(attachment["path"])
            if owner is not None and owner != item_id:
                conflicts.append({
                    "reason": "ambiguous_attachment_ownership",
                    "path": attachment["path"],
                    "detail": f"claimed by both {owner} and {item_id}",
                })
            else:
                by_path[attachment["path"]] = item_id
        record["attachments"] = contained_paths

    return records, conflicts


def build_inventory(root: Path) -> dict:
    """Build the validated, side-effect-free inventory used by every consumer.

    The inventory classifies every file by role, maps records to attachments,
    detects legacy item directories, and collects classified conflicts and
    warnings.  It never calls the ledger scanner/loader and never writes
    hash caches, manifests, or any other runtime state.
    """
    root = Path(root)
    records, conflicts = load_records(root)
    warnings: list[dict] = []
    by_path: dict[str, str] = {}
    for item_id, record in records.items():
        for attachment in record["attachments"]:
            by_path.setdefault(attachment["path"], item_id)

    files: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        try:
            rel = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            continue
        if not path.is_file():
            continue
        role = structure.classify_file(path, root)
        if role == "navigation":
            continue
        files[rel] = {"path": path, "role": role}

    # Missing declared attachments must not attach metadata to unrelated files.
    for item_id, record in records.items():
        for attachment in record["attachments"]:
            if attachment["path"] not in files:
                warnings.append({
                    "reason": "missing_declared_attachment",
                    "path": attachment["path"],
                    "item_id": item_id,
                })

    # Deterministic generated-note mapping works even when the note is not
    # listed in attachments, and a missing optional note is never an error.
    note_by_item: dict[str, str] = {}
    for item_id in records:
        rel = generated_note_relpath(item_id)
        if rel in files:
            note_by_item[item_id] = rel

    # Genuine per-item directories: a supported content file nested below a
    # shelf lives in an item directory; flat shelf files do not.
    legacy_items: dict[str, dict] = {}
    unmapped: list[str] = []
    for rel, info in files.items():
        if info["role"] != "content":
            continue
        parts = rel.split("/")
        if parts[0] not in SHELF_TYPES or len(parts) < 3:
            if parts[0] in SHELF_TYPES and rel not in by_path:
                unmapped.append(rel)
            continue
        item_dir = "/".join(parts[:2])
        if rel in by_path:
            continue
        entry = legacy_items.setdefault(item_dir, {"files": [], "dir": item_dir})
        entry["files"].append(rel)

    for rel, info in files.items():
        if info["role"] != "content":
            continue
        parts = rel.split("/")
        if parts[0] == "Notes" and len(parts) >= 2:
            if rel not in by_path and rel not in note_by_item.values():
                warnings.append({
                    "reason": "unmapped_note",
                    "path": rel,
                    "detail": "note is not mapped to an item; indexed with type 'other'",
                })

    return {
        "root": root,
        "records": records,
        "by_path": by_path,
        "note_by_item": note_by_item,
        "files": files,
        "legacy_items": legacy_items,
        "unmapped": sorted(unmapped),
        "conflicts": conflicts,
        "warnings": warnings,
    }


def item_for_path(inventory: dict, rel_path: str) -> str | None:
    """Return the item ID claiming a repository-relative path, if any."""
    return inventory["by_path"].get(rel_path)


def effective_citekey(inventory: dict, record: dict, bib_cache: dict | None = None) -> str | None:
    """Resolve the citekey used to fetch an item's child notes.

    An explicit record citekey wins; otherwise a catalog entry correlated by a
    source attachment path supplies it.  Missing citekeys are reported by the
    caller, never guessed from titles.
    """
    citekey = record_facts(record).get("citekey")
    if citekey:
        return citekey
    if bib_cache is None:
        bib_file = inventory["root"] / "catalog.bib"
        bib_cache = structure.parse_catalog_bib(bib_file) if bib_file.is_file() else {}
    for attachment in record["attachments"]:
        if attachment["role"] != "source":
            continue
        entry = bib_cache.get(attachment["path"])
        if entry and entry.get("citekey"):
            return entry["citekey"]
    return None
