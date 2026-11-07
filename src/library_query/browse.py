"""Offline human browsing and explicitly generated Markdown navigation.

`browse` renders a readable, side-effect-free inventory from the shared item
mapping — no Qdrant, embeddings, Zotero, Git, or runtime manifest required.
`browse --write` opts into updating reserved managed blocks in the root and
shelf READMEs plus ``Collections/subjects.md``: user prose outside the markers
is preserved byte-for-byte, all targets are validated before anything is
written, each changed file is replaced atomically, and unchanged inputs leave
file bytes and modification times untouched.
"""
from pathlib import Path
import os
import posixpath
import unicodedata

from . import items, structure

BROWSE_JSON_VERSION = 1

MANAGED_START = "<!-- library-query:browse:start -->"
MANAGED_END = "<!-- library-query:browse:end -->"

#: RFC 3986 unreserved characters; everything else is percent-encoded as UTF-8.
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


# --- portable links ------------------------------------------------------------


def encode_segment(segment: str) -> str:
    """Percent-encode one path segment as UTF-8, leaving unreserved characters.

    Encoding starts from the filesystem name exactly once: a literal ``%`` in
    a filename becomes ``%25``, so ``study%20results.pdf`` never aliases
    ``study results.pdf``.
    """
    out: list[str] = []
    for char in segment:
        if char in _UNRESERVED:
            out.append(char)
        else:
            out.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
    return "".join(out)


def encode_rel_path(rel_path: str) -> str:
    """Encode each segment of a repository-relative POSIX path, keeping separators."""
    return "/".join(encode_segment(part) for part in rel_path.split("/"))


def relative_link(from_dir: str, to_rel: str) -> str:
    """Return the encoded relative Markdown destination from a directory to a file."""
    start = from_dir if from_dir else "."
    rel = posixpath.relpath(to_rel, start=start)
    return encode_rel_path(rel.replace("\\", "/"))


def _escape_label(label: str) -> str:
    """Escape a table-cell label so it cannot break the Markdown table."""
    return label.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


# --- inventory -----------------------------------------------------------------


def _item_warnings(inventory: dict, item_id: str | None, paths: list[str]) -> list[str]:
    warnings = []
    for warning in inventory["warnings"]:
        if warning.get("item_id") and warning["item_id"] == item_id:
            warnings.append(f"missing declared attachment: {warning['path']}")
    for path in paths:
        if path in inventory["unmapped"]:
            warnings.append(f"unmapped source uses catalog/filename fallbacks: {path}")
    return warnings


def _resolve_item_metadata(root: Path, inventory: dict, first_source: str | None) -> dict:
    if first_source is None:
        return {}
    return structure.resolve_metadata(
        root / first_source, root, inventory=inventory, diagnostics=inventory["conflicts"]
    )


def inventory_items(root: Path) -> tuple[list[dict], list[dict]]:
    """Assemble browse items from the shared inventory.

    Mapped records, legacy item directories (presented as one item per
    directory, not one per format), and unmapped flat files are returned with
    stable ordering: case-folded title, year (unknown after known), then item
    ID or relative path.
    """
    root = Path(root)
    inventory = items.build_inventory(root)
    by_rel = inventory["files"]
    out: list[dict] = []

    for item_id, record in sorted(inventory["records"].items()):
        existing = [a for a in record["attachments"] if a["path"] in by_rel]
        sources = [a["path"] for a in existing if a["role"] == "source"]
        supplements = [a["path"] for a in existing if a["role"] == "supplement"]
        note_paths = [a["path"] for a in existing if a["role"] == "note"]
        generated = inventory["note_by_item"].get(item_id)
        if generated and generated not in note_paths:
            note_paths.append(generated)
        metadata = _resolve_item_metadata(root, inventory, sources[0] if sources else None)
        metadata.setdefault("type", record["type"])
        out.append({
            "item_id": item_id,
            "type": record["type"],
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "year": metadata.get("year"),
            "subjects": metadata.get("subjects"),
            "tags": metadata.get("tags"),
            "citekey": metadata.get("citekey"),
            "sources": sources,
            "supplements": supplements,
            "notes": note_paths,
            "detail": record["record_path"],
            "legacy": False,
            "warnings": _item_warnings(inventory, item_id, sources + supplements),
        })

    for dir_rel, entry in sorted(inventory["legacy_items"].items()):
        content = sorted(entry["files"])
        if not content:
            continue
        metadata = _resolve_item_metadata(root, inventory, content[0])
        legacy_note = f"{dir_rel}/zotero-notes.md"
        note_paths = [legacy_note] if legacy_note in by_rel else []
        detail = f"{dir_rel}/README.md"
        out.append({
            "item_id": None,
            "type": metadata.get("type") or items.SHELF_TYPES.get(dir_rel.split("/")[0], "other"),
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "year": metadata.get("year"),
            "subjects": metadata.get("subjects"),
            "tags": metadata.get("tags"),
            "citekey": metadata.get("citekey"),
            "sources": content,
            "supplements": [],
            "notes": note_paths,
            "detail": detail if detail in by_rel else None,
            "legacy": True,
            "warnings": _item_warnings(inventory, None, content),
        })

    for rel in inventory["unmapped"]:
        info = by_rel.get(rel)
        if info is None or info["role"] != "content":
            continue
        shelf = rel.split("/", 1)[0]
        if shelf not in items.SHELF_TYPES:
            continue
        metadata = _resolve_item_metadata(root, inventory, rel)
        out.append({
            "item_id": None,
            "type": items.SHELF_TYPES[shelf],
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "year": metadata.get("year"),
            "subjects": metadata.get("subjects"),
            "tags": metadata.get("tags"),
            "citekey": metadata.get("citekey"),
            "sources": [rel],
            "supplements": [],
            "notes": [],
            "detail": None,
            "legacy": False,
            "warnings": [f"unmapped source uses catalog/filename fallbacks: {rel}"],
        })

    def sort_key(item: dict):
        title = (item["title"] or "").casefold()
        year = item["year"] if item["year"] is not None else 10**9
        tie = item["item_id"] or (item["sources"][0] if item["sources"] else "")
        return (title, year, tie)

    out.sort(key=sort_key)
    return out, inventory["warnings"] + inventory["conflicts"]


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _fold(text: str) -> str:
    """Casefold, drop accents, turn punctuation into spaces, collapse whitespace.

    ``"$100 Startup"`` becomes ``"100 startup"``.
    """
    plain = _plain(text)
    return " ".join("".join(c if c.isalnum() else " " for c in plain).split())


def _words(text: str) -> set[str]:
    """Words of ``text`` with punctuation both as a separator and as nothing.

    ``"F*ck"`` yields ``f``, ``ck`` and ``fck``, so a search typed either way finds it.
    """
    plain = _plain(text)
    joined = "".join(c for c in plain if c.isalnum() or c.isspace())
    return set(_fold(text).split()) | set(joined.split())


def _search_words(item: dict) -> set[str]:
    parts = [item.get("title") or "", item.get("citekey") or ""]
    parts += [str(p) for p in (item.get("authors") or [])]
    parts += [str(p) for p in (item.get("subjects") or [])]
    parts += [str(p) for p in (item.get("tags") or [])]
    return _words(" ".join(parts))


def matches_search(item: dict, query: str) -> bool:
    """True when every word of ``query`` starts a word in the item's title,
    authors, subjects, tags or citekey. Case, accents and punctuation are ignored.
    """
    query_words = _fold(query).split()
    if not query_words:
        return False
    haystack = _search_words(item)
    return all(any(h.startswith(w) for h in haystack) for w in query_words)


def browse_json(root: Path, search: str | None = None) -> dict:
    """Version-1 machine-readable browse output."""
    item_list, warnings = inventory_items(root)
    if search is not None:
        item_list = [i for i in item_list if matches_search(i, search)]
    return {"version": BROWSE_JSON_VERSION, "items": item_list, "warnings": warnings}


def format_human(root: Path, search: str | None = None) -> str:
    """Render the inventory as readable plain text grouped by shelf."""
    result = browse_json(root, search)
    item_list, warnings = result["items"], result["warnings"]
    lines: list[str] = []
    by_shelf: dict[str, list[dict]] = {}
    for item in item_list:
        by_shelf.setdefault(item["type"], []).append(item)
    for shelf, shelf_type in (("Books", "book"), ("Papers", "paper"), ("Videos", "video")):
        entries = by_shelf.get(shelf_type, [])
        lines.append(f"{shelf} ({len(entries)} items)")
        for item in entries:
            meta = []
            if item["authors"]:
                meta.append(", ".join(item["authors"]))
            if item["year"]:
                meta.append(str(item["year"]))
            meta_info = f" ({' | '.join(meta)})" if meta else ""
            suffix = " (legacy)" if item["legacy"] else ""
            lines.append(f"- {item['title'] or '(untitled)'}{meta_info}{suffix}")
            for label, paths in (("file", item["sources"]), ("supplement", item["supplements"])):
                for path in paths:
                    ext = path.rsplit(".", 1)[-1].lower()
                    name = f"{label}: {ext}" if label == "file" else f"supplement: {Path(path).stem}"
                    lines.append(f"    [{name}] {path}")
            for path in item["notes"]:
                lines.append(f"    [note] {path}")
            if item["detail"]:
                lines.append(f"    [metadata] {item['detail']}")
            for warning in item["warnings"]:
                lines.append(f"    ! {warning}")
        lines.append("")
    other = [i for i in item_list if i["type"] not in set(items.SHELF_TYPES.values())]
    if other:
        lines.append(f"Other ({len(other)} items)")
        for item in other:
            lines.append(f"- {item['title'] or item['sources'][0] if item['sources'] else '(untitled)'}")
        lines.append("")
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            if isinstance(warning, dict):
                lines.append(f"- {warning.get('reason')}: {warning.get('path')} {warning.get('detail', '')}")
            else:
                lines.append(f"- {warning}")
    return "\n".join(lines).rstrip() + "\n"


# --- managed navigation generation ---------------------------------------------


def _format_links(from_dir: str, item: dict, existing: set[str]) -> str:
    links = []
    for path in item["sources"]:
        if path not in existing:
            continue
        ext = path.rsplit(".", 1)[-1].lower()
        links.append(f"[{ext}]({relative_link(from_dir, path)})")
    for path in item["supplements"]:
        if path not in existing:
            continue
        label = Path(path).stem
        links.append(
            f"[supplement: {_escape_label(label)}]({relative_link(from_dir, path)})"
        )
    if item["detail"] and item["detail"] in existing:
        links.append(f"[metadata]({relative_link(from_dir, item['detail'])})")
    for path in item["notes"]:
        if path not in existing:
            continue  # a missing optional note never produces a broken link
        links.append(f"[notes]({relative_link(from_dir, path)})")
    return " ".join(links)


def _table_rows(from_dir: str, entries: list[dict], existing: set[str]) -> list[str]:
    rows = ["| Title | Authors | Year | Subjects | Files |", "| --- | --- | --- | --- | --- |"]
    for item in entries:
        title = _escape_label(str(item["title"] or "(untitled)"))
        authors = _escape_label(", ".join(item["authors"] or []))
        year = str(item["year"]) if item["year"] is not None else ""
        subjects = _escape_label(", ".join(item["subjects"] or []))
        files = _format_links(from_dir, item, existing)
        rows.append(f"| {title} | {authors} | {year} | {subjects} | {files} |")
    return rows


def _managed_block(body: str) -> str:
    # No trailing newline inside the block: the suffix after the end marker is
    # preserved verbatim on regeneration, which keeps repeated generation
    # byte-identical.
    return f"{MANAGED_START}\n{body.rstrip()}\n{MANAGED_END}"


def _subject_block(from_dir: str, item_list: list[dict], existing: set[str]) -> str:
    rows = []
    subjects: dict[str, list[dict]] = {}
    for item in item_list:
        for subject in item["subjects"] or []:
            subjects.setdefault(subject, []).append(item)
    for subject in sorted(subjects, key=str.casefold):
        rows.append(f"### {_escape_label(subject)}")
        rows.extend(_table_rows(from_dir, subjects[subject], existing))
        rows.append("")
    return _managed_block("\n".join(rows).rstrip() or "_No repository-local subjects yet._")


def _parse_managed(content: str) -> tuple[str, str] | None:
    """Split content around one well-formed managed block.

    Returns ``(prefix, suffix)`` (each including everything outside the block,
    with the block markers removed) or None when no complete block exists.
    """
    starts = content.count(MANAGED_START)
    ends = content.count(MANAGED_END)
    if starts == 0 and ends == 0:
        return None
    if starts != 1 or ends != 1:
        raise ValueError("malformed or duplicate managed markers")
    start = content.index(MANAGED_START)
    end = content.index(MANAGED_END)
    if end < start:
        raise ValueError("managed end marker precedes start marker")
    prefix = content[:start]
    suffix = content[end + len(MANAGED_END):]
    return prefix, suffix


def _atomic_replace(path: Path, content: str) -> None:
    """Replace a file atomically via a same-directory temp file."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_navigation(root: Path) -> dict:
    """Generate managed navigation blocks, preflighting every target first.

    Only managed blocks in the root/shelf READMEs and ``Collections/subjects.md``
    are written; source files, item records, manifests, and hash caches are
    never touched.  A conflict anywhere aborts before any write.
    """
    root = Path(root)
    inventory = items.build_inventory(root)
    item_list, _ = inventory_items(root)
    existing = set(inventory["files"])
    conflicts: list[str] = []

    shelf_targets = {
        "Books": [i for i in item_list if i["type"] == "book"],
        "Papers": [i for i in item_list if i["type"] == "paper"],
        "Videos": [i for i in item_list if i["type"] == "video"],
    }

    # Root overview: shelf counts and links to shelves, the subject index, and
    # existing curated collection pages (curated pages are navigation, so they
    # live outside the inventory's file map).
    collections_dir = root / "Collections"
    curated = []
    if collections_dir.is_dir():
        curated = [
            f"Collections/{name}"
            for name in sorted(os.listdir(collections_dir))
            if name.endswith(".md")
            and name != "subjects.md"
            and (collections_dir / name).is_file()
        ]
    counts = {shelf: len(entries) for shelf, entries in shelf_targets.items()}
    overview = ["## Library overview", ""]
    for shelf in ("Books", "Papers", "Videos"):
        overview.append(f"- [{shelf}]({shelf}/README.md) — {counts[shelf]} items")
    overview.append("- [Subjects](Collections/subjects.md)")
    for rel in curated:
        overview.append(f"- [{Path(rel).stem}]({encode_rel_path(rel)})")
    overview_body = _managed_block("\n".join(overview))

    subjects_body = _subject_block("Collections", item_list, existing)

    planned: dict[str, str] = {}
    for shelf, entries in shelf_targets.items():
        if not (root / shelf).is_dir():
            continue
        body = _managed_block("\n".join(_table_rows(shelf, entries, existing)))
        planned[f"{shelf}/README.md"] = body
    planned["README.md"] = overview_body
    planned["Collections/subjects.md"] = subjects_body

    # Preflight every target before any write: markers, ownership, permissions.
    finals: dict[str, str] = {}
    for rel, body in sorted(planned.items()):
        path = root / rel
        is_subject_page = rel == "Collections/subjects.md"
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                conflicts.append(f"{rel}: unreadable: {exc}")
                continue
            try:
                split = _parse_managed(content)
            except ValueError as exc:
                conflicts.append(f"{rel}: {exc}")
                continue
            if split is None:
                if is_subject_page:
                    conflicts.append(
                        f"{rel}: existing file has no managed markers; refusing to overwrite manual writing"
                    )
                    continue
                addition = content
                if addition and not addition.endswith("\n"):
                    addition += "\n"
                if addition and not addition.endswith("\n\n"):
                    addition += "\n"
                finals[rel] = addition + body + "\n"
            else:
                prefix, suffix = split
                finals[rel] = prefix + body + suffix
            if not os.access(path, os.W_OK):
                conflicts.append(f"{rel}: not writable")
        else:
            parent = path.parent
            if not parent.is_dir():
                if is_subject_page:
                    parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Never create shelf directories just to hold navigation.
                    conflicts.append(f"{rel}: directory {parent.relative_to(root)} does not exist")
                    continue
            if not os.access(parent, os.W_OK):
                conflicts.append(f"{rel}: not writable")
            finals[rel] = body + "\n"

    if conflicts:
        return {"written": [], "unchanged": [], "conflicts": conflicts}

    written: list[str] = []
    unchanged: list[str] = []
    for rel, content in sorted(finals.items()):
        path = root / rel
        if path.exists() and path.read_text(encoding="utf-8") == content:
            unchanged.append(rel)  # byte-identical: leave bytes and mtime alone
            continue
        _atomic_replace(path, content)
        written.append(rel)
    return {"written": written, "unchanged": unchanged, "conflicts": []}
