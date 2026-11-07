"""Read-only, offline migration planning for the flat-shelf layout.

``plan-layout`` inventories legacy per-item directories and flat shelves,
proposes title-first NFC filenames, derives stable UUIDv5 preview item IDs
from scanned normalized relative paths, and emits a versioned, hash-checked
migration map with conflicts and owner-project prerequisites.

The planner never moves or deletes anything, never writes metadata/catalog or
runtime-state files, never updates the persisted hash cache, never migrates
or repairs the ledger manifest, and never contacts Zotero, Qdrant, embedding
services, or Git.  Hashing uses the pure ``ledger.file_hash`` reader, not the
auto-writing ledger scanner/loader.
"""
from pathlib import Path
import os
import re
import unicodedata
from urllib.parse import unquote

import yaml

from . import browse, items, ledger, structure, zotero

PLAN_VERSION = 1
NAMING_ALGORITHM_VERSION = 1
BASE_STEM_CEILING_BYTES = 180

GENERIC_SOURCE_STEMS = {
    "book", "ebook", "paper", "article", "document", "doc", "file",
    "source", "text", "full", "main", "monograph",
}

ACQUISITION_NOISE = re.compile(
    r"(?i)\b(z-?lib(?:\.org)?|libgen(?:\.fresh)?|annas[ -]?archive|archive\.org)\b"
)
DUPLICATE_MARKER = re.compile(r"-\(?\d+\)?$")
YEAR_TOKEN = re.compile(r"^(?P<head>.*)-(?P<year>\d{4})$")
VIDEO_ID = re.compile(r"\[([^\]]+)\]")
LANG_SUFFIX = re.compile(r"\.(?P<lang>[a-z]{2,3}(?:-[a-z0-9]{2,8})*)\.md$", re.IGNORECASE)

WINDOWS_RESERVED = (
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)

#: Fields the version-1 record owns; everything else in a legacy README's
#: frontmatter is preserved as an extra field.
RECORD_FIELDS = {
    "schema_version", "item_id", "type", "citekey", "title", "authors",
    "author", "year", "subjects", "status", "identifier", "attachments",
}


# --- naming --------------------------------------------------------------------

def slugify(value: str) -> str:
    """Lowercase hyphen-separated NFC slug retaining diacritics.

    Unsafe and reserved characters (Windows ``<>:"/\\|?*``), control
    characters, and trailing spaces/dots are removed rather than escaped; the
    result is always safe to emit on every supported filesystem.
    """
    value = unicodedata.normalize("NFC", value).lower()
    out = []
    for char in value:
        if ord(char) < 32 or ord(char) == 127 or char in '<>:"/\\|?*':
            out.append(" ")
        else:
            out.append(char)
    slug = re.sub(r"\s+", "-", "".join(out).strip())
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip(".-")


def clean_label(stem: str) -> str:
    """Slugify an original filename stem and strip acquisition-site noise."""
    slug = slugify(stem)
    slug = ACQUISITION_NOISE.sub("", slug)
    slug = re.sub(r"\(\s*\)", "", slug)  # empty acquisition parens
    slug = DUPLICATE_MARKER.sub("", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-")


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate to at most ``max_bytes`` UTF-8 bytes at a code-point boundary."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    cut = encoded[:max_bytes]
    while cut:
        try:
            return cut.decode("utf-8")
        except UnicodeDecodeError:
            cut = cut[:-1]
    return ""


def source_stem(title_slug: str, year: int | None, filename_stem: str) -> str:
    """Title-first stem for a book/paper source or supplement.

    Supplements retain a meaningful distinguishing label from their original
    filename; primary formats of the same item collapse onto the title slug
    and are told apart by their extension.
    """
    label = clean_label(filename_stem)
    if label in GENERIC_SOURCE_STEMS:
        label = ""
    if label == title_slug:
        label = ""
    elif label.startswith(title_slug + "-"):
        label = label[len(title_slug) + 1:]
    stem = title_slug
    if year and not stem.endswith(f"-{year}"):
        stem += f"-{year}"
    if label:
        stem += f"-{label}"
    if stem.lower() in WINDOWS_RESERVED:
        # Windows reserves these names with or without an extension.
        stem += "-doc"
    return stem


def video_stem(title_slug: str, filename: str) -> str:
    """Stem retaining video IDs and transcript language/provenance distinctions."""
    stem_part = filename.rsplit(".", 1)[0]
    parts = [title_slug]
    vid = VIDEO_ID.search(stem_part)
    if vid:
        parts.append(slugify(vid.group(1)))
    if "transcript" in stem_part.lower():
        parts.append("transcript")
    lang = LANG_SUFFIX.search(filename)
    if lang:
        parts.append(slugify(lang.group("lang")))
    if len(parts) == 1:
        label = clean_label(stem_part)
        if label and label != title_slug:
            parts.append(label)
    stem = "-".join(part for part in parts if part)
    if stem.lower() in WINDOWS_RESERVED:
        stem += "-doc"
    return stem


def fit_component(base: str, suffix: str, ext: str, limit_bytes: int | None) -> str | None:
    """Fit ``base + suffix + "." + ext`` into the component limit.

    Truncation happens only at complete code-point boundaries and preserves
    the trailing year token whenever it can fit; when even the required parts
    (year, format/provenance, extension) cannot fit, the fit fails so the
    caller reports a conflict instead of inventing a valid-looking target.
    """
    tail = f"{suffix}.{ext}"
    tail_len = len(tail.encode("utf-8"))
    if limit_bytes is not None and tail_len >= limit_bytes:
        return None
    budget = BASE_STEM_CEILING_BYTES if limit_bytes is None else min(
        BASE_STEM_CEILING_BYTES, limit_bytes - tail_len
    )
    if budget < 1:
        return None
    match = YEAR_TOKEN.match(base)
    head, year = (match.group("head"), match.group("year")) if match else (base, None)
    year_len = len(year.encode("utf-8")) + 1 if year else 0
    fitted = truncate_utf8(head, budget - year_len)
    stem = f"{fitted}-{year}" if year else fitted
    if not stem:
        return None
    if limit_bytes is not None and len((stem + tail).encode("utf-8")) > limit_bytes:
        return None
    return stem


def assign_unique_component(
    taken: dict[str, str], base: str, ext: str, digest: str, limit_bytes: int | None
) -> str | None:
    """Reserve a unique component name, escaping collisions with a hash suffix.

    ``taken`` maps the NFC+casefolded comparison key to the chosen component;
    the suffix starts at eight hex characters of the source SHA-256 and
    extends until the target is unique.
    """
    suffix = ""
    for width in range(8, 40):
        stem = fit_component(base, suffix, ext, limit_bytes)
        if stem is None:
            return None
        component = f"{stem}{suffix}.{ext}"
        key = unicodedata.normalize("NFC", component).casefold()
        if key not in taken:
            taken[key] = component
            return component
        suffix = f"-{digest[:width]}"
    return None


def probe_name_max(directory: Path) -> int | None:
    """Return the filesystem's byte-based NAME_MAX for a shelf, or None.

    A missing directory or an unsupported probe reports unknown; the plan
    never silently substitutes 255.
    """
    if not Path(directory).is_dir():
        return None
    try:
        return os.pathconf(str(directory), "PC_NAME_MAX")
    except (OSError, ValueError):
        return None


def build_naming(root: Path, probe=None) -> dict:
    """Assemble the portable naming contract for the plan."""
    probe = probe or probe_name_max
    component_limits = {}
    for shelf in items.SHELF_TYPES:
        value = probe(Path(root) / shelf)
        if value is None:
            component_limits[shelf] = {"value": None, "unit": None, "provenance": "unknown"}
        else:
            component_limits[shelf] = {
                "value": value,
                "unit": "bytes",
                "provenance": "planning-host-probe",
            }
    return {
        "algorithm_version": NAMING_ALGORITHM_VERSION,
        "emitted_normalization": "NFC",
        "collision_comparison": "NFC then full casefold",
        "base_stem_ceiling_bytes": BASE_STEM_CEILING_BYTES,
        "component_limits": component_limits,
        # Local probing cannot validate a different Zotero/applying host, and
        # no plan can promise the approved map never hits a length or path
        # limitation on that host.
        "applying_host_verified": False,
    }


# --- record conversion ---------------------------------------------------------


def build_record_content(
    item_id: str,
    item_type: str,
    metadata: dict,
    local_facts: dict,
    extra_fields: dict,
    body: str,
    attachments: list[dict],
) -> str:
    """Render the complete intended ``Metadata/<item_id>.md`` content.

    Bibliographic facts follow the resolution precedence; local subjects and
    status are carried only when locally present (including intentionally
    empty values); unknown legacy frontmatter fields are preserved.
    """
    data: dict = dict(extra_fields)
    data["schema_version"] = items.SCHEMA_VERSION
    data["item_id"] = item_id
    data["type"] = item_type
    if metadata.get("citekey"):
        data["citekey"] = metadata["citekey"]
    data["title"] = metadata.get("title")
    if metadata.get("authors"):
        data["authors"] = metadata["authors"]
    if metadata.get("year"):
        data["year"] = metadata["year"]
    if "subjects" in local_facts:
        data["subjects"] = local_facts["subjects"]
    if metadata.get("identifier"):
        data["identifier"] = metadata["identifier"]
    data["attachments"] = attachments
    frontmatter = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    text = f"---\n{frontmatter}---\n"
    body = body.strip("\n")
    if body:
        text += f"\n{body}\n"
    return text


def split_readme(path: Path) -> tuple[dict, str]:
    """Return ``(frontmatter, body)`` of a legacy README, both possibly empty."""
    if not path.is_file():
        return {}, ""
    content = path.read_text(encoding="utf-8", errors="replace")
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        data = yaml.safe_load(parts[1])
    except Exception:
        return {}, content
    return (data if isinstance(data, dict) else {}), parts[2]


# --- navigation rewrites -------------------------------------------------------


MARKDOWN_LINK = re.compile(r"(?P<bang>!?)\[[^\]]*\]\((?P<target>[^)\s]+)[^)]*\)")
WIKI_LINK = re.compile(r"\[\[(?P<target>[^\]|]+)(?:\|[^\]]*)?\]\]")


def _link_variants(target: str) -> str:
    """Normalize a link target for matching against repository paths."""
    target = unquote(target.split("#", 1)[0]).replace("\\", "/").lstrip("./")
    return unicodedata.normalize("NFC", target).strip("/")


def navigation_rewrites(root: Path, file_moves: dict[str, str], item_pages: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Propose curated-link conversions for the migration map.

    Only links pointing at affected legacy item directories or files are
    proposed; everything else stays untouched.  A directory link maps to the
    item's metadata/detail page, a file link to that file's planned flat
    target, and anything ambiguous becomes a conflict for owner review.
    """
    root = Path(root)
    rewrites: list[dict] = []
    conflicts: list[dict] = []
    candidates = ["README.md"]
    collections = root / "Collections"
    if collections.is_dir():
        candidates += [
            f"Collections/{name}"
            for name in sorted(os.listdir(collections))
            if name.endswith(".md") and name != "subjects.md"
        ]
    for rel in candidates:
        path = root / rel
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        # Generated navigation links live inside managed blocks and are
        # regenerated after migration; only human prose links are curated.
        while browse.MANAGED_START in content and browse.MANAGED_END in content:
            start = content.index(browse.MANAGED_START)
            end = content.index(browse.MANAGED_END, start) + len(browse.MANAGED_END)
            content = content[:start] + content[end:]
        for match in list(MARKDOWN_LINK.finditer(content)) + list(WIKI_LINK.finditer(content)):
            target = match.group("target")
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            normalized = _link_variants(target)
            if not normalized:
                continue
            proposal = None
            if normalized in file_moves:
                proposal = file_moves[normalized]
            elif normalized in item_pages:
                proposal = item_pages[normalized]
            else:
                # A link naming a file inside a legacy item directory.
                for old_dir, page in item_pages.items():
                    if normalized.startswith(old_dir + "/"):
                        proposal = page
                        break
            if proposal is None:
                if not (root / normalized).exists():
                    conflicts.append({
                        "reason": "ambiguous_curated_link",
                        "path": rel,
                        "detail": f"link target {target!r} cannot be resolved to a migrated item",
                    })
                continue
            rewrites.append({
                "page": rel,
                "link": target,
                "proposed_target": proposal,
            })
    return rewrites, conflicts


# --- the plan ------------------------------------------------------------------

PREREQUISITES = [
    {
        "id": "dual-layout-readers",
        "description": "Deploy module dual-layout readers and verify old-layout regression before moving anything.",
    },
    {
        "id": "inbox-processing-contract",
        "description": "Update the owning ingestion/Hermes inbox-processing instructions for flat shelves and one item record.",
    },
    {
        "id": "owner-preview-non-mutating",
        "description": "The owner-side import/preview must be genuinely non-mutating (no collection creation during previews).",
    },
    {
        "id": "git-clean-reviewed",
        "description": "Check the plan's source revision and a clean, reviewed working tree on the applying host.",
    },
    {
        "id": "file-hash-verification",
        "description": "Verify every planned file exists with the planned SHA-256 on the applying host before the first move.",
    },
    {
        "id": "applying-host-naming-fullpath",
        "applying_host_verified": False,
        "description": "Revalidate every complete target component, native full path, portability conflicts, and collisions "
                       "on the applying host. If its limits require different targets, regenerate and review the map "
                       "instead of renaming ad hoc.",
    },
    {
        "id": "zotero-relinking-verified",
        "status": "unverified",
        "description": "A relinking mechanism tested against the installed Zotero version must update existing linked-file "
                       "attachment keys in place (preferring relinkAttachmentFile()), preserving parent/attachment keys, "
                       "notes, and annotations. Direct database edits and delete/recreate are never acceptable.",
    },
    {
        "id": "zotero-backup-and-key-inventory",
        "description": "Take a normal Zotero backup and a local inventory of parent/attachment keys, versions, old paths, "
                       "and annotation/note relationships before moving anything.",
    },
    {
        "id": "linked-file-base-directory",
        "description": "Configure the linked-attachment base directory on the applying host and keep automatic linked-file "
                       "renaming disabled.",
    },
    {
        "id": "pause-ingestion-and-note-sync",
        "description": "Pause inbox processing and note synchronization for the batch; keep one authoring authority.",
    },
    {
        "id": "inverse-map-and-journal",
        "description": "Retain the inverse path map and a per-action journal so rollback can restore only verified files "
                       "and relink the same attachment keys to old paths.",
    },
    {
        "id": "catalog-export-after-relinking",
        "description": "Export the complete attachment catalog from Zotero after relinking; never restore a stale generated bibliography.",
    },
    {
        "id": "shallow-collections-cleanup-approval",
        "description": "Add required shallow type memberships first; remove only separately approved obsolete collection keys, "
                       "never items, and never infer deletion authority from names alone.",
    },
]


def plan_layout(root: Path, probe=None) -> dict:
    """Produce the version-1, read-only flat-layout migration plan."""
    root = Path(root)
    inventory = items.build_inventory(root)
    bib_file = root / "catalog.bib"
    bib_cache = structure.parse_catalog_bib(bib_file) if bib_file.is_file() else {}
    naming = build_naming(root, probe=probe)
    limits = {
        shelf: (entry["value"] if entry["value"] is not None else None)
        for shelf, entry in naming["component_limits"].items()
    }

    plan_items: list[dict] = []
    file_actions: list[dict] = []
    conflicts: list[dict] = list(inventory["conflicts"])
    generated_note_moves: dict[str, str] = {}

    # Per-shelf reserved name comparison keys for collision detection.
    taken: dict[str, dict[str, str]] = {shelf: {} for shelf in items.SHELF_TYPES}
    file_moves: dict[str, str] = {}
    item_pages: dict[str, str] = {}

    def hash_of(path: Path) -> tuple[str, int]:
        return ledger.file_hash(path), path.stat().st_size

    def retain(rel: str, path: Path, note: str | None = None) -> None:
        if path.is_symlink() or not path.is_file():
            return
        digest, size = hash_of(path)
        action = {
            "path": rel,
            "action": "retain",
            "sha256": digest,
            "size": size,
        }
        if note:
            action["note"] = note
        file_actions.append(action)

    # ---- legacy item directories ------------------------------------------------
    for dir_rel, entry in sorted(inventory["legacy_items"].items()):
        shelf, _ = dir_rel.split("/", 1)
        item_type = items.SHELF_TYPES[shelf]
        dir_path = root / dir_rel
        files = sorted(entry["files"])
        item_id = items.legacy_preview_item_id(dir_rel)
        record_rel = f"Metadata/{item_id}.md"
        note_target_rel = items.generated_note_relpath(item_id)

        # Anchor metadata on the file the catalog correlates when there is
        # one, so the item record carries corrected bibliographic facts even
        # when a supplement sorts first.
        anchor_rel = next(
            (rel for rel in files if rel in bib_cache),
            files[0],
        )
        metadata = structure.resolve_metadata(
            root / anchor_rel, root, bib_cache, inventory=inventory,
            diagnostics=inventory["conflicts"],
        )
        local_facts = structure.parse_readme_frontmatter(dir_path / "README.md")
        extra_fields, readme_body = split_readme(dir_path / "README.md")
        extra_fields = {
            key: value for key, value in extra_fields.items() if key not in RECORD_FIELDS
        }

        title_slug = slugify(str(metadata.get("title") or dir_rel.split("/", 1)[1]))
        if not title_slug:
            conflicts.append({
                "reason": "unnamable_item",
                "path": dir_rel,
                "detail": "no usable title for naming proposals",
            })
            continue
        year = metadata.get("year")

        seen_hashes: dict[str, str] = {}
        moves: list[dict] = []
        attachments: list[dict] = []
        dir_conflicts = False

        for rel in files:
            if rel.rsplit("/", 1)[-1] in {"README.md", "zotero-notes.md"}:
                continue  # handled as metadata conversion / note move below
            path = root / rel
            if path.is_symlink():
                conflicts.append({
                    "reason": "unknown_file",
                    "path": rel,
                    "detail": "symlinked file blocks retiring the legacy directory until reviewed",
                })
                dir_conflicts = True
                continue
            digest, size = hash_of(path)
            if digest in seen_hashes:
                conflicts.append({
                    "reason": "duplicate_bytes",
                    "path": rel,
                    "detail": f"identical bytes also at {seen_hashes[digest]}; review, never silently deduplicated",
                })
                dir_conflicts = True
                continue
            seen_hashes[digest] = rel
            ext = path.name.rsplit(".", 1)[-1].lower()
            if item_type == "video":
                base = video_stem(title_slug, path.name)
            else:
                base = source_stem(title_slug, year, path.name.rsplit(".", 1)[0])
            component = assign_unique_component(taken[shelf], base, ext, digest, limits[shelf])
            if component is None:
                conflicts.append({
                    "reason": "component_limit",
                    "path": rel,
                    "detail": "required year/format/provenance parts cannot fit the shelf component limit",
                })
                dir_conflicts = True
                continue
            target_rel = f"{shelf}/{component}"
            role = "supplement"
            if item_type == "video":
                role = "source"
            else:
                label = clean_label(path.name.rsplit(".", 1)[0])
                if label in GENERIC_SOURCE_STEMS or label == title_slug:
                    label = ""
                elif label.startswith(title_slug + "-"):
                    label = label[len(title_slug) + 1:]
                # A file with no distinguishing label beyond the title is a
                # source format; anything else is a meaningful supplement.
                role = "source" if not label else "supplement"
            attachments.append({"path": target_rel, "role": role})
            moves.append({
                "path": rel,
                "action": "move",
                "target": target_rel,
                "sha256": digest,
                "size": size,
                "reverse": {"from": target_rel, "to": rel},
            })
            file_moves[rel] = target_rel

        note_path = dir_path / "zotero-notes.md"
        if note_path.is_file():
            digest, size = hash_of(note_path)
            head = note_path.read_text(encoding="utf-8", errors="replace")[:400]
            if not head.startswith(zotero.GENERATED_MARKER):
                conflicts.append({
                    "reason": "manual_note_target",
                    "path": f"{dir_rel}/zotero-notes.md",
                    "detail": "manually authored file at a generated-note location; review before migration",
                })
                dir_conflicts = True
            elif (root / note_target_rel).exists():
                conflicts.append({
                    "reason": "note_target_exists",
                    "path": f"{dir_rel}/zotero-notes.md",
                    "detail": f"generated-note target {note_target_rel} already exists",
                })
                dir_conflicts = True
            else:
                generated_note_moves[f"{dir_rel}/zotero-notes.md"] = note_target_rel
                attachments.append({"path": note_target_rel, "role": "note"})
                moves.append({
                    "path": f"{dir_rel}/zotero-notes.md",
                    "action": "move",
                    "target": note_target_rel,
                    "sha256": digest,
                    "size": size,
                    "note": "preserve rendered note bytes so byte-identical notes keep document identities",
                    "reverse": {"from": note_target_rel, "to": f"{dir_rel}/zotero-notes.md"},
                })
                file_moves[f"{dir_rel}/zotero-notes.md"] = note_target_rel

        # Unknown files inside the directory block retiring it until reviewed.
        known = set(files) | {f"{dir_rel}/README.md", f"{dir_rel}/zotero-notes.md"}
        for child in sorted(dir_path.rglob("*")):
            child_rel = str(child.relative_to(root)).replace("\\", "/")
            if child_rel in known:
                continue
            if any(part.startswith(".") for part in child_rel.split("/")):
                continue
            if child.is_file():
                conflicts.append({
                    "reason": "unknown_file",
                    "path": child_rel,
                    "detail": "unsupported file blocks retiring the legacy directory until reviewed",
                })
                dir_conflicts = True

        readme_rel = f"{dir_rel}/README.md"
        if (dir_path / "README.md").is_file():
            digest, size = hash_of(dir_path / "README.md")
            moves.append({
                "path": readme_rel,
                "action": "convert-metadata",
                "target": record_rel,
                "sha256": digest,
                "size": size,
                "reverse": {"from": record_rel, "to": readme_rel},
            })

        record_content = build_record_content(
            item_id, item_type, metadata, local_facts, extra_fields, readme_body, attachments
        )
        plan_items.append({
            "item_id": item_id,
            "legacy_dir": dir_rel,
            "record_path": record_rel,
            "record": record_content,
            "blocked": dir_conflicts,
        })
        item_pages[dir_rel] = record_rel
        file_actions.extend(moves)

    # ---- everything else is retained --------------------------------------------
    legacy_dirs = set(inventory["legacy_items"])
    for rel, info in sorted(inventory["files"].items()):
        path = root / rel
        parts = rel.split("/")
        in_legacy_dir = (
            len(parts) >= 3
            and parts[0] in items.SHELF_TYPES
            and "/".join(parts[:2]) in legacy_dirs
        )
        if in_legacy_dir and rel not in file_moves:
            continue  # handled (or conflicted) with the legacy item above
        if rel in file_moves or rel in generated_note_moves.values():
            continue
        if parts[0] == "_Inbox":
            retain(rel, path, note="inbox stays the intake boundary; catalogue through the existing workflow")
        else:
            retain(rel, path)

    # Navigation files are excluded from the inventory roles; retain them (the
    # generated ones are regenerated by `browse --write` after migration).
    for rel in (
        "README.md",
        "Books/README.md", "Papers/README.md", "Videos/README.md", "Notes/README.md",
        "Collections/subjects.md",
    ):
        path = root / rel
        if path.is_file():
            retain(rel, path, note="regenerate managed navigation with browse --write after migration")

    rewrites, link_conflicts = navigation_rewrites(root, file_moves, item_pages)
    conflicts.extend(link_conflicts)

    return {
        "version": PLAN_VERSION,
        "items": plan_items,
        "file_actions": file_actions,
        "navigation_rewrites": rewrites,
        "conflicts": conflicts,
        "warnings": list(inventory["warnings"]),
        "prerequisites": PREREQUISITES,
        "naming": naming,
    }
