"""Structure classification, role separation, and metadata resolution.

Enforces:
- File roles: content (embeddable), metadata (enrichment only), navigation (ignored).
- Scope and type derived from directory hierarchy plus explicit item mapping.
- Metadata precedence: catalog.bib > explicit item record / README frontmatter >
  folder parsing (true legacy item directories only) > embedded > filename.

`catalog.bib` is the bibliographic authority when it is maintained by a
reference manager.  Explicit item records and README sidecars remain the home
for repository-local reading state and controlled subjects, including
intentionally empty values.
"""
from pathlib import Path
import re
import yaml

from . import config

# Permissive regex for folder-name parsing: Author_Year_Title
# Non-greedy author allows underscores in author names (e.g. García_Márquez_1967_Title)
FOLDER_PATTERN = re.compile(r"^(?P<author>.+?)_(?P<year>\d{4})_(?P<title>.+)$")

#: Source shelves; a file's parent being one of these means there is no item
#: sidecar above it (a shelf README is navigation, never a metadata sidecar).
SHELVES = ("Books", "Papers", "Videos", "Notes")


def normalize_tag(value: str) -> str:
    """Return a stable, case-insensitive representation of a Zotero tag."""
    return " ".join(str(value).split()).casefold()


def _split_tags(value: str) -> list[str]:
    """Parse BibTeX keyword/subject fields into unique normalized tags."""
    tags = []
    for raw_tag in re.split(r"[,;]", value):
        tag = normalize_tag(raw_tag)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def classify_file(path: Path, root: Path) -> str:
    """Classify file as 'content', 'metadata', or 'navigation'.
    
    - 'navigation': Collections/**, Templates/**, root README.md, .gitkeep, .DS_Store, hidden files
    - 'metadata': catalog.bib, item README.md sidecars, _Inbox/*.md
    - 'content': supported source files, including Videos/** timestamped Markdown
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        # Outside the library root: not ours to embed.
        return "navigation"
    parts = rel.parts

    # Any hidden component or common operational dotfile
    for part in parts:
        if part.startswith("."):
            return "navigation"

    # graphify-out/ is knowledge-graph tool output (a single 45k-char report that
    # the embedding API rejects), not library material. Without this rule one
    # unembeddable file aborts the whole index run.
    if parts and parts[0] == "graphify-out":
        return "navigation"

    # Collections and templates are navigation only.
    if parts and parts[0] in {"Collections", "Templates"}:
        return "navigation"

    # Repository root README.md is navigation
    if len(parts) == 1 and parts[0].lower() == "readme.md":
        return "navigation"

    # Item records are metadata: they enrich mapped sources only and are never
    # embedded as independent documents.
    if parts and parts[0] == "Metadata":
        return "metadata"

    # catalog.bib is metadata enrichment
    if path.name == "catalog.bib":
        return "metadata"

    # Shelf README files are generated/navigation content: they SHALL NOT
    # enrich every neighboring flat source.  Deeper README.md files remain
    # metadata sidecars (legacy item directories, Operations/, ...).
    if (
        len(parts) == 2
        and parts[0] in SHELVES
        and path.name.lower() == "readme.md"
    ):
        return "navigation"

    # README.md inside folders is a metadata sidecar
    if path.name.lower() == "readme.md":
        return "metadata"

    # Markdown in the inbox is pipeline bookkeeping, not library material. `_Inbox`
    # holds the notes the processing skill writes about its own work -- UNSORTED.md
    # parking notes and the like. Indexed as content, `_Inbox/UNSORTED.md` was
    # retrievable at 0.64 for "unsorted inbox items" and served operational
    # bookkeeping back as evidence. The books themselves arrive as PDF/EPUB and are
    # still classified as content, so nothing real is excluded here.
    if parts and parts[0] == "_Inbox" and path.suffix.lower() == ".md":
        return "metadata"

    # Operations/ holds operator guides in a separate evidence scope. Supported
    # documents here are content, but the exclusions above still win: hidden path
    # components, navigation roots, and README.md/metadata sidecars never become
    # standalone operational documents merely because of the directory.
    if parts and parts[0] == "Operations" and path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
        return "content"

    # Supported content files
    if path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
        return "content"

    return "navigation"


def _note_item_type(root: Path, rel_name: str) -> str | None:
    """Return the shelf type of the item owning a Notes/ file, if mapped.

    Mapped notes follow either an explicit ``role: note`` attachment path or
    the deterministic ``Notes/<item_id>-zotero.md`` naming.  Building the
    record set only scans ``Metadata/`` — never the ledger or the tree.
    """
    from . import items

    records, _ = items.load_records(root)
    item_id = rel_name[: -len(items.GENERATED_NOTE_SUFFIX)]
    record = records.get(item_id)
    if record is not None:
        return record["type"]
    for candidate_id, candidate in records.items():
        for attachment in candidate["attachments"]:
            if (
                attachment["role"] == "note"
                and attachment["path"] == f"Notes/{rel_name}"
            ):
                return candidate["type"]
    return None


def get_scope_and_type(path: Path, root: Path, inventory: dict | None = None) -> tuple[str, str]:
    """Derive workflow scope and document type from location and item mapping.

    Mapped note content under ``Notes/`` inherits the owning item's validated
    shelf type; unmapped supported notes stay ``type="other"`` and are flagged
    as warnings by the inventory, never merged into a neighboring item.  An
    item record never overrides inbox or operations scope.
    """
    rel = path.relative_to(root)
    parts = rel.parts
    if not parts:
        return ("catalogued", "other")

    top = parts[0]
    # Legacy libraries keep the catalog operator guide at the repository root.
    # It remains searchable explicitly, but never as catalogued source evidence.
    if len(parts) == 1 and top.casefold() == "zotero.md":
        return ("operations", "operation")
    if top == "_Inbox":
        return ("inbox", "inbox")
    elif top == "Books":
        return ("catalogued", "book")
    elif top == "Papers":
        return ("catalogued", "paper")
    elif top == "Videos":
        return ("catalogued", "video")
    elif top == "Operations":
        # Operator guides live in their own evidence scope, excluded from the
        # catalogued default research sees unless a caller asks for them.
        return ("operations", "operation")
    elif top == "Notes":
        rel_str = str(rel).replace("\\", "/")
        if inventory is not None:
            item_id = inventory["by_path"].get(rel_str)
            if item_id is None:
                for candidate_id, note_rel in inventory["note_by_item"].items():
                    if note_rel == rel_str:
                        item_id = candidate_id
                        break
            if item_id is not None and item_id in inventory["records"]:
                return ("catalogued", inventory["records"][item_id]["type"])
            return ("catalogued", "other")
        note_type = _note_item_type(root, rel_str.split("/")[-1])
        return ("catalogued", note_type or "other")
    else:
        return ("catalogued", "other")


def parse_folder_name(folder_name: str) -> dict:
    """Parse Author_Year_Title folder name fallback.
    
    If format matches, returns parsed author, year, title.
    Otherwise, returns title=folder_name, authors=None, year=None.
    """
    match = FOLDER_PATTERN.match(folder_name)
    if match:
        author_raw = match.group("author")
        year_raw = int(match.group("year"))
        title_raw = match.group("title").replace("-", " ").strip()
        authors = [author_raw.replace("_", " ").strip()]
        return {
            "title": title_raw,
            "authors": authors,
            "year": year_raw,
        }
    return {
        "title": folder_name.replace("-", " ").strip(),
        "authors": None,
        "year": None,
    }


def parse_readme_frontmatter(readme_path: Path) -> dict:
    """Extract YAML frontmatter from a sidecar README.md."""
    if not readme_path.is_file():
        return {}
    try:
        content = readme_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    if not content.startswith("---"):
        return {}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}

    try:
        data = yaml.safe_load(parts[1])
        if not isinstance(data, dict):
            return {}
    except Exception:
        return {}

    metadata = {}
    if "title" in data and data["title"]:
        metadata["title"] = str(data["title"]).strip()
    if "authors" in data and data["authors"]:
        authors = data["authors"]
        if isinstance(authors, list):
            metadata["authors"] = [str(a).strip() for a in authors if a]
        else:
            metadata["authors"] = [str(authors).strip()]
    elif "author" in data and data["author"]:
        author = data["author"]
        if isinstance(author, list):
            metadata["authors"] = [str(a).strip() for a in author if a]
        else:
            metadata["authors"] = [str(author).strip()]

    if "year" in data and data["year"]:
        try:
            metadata["year"] = int(data["year"])
        except (ValueError, TypeError):
            pass

    # Presence, not truthiness: an explicit empty subjects list is an
    # authoritative local value and must not fall back to catalog keywords.
    if "subjects" in data and data["subjects"] is not None:
        subjects = data["subjects"]
        if isinstance(subjects, list):
            metadata["subjects"] = [str(s).strip() for s in subjects if s]
        else:
            metadata["subjects"] = [s.strip() for s in str(subjects).split(",") if s.strip()]

    if "identifier" in data and isinstance(data["identifier"], dict):
        metadata["identifier"] = {str(k): str(v) for k, v in data["identifier"].items()}

    if "citekey" in data and data["citekey"]:
        metadata["citekey"] = str(data["citekey"]).strip()

    return metadata


def _catalog_file_lookup_keys(value: str) -> set[str]:
    """Return portable content paths from a BibTeX ``file`` field.

    Plain BibTeX exports may contain a path directly; Better BibTeX can also
    encode attachments as ``title:/absolute/path:media/type`` and export
    several attachments separated by ``;``.  In every case indexing is
    portable when the path below the library root is used as the lookup key;
    the absolute host prefix is deliberately ignored, so every attachment
    correlates to its exported parent without depending on where the library
    lives on the exporting machine.
    """
    normalized = value.replace("\\\\", "/")
    keys = {normalized.lstrip("./")}
    extensions = "|".join(re.escape(suffix.lstrip(".")) for suffix in config.SUPPORTED_EXTENSIONS)
    content_path = re.compile(
        rf"((?:Books|Papers|Videos|_Inbox|Operations)/[^;]*?\.(?:{extensions}))",
        re.IGNORECASE,
    )
    for match in content_path.finditer(normalized):
        keys.add(match.group(1))
    return keys


def parse_catalog_bib(bib_path: Path) -> dict[str, dict]:
    """Parse catalog.bib entries into a lookup mapping by citekey and file path."""
    if not bib_path.is_file():
        return {}
    try:
        content = bib_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    entries: dict[str, dict] = {}
    # Simple regex-based BibTeX entry extractor
    entry_pattern = re.compile(r"@(\w+)\s*\{\s*([^,]+),([\s\S]*?)\n\}(?=\s*@|\s*$)", re.MULTILINE)
    field_pattern = re.compile(r"(\w+)\s*=\s*(?:\{([\s\S]*?)\}|\"([\s\S]*?)\"|(\d+))", re.MULTILINE)

    for match in entry_pattern.finditer(content):
        citekey = match.group(2).strip()
        body = match.group(3)
        fields = {}
        for fmatch in field_pattern.finditer(body):
            name = fmatch.group(1).lower()
            val = fmatch.group(2) or fmatch.group(3) or fmatch.group(4) or ""
            fields[name] = val.strip()

        meta = {}
        if "title" in fields:
            meta["title"] = fields["title"]
        if "author" in fields:
            # BibTeX separates authors with 'and'
            authors = [a.strip() for a in fields["author"].split(" and ") if a.strip()]
            meta["authors"] = authors
        if "year" in fields:
            try:
                meta["year"] = int(fields["year"])
            except (ValueError, TypeError):
                pass
        if "keywords" in fields or "subjects" in fields:
            raw_sub = fields.get("keywords") or fields.get("subjects", "")
            tags = _split_tags(raw_sub)
            # Zotero tags are represented as BibTeX keywords. Keep them separate
            # from local README subjects so callers can filter on Zotero's full
            # vocabulary without changing the repository's controlled taxonomy.
            meta["tags"] = tags
            meta["subjects"] = tags

        identifier = {}
        if "doi" in fields:
            identifier["doi"] = fields["doi"]
        if "isbn" in fields:
            identifier["isbn"] = fields["isbn"]
        if identifier:
            meta["identifier"] = identifier

        meta["citekey"] = citekey
        entries[citekey] = meta
        if "file" in fields:
            for file_key in _catalog_file_lookup_keys(fields["file"]):
                entries[file_key] = meta

    return entries


def parse_pymupdf_metadata(doc_path: Path) -> dict:
    """Extract title and author from embedded PDF/EPUB metadata via PyMuPDF."""
    if doc_path.suffix.lower() not in {".pdf", ".epub"}:
        return {}
    try:
        import pymupdf
        with pymupdf.open(doc_path) as doc:
            meta = doc.metadata or {}
            res = {}
            title = meta.get("title")
            if title and title.strip():
                res["title"] = title.strip()
            author = meta.get("author")
            if author and author.strip():
                res["authors"] = [author.strip()]
            return res
    except Exception:
        return {}


def parse_pdf_metadata(pdf_path: Path) -> dict:
    """Extract title and author from embedded PDF/EPUB metadata (backwards-compatibility alias)."""
    return parse_pymupdf_metadata(pdf_path)


def resolve_metadata(
    content_path: Path,
    root: Path,
    bib_cache: dict[str, dict] | None = None,
    inventory: dict | None = None,
    diagnostics: list | None = None,
) -> dict:
    """Resolve metadata for a content document following strict precedence:

    1. ``catalog.bib`` entry, matched by linked attachment path, then an
       explicit item-record/legacy-sidecar citekey, then legacy folder name or
       filename fallback
    2. Explicit item metadata from ``Metadata/<item_id>.md``, or YAML
       frontmatter in the legacy item's ``README.md`` when no explicit mapping
       exists
    3. Folder-name parsing (true legacy item directories only)
    4. Embedded document metadata
    5. Filename fallback

    Bibliographic facts (title, authors, year, identifiers) and Zotero tags
    come from the catalog.  Explicit local ``subjects`` and ``status`` remain
    authoritative, including intentionally empty values; catalog values apply
    only when the local field is absent.  Identifier subfields resolve
    individually so a catalog DOI correction never erases an independent local
    URL.  A catalog path match and an explicit citekey that unambiguously name
    different parents are reported as an association conflict (when a
    ``diagnostics`` list is given) rather than mixing bibliographic facts.
    """
    from . import items

    rel = content_path.relative_to(root)
    parts = rel.parts
    rel_str = str(rel).replace("\\", "/")
    parent_dir = content_path.parent

    if inventory is None:
        records, _ = items.load_records(root)
        by_path: dict[str, dict] = {}
        for record in records.values():
            for attachment in record["attachments"]:
                by_path.setdefault(attachment["path"], record)
    else:
        records = inventory["records"]
        by_path = {
            path: records[item_id]
            for path, item_id in inventory["by_path"].items()
            if item_id in records
        }
    record = by_path.get(rel_str)

    # A shelf README is navigation, never a sidecar for flat sources: read the
    # parent README only when the file lives in a genuine item directory (or
    # anywhere outside the source shelves, preserving historical behavior).
    parent_is_shelf_root = (
        len(parts) >= 2 and parent_dir == root / parts[0] and parts[0] in SHELVES
    )
    sidecar_path = parent_dir / "README.md"
    if parent_is_shelf_root or not sidecar_path.is_file():
        readme_meta = {}
    else:
        readme_meta = parse_readme_frontmatter(sidecar_path)

    local_meta = dict(readme_meta)
    if record is not None:
        # Explicit records take precedence over legacy sidecars for local
        # metadata; their presence never merges items by title or filename.
        local_meta = {**readme_meta, **items.record_facts(record)}

    # catalog.bib: the linked attachment path is the strongest association, then
    # an explicit citekey.  Contradictory unambiguous matches are a conflict.
    if bib_cache is None:
        bib_file = root / "catalog.bib"
        bib_cache = parse_catalog_bib(bib_file) if bib_file.is_file() else {}
    bib_meta: dict = {}
    citekey = local_meta.get("citekey")
    if bib_cache:
        by_path_entry = bib_cache.get(rel_str)
        by_citekey_entry = bib_cache.get(citekey) if citekey else None
        if (
            by_path_entry
            and by_citekey_entry
            and by_path_entry.get("citekey") != by_citekey_entry.get("citekey")
        ):
            if diagnostics is not None:
                diagnostics.append({
                    "reason": "association_conflict",
                    "path": rel_str,
                    "detail": (
                        f"attachment path matches catalog parent "
                        f"{by_path_entry.get('citekey')} but citekey {citekey} matches "
                        f"{by_citekey_entry.get('citekey')}"
                    ),
                })
        elif by_path_entry or by_citekey_entry:
            bib_meta = by_path_entry or by_citekey_entry
        else:
            folder_name = parent_dir.name
            for lookup_key in (folder_name, content_path.stem):
                if lookup_key and lookup_key in bib_cache:
                    bib_meta = bib_cache[lookup_key]
                    break

    # 3. Folder-name parsing (only for genuine legacy item directories; flat
    #    shelf filenames are presentation, not item identity).
    folder_meta = {}
    if len(parts) > 2 and parts[0] in ("Books", "Papers", "Videos"):
        folder_meta = parse_folder_name(parts[1])

    # 4. Embedded PDF/EPUB metadata
    doc_meta = parse_pymupdf_metadata(content_path) if content_path.suffix.lower() in {".pdf", ".epub"} else {}

    # Merge according to precedence.  Local subjects/status use presence, not
    # truthiness, so explicit empty values stay authoritative.
    title = (
        bib_meta.get("title")
        or local_meta.get("title")
        or folder_meta.get("title")
        or doc_meta.get("title")
        or content_path.stem
    )
    authors = (
        bib_meta.get("authors")
        or local_meta.get("authors")
        or folder_meta.get("authors")
        or doc_meta.get("authors")
        or None
    )
    year = (
        bib_meta.get("year")
        or local_meta.get("year")
        or folder_meta.get("year")
        or None
    )
    if "subjects" in local_meta:
        subjects = local_meta["subjects"]
    else:
        subjects = bib_meta.get("subjects") or None
    local_identifier = local_meta.get("identifier") or {}
    bib_identifier = bib_meta.get("identifier") or {}
    identifier = {**local_identifier, **bib_identifier} or None
    tags = bib_meta.get("tags") or None
    resolved_citekey = bib_meta.get("citekey") or local_meta.get("citekey") or None

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "subjects": subjects,
        "identifier": identifier,
        "tags": tags,
        "citekey": resolved_citekey,
        "item_id": record["item_id"] if record is not None else None,
    }
