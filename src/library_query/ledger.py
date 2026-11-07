"""Content-hash document manifest (v2) and reconciliation engine.

Core principles:
- content hash = identity (document_id = doc_<content_hash>)
- path = location / state
- metadata = description
"""
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import config, structure


def file_hash(path: Path) -> str:
    """Calculate SHA-256 digest for file content."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_key(path: Path) -> tuple[int, int] | None:
    """Return the (size, mtime_ns) identity of a file, or None if it cannot be stat'd.

    Size plus mtime is the cheap proxy for "these bytes are the ones we already
    hashed". It is not a cryptographic guarantee -- a write that preserves both
    would slip through -- but every ordinary edit, re-save, re-download or
    conversion changes at least one of them, and the alternative is reading every
    byte of the corpus on every `status` call.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_size, info.st_mtime_ns)


def _hash_with_cache(path: Path, rel: str, cache: dict, updated: dict) -> str:
    """Hash `path`, reusing the cached digest when (size, mtime_ns) still matches.

    `updated` accumulates the cache entries to persist: every scanned file lands
    there, so stale paths fall out of the cache simply by not being re-added.
    """
    key = _stat_key(path)
    entry = cache.get(rel)
    if (
        key is not None
        and isinstance(entry, dict)
        and entry.get("size") == key[0]
        and entry.get("mtime_ns") == key[1]
        and isinstance(entry.get("content_hash"), str)
    ):
        updated[rel] = entry
        return entry["content_hash"]

    digest = file_hash(path)
    if key is not None:
        updated[rel] = {"size": key[0], "mtime_ns": key[1], "content_hash": digest}
    return digest


def _migrate_v1_to_v2(state: dict, root: Path | None = None) -> dict:
    """Auto-migrate legacy v1 ledger to v2 document manifest."""
    root = root or config.LIBRARY_ROOT
    files = state.get("files", {})
    documents: dict[str, Any] = {}

    from . import store

    for file_rel, hash_val in list(files.items()):
        if file_rel.startswith("_meta:"):
            continue

        meta = files.get(f"_meta:{file_rel}", {})
        chunks_count = meta.get("chunks", 1)
        doc_id = f"doc_{hash_val}"
        file_path = root / file_rel

        scope, doc_type = structure.get_scope_and_type(file_path, root)
        metadata = structure.resolve_metadata(file_path, root) if file_path.is_file() else {}

        # Migration is a pure offline transformation: point IDs are always recomputed
        # deterministically from (document_id, chunk_index). No live Qdrant lookups.
        point_ids = [store.make_point_id(doc_id, i) for i in range(chunks_count)]

        documents[doc_id] = {
            "content_hash": hash_val,
            "path": file_rel,
            "scope": scope,
            "type": doc_type,
            "metadata": metadata,
            "chunks": chunks_count,
            "point_ids": point_ids,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

    return {"version": 2, "documents": documents}


def _backup_corrupt_ledger() -> Path | None:
    """Move an unreadable ledger aside, returning where it went.

    An empty manifest means the next index re-embeds the entire corpus, so the file
    that caused it is the only evidence of what was already indexed. Losing it to a
    silent overwrite turns a recoverable parse error into a full re-embed with no way
    to diff what changed. Never raises: the caller's fallback must still happen.
    """
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = config.LEDGER_PATH.with_name(f"{config.LEDGER_PATH.name}.corrupt-{stamp}")
        config.LEDGER_PATH.replace(backup)
        print(f"ledger: unreadable manifest backed up to {backup}", file=sys.stderr)
        return backup
    except Exception as exc:
        print(f"ledger: could not back up corrupt manifest: {exc}", file=sys.stderr)
        return None


def _load(root: Path | None = None) -> dict:
    """Load the v2 document manifest, automatically migrating v1 ledgers if necessary."""
    root = root or config.LIBRARY_ROOT
    if config.LEDGER_PATH.exists():
        try:
            state = json.loads(config.LEDGER_PATH.read_text())
            if state.get("version") != 2:
                state = _migrate_v1_to_v2(state, root)
                _save(state)
            return state
        except Exception:
            # Preserve the unreadable file before falling back to an empty manifest.
            _backup_corrupt_ledger()
            return {"version": 2, "documents": {}}
    return {"version": 2, "documents": {}}


def _save(state: dict) -> None:
    """Write the manifest atomically: temp file in the same directory, then replace.

    `write_text` truncates in place, so an interruption mid-write leaves a truncated
    JSON file -- which `_load` reads as an empty manifest and the next index answers by
    re-embedding every document. `os.replace` on the same filesystem is atomic, so a
    reader sees either the old manifest or the new one, never a partial one.
    """
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, sort_keys=True)
    tmp = config.LEDGER_PATH.with_name(f"{config.LEDGER_PATH.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, config.LEDGER_PATH)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def scan_content_files(root: Path | None = None) -> dict[str, Path]:
    """Scan library root for embeddable content files, returning {content_hash: file_path}.

    One path per hash: byte-identical duplicates collapse onto the first path in sorted
    order. `scan_content_paths` is the lossless form, used where duplicates matter.
    """
    return {
        digest: paths[0] for digest, paths in scan_content_paths(root).items()
    }


def scan_content_paths(root: Path | None = None) -> dict[str, list[Path]]:
    """Scan library root for content files, returning {content_hash: [file_path, ...]}.

    Keeps every path a hash was found at, in sorted order. Content hash is document
    identity, so two paths under one hash is genuinely ambiguous -- which one is "the"
    document cannot be answered from the hash alone, and reconciliation must report the
    ambiguity rather than pick.
    """
    root = root or config.LIBRARY_ROOT
    content_paths: dict[str, list[Path]] = {}
    if not root.exists():
        return content_paths

    # (size, mtime_ns) -> content_hash cache, persisted in the ledger. Unchanged
    # files are not re-read; new or modified ones are hashed and re-cached.
    state = _load(root)
    cache = state.get("hash_cache") or {}
    updated: dict[str, dict] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        role = structure.classify_file(path, root)
        if role == "content":
            rel = str(path.relative_to(root))
            digest = _hash_with_cache(path, rel, cache, updated)
            content_paths.setdefault(digest, []).append(path)

    if updated != cache:
        state["hash_cache"] = updated
        _save(state)

    return content_paths


def reconcile(root: Path | None = None) -> dict[str, list]:
    """Reconcile active filesystem content files against the manifest using 5-way classification.

    Returns:
      {
        "new": [...],
        "changed": [...],
        "moved": [...],
        "deleted": [...],
        "unchanged": [...],
        "conflicts": [...]
      }

    `conflicts` holds hashes found at more than one path plus classified
    metadata-association diagnostics from the shared inventory. Those are
    reported and left alone: the manifest binding is not touched and no move
    is planned for them.
    """
    root = root or config.LIBRARY_ROOT
    state = _load(root)
    documents = state.get("documents", {})

    # One validated item-mapping index per pass: mapped notes inherit their
    # item's shelf type and records supply local metadata precedence.  The
    # inventory itself is side-effect-free; persistence stays with the
    # hash-cache scanner below.
    from . import items

    inventory = items.build_inventory(root)
    diagnostics: list[dict] = []

    scanned_paths = scan_content_paths(root)
    # Hashes present at more than one path. Content hash is document identity, so a
    # duplicate makes "where is this document?" unanswerable; previously reconciliation
    # answered it by silently rebinding the manifest to whichever path sorted last,
    # which orphaned the original file and made the pair oscillate between the two
    # locations on every pass, re-writing Qdrant payloads each time.
    duplicate_hashes = {h for h, paths in scanned_paths.items() if len(paths) > 1}
    scanned_content = {digest: paths[0] for digest, paths in scanned_paths.items()}
    scanned_hashes = set(scanned_content.keys())

    manifest_by_path: dict[str, tuple[str, dict]] = {
        doc["path"]: (doc_id, doc) for doc_id, doc in documents.items()
    }

    plan = {
        "new": [],
        "changed": [],
        "moved": [],
        "deleted": [],
        "unchanged": [],
        "conflicts": [],
    }

    handled_scanned_hashes = set()
    handled_manifest_docs = set()

    # Report every duplicated hash once, before any classification uses it.
    for content_hash in sorted(duplicate_hashes):
        paths = [str(p.relative_to(root)) for p in scanned_paths[content_hash]]
        recorded = documents.get(f"doc_{content_hash}", {}).get("path")
        plan["conflicts"].append({
            "doc_id": f"doc_{content_hash}",
            "content_hash": content_hash,
            "paths": paths,
            "recorded_path": recorded,
            "reason": "duplicate_content",
        })
        print(
            f"ledger: duplicate content {content_hash[:12]} at {len(paths)} paths "
            f"({', '.join(paths)}); keeping existing binding "
            f"{recorded or '<none>'} and planning no move",
            file=sys.stderr,
        )

    # 1. Detect MOVED vs DELETED for recorded documents.
    #
    # The trigger is "the recorded path no longer holds this document's content",
    # not merely "the recorded path is gone": a document can also vacate its slot by
    # having unrelated content overwrite it in the same pass (Scenario 15).
    scanned_hash_by_rel = {
        str(p.relative_to(root)): h for h, p in scanned_content.items()
    }
    for doc_id, doc in documents.items():
        recorded_path = doc["path"]
        still_at_recorded_path = scanned_hash_by_rel.get(recorded_path) == doc["content_hash"]
        if not still_at_recorded_path:
            # This document's content is no longer at the path we recorded for it.
            if doc["content_hash"] in duplicate_hashes:
                # Ambiguous: the content sits at several paths, so "moved where?" has
                # no answer. Already reported as a conflict above. Hold the existing
                # binding and treat the document as untouched this pass -- marking it
                # handled is what keeps the purge loop below from deleting it.
                handled_scanned_hashes.add(doc["content_hash"])
                handled_manifest_docs.add(doc_id)
            elif doc["content_hash"] in scanned_hashes:
                # Content hash exists elsewhere: MOVED!
                new_path = scanned_content[doc["content_hash"]]
                new_rel = str(new_path.relative_to(root))
                scope, doc_type = structure.get_scope_and_type(new_path, root, inventory)
                metadata = structure.resolve_metadata(new_path, root, inventory=inventory, diagnostics=diagnostics)
                plan["moved"].append({
                    "doc_id": doc_id,
                    "old_path": recorded_path,
                    "new_path": new_rel,
                    "file_path": new_path,
                    "content_hash": doc["content_hash"],
                    "point_ids": doc.get("point_ids", []),
                    "scope": scope,
                    "type": doc_type,
                    "metadata": metadata,
                })
                handled_scanned_hashes.add(doc["content_hash"])
                handled_manifest_docs.add(doc_id)
            elif not (root / recorded_path).exists():
                # Content hash does not exist anywhere on disk and the recorded path
                # is gone: DELETED!
                plan["deleted"].append({
                    "doc_id": doc_id,
                    "path": recorded_path,
                    "content_hash": doc["content_hash"],
                    "point_ids": doc.get("point_ids", []),
                })
                handled_manifest_docs.add(doc_id)

    # 2. Check scanned files
    for content_hash, file_path in scanned_content.items():
        if content_hash in handled_scanned_hashes:
            continue

        rel_path = str(file_path.relative_to(root))
        doc_id = f"doc_{content_hash}"
        scope, doc_type = structure.get_scope_and_type(file_path, root, inventory)
        metadata = structure.resolve_metadata(file_path, root, inventory=inventory, diagnostics=diagnostics)

        if doc_id in documents:
            # Document known by hash
            doc = documents[doc_id]
            recorded_path = doc["path"]
            if recorded_path == rel_path:
                # Same path, same hash: UNCHANGED
                old_meta = doc.get("metadata", {})
                meta_changed = (old_meta != metadata)
                plan["unchanged"].append({
                    "doc_id": doc_id,
                    "path": rel_path,
                    "file_path": file_path,
                    "content_hash": content_hash,
                    "scope": scope,
                    "type": doc_type,
                    "metadata": metadata,
                    "metadata_changed": meta_changed,
                    "point_ids": doc.get("point_ids", []),
                })
                handled_manifest_docs.add(doc_id)
            else:
                # Same hash, different path (e.g. duplicate or move where old path was removed)
                plan["unchanged"].append({
                    "doc_id": doc_id,
                    "path": rel_path,
                    "file_path": file_path,
                    "content_hash": content_hash,
                    "scope": scope,
                    "type": doc_type,
                    "metadata": metadata,
                    "metadata_changed": False,
                    "point_ids": doc.get("point_ids", []),
                })
                handled_manifest_docs.add(doc_id)
        else:
            # Document unknown by hash
            prior = manifest_by_path.get(rel_path)
            if prior is not None and prior[0] not in handled_manifest_docs:
                # Path existed with an older hash, and that document has not already
                # been handled (e.g. as MOVED) in this pass: CHANGED
                old_doc_id, old_doc = prior
                plan["changed"].append({
                    "doc_id": doc_id,
                    "path": rel_path,
                    "file_path": file_path,
                    "content_hash": content_hash,
                    "scope": scope,
                    "type": doc_type,
                    "metadata": metadata,
                    "old_doc_id": old_doc_id,
                    "old_point_ids": old_doc.get("point_ids", []),
                })
                handled_manifest_docs.add(old_doc_id)
            else:
                # Completely new: NEW
                plan["new"].append({
                    "doc_id": doc_id,
                    "path": rel_path,
                    "file_path": file_path,
                    "content_hash": content_hash,
                    "scope": scope,
                    "type": doc_type,
                    "metadata": metadata,
                })

    # Association diagnostics from the shared inventory are surfaced the same
    # way as duplicate-content conflicts: reported, with no state change.
    for diagnostic in diagnostics:
        print(
            json.dumps({"action": "metadata_conflict", **diagnostic}),
            file=sys.stderr,
        )
    plan["conflicts"].extend(diagnostics)

    # 3. Purge remaining unhandled manifest entries: either the path is gone, or the
    #    file is no longer embeddable content. A v1 ledger indexed navigation and
    #    metadata files (root README.md, Collections/**) that the v2 role rules
    #    exclude; those are ghosts and must not linger as catalogued vectors.
    for doc_id, doc in documents.items():
        if doc_id in handled_manifest_docs:
            continue
        full_path = root / doc["path"]
        if not full_path.exists() or structure.classify_file(full_path, root) != "content":
            plan["deleted"].append({
                "doc_id": doc_id,
                "path": doc["path"],
                "content_hash": doc["content_hash"],
                "point_ids": doc.get("point_ids", []),
            })

    return plan


def pending_files(root: Path | None = None) -> list[Path]:
    """Return paths of files that require embedding (NEW or CHANGED)."""
    plan = reconcile(root)
    return [item["file_path"] for item in plan["new"] + plan["changed"]]


def mark_done(
    path: Path,
    chunk_count: int,
    doc_id: str | None = None,
    content_hash: str | None = None,
    scope: str | None = None,
    doc_type: str | None = None,
    metadata: dict | None = None,
    point_ids: list[int] | None = None,
    root: Path | None = None,
) -> None:
    """Record or update a document in the v2 manifest."""
    root = root or config.LIBRARY_ROOT
    state = _load(root)
    rel = str(path.relative_to(root))
    c_hash = content_hash or file_hash(path)
    d_id = doc_id or f"doc_{c_hash}"
    sc, dt = (scope, doc_type) if scope and doc_type else structure.get_scope_and_type(path, root)
    meta = metadata if metadata is not None else structure.resolve_metadata(path, root)

    state["documents"][d_id] = {
        "content_hash": c_hash,
        "path": rel,
        "scope": sc,
        "type": dt,
        "metadata": meta,
        "chunks": chunk_count,
        "point_ids": point_ids or [],
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(state)


def stats(root: Path | None = None) -> dict:
    """Return statistics on corpus breakdown across scopes and pending files."""
    root = root or config.LIBRARY_ROOT
    state = _load(root)
    docs = state.get("documents", {})

    inbox_count = sum(1 for d in docs.values() if d.get("scope") == "inbox")
    catalogued_count = sum(1 for d in docs.values() if d.get("scope") == "catalogued")

    plan = reconcile(root)
    pending_count = len(plan["new"]) + len(plan["changed"])
    total_content = sum(1 for p in root.rglob("*") if p.is_file() and structure.classify_file(p, root) == "content")

    return {
        "files_in_library": total_content,
        "indexed": len(docs),
        "inbox": inbox_count,
        "catalogued": catalogued_count,
        "pending": pending_count,
    }
