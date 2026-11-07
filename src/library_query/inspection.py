"""Bounded current-index reading without embeddings, extraction, or writes."""
import hashlib
from pathlib import Path
import re

from . import browse, config, errors, items, research, store, targets

SOURCE_LIMIT = 20
SOURCE_MAX_LIMIT = 100
PREVIEW_CHARS = 200
CONTEXT_COUNT = 1
CONTEXT_MAX_COUNT = 5
CONTEXT_CHARS = 8000
CONTEXT_MAX_CHARS = 32000


def bounded(value, field, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        targets.invalid(field, f"must be an integer in [{low}, {high}]")


def inspect_item(identity, root=None):
    targets.parse({"item_ids": [identity]})
    inventory = browse.browse_json(Path(root or config.LIBRARY_ROOT))
    item = next((i for i in inventory["items"] if i["item_id"] == identity), None)
    if item is None:
        raise errors.LibraryQueryError("target_not_found", "Selected item was not found")
    item["attachments"] = items.build_inventory(Path(root or config.LIBRARY_ROOT))["records"][identity]["attachments"]
    return {"schema_version": 1, "item": item, "warnings": inventory["warnings"]}


def read_chunks(document_id, indexes):
    records = store.retrieve_chunks(document_id, indexes)
    result = {}
    for record in records:
        p = record.payload or {}
        index = p.get("chunk_index")
        if p.get("document_id") != document_id or index not in indexes or not isinstance(p.get("text"), str):
            raise errors.LibraryQueryError("target_not_indexed", "Indexed chunk identity invalid; run library-query index")
        result[index] = p
    return result


def descriptor(payload, max_chars):
    text = payload["text"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"evidence_id": f"{payload['document_id']}:{payload['chunk_index']}:{digest}",
            "chunk_index": payload["chunk_index"], "location": research._location_hint(payload),
            "text": text[:max_chars], "text_truncated": len(text) > max_chars}


def source(document_id=None, path=None, offset=0, limit=SOURCE_LIMIT):
    bounded(offset, "offset", 0, 2**63 - 1)
    bounded(limit, "limit", 1, SOURCE_MAX_LIMIT)
    if (document_id is None) == (path is None):
        targets.invalid("source", "requires exactly one document or path")
    documents = targets.manifest()
    if path is not None:
        selected_path = targets.contained(config.LIBRARY_ROOT, path)
        if not selected_path.exists() and not any(d.get("path") == path for d in documents.values()):
            raise errors.LibraryQueryError("target_not_found", "Selected source path was not found")
        document_id = targets.document_for_path(config.LIBRARY_ROOT, path, documents)
    selected = targets.resolve(targets.parse({"document_ids": [document_id]}))
    total = documents[document_id]["chunks"]
    indexes = list(range(min(offset, total), min(offset + limit, total)))
    chunks = read_chunks(document_id, indexes) if indexes else {}
    if len(chunks) != len(indexes):
        raise errors.LibraryQueryError("target_not_indexed", "Indexed chunks missing; run library-query index")
    inventory = items.build_inventory(config.LIBRARY_ROOT)
    relative = documents[document_id]["path"]
    associations = sorted({i for i, r in inventory["records"].items() if any(a["path"] == relative for a in r["attachments"])})
    return {"schema_version": 1, "document_id": document_id, "target": selected, "item_ids": associations,
            "total_chunks": total, "chunks": [descriptor(chunks[i], PREVIEW_CHARS) for i in indexes],
            "next_offset": offset + limit if offset + limit < total else None}


def passage(evidence_id, before=CONTEXT_COUNT, after=CONTEXT_COUNT, max_chars=CONTEXT_CHARS):
    bounded(before, "before", 0, CONTEXT_MAX_COUNT)
    bounded(after, "after", 0, CONTEXT_MAX_COUNT)
    bounded(max_chars, "max_chars", 1, CONTEXT_MAX_CHARS)
    match = re.fullmatch(r"(" + targets.DOCUMENT_PATTERN + r"):(0|[1-9][0-9]*):([0-9a-f]{64})", evidence_id or "")
    if match is None:
        targets.invalid("evidence_id", "must be a canonical content-bound passage identity")
    document_id, coordinate, digest = match.groups()
    index = int(coordinate)
    bounded(index, "chunk_index", 0, 2**63 - 1)
    indexes = list(range(max(0, index - before), index + after + 1))
    chunks = read_chunks(document_id, indexes)
    center = chunks.get(index)
    if center is None:
        raise errors.LibraryQueryError("passage_not_found", "Requested passage is not in the current index")
    if hashlib.sha256(center["text"].encode("utf-8")).hexdigest() != digest:
        raise errors.LibraryQueryError("stale_evidence", "Requested passage text changed; obtain a current evidence ID")
    remaining = max_chars
    emitted, omitted = {}, []
    for i in sorted(chunks, key=lambda i: (abs(i - index), i)):
        if remaining == 0:
            omitted.append(i)
            continue
        emitted[i] = descriptor(chunks[i], remaining)
        remaining -= len(emitted[i]["text"])
    return {"schema_version": 1, "evidence_id": evidence_id, "document_id": document_id,
            "center_chunk_index": index, "passages": [emitted[i] for i in sorted(emitted)],
            "omitted_chunk_indexes": sorted(omitted), "absent_chunk_indexes": sorted(set(indexes) - chunks.keys())}


def schema():
    return {"schema_version": 1, "source": {"default_limit": SOURCE_LIMIT, "max_limit": SOURCE_MAX_LIMIT,
            "preview_chars": PREVIEW_CHARS, "fields": ["schema_version", "document_id", "target", "item_ids", "total_chunks", "chunks", "next_offset"]},
            "inspect": {"fields": ["schema_version", "item", "warnings"]},
            "passage": {"default_before_after": CONTEXT_COUNT, "max_before_after": CONTEXT_MAX_COUNT,
            "default_max_chars": CONTEXT_CHARS, "max_chars": CONTEXT_MAX_CHARS,
            "fields": ["schema_version", "evidence_id", "document_id", "center_chunk_index", "passages", "omitted_chunk_indexes", "absent_chunk_indexes"]},
            "chunk_fields": ["evidence_id", "chunk_index", "location", "text", "text_truncated"]}
