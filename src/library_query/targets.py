"""Exact work selection through portable inventory and read-only index mappings."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

from . import config, errors, items, ledger

MAX_TARGETS = 10
ROLES = ("source", "supplement", "note")
ITEM_PATTERN = r"item-[0-9a-f]{32}"
DOCUMENT_PATTERN = r"doc_[0-9a-f]{64}"


def invalid(field, message):
    raise errors.InvalidResearchRequestError(f"{field}: {message}")


@dataclass(frozen=True)
class Target:
    kind: str
    ids: tuple[str, ...]
    roles: tuple[str, ...] = ()

    def to_dict(self):
        result = {self.kind: list(self.ids)}
        if self.kind == "item_ids":
            result["roles"] = list(self.roles)
        return result


def parse(value, field="target"):
    if not isinstance(value, dict) or set(value) - {"item_ids", "document_ids", "roles"}:
        invalid(field, "must be a target object with known fields")
    kinds = [k for k in ("item_ids", "document_ids") if k in value]
    if len(kinds) != 1:
        invalid(field, "requires exactly one identity kind")
    kind = kinds[0]
    ids = value[kind]
    pattern = ITEM_PATTERN if kind == "item_ids" else DOCUMENT_PATTERN
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_TARGETS or any(
        not isinstance(i, str) or re.fullmatch(pattern, i) is None for i in ids
    ):
        invalid(field, "requires 1–10 canonical identities")
    roles = value.get("roles", ["source"] if kind == "item_ids" else [])
    if kind == "document_ids" and "roles" in value:
        invalid(field, "roles require item identities")
    if not isinstance(roles, list) or (kind == "item_ids" and not roles) or any(r not in ROLES for r in roles):
        invalid(field, "roles must be source, supplement, or note")
    return Target(kind, tuple(sorted(set(ids))), tuple(sorted(set(roles))))


def manifest():
    """Do not call the migrating ledger loader on a read path."""
    try:
        state = json.loads(config.LEDGER_PATH.read_text(encoding="utf-8"))
        if state.get("version") != 2 or not isinstance(state.get("documents"), dict):
            raise ValueError("manifest version")
        return state["documents"]
    except (OSError, ValueError, AttributeError):
        raise errors.LibraryQueryError("target_not_indexed", "Current manifest required; run library-query index") from None


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        invalid("path", "must be a contained relative path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        invalid("path", "must be a contained relative path")
    return resolved


def document_for_path(root, relative, documents):
    path = contained(root, relative)
    if not path.is_file():
        raise errors.LibraryQueryError("target_not_indexed", "Selected attachment missing; restore it and run library-query index")
    digest = ledger.file_hash(path)
    doc_id = "doc_" + digest
    doc = documents.get(doc_id)
    if not isinstance(doc, dict) or doc.get("content_hash") != digest or not doc.get("point_ids"):
        raise errors.LibraryQueryError("target_not_indexed", "Selected attachment requires library-query index")
    return doc_id


def resolve(target, root=None):
    root = Path(root or config.LIBRARY_ROOT)
    if not root.is_dir():
        raise errors.InvalidLibraryRootError("LIBRARY_ROOT must be an existing directory")
    inventory = items.build_inventory(root)
    if inventory["conflicts"]:
        raise errors.LibraryQueryError("target_not_indexed", "Resolve inventory conflicts and run library-query index")
    # Reject unknown items before requiring index state.
    if target.kind == "item_ids" and any(i not in inventory["records"] for i in target.ids):
        raise errors.LibraryQueryError("target_not_found", "Selected item was not found")
    documents = manifest()
    members = []
    for identity in target.ids:
        paths = []
        if target.kind == "item_ids":
            record = inventory["records"][identity]
            paths = [a["path"] for a in record["attachments"] if a["role"] in target.roles]
            generated = inventory["note_by_item"].get(identity)
            if "note" in target.roles and generated and generated not in paths:
                paths.append(generated)
            doc_ids = sorted({document_for_path(root, p, documents) for p in paths})
        else:
            doc = documents.get(identity)
            if doc is None:
                raise errors.LibraryQueryError("target_not_found", "Selected document was not found")
            relative = doc.get("path")
            if not isinstance(relative, str) or document_for_path(root, relative, documents) != identity:
                raise errors.LibraryQueryError("target_not_indexed", "Selected document requires library-query index")
            paths = [relative]
            doc_ids = [identity]
        members.append({"kind": target.kind, "id": identity, "document_ids": doc_ids, "paths": sorted(paths)})
    return {**target.to_dict(), "members": members,
            "document_ids_resolved": sorted({d for m in members for d in m["document_ids"]})}


def schema():
    return {"type": "object", "identity_kinds": ["item_ids", "document_ids"],
            "max_targets": MAX_TARGETS, "item_pattern": ITEM_PATTERN, "document_pattern": DOCUMENT_PATTERN,
            "roles": list(ROLES), "default_item_roles": ["source"],
            "semantics": "exactly one non-empty identity list; question target replaces shared target; omitted inherits; null invalid"}


def verify_index(resolved, collection_name=None, timeout=None):
    from . import store
    for document_id in resolved["document_ids_resolved"]:
        records = store.retrieve_chunks(document_id, [0], collection_name, timeout)
        if len(records) != 1 or (records[0].payload or {}).get("document_id") != document_id:
            raise errors.LibraryQueryError("target_not_indexed", "Selected document is absent from the index; run library-query index")
