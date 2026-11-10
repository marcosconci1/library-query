"""library-query CLI: index a document library with 5-way reconciliation, query it for places."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from . import browse, cache, config, doctor, errors, exact, expansion, extract, inspection, journal, layout_plan, ledger, readiness, research, revisions, store, structure, targets, zotero

#: The query result fields guaranteed by the agent-tool-contract capability. Every
#: result object carries all of these, using `null` where a value did not resolve.
#: Removing or renaming one is a contract break, not a refactor.
QUERY_OUTPUT_FIELDS = [
    "doc",
    "doc_name",
    "page",
    "score",
    "text",
    "scope",
    "type",
    "title",
    "authors",
    "year",
    "subjects",
    "tags",
    "identifier",
]


def require_library_root() -> Path:
    """Validate LIBRARY_ROOT before any command that reads the library.

    `ledger.scan_content_files` treats a missing root as an empty library, which would
    silently report zero files instead of a configuration error — exactly the ambiguity
    an agent caller can't recover from.
    """
    root = config.LIBRARY_ROOT
    if not root.exists():
        raise errors.InvalidLibraryRootError(f"LIBRARY_ROOT does not exist: {root}")
    if not root.is_dir():
        raise errors.InvalidLibraryRootError(f"LIBRARY_ROOT is not a directory: {root}")
    return root


def _structural_payload(scope: str, doc_type: str, metadata: dict, path: str | None = None) -> dict:
    """Build the payload fields that describe where a document sits and what it is."""
    payload = {
        "scope": scope,
        "status": scope,  # legacy compatibility
        "type": doc_type,
        "doc_type": doc_type,  # legacy compatibility
        "title": metadata.get("title"),
        "authors": metadata.get("authors"),
        "year": metadata.get("year"),
        "subjects": metadata.get("subjects"),
        "tags": metadata.get("tags"),
        "citekey": metadata.get("citekey"),
        "identifier": metadata.get("identifier"),
        # Additive: immutable bibliographic item grouping, never a replacement
        # for content-hash document identity.
        "item_id": metadata.get("item_id"),
    }
    if path is not None:
        payload["path"] = path
        payload["doc"] = path
        payload["doc_name"] = Path(path).name
    return payload


def backfill_legacy_payloads(state: dict) -> int:
    """One-time backfill of structural payloads onto legacy points lacking them.

    Points indexed before the v2 manifest carry only `doc`/`doc_name`/`page`/`text`,
    so they are invisible to `--scope` filtering. The manifest already holds the
    correct scope/type/metadata per document, so this walks it keyed by
    document_id -> point_ids and sets the missing fields in place. No re-embedding.

    Returns the number of documents whose points were backfilled.
    """
    if store.count_points_missing_field("scope") == 0:
        return 0

    backfilled = 0
    for doc_id, doc in state.get("documents", {}).items():
        payload = _structural_payload(
            doc.get("scope", "inbox"),
            doc.get("type", "other"),
            doc.get("metadata") or {},
            doc.get("path"),
        )
        payload["document_id"] = doc_id
        payload["content_hash"] = doc.get("content_hash")

        # Legacy points predate deterministic IDs, so address them by recorded path.
        legacy_ids = store.find_legacy_point_ids(doc.get("path", ""))
        target_ids = sorted(set(doc.get("point_ids", [])) | set(legacy_ids))
        if not target_ids:
            continue
        store.update_payloads(target_ids, payload)
        backfilled += 1
        print(
            json.dumps({"file": doc.get("path"), "action": "payload_backfilled", "points": len(target_ids)}),
            file=sys.stderr,
        )
    return backfilled


def _reindex_document(doc_id: str, doc: dict) -> None:
    """Staged reindex of one document: new points first, old points retired only on success.

    This is the only safe repair for a legacy identity the manifest cannot prove
    (or a point whose stored text is gone). The old indexed version keeps serving
    the legacy locator until the replacement upsert succeeds; a failure leaves
    every old point in place.
    """
    root = require_library_root()
    file_path = root / doc["path"]
    pages = extract.extract(file_path)
    chunks = extract.chunk(pages)
    if not chunks:
        raise ValueError("no extractable text")
    vectors = store.embed([c["text"] for c in chunks])
    point_ids = store.upsert_chunks(
        doc_id=doc_id,
        content_hash=doc["content_hash"],
        file_rel=doc["path"],
        scope=doc.get("scope", "catalogued"),
        doc_type=doc.get("type", "other"),
        metadata=doc.get("metadata") or {},
        chunks=chunks,
        vectors=vectors,
    )
    stale = sorted(set(doc.get("point_ids") or []) - set(point_ids))
    store.delete_points(stale)
    doc["point_ids"] = point_ids
    doc["chunks"] = len(chunks)


def _documents_needing_point_id_repair(state: dict) -> dict:
    """Find manifests whose ID order cannot prove source chunk order.

    Legacy point IDs may have been recorded in arbitrary scroll/ID order.
    Even syntactically valid chunk indexes backfilled from that order cannot
    establish passage identity. Rebuild these documents from source text.
    """
    return {
        doc_id: doc for doc_id, doc in state.get("documents", {}).items()
        if doc.get("point_ids") != [
            store.make_point_id(doc_id, i) for i in range(doc.get("chunks", 0))
        ]
    }


def _repair_research_identities(state: dict) -> dict:
    """Repair research identities, using only manifest-proven point mappings.

    Runs during explicit indexing, independently of scope backfill: a point with
    valid scope but a missing chunk identity is repaired here. Identity backfill
    is limited to points the manifest itself maps to a chunk index — manifest
    point_ids are deterministic `make_point_id(doc_id, i)` sequences, so the
    position *is* the proof. Arbitrary legacy point order is never assumed to be
    chunk order; unproven or textless identities get a staged reindex, and
    points no manifest document can account for are left untouched — readiness
    reports them rather than this pass purging them.
    """
    facts = store.scan_identity_payloads()
    documents = state.get("documents", {})
    to_reindex = _documents_needing_point_id_repair(state)
    proven: dict[object, tuple[str, int]] = {}
    path_to_doc: dict[str, str] = {}
    for doc_id, doc in documents.items():
        if doc.get("path") is not None:
            path_to_doc[doc["path"]] = doc_id
        if doc_id not in to_reindex:
            for index, pid in enumerate(doc.get("point_ids") or []):
                proven[pid] = (doc_id, index)

    backfilled = 0
    unresolvable = 0
    for pid, point_facts in facts.items():
        if (point_facts.get("document_id") in to_reindex
                or path_to_doc.get(point_facts.get("doc")) in to_reindex):
            continue
        if store.invalid_identity_reason(point_facts) is None:
            continue
        owner = proven.get(pid)
        if owner is None:
            owner_doc = path_to_doc.get(point_facts.get("doc"))
            owner = (owner_doc, None) if owner_doc else None
        if owner is None:
            unresolvable += 1
            continue
        doc_id, chunk_index = owner
        if chunk_index is not None and point_facts.get("text_present"):
            # Proven mapping and stored text: restore identity in place, no embedding.
            store.update_payloads([pid], {"document_id": doc_id, "chunk_index": chunk_index})
            backfilled += 1
        else:
            to_reindex.setdefault(doc_id, documents[doc_id])

    reindexed, failed = [], []
    for doc_id, doc in sorted(to_reindex.items()):
        try:
            _reindex_document(doc_id, doc)
            reindexed.append(doc_id)
            print(
                json.dumps({"doc_id": doc_id, "action": "reindexed_for_research", "chunks": doc["chunks"]}),
                file=sys.stderr,
            )
        except Exception as exc:  # old version must survive a failed replacement
            failed.append(doc_id)
            print(
                json.dumps({"doc_id": doc_id, "action": "research_reindex_failed", "error": str(exc)}),
                file=sys.stderr,
            )
    return {
        "backfilled": backfilled,
        "reindexed": reindexed,
        "failed": failed,
        "unresolvable": unresolvable,
    }


def _prepare_research_index(state: dict) -> dict:
    """Idempotent grouped-retrieval preparation, run on every explicit indexing pass.

    Ensures the keyword grouping payload index and repairs verified legacy
    identities. Inspection failures (Qdrant down) propagate as classified service
    errors; a corpus that is still unready afterwards is reported, not hidden.
    """
    created = store.ensure_grouping_index()
    readiness.ensure_filter_indexes()
    repair = _repair_research_identities(state)
    ready_info: dict = {"grouping_index_created": created, **repair}
    try:
        ready_info["point_count"] = store.check_research_ready()
        ready_info["ready"] = not repair["failed"]
        if repair["failed"]:
            ready_info["reason"] = "source-order identity repair failed; retry `library-query index`"
    except errors.ResearchIndexNotReadyError as exc:
        ready_info["ready"] = False
        ready_info["reason"] = str(exc)
        print(json.dumps({"action": "research_not_ready", "reason": str(exc)}), file=sys.stderr)
    return ready_info


def _research_repairs_needed() -> bool:
    """Read-only determination: would research preparation mutate the corpus?

    Mirrors what `_prepare_research_index` may write (grouping payload index,
    legacy scope backfill, identity repair/reindex) without performing any of
    it, so the revision protocol can publish dirty state before the first
    corpus write. Inspection failures propagate as the classified service
    errors they are.
    """
    from qdrant_client.models import PayloadSchemaType

    if _documents_needing_point_id_repair(ledger._load()):
        return True

    try:
        info = store.qdrant().get_collection(collection_name=config.COLLECTION)
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise errors.QdrantUnreachableError(
            f"Could not reach Qdrant at {config.QDRANT_URL}: {exc}"
        ) from exc
    payload_schema = getattr(info, "payload_schema", None) or {}
    entry = payload_schema.get(store.GROUPING_FIELD) if isinstance(payload_schema, dict) else None
    if entry is None or store._payload_index_data_type(entry) != str(PayloadSchemaType.KEYWORD):
        return True
    if store.count_points_missing_field("scope") > 0:
        return True
    facts = store.scan_identity_payloads()
    return any(store.invalid_identity_reason(point_facts) is not None for point_facts in facts.values())


def cmd_readiness(args: argparse.Namespace) -> int:
    """Cheap operational probe using the same mandatory proof as research."""
    deadline = research._Deadline(config.RESEARCH_TIMEOUT, time.monotonic)
    repaired = False
    try:
        with readiness.reader(config.COLLECTION, deadline) as validation:
            if validation is None:
                raise readiness.not_ready("unmanaged_index")
            verified = validation.revision
    except errors.ResearchIndexNotReadyError:
        if not args.repair:
            raise
        record, repaired = readiness.repair_markers(config.COLLECTION)
        verified = revisions.validate_stable_record(record, revisions.managed_context())
        if verified is None:
            raise readiness.not_ready("repair_verification_failed")
    result = {"research_ready": True, "repaired": repaired,
              "index_revision": verified.revision,
              "index_incarnation": verified.incarnation}
    print(json.dumps(result))
    return 0


def _index_counts(summary: dict) -> dict:
    return {key: summary[key] for key in ("indexed", "new", "changed", "moved", "deleted", "unchanged")}


def cmd_index(args: argparse.Namespace) -> int:
    """Run indexing and journal its outcome (error class name only, never a message)."""
    counts: dict = {}
    try:
        code = _run_index(args, counts)
    except BaseException as exc:  # noqa: BLE001 — recorded, then re-raised untouched
        journal.append_event("index_finished", outcome="error", error=type(exc).__name__)
        raise
    journal.append_event("index_finished", outcome="ok" if code == 0 else "not_ready", exit_code=code, **counts)
    return code


def _run_index(args: argparse.Namespace, counts: dict) -> int:
    extract.validate_supported_extensions()
    require_library_root()
    # Revision publication is part of indexing for every writer, cache-enabled
    # or not: an exclusive guard wraps every retrieval-affecting corpus write,
    # durable dirty state publishes before the first mutation, and a stable
    # revision publishes only after the mutations and readiness checks succeed.
    ctx = revisions.managed_context()
    with revisions.writer_session(ctx) as session:
        plan = ledger.reconcile()
        journal.append_event(
            "index_started",
            files_seen=sum(len(plan[key]) for key in ("new", "changed", "moved", "unchanged")),
        )

        has_actions = bool(
            plan["new"]
            or plan["changed"]
            or plan["moved"]
            or plan["deleted"]
            or any(item.get("metadata_changed") for item in plan["unchanged"])
        )

        state = ledger._load()
        collection_existed = store.collection_exists()
        record = revisions.read_record(ctx)
        compatible = readiness.valid_local(record, ctx)
        will_mutate = has_actions or not collection_existed or compatible is None
        if not will_mutate:
            will_mutate = _research_repairs_needed()
        if not will_mutate:
            try:
                readiness.verify_info(readiness.collection_info(ctx.collection), record["research_validation"])
            except errors.ResearchIndexNotReadyError:
                will_mutate = True
        if will_mutate:
            readiness.invalidate(session, recreated=not collection_existed)

        store.ensure_collection()

        # A new embedding/representation deployment requires all managed
        # documents to be rebuilt before this fingerprint can be trusted.
        rebuild = (not collection_existed or (isinstance(record, dict) and
                   (record.get("embedding_fingerprint") != ctx.embedding_fingerprint or
                    record.get("representation_fingerprint") != ctx.representation_fingerprint)))
        if not has_actions:
            if rebuild:
                for doc_id, doc in state.get("documents", {}).items():
                    _reindex_document(doc_id, doc)
            # Even an otherwise-idle pass must reconcile legacy points that predate the
            # v2 manifest and still lack scope/type payloads, and must keep the grouped
            # research index and identities ready.
            backfilled = backfill_legacy_payloads(state)
            research_info = _prepare_research_index(state)
            if not research_info["ready"] and not will_mutate:
                readiness.invalidate(session, recreated=False)
            if research_info["reindexed"] or rebuild:
                ledger._save(state)
            published = None
            if will_mutate and research_info["ready"]:
                store.verify_collection_configured()
                published = readiness.publish(session)
            summary = {
                "indexed": 0,
                "new": 0,
                "changed": 0,
                "moved": 0,
                "deleted": 0,
                "unchanged": len(plan["unchanged"]),
                "conflicts": plan.get("conflicts", []),
                "backfilled": backfilled,
                "research_ready": research_info["ready"],
                "message": "library up to date" if research_info["ready"] else "research validation failed",
            }
            if published is not None:
                summary["index_revision"] = published["revision"]
                summary["index_incarnation"] = published["incarnation"]
            counts.update(_index_counts(summary))
            print(json.dumps(summary))
            return 0 if research_info["ready"] else errors.EXIT_CONFIG_ERROR

        backfilled = backfill_legacy_payloads(state)
        new_count = 0
        changed_count = 0
        moved_count = 0
        deleted_count = 0

        # 1. Deletions (purge from Qdrant and manifest)
        for item in plan["deleted"]:
            store.delete_points(item.get("point_ids", []))
            state.get("documents", {}).pop(item["doc_id"], None)
            deleted_count += 1
            print(json.dumps({"file": item.get("path"), "action": "purged", "doc_id": item["doc_id"]}), file=sys.stderr)

        # 2. Moves (update path, scope, type, metadata payloads in place with zero re-embedding)
        for item in plan["moved"]:
            payload_update = _structural_payload(
                item["scope"], item["type"], item["metadata"], item["new_path"]
            )
            store.update_payloads(item.get("point_ids", []), payload_update)
            if item["doc_id"] in state.get("documents", {}):
                doc_record = state["documents"][item["doc_id"]]
                doc_record["path"] = item["new_path"]
                doc_record["scope"] = item["scope"]
                doc_record["type"] = item["type"]
                doc_record["metadata"] = item["metadata"]
            moved_count += 1
            print(json.dumps({"file": item["new_path"], "action": "moved", "from": item["old_path"]}), file=sys.stderr)

        # 3. Metadata updates for unchanged files (zero re-embedding)
        for item in plan["unchanged"]:
            if item.get("metadata_changed"):
                payload_update = _structural_payload(item["scope"], item["type"], item["metadata"])
                store.update_payloads(item.get("point_ids", []), payload_update)
                if item["doc_id"] in state.get("documents", {}):
                    state["documents"][item["doc_id"]]["metadata"] = item["metadata"]
                print(json.dumps({"file": item.get("path"), "action": "metadata_updated"}), file=sys.stderr)

        # 4. Changed content (re-extract, re-embed, replace points)
        for item in plan["changed"]:
            store.delete_points(item.get("old_point_ids", []))
            state.get("documents", {}).pop(item.get("old_doc_id"), None)

            pages = extract.extract(item["file_path"])
            chunks = extract.chunk(pages)
            if not chunks:
                print(json.dumps({"file": item["path"], "skipped": "no extractable text"}), file=sys.stderr)
                continue
            vectors = store.embed([c["text"] for c in chunks])
            point_ids = store.upsert_chunks(
                doc_id=item["doc_id"],
                content_hash=item["content_hash"],
                file_rel=item["path"],
                scope=item["scope"],
                doc_type=item["type"],
                metadata=item["metadata"],
                chunks=chunks,
                vectors=vectors,
            )
            state.setdefault("documents", {})[item["doc_id"]] = {
                "content_hash": item["content_hash"],
                "path": item["path"],
                "scope": item["scope"],
                "type": item["type"],
                "metadata": item["metadata"],
                "chunks": len(chunks),
                "point_ids": point_ids,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            changed_count += 1
            print(json.dumps({"file": item["path"], "action": "reindexed", "chunks": len(chunks)}), file=sys.stderr)

        # 5. New content (extract, embed, upsert)
        for item in plan["new"]:
            pages = extract.extract(item["file_path"])
            chunks = extract.chunk(pages)
            if not chunks:
                print(json.dumps({"file": item["path"], "skipped": "no extractable text"}), file=sys.stderr)
                continue
            vectors = store.embed([c["text"] for c in chunks])
            point_ids = store.upsert_chunks(
                doc_id=item["doc_id"],
                content_hash=item["content_hash"],
                file_rel=item["path"],
                scope=item["scope"],
                doc_type=item["type"],
                metadata=item["metadata"],
                chunks=chunks,
                vectors=vectors,
            )
            state.setdefault("documents", {})[item["doc_id"]] = {
                "content_hash": item["content_hash"],
                "path": item["path"],
                "scope": item["scope"],
                "type": item["type"],
                "metadata": item["metadata"],
                "chunks": len(chunks),
                "point_ids": point_ids,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            new_count += 1
            print(json.dumps({"file": item["path"], "action": "indexed", "chunks": len(chunks)}), file=sys.stderr)

        if rebuild:
            freshly_indexed = {item["doc_id"] for item in plan["new"] + plan["changed"]}
            for doc_id, doc in state.get("documents", {}).items():
                if doc_id not in freshly_indexed:
                    _reindex_document(doc_id, doc)

        # 6. Grouped-retrieval preparation and verified identity repair. Runs after all
        #    content mutations so staged reindexes see the final corpus state.
        research_info = _prepare_research_index(state)

        ledger._save(state)

        published = None
        if will_mutate and research_info["ready"]:
            store.verify_collection_configured()
            published = readiness.publish(session)

        summary = {
            "indexed": new_count + changed_count,
            "new": new_count,
            "changed": changed_count,
            "moved": moved_count,
            "deleted": deleted_count,
            "unchanged": len(plan["unchanged"]),
            # Duplicated content hashes, reported rather than silently rebound.
            "conflicts": plan.get("conflicts", []),
            "backfilled": backfilled,
            "research_ready": research_info["ready"],
        }
        if published is not None:
            summary["index_revision"] = published["revision"]
            summary["index_incarnation"] = published["incarnation"]
        counts.update(_index_counts(summary))
        print(json.dumps(summary))
        return 0 if research_info["ready"] else errors.EXIT_CONFIG_ERROR


def _complete_result(place: dict) -> dict:
    """Guarantee the stable contract fields are present, in documented order.

    `store.query_places` already populates them, but the contract is enforced here so
    a future change to the store cannot silently drop a key from the CLI's output.
    """
    return {field: place.get(field) for field in QUERY_OUTPUT_FIELDS}


def cmd_query(args: argparse.Namespace) -> int:
    """Run a query and journal its outcome; never the question text or the error message."""
    started = time.monotonic()
    result: dict = {}
    try:
        code = _run_query(args, result)
    except BaseException as exc:  # noqa: BLE001 — recorded, then re-raised untouched
        journal.append_event(
            "query_finished", outcome="error", error=type(exc).__name__,
            duration=journal.duration_bucket(time.monotonic() - started),
        )
        raise
    journal.append_event(
        "query_finished", outcome="ok" if code == 0 else "error",
        duration=journal.duration_bucket(time.monotonic() - started), **result,
    )
    return code


def _run_query(args: argparse.Namespace, result: dict) -> int:
    expand = not args.no_expand
    target = query_target(args)
    extra = {"target": target} if target is not None else {}
    tags = list(dict.fromkeys(
        tag for tag in (structure.normalize_tag(raw) for raw in args.tag) if tag
    ))
    session = cache.session_for(args.cache_mode)
    if session is None:
        places = [
            _complete_result(p)
            for p in store.query_places(
                args.question,
                top_k=args.top_k,
                scope=args.scope,
                tags=tags,
                expand=expand,
                min_score=args.min_score,
                **extra,
            )
        ]
        cache_diagnostics = cache.off_diagnostics()
        probes = expansion.probes(args.question, expand=expand)
        cache_diagnostics["embedding_request_count"] = (len(probes) + config.EMBED_BATCH_SIZE - 1) // config.EMBED_BATCH_SIZE
        cache_diagnostics["search_request_count"] = len(probes)
    else:
        with session:
            places = [
                _complete_result(p)
                for p in store.query_places(
                    args.question,
                    top_k=args.top_k,
                    scope=args.scope,
                    tags=tags,
                    expand=expand,
                    min_score=args.min_score,
                    cache=session,
                    **extra,
                )
            ]
        cache_diagnostics = session.diagnostics()
    result["results"] = len(places)
    result["top_score_band"] = journal.score_band(places[0].get("score") if places else None)
    if args.json:
        # `expansion` is a sibling of `places`, never a field inside a result: the
        # The result-field contract predates this change and consumers reading
        # `places[]` keep working without knowing expansion happened. It is reported
        # rather than hidden because expansion is on by default, so a surprising
        # result needs to be traceable to the cue that produced it. `cache` is
        # likewise an additive sibling and never changes the thirteen place fields.
        output = {
            "question": args.question,
            "scope": args.scope,
            "tags": tags,
            "expansion": expansion.diagnostics(args.question, expand=expand),
            "places": places,
            "cache": cache_diagnostics,
        }
        if target is not None:
            output["target"] = target
        print(json.dumps(output, ensure_ascii=False))
    else:
        for place in places:
            title = place.get("title") or place.get("doc_name") or place.get("doc") or "(unknown document)"
            meta_items = []
            if place.get("authors"):
                auth = place["authors"]
                meta_items.append(", ".join(auth) if isinstance(auth, list) else str(auth))
            if place.get("year"):
                meta_items.append(str(place["year"]))
            meta_info = f" ({' | '.join(meta_items)})" if meta_items else ""
            scope_info = f" [{place.get('scope', '')}]" if place.get("scope") else ""
            loc = ""
            if place.get("page") is not None:
                doc_str = place.get("doc_name") or place.get("doc") or ""
                suffix = Path(doc_str).suffix.lower()
                if suffix == ".epub":
                    loc = f" section {place['page']}"
                elif suffix == ".pdf" or suffix not in {".md", ".txt"}:
                    loc = f" PDF p.{place['page']}"
            where = f"{title}{meta_info}{loc}"
            score = place.get("score")
            score_info = f"[{score:.3f}]" if isinstance(score, (int, float)) else "[  -  ]"
            print(f"{score_info}{scope_info} {where}")
            print(f"  {(place.get('text') or '')[:300].replace(chr(10), ' ')}")
            print()
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    """Exact text lookup: where does this literal string appear, verbatim?"""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    wanted = [lit for lit in dict.fromkeys(args.literal) if lit.strip()]
    if not wanted:
        raise errors.InvalidResearchRequestError("find: provide at least one non-blank literal")
    client = store.qdrant()
    if not exact.is_ready(client, config.COLLECTION):
        raise errors.ResearchIndexNotReadyError("exact-text index is not built; run `library-query index`")
    base = None
    if args.scope != "all":
        base = Filter(must=[FieldCondition(key="scope", match=MatchValue(value=args.scope))])
    hits = exact.search(client, config.COLLECTION, wanted, base, max_hits=args.limit)
    places = []
    for hit in hits:
        place = store._place_from_hit(hit)
        place["matches"] = hit.exact_matches
        place["text"] = exact.snippet(hit.payload.get("text") or "", next(
            (lit for lit in wanted if lit in (hit.payload.get("text") or "")), wanted[0]))
        places.append({k: place.get(k) for k in (
            "doc", "doc_name", "page", "matches", "text", "scope", "type", "title", "authors", "year")})
    if args.json:
        print(json.dumps({"literals": wanted, "places": places}, ensure_ascii=False))
        return 0
    if not places:
        print(f"No verbatim match for: {', '.join(wanted)}")
        return 0
    for place in places:
        authors = ", ".join(place["authors"]) if isinstance(place["authors"], list) else (place["authors"] or "")
        meta = " | ".join(x for x in (authors, str(place["year"]) if place["year"] else "") if x)
        loc = f" PDF p.{place['page']}" if place["page"] is not None else ""
        print(f"[{place['matches']}x] {place['title'] or place['doc_name']}{f' ({meta})' if meta else ''}{loc}")
        print(f"  {place['text']}")
        print()
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    """Versioned multi-question evidence retrieval: bounded JSON in, bundle out.

    The whole request validates before any service call or cache access;
    retrieval is read-only and shares one total service-work deadline. Failures
    emit the classified error contract with no partial stdout bundle. The cache
    session (when enabled) aborts staged candidate publication on any failure.
    """
    data = research.read_input(args.input)
    request = research.parse_request(data)
    resolved = research.resolve_targets(request)
    extra = {"resolved_targets": resolved} if request.schema_version == 2 else {}
    session = cache.session_for(args.cache_mode)
    if session is None:
        bundle = research.run(request, **extra)
    else:
        with session:
            bundle = research.run(request, cache=session, **extra)
    print(json.dumps(bundle, ensure_ascii=False))
    return 0


def cmd_cache_status(args: argparse.Namespace) -> int:
    """Read-only local cache inspection: creates, recovers, and contacts nothing."""
    settings = cache.resolve_settings(None, strict=True)
    print(json.dumps(cache.status_report(settings), ensure_ascii=False))
    return 0


def cmd_cache_clear(args: argparse.Namespace) -> int:
    """Targeted scoped cache deletion; never touches directories or index records."""
    settings = cache.resolve_settings(None, strict=True)
    print(json.dumps(cache.clear_storage(settings, args.kind), ensure_ascii=False))
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    require_library_root()
    print(json.dumps(ledger.stats()))
    return 0


def cmd_browse(args: argparse.Namespace) -> int:
    """Offline inventory, or explicitly requested managed navigation writes."""
    root = require_library_root()
    search = getattr(args, "search", None)
    if search is not None and (not search.strip() or args.write):
        raise errors.InvalidResearchRequestError("search: requires non-blank text and cannot be combined with write")
    if args.write:
        result = browse.write_navigation(root)
        print(json.dumps(result, ensure_ascii=False))
        if result["conflicts"]:
            raise errors.NavigationWriteBlockedError(
                f"{len(result['conflicts'])} navigation conflict(s); no files were written"
            )
        return 0
    if args.json:
        print(json.dumps(browse.browse_json(root, search), ensure_ascii=False, indent=2))
    else:
        print(browse.format_human(root, search), end="")
    return 0


def cmd_plan_layout(args: argparse.Namespace) -> int:
    """Read-only migration planning: JSON map out, nothing moved or written."""
    root = require_library_root()
    plan = layout_plan.plan_layout(root)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if plan["conflicts"]:
        raise errors.LayoutPlanBlockedError(
            f"{len(plan['conflicts'])} blocking conflict(s); see the plan JSON on stdout"
        )
    return 0


def cmd_sync_zotero_notes(args: argparse.Namespace) -> int:
    """Materialize Zotero child notes as generated, indexable Markdown."""
    root = require_library_root()
    summary = zotero.sync_notes(root, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def query_target(args):
    item_ids, document_ids, roles = (getattr(args, name, None) for name in ("item", "document", "role"))
    if roles and not item_ids:
        targets.invalid("role", "requires item selection")
    if not item_ids and not document_ids:
        return None
    value = {"item_ids": item_ids} if item_ids else {"document_ids": document_ids}
    if roles:
        value["roles"] = roles
    resolved = targets.resolve(targets.parse(value))
    targets.verify_index(resolved)
    return resolved


def cmd_doctor(args: argparse.Namespace) -> int:
    results = doctor.run_checks()
    code = doctor.exit_code(results)
    if args.json:
        print(json.dumps(doctor.to_json(results), ensure_ascii=False))
    else:
        for name, ok, detail in results:
            print(f"[{'ok' if ok else 'fail'}] {name}: {detail}")
    return code


def cmd_diagnose(_: argparse.Namespace) -> int:
    print(doctor.build_bundle(doctor.run_checks()), end="")
    return 0


def cmd_inspect(args):
    require_library_root()
    print(json.dumps(inspection.inspect_item(args.item), ensure_ascii=False))
    return 0


def cmd_source(args):
    require_library_root()
    print(json.dumps(inspection.source(args.document, args.path, args.offset, args.limit), ensure_ascii=False))
    return 0


def cmd_passage(args):
    print(json.dumps(inspection.passage(args.evidence_id, args.before, args.after, args.max_chars), ensure_ascii=False))
    return 0


def _describe_type(action: argparse.Action) -> str:
    """Name the argument's value type from the parser's own `type` configuration."""
    if isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
        return "boolean"
    if action.type is int:
        return "integer"
    if action.type is float:
        return "number"
    return "string"


def _describe_action(action: argparse.Action) -> dict:
    """Serialize one argparse action into the machine-readable description."""
    return {
        "name": action.dest,
        "flags": list(action.option_strings),
        "positional": not action.option_strings,
        "type": _describe_type(action),
        "choices": list(action.choices) if action.choices else None,
        "default": action.default,
        "required": bool(action.required),
        "help": action.help,
    }


def describe_parser(parser: argparse.ArgumentParser) -> dict:
    """Derive the tool schema from the live parser, so it can never drift from the CLI.

    A flag added to `main()`'s parser shows up here automatically; there is no second
    copy of the interface to keep in sync.
    """
    subcommands = {}
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, subparser in action.choices.items():
            arguments = [
                _describe_action(sub_action)
                for sub_action in subparser._actions
                if not isinstance(sub_action, argparse._HelpAction)
            ]
            entry = {
                "name": name,
                "help": subparser.description or None,
                "arguments": arguments,
            }
            if name == "query":
                entry["output_fields"] = list(QUERY_OUTPUT_FIELDS)
                entry["cache_diagnostics"] = (
                    "additive top-level `cache` sibling in --json output; "
                    "typed contract in the top-level cache section"
                )
            if name == "browse":
                entry["inventory_json_version"] = browse.BROWSE_JSON_VERSION
            if name == "plan-layout":
                entry["plan_version"] = layout_plan.PLAN_VERSION
                entry["naming"] = {
                    "algorithm_version": layout_plan.NAMING_ALGORITHM_VERSION,
                    "base_stem_ceiling_bytes": layout_plan.BASE_STEM_CEILING_BYTES,
                    "applying_host_verified": False,
                }
            if name == "research":
                # argparse cannot describe the contents of a JSON file, so the
                # nested schemas come from the same runtime definitions validation
                # and serialization use — one source of truth, no drift.
                entry["request_schema"] = research.request_schema()
                entry["response_schema"] = research.response_schema()
                entry["research_timeout_seconds"] = config.RESEARCH_TIMEOUT
                entry["coverage_statuses"] = list(research.COVERAGE_STATUSES)
                entry["evidence_id_format"] = "<source_id>:<chunk_index>:<passage_hash>"
                entry["supported_versions"] = [1, 2]
                entry["version_2"] = research.version2_schema()
            subcommands[name] = entry

    return {
        "name": parser.prog,
        "description": parser.description,
        "exit_codes": {
            "0": "success, including zero query results",
            "2": "invalid CLI arguments",
            "3": "configuration error",
            "4": "external service unreachable or erroring",
            "5": "internal or unexpected error",
        },
        "error_categories": dict(sorted(errors.ERROR_EXIT_CODES.items())),
        "commands": subcommands,
        # The typed cache contract, from the same runtime definitions that
        # validation, serialization, and the cache commands use. Discovery
        # reads and creates nothing.
        "cache": cache.describe_schema(),
        "targets": targets.schema(),
        "inspection": inspection.schema(),
    }


def cmd_describe(args: argparse.Namespace) -> int:
    """Emit the tool schema. Reads nothing, connects to nothing."""
    print(json.dumps(describe_parser(args.parser), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Single source of truth for both parsing and `describe`."""
    parser = argparse.ArgumentParser(prog="library-query", description="Semantic locator for a document library")
    sub = parser.add_subparsers(dest="command", required=True)

    index_help = "reconcile library files against vector store"
    p_index = sub.add_parser("index", help=index_help, description=index_help)
    p_index.set_defaults(func=cmd_index)

    p_readiness = sub.add_parser("readiness", help="verify managed research readiness without a corpus scan")
    p_readiness.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_readiness.add_argument("--repair", action="store_true", help="explicitly audit and republish missing markers; never re-embed or repair incompatible/dirty indexes")
    p_readiness.set_defaults(func=cmd_readiness)

    query_help = "question in, places out"
    p_query = sub.add_parser("query", help=query_help, description=query_help)
    p_query.add_argument("question", help="the question to locate in the library")
    p_query.add_argument(
        "--scope",
        choices=["all", "inbox", "catalogued", "operations"],
        default="all",
        help="workflow scope",
    )
    p_query.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="TAG",
        help="require a Zotero tag; repeat to require every listed tag",
    )
    p_query.add_argument("--json", action="store_true", help="machine-readable output")
    p_query.add_argument("--top-k", type=int, default=5, help="number of results to return")
    p_query.add_argument(
        "--no-expand",
        action="store_true",
        help="search the question verbatim, without colloquial-to-clinical expansion",
    )
    p_query.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="drop merged results scoring below this floor; returning none of them is valid output",
    )
    p_query.add_argument(
        "--cache-mode",
        choices=list(cache.CACHE_MODES),
        default=None,
        help="retrieval cache mode for this request; overrides LIBRARY_QUERY_CACHE_MODE (default off)",
    )
    p_query.set_defaults(func=cmd_query)
    selection = p_query.add_mutually_exclusive_group()
    selection.add_argument("--item", action="append", help="exact immutable item ID; repeat to select works")
    selection.add_argument("--document", action="append", help="exact content document ID; repeat to select documents")
    p_query.add_argument("--role", action="append", choices=targets.ROLES, help="item attachment role; default source")

    research_help = "versioned multi-question evidence retrieval (JSON request in, evidence bundle out)"
    p_research = sub.add_parser("research", help=research_help, description=research_help)
    p_research.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="request JSON file, or '-' to read standard input",
    )
    p_research.add_argument(
        "--cache-mode",
        choices=list(cache.CACHE_MODES),
        default=None,
        help="retrieval cache mode for this request; overrides LIBRARY_QUERY_CACHE_MODE (default off)",
    )
    p_research.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_research.set_defaults(func=cmd_research)

    cache_help = "scoped maintenance for the opt-in retrieval cache (read-only status; targeted clear)"
    p_cache = sub.add_parser("cache", help=cache_help, description=cache_help)
    cache_sub = p_cache.add_subparsers(dest="cache_command", required=True)
    cache_status_help = "read-only local cache inspection (creates and recovers nothing)"
    p_cache_status = cache_sub.add_parser("status", help=cache_status_help, description=cache_status_help)
    p_cache_status.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_cache_status.set_defaults(func=cmd_cache_status)
    cache_clear_help = "delete current-namespace cache entries by kind; never touches directories or index records"
    p_cache_clear = cache_sub.add_parser("clear", help=cache_clear_help, description=cache_clear_help)
    p_cache_clear.add_argument(
        "--kind",
        choices=["embeddings", "evidence", "all"],
        default="all",
        help="which current-context entries to remove (default: all)",
    )
    p_cache_clear.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_cache_clear.set_defaults(func=cmd_cache_clear)

    status_help = "files in library / indexed / pending"
    p_status = sub.add_parser("status", help=status_help, description=status_help)
    p_status.set_defaults(func=cmd_status)

    sync_notes_help = "materialize Zotero child notes as generated Markdown (run index afterward)"
    p_sync_notes = sub.add_parser("sync-zotero-notes", help=sync_notes_help, description=sync_notes_help)
    p_sync_notes.add_argument("--dry-run", action="store_true", help="report note changes without writing files")
    p_sync_notes.set_defaults(func=cmd_sync_zotero_notes)

    browse_help = "offline human-readable or JSON inventory of library items"
    p_browse = sub.add_parser("browse", help=browse_help, description=browse_help)
    p_browse.add_argument("--json", action="store_true", help="machine-readable version-1 inventory")
    p_browse.add_argument("--search", help="offline lookup by title, author, subject or citekey words; ignores case, accents and punctuation; returns every match")
    p_browse.add_argument(
        "--write",
        action="store_true",
        help="update managed navigation blocks in root/shelf READMEs and Collections/subjects.md",
    )
    p_browse.set_defaults(func=cmd_browse)

    plan_help = "read-only flat-layout migration plan (JSON out; nothing is moved or written)"
    p_plan = sub.add_parser("plan-layout", help=plan_help, description=plan_help)
    p_plan.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_plan.set_defaults(func=cmd_plan_layout)

    describe_help = "emit the machine-readable tool schema"
    p_describe = sub.add_parser("describe", help=describe_help, description=describe_help)
    p_describe.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_describe.set_defaults(func=cmd_describe)

    find_help = "find a literal string verbatim (identifiers, symbols, quoted code); exact, not semantic"
    p_find = sub.add_parser("find", help=find_help, description=find_help)
    p_find.add_argument("literal", nargs="+", help="text to find, matched case-sensitively as written")
    p_find.add_argument("--scope", choices=["all", "inbox", "catalogued", "operations"], default="all")
    p_find.add_argument("--limit", type=int, default=exact.MAX_HITS, help="maximum places returned")
    p_find.add_argument("--json", action="store_true", help="machine-readable output")
    p_find.set_defaults(func=cmd_find)

    doctor_help = "check environment, library folder, Qdrant, and ledger (exit 1 on any failure)"
    p_doctor = sub.add_parser("doctor", help=doctor_help, description=doctor_help)
    p_doctor.add_argument("--json", action="store_true", help="machine-readable output")
    p_doctor.set_defaults(func=cmd_doctor)

    diagnose_help = "print a redacted diagnostic report for bug reports (no queries, paths, or keys)"
    p_diagnose = sub.add_parser("diagnose", help=diagnose_help, description=diagnose_help)
    p_diagnose.set_defaults(func=cmd_diagnose)

    p_inspect = sub.add_parser("inspect", help="offline details for an exact bibliographic item")
    p_inspect.add_argument("--item", required=True)
    p_inspect.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_inspect.set_defaults(func=cmd_inspect)
    p_source = sub.add_parser("source", help="bounded indexed chunks in source order")
    source_target = p_source.add_mutually_exclusive_group(required=True)
    source_target.add_argument("--document")
    source_target.add_argument("--path", help="contained library-relative attachment path")
    p_source.add_argument("--offset", type=int, default=0)
    p_source.add_argument("--limit", type=int, default=inspection.SOURCE_LIMIT)
    p_source.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_source.set_defaults(func=cmd_source)
    p_passage = sub.add_parser("passage", help="verify and read an indexed passage with neighboring context")
    p_passage.add_argument("--evidence-id", required=True)
    p_passage.add_argument("--before", type=int, default=inspection.CONTEXT_COUNT)
    p_passage.add_argument("--after", type=int, default=inspection.CONTEXT_COUNT)
    p_passage.add_argument("--max-chars", type=int, default=inspection.CONTEXT_CHARS)
    p_passage.add_argument("--json", action="store_true", help="machine-readable output (always on)")
    p_passage.set_defaults(func=cmd_passage)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch, converting every failure into the error contract.

    `parse_args` is deliberately outside the try block: argparse already exits `2` on
    malformed input via SystemExit, and that behavior is part of the contract, not
    something to reclassify.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.parser = parser

    try:
        return args.func(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        return errors.emit("internal_error", "Interrupted", errors.EXIT_INTERNAL_ERROR)
    except BaseException as exc:  # noqa: BLE001 — the point is that nothing escapes
        return errors.report(exc)


if __name__ == "__main__":
    raise SystemExit(main())
