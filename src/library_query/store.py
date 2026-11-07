"""Embeddings via OpenRouter (openai-compat) and the Qdrant store."""
from pathlib import Path
import json
import uuid
from contextvars import ContextVar
from contextlib import closing, contextmanager


import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchValue,
    PayloadField,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from . import config, errors, exact, expansion

_VALIDATION_CALLS = ContextVar("validation_calls", default=None)
SERVICE_DEADLINE = ContextVar("service_deadline", default=None)


def _validation_timeout():
    deadline = SERVICE_DEADLINE.get()
    if deadline is None:
        return {}
    deadline.check()
    return {"timeout": deadline.service_timeout()}


@contextmanager
def _validation_client():
    deadline = SERVICE_DEADLINE.get()
    if deadline is None:
        yield qdrant()
    else:
        deadline.check()
        with closing(QdrantClient(url=config.QDRANT_URL, timeout=deadline.remaining(),
                                  check_compatibility=False)) as client:
            yield client
        deadline.check()


def _validation_call():
    counter = _VALIDATION_CALLS.get()
    if counter is not None:
        counter[0] += 1


def count_validation_calls(callback):
    counter = [0]
    token = _VALIDATION_CALLS.set(counter)
    try:
        callback()
        return counter[0]
    finally:
        _VALIDATION_CALLS.reset(token)


def _confirm_write(result):
    status = getattr(result, "status", None)
    if isinstance(status, str) and status.lower() != "completed":
        raise errors.IndexRevisionUnavailableError("index mutation completion was not confirmed")


def _point_wire_size(point: PointStruct) -> int:
    """Serialized bytes one point contributes to an upsert request body.

    Measured with the standard JSON encoder over the exact `{"id", "vector",
    "payload"}` shape the client puts on the wire, plus one separator byte.
    The client's encoder may produce slightly different bytes (UTF-8 vs escaped
    non-ASCII); the ceiling's margin absorbs that drift.
    """
    return len(json.dumps({"id": point.id, "vector": point.vector, "payload": point.payload})) + 1


def _batches_by_wire_size(points: list[PointStruct], max_bytes: int, context: str) -> list[list[PointStruct]]:
    """Pack points into sequential batches whose bodies stay under `max_bytes`.

    Order and membership are preserved exactly; every caller-visible point lands
    in exactly one batch. A single point that cannot fit even alone is never
    truncated or dropped: it raises the classified configuration error, because
    no batching can make it acceptable to the server.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    wrapper_bytes = len('{"points":[]}') + 1  # framing + trailing safety byte
    batches: list[list[PointStruct]] = []
    current: list[PointStruct] = []
    current_bytes = wrapper_bytes
    for point in points:
        size = _point_wire_size(point)
        if current and current_bytes + size > max_bytes:
            batches.append(current)
            current = []
            current_bytes = wrapper_bytes
        if size + wrapper_bytes > max_bytes:
            raise errors.QdrantBatchTooLargeError(
                f"point for '{context}' serializes to {size} bytes, exceeding the "
                f"configured Qdrant batch ceiling of {max_bytes} bytes; raise "
                "LIBRARY_QUERY_QDRANT_MAX_BATCH_BYTES or inspect the extractor"
            )
        current.append(point)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


_client: QdrantClient | None = None


def qdrant() -> QdrantClient:
    global _client
    if _client is None:
        # The client default is 5s, which bulk upserts during a full re-index and
        # queries contending with them exceed often enough to surface as
        # QdrantUnreachableError. The server is local: a slow call here means disk
        # pressure, not an unreachable host, so waiting is the right response.
        _client = QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT)
    return _client


def ensure_collection(collection_name: str | None = None) -> None:
    coll = collection_name or config.COLLECTION
    try:
        if not qdrant().collection_exists(coll):
            qdrant().create_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
            )
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise errors.QdrantUnreachableError(
            f"Could not reach Qdrant at {config.QDRANT_URL}: {exc}"
        ) from exc
    # Exact-text lookup needs this index; failing to build it only disables that leg.
    exact.ensure_text_index(qdrant(), coll)


def make_point_id(document_id: str, chunk_index: int) -> int:
    """Generate a deterministic 63-bit integer point ID invariant to file path."""
    return int(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}").int & 0x7FFFFFFFFFFFFFFF)


def embed(texts: list[str], timeout: float | None = None) -> list[list[float]]:
    """Embed texts via the provider's batch endpoint, splitting oversized inputs.

    Embedding providers cap both the request token budget and the per-text
    token count; a whole book's chunks exceed both, so texts are sent in
    bounded batches and concatenated in order.

    `timeout` caps each provider wait. Indexing passes None and inherits the
    historical 120s; the research path passes its remaining service-work
    deadline so one slow batch cannot blow the total research budget.
    """
    if not config.OPENROUTER_API_KEY:
        raise errors.MissingApiKeyError()
    wait = 120.0 if timeout is None else timeout
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        chunk_batch = texts[start : start + config.EMBED_BATCH_SIZE]
        try:
            response = httpx.post(
                f"{config.EMBEDDING_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
                json={"model": config.EMBEDDING_MODEL, "input": chunk_batch},
                timeout=wait,
            )
        except httpx.HTTPError as exc:
            raise errors.EmbeddingApiError(
                f"Could not reach the embedding API at {config.EMBEDDING_BASE_URL}: {exc}"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise errors.EmbeddingApiError(
                f"Embedding API returned HTTP {response.status_code} from {config.EMBEDDING_BASE_URL}"
            )
        data = response.json()["data"]
        vectors.extend(item["embedding"] for item in sorted(data, key=lambda d: d["index"]))
    return vectors


def upsert_chunks(
    doc_id: str,
    content_hash: str,
    file_rel: str,
    scope: str,
    doc_type: str,
    metadata: dict,
    chunks: list[dict],
    vectors: list[list[float]],
    collection_name: str | None = None,
) -> list[int]:
    coll = collection_name or config.COLLECTION
    doc_name = Path(file_rel).name
    points = []
    point_ids = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        pid = make_point_id(doc_id, index)
        point_ids.append(pid)
        payload = {
            "document_id": doc_id,
            "chunk_index": index,
            "content_hash": content_hash,
            "path": file_rel,
            "doc": file_rel,  # legacy compatibility
            "doc_name": doc_name,
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
            # Additive enrichment: the immutable item ID groups attachments of
            # one bibliographic item; document identity stays content-hash based.
            "item_id": metadata.get("item_id"),
            "page": chunk.get("page"),
            "text": chunk.get("text"),
        }
        points.append(PointStruct(id=pid, vector=vector, payload=payload))

    # Qdrant rejects request bodies over 32 MiB; a whole document's points can
    # exceed that (vectors dominate), so upsert in size-bounded batches, each
    # confirmed before the next. Deterministic IDs make a re-run of an
    # interrupted document an in-place overwrite, never duplicates.
    batches = _batches_by_wire_size(points, config.QDRANT_MAX_BATCH_BYTES, file_rel)
    for batch in batches:
        _confirm_write(qdrant().upsert(collection_name=coll, points=batch, wait=True))
    return point_ids


def update_payloads(
    point_ids: list[int | str],
    payload: dict,
    collection_name: str | None = None,
) -> None:
    """Update payload fields in place for the given point IDs, without re-embedding.

    `point_ids` is the sole authoritative selector: they are derived deterministically
    from (document_id, chunk_index), so no path- or doc_id-filter fallback is needed.
    `wait=True` keeps the managed-revision contract: an acknowledged-but-queued
    payload update is not completion.
    """
    if not point_ids:
        return
    coll = collection_name or config.COLLECTION
    _confirm_write(qdrant().set_payload(collection_name=coll, payload=payload, points=point_ids, wait=True))


def delete_points(
    point_ids: list[int | str],
    collection_name: str | None = None,
) -> None:
    """Delete vector points by point ID.

    `point_ids` is the sole authoritative selector; payload-filter fallbacks are a
    latent hazard if ever invoked out of strict delete-before-insert order.
    `wait=True` confirms completion before a stable indexed revision may publish.
    """
    if not point_ids:
        return
    coll = collection_name or config.COLLECTION
    _confirm_write(qdrant().delete(
        collection_name=coll,
        points_selector=PointIdsList(points=point_ids),
        wait=True,
    ))


def collection_exists(collection_name: str | None = None) -> bool:
    """Read-only existence check; transport failures stay classified errors."""
    coll = collection_name or config.COLLECTION
    try:
        _validation_call()
        with _validation_client() as client:
            return bool(client.collection_exists(coll))
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise _wrap_qdrant_error(exc) from exc


def verify_collection_configured(collection_name: str | None = None) -> None:
    """Live configuration check for candidate-cache reads (never a mutation).

    The collection must exist and its vector size must match the configured
    embedding dimensions. Runs before cached candidates can be admitted so a
    hit can never conceal a missing/misconfigured collection; on transport
    failure it raises the ordinary classified service error.
    """
    coll = collection_name or config.COLLECTION
    if not collection_exists(coll):
        raise errors.ResearchIndexNotReadyError(
            f"collection '{coll}' does not exist; run `library-query index` to build it"
        )
    try:
        _validation_call()
        with _validation_client() as client:
            info = client.get_collection(collection_name=coll)
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise _wrap_qdrant_error(exc) from exc
    size = _configured_vector_size(info)
    if size != config.EMBEDDING_DIM:
        raise errors.ResearchIndexNotReadyError(
            f"collection '{coll}' has vector size {size}, expected {config.EMBEDDING_DIM}; "
            "reindex with the configured embedding model"
        )


def _configured_vector_size(info) -> int | None:
    """The collection's configured vector size, or None when it cannot be read."""
    vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    size = getattr(vectors, "size", None)
    if isinstance(size, int) and not isinstance(size, bool):
        return size
    vectors = getattr(info, "vectors_config", None)
    params = getattr(vectors, "params", None)
    size = getattr(params, "size", None)
    if isinstance(size, bool) or not isinstance(size, int):
        return None
    return size


def get_point_ids_for_doc(file_rel: str, collection_name: str | None = None) -> list[int]:
    """Retrieve point IDs for a given document path from Qdrant."""
    coll = collection_name or config.COLLECTION
    try:
        points = []
        next_page = None
        seen_offsets: set = set()
        while True:
            _validation_call()
            res = qdrant().scroll(
                collection_name=coll,
                scroll_filter=Filter(must=[FieldCondition(key="doc", match=MatchValue(value=file_rel))]),
                limit=250,
                offset=next_page,
                with_payload=False,
                with_vectors=False,
            )
            points.extend(p.id for p in res[0])
            next_page = res[1]
            # Explicit None check: a falsy-but-not-None cursor (e.g. 0, or a mocked
            # client's sentinel) must not be mistaken for exhaustion, and repeating
            # an offset we have already followed means the cursor is not advancing.
            if next_page is None or next_page in seen_offsets:
                break
            seen_offsets.add(next_page)
        return points
    except Exception:
        return []


def count_points_missing_field(field: str = "scope", collection_name: str | None = None) -> int:
    """Count points in the collection whose payload lacks `field`.

    Used to detect legacy (pre-manifest) points that still need a payload backfill.
    A service failure here is never a successful zero: the caller cannot tell
    "nothing missing" from "Qdrant unreachable", so exceptions propagate as
    classified service errors instead of being swallowed into a count of 0.
    """
    coll = collection_name or config.COLLECTION
    try:
        return qdrant().count(
            collection_name=coll,
            count_filter=Filter(must=[IsEmptyCondition(is_empty=PayloadField(key=field))]),
        ).count
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise errors.QdrantUnreachableError(
            f"Could not reach Qdrant at {config.QDRANT_URL}: {exc}"
        ) from exc


def find_legacy_point_ids(file_rel: str, field: str = "scope", collection_name: str | None = None) -> list[int]:
    """Return IDs of points recorded at `file_rel` whose payload is missing `field`.

    Legacy points predate deterministic point IDs, so they cannot be addressed by
    `make_point_id`; the recorded path is the only handle onto them.
    """
    coll = collection_name or config.COLLECTION
    try:
        points: list[int] = []
        next_page = None
        seen_offsets: set = set()
        scroll_filter = Filter(
            must=[
                FieldCondition(key="doc", match=MatchValue(value=file_rel)),
                IsEmptyCondition(is_empty=PayloadField(key=field)),
            ]
        )
        while True:
            _validation_call()
            res = qdrant().scroll(
                collection_name=coll,
                scroll_filter=scroll_filter,
                limit=250,
                offset=next_page,
                with_payload=False,
                with_vectors=False,
            )
            points.extend(p.id for p in res[0])
            next_page = res[1]
            if next_page is None or next_page in seen_offsets:
                break
            seen_offsets.add(next_page)
        return points
    except Exception:
        return []


def query_places(
    question: str,
    top_k: int = 5,
    scope: str = "all",
    tags: list[str] | None = None,
    collection_name: str | None = None,
    expand: bool = True,
    min_score: float | None = None,
    cache=None,
    target=None,
) -> list[dict]:
    """Retrieve the top-k places for `question`, expanding its register by default.

    A cued colloquial query is searched once per clinical variant *in addition to*
    the original, and the results merged by best score per chunk. Probes are kept
    separate rather than concatenated into one query string: a blended vector both
    dilutes genuine matches and drags unrelated queries toward the corpus. An uncued query
    produces exactly one probe with the same input as unexpanded retrieval.
    Independent live calls may yield different scores or ANN candidates.

    `min_score`, when given, is a floor on the *merged* score: it is applied after
    per-chunk best-score merging and before the top-k cut. Filtering per probe
    instead would defeat expansion -- a chunk the raw query scores under the floor
    but an expanded probe scores above it is exactly the register-gap recovery
    expansion exists to make. Nothing clearing the floor is an empty result, which
    is a valid answer ("no place in the library meets this bar"), never an error.

    With a cache session, candidate lookups run before embedding lookups: a
    candidate hit supplies raw per-probe hits and needs no vector at all. Cached
    raw hits are re-merged, re-floored, and re-cut for this request exactly like
    fresh ones; nothing downstream of the search is replayed.
    """
    from . import cache as cache_mod

    coll = collection_name or config.COLLECTION
    probes = expansion.probes(question, expand=expand)
    filters = cache_mod.flat_filters(scope, tags)
    if target is not None:
        filters["target"] = target
        if not target["document_ids_resolved"]:
            return []
    window = {"limit": top_k}
    if cache is not None:
        cache.verify_index(live_check=lambda: verify_collection_configured(coll))

    candidate_payloads: list | None = None
    if cache is not None:
        candidate_payloads = [
            cache.lookup_candidates(probe_text=probe, method=cache_mod.METHOD_FLAT, filters=filters, window=window)
            for probe in probes
        ]

    filter_conditions = []
    if scope in ("inbox", "catalogued", "operations"):
        # `scope` is authoritative: the reconciliation backfill guarantees every point
        # carries it, so no legacy `status` fallback is needed (and matching `status`
        # too would over-match, since it mirrors a document's bibliographic read state).
        filter_conditions.append(FieldCondition(key="scope", match=MatchValue(value=scope)))
    for tag in tags or []:
        # A separate must condition per tag means repeated --tag flags are an
        # intersection: every requested Zotero tag must be present on a result.
        filter_conditions.append(FieldCondition(key="tags", match=MatchValue(value=tag)))
    query_filter = Filter(must=filter_conditions) if filter_conditions else None
    if target is not None:
        from qdrant_client.models import MatchAny
        filter_conditions.append(FieldCondition(key="document_id", match=MatchAny(any=target["document_ids_resolved"])))
        query_filter = Filter(must=filter_conditions)

    probes_needing_vectors = [
        probe for probe, payload in zip(probes, candidate_payloads or [None] * len(probes))
        if payload is None
    ]
    if cache is not None:
        vector_by_probe = cache.embed_probes(probes_needing_vectors, embed)
    else:
        vectors = embed(probes)
        vector_by_probe = dict(zip(probes, vectors))

    best: dict[object, object] = {}
    for index, probe in enumerate(probes):
        payload = candidate_payloads[index] if candidate_payloads is not None else None
        if payload is not None:
            hits = cache_mod.materialize_flat(payload)
        else:
            try:
                hits = qdrant().query_points(
                    collection_name=coll,
                    query=vector_by_probe[probe],
                    limit=top_k,
                    query_filter=query_filter,
                ).points
            except errors.LibraryQueryError:
                raise
            except Exception as exc:
                # The embedding call above already raised its own classified error, so any
                # failure surfacing here is the Qdrant transport — including bare `httpx`
                # errors the client lets through, which would otherwise be misattributed.
                raise errors.QdrantUnreachableError(
                    f"Could not reach Qdrant at {config.QDRANT_URL}: {exc}"
                ) from exc
            if cache is not None:
                cache.stage_candidates(
                    probe_text=probe,
                    method=cache_mod.METHOD_FLAT,
                    filters=filters,
                    window=window,
                    payload=cache_mod.flat_payload_from_hits(hits, filters=filters, window=window),
                )
                cache.diag.search_requests += 1

        for hit_index, hit in enumerate(hits):
            if target is not None and (hit.payload or {}).get("document_id") not in target["document_ids_resolved"]:
                raise errors.LibraryQueryError("target_not_indexed", "Search returned an off-target document; run library-query index")
            # Keep each chunk once, at the best score any probe gave it. Because the
            # original query is probe 0, its observed score is retained here.
            # Additional candidates can still displace it at final top-k.
            key = _hit_key(hit, hit_index)
            previous = best.get(key)
            if previous is None or _hit_score(hit) > _hit_score(previous):
                best[key] = hit

    ranked = sorted(best.values(), key=_hit_score, reverse=True)
    if min_score is not None:
        # Before the top-k cut, so the floor never lets a weak result occupy a slot
        # a stronger one was denied: top-k of the survivors, not survivors of top-k.
        ranked = [hit for hit in ranked if _hit_score(hit) >= min_score]
    return [_place_from_hit(hit) for hit in ranked[:top_k]]


def retrieve_chunks(document_id, indexes, collection_name=None, timeout=None):
    """Read only the bounded deterministic chunk coordinates requested."""
    try:
        return qdrant().retrieve(collection_name=collection_name or config.COLLECTION,
                                 ids=[make_point_id(document_id, i) for i in indexes],
                                 with_payload=True, with_vectors=False,
                                 **({"timeout": timeout} if timeout is not None else {}))
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise errors.QdrantUnreachableError("Could not read indexed passages") from exc


def _hit_key(hit, index: int):
    """Identity for de-duplicating a chunk across probes.

    Qdrant hits carry a stable point ID, which is the right key. Falling back to
    payload coordinates keeps the merge working for any hit-like object that
    omits it, rather than making point identity a new requirement on a shape the
    rest of this module never depended on.
    """
    point_id = getattr(hit, "id", None)
    if point_id is not None:
        return point_id
    payload = getattr(hit, "payload", None) or {}
    return (payload.get("path") or payload.get("doc"), payload.get("page"),
            payload.get("text"), index)


def _hit_score(hit) -> float:
    """A hit's score as a sortable number; an absent score ranks last, never crashes."""
    score = getattr(hit, "score", None)
    return score if isinstance(score, (int, float)) else float("-inf")


def _place_from_hit(hit) -> dict:
    """Build a result object carrying every contract field, `None` where unresolved.

    A key is never omitted: an agent doing `result["year"]` gets `None`, not a KeyError.
    """
    payload = hit.payload or {}
    return {
        "doc": payload.get("path") or payload.get("doc"),
        "doc_name": payload.get("doc_name"),
        "page": payload.get("page"),
        "score": round(hit.score, 4) if hit.score is not None else None,
        "text": (payload.get("text") or "")[:600],
        "scope": payload.get("scope"),
        "type": payload.get("type") or payload.get("doc_type"),
        "title": payload.get("title"),
        "authors": payload.get("authors"),
        "year": payload.get("year"),
        "subjects": payload.get("subjects"),
        "tags": payload.get("tags"),
        "identifier": payload.get("identifier"),
    }


# --- Research readiness and grouped retrieval -------------------------------------
#
# Research is strictly read-only: nothing in this section creates a collection,
# creates an index, mutates a payload, or embeds content. Preparation (grouping
# index, identity repair, staged reindexing) belongs to the explicit `index`
# workflow in cli.py.

#: The single scalar source identity research groups by. One keyword payload index
#: on this field is the compatibility requirement for grouped retrieval.
GROUPING_FIELD = "document_id"

_IDENTITY_PAYLOAD_FIELDS = [GROUPING_FIELD, "chunk_index", "text", "doc"]
_IDENTITY_FIELDS_NO_TEXT = [GROUPING_FIELD, "chunk_index", "doc"]


def _wrap_qdrant_error(exc: Exception) -> errors.QdrantUnreachableError:
    return errors.QdrantUnreachableError(f"Could not reach Qdrant at {config.QDRANT_URL}: {exc}")


def _payload_index_data_type(entry) -> str | None:
    data_type = getattr(entry, "data_type", None)
    if data_type is None and isinstance(entry, dict):
        data_type = entry.get("data_type")
    return str(data_type) if data_type is not None else None


def _require_grouping_index(collection: str) -> None:
    """Raise research_index_not_ready unless a keyword index on document_id exists."""
    try:
        _validation_call()
        with _validation_client() as client:
            info = client.get_collection(collection_name=collection)
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise _wrap_qdrant_error(exc) from exc
    size = _configured_vector_size(info)
    if size is not None and size != config.EMBEDDING_DIM:
        raise errors.ResearchIndexNotReadyError("indexed vector dimensions do not match the embedding configuration; reindex required")
    payload_schema = getattr(info, "payload_schema", None) or {}
    entry = payload_schema.get(GROUPING_FIELD) if isinstance(payload_schema, dict) else None
    if entry is None or _payload_index_data_type(entry) != str(PayloadSchemaType.KEYWORD):
        raise errors.ResearchIndexNotReadyError(
            f"collection '{collection}' has no keyword payload index on '{GROUPING_FIELD}'; "
            "run `library-query index` to prepare it"
        )


def ensure_grouping_index(collection_name: str | None = None) -> bool:
    """Idempotently create the keyword source-grouping payload index.

    Runs during `index`, including otherwise-idle passes. Returns True when the
    index was created, False when it already existed with the compatible type;
    an incompatible existing index is an error, never silently reused.
    """
    coll = collection_name or config.COLLECTION
    try:
        _validation_call()
        info = qdrant().get_collection(collection_name=coll)
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise _wrap_qdrant_error(exc) from exc
    payload_schema = getattr(info, "payload_schema", None) or {}
    entry = payload_schema.get(GROUPING_FIELD) if isinstance(payload_schema, dict) else None
    if entry is not None:
        if _payload_index_data_type(entry) != str(PayloadSchemaType.KEYWORD):
            raise errors.ResearchIndexNotReadyError(
                f"payload index on '{GROUPING_FIELD}' is {_payload_index_data_type(entry)}, not keyword"
            )
        return False
    try:
        _confirm_write(qdrant().create_payload_index(
            collection_name=coll,
            field_name=GROUPING_FIELD,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        ))
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise _wrap_qdrant_error(exc) from exc
    return True


def scan_identity_payloads(collection_name: str | None = None, include_text: bool = True) -> dict[int, dict]:
    """Fetch the research-relevant payload facts of every point, keyed by point id.

    `include_text=False` skips transferring each chunk's full text (about 7x
    faster on a 54k-point corpus); `text_present` is then omitted from the facts
    and callers must verify text separately with `count_points_without_text`.

    Identity-only pages contain small scalar fields, so use larger pages to
    avoid hundreds of round trips on a book-sized corpus. Full-text scans keep
    their smaller pages to bound response memory.

    One paginated scroll per call. A service failure propagates as
    qdrant_unreachable: inspection is never reported as a successful empty scan,
    which would masquerade a ready corpus.
    """
    coll = collection_name or config.COLLECTION
    facts: dict[int, dict] = {}
    next_page = None
    seen_offsets: set = set()
    while True:
        try:
            _validation_call()
            with _validation_client() as client:
                res = client.scroll(
                    collection_name=coll,
                    limit=250 if include_text else 10000,
                    offset=next_page,
                    with_payload=_IDENTITY_PAYLOAD_FIELDS if include_text else _IDENTITY_FIELDS_NO_TEXT,
                    with_vectors=False,
                    **_validation_timeout(),
                )
        except errors.LibraryQueryError:
            raise
        except Exception as exc:
            raise _wrap_qdrant_error(exc) from exc
        points, next_page = res[0], res[1]
        for point in points:
            payload = point.payload or {}
            facts[point.id] = {
                "document_id": payload.get(GROUPING_FIELD),
                "chunk_index": payload.get("chunk_index"),
                "doc": payload.get("doc"),
            }
            if include_text:
                facts[point.id]["text_present"] = isinstance(payload.get("text"), str) and bool(payload.get("text"))
        if next_page is None:
            break
        if next_page in seen_offsets:
            raise errors.ResearchIndexNotReadyError("index scan repeated a pagination cursor; retry explicit indexing")
        seen_offsets.add(next_page)
    return facts


def count_points_without_text(collection_name: str | None = None) -> int:
    """Count points whose stored passage text is missing, null, or empty (server-side)."""
    from qdrant_client import models

    coll = collection_name or config.COLLECTION
    try:
        _validation_call()
        with _validation_client() as client:
            return client.count(
                collection_name=coll,
                count_filter=models.Filter(should=[
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="text")),
                    models.IsNullCondition(is_null=models.PayloadField(key="text")),
                    models.FieldCondition(key="text", match=models.MatchValue(value="")),
                ]),
                exact=True,
                **_validation_timeout(),
            ).count
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise _wrap_qdrant_error(exc) from exc


def invalid_identity_reason(facts: dict) -> str | None:
    """Why this point cannot serve research evidence, or None when it is valid.

    A valid identity is a non-empty scalar document_id, a non-negative integer
    chunk index (booleans are rejected, not coerced), and stored passage text.
    """
    document_id = facts.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        return "document_id"
    chunk_index = facts.get("chunk_index")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        return "chunk_index"
    if not facts.get("text_present"):
        return "text"
    return None


def check_research_ready(collection_name: str | None = None) -> int:
    """Verify the corpus can serve grouped research; return the point count.

    A missing collection, a missing/incompatible grouping payload index, or any
    point with an invalid identity raises research_index_not_ready (exit 3) with
    the remediation action. Qdrant transport failures raise qdrant_unreachable
    (exit 4) — never a zero count, never an empty success. An initialized empty
    collection is ready and returns 0.
    """
    coll = collection_name or config.COLLECTION
    try:
        _validation_call()
        with _validation_client() as client:
            exists = client.collection_exists(coll)
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise _wrap_qdrant_error(exc) from exc
    if not exists:
        raise errors.ResearchIndexNotReadyError(
            f"collection '{coll}' does not exist; run `library-query index` to build it"
        )
    _require_grouping_index(coll)
    # Text presence is checked server-side so the scan never ships every chunk's text.
    facts = scan_identity_payloads(coll, include_text=False)
    invalid = sum(
        1 for point_facts in facts.values()
        if invalid_identity_reason({**point_facts, "text_present": True}) is not None
    ) + count_points_without_text(coll)
    if invalid:
        raise errors.ResearchIndexNotReadyError(
            f"{invalid} of {len(facts)} points in '{coll}' lack a valid document_id/chunk_index/text "
            "payload; run `library-query index` to repair or reindex them"
        )
    return len(facts)


def audit_research_text(collection_name: str | None = None) -> None:
    """Full write-side audit also rejects non-string and empty passage payloads."""
    facts = scan_identity_payloads(collection_name, include_text=True)
    if any(invalid_identity_reason(f) is not None for f in facts.values()):
        raise errors.ResearchIndexNotReadyError(
            "indexed identities or passage text are invalid; run `library-query index`")


def query_points_grouped(
    vector: list[float],
    *,
    groups: int,
    group_size: int,
    query_filter=None,
    score_threshold: float | None = None,
    timeout: int | None = None,
    collection_name: str | None = None,
):
    """One grouped source-diversity search over the shared Qdrant client.

    Groups by document identity so repeated chunks from one document cannot
    consume the candidate window. A failed required probe fails the request:
    transport errors surface as qdrant_unreachable, never as empty groups.
    """
    coll = collection_name or config.COLLECTION
    try:
        with _validation_client() as client:
            result = client.query_points_groups(
                collection_name=coll,
                query=vector,
                group_by=GROUPING_FIELD,
                limit=groups,
                group_size=group_size,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
                timeout=timeout,
            )
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise _wrap_qdrant_error(exc) from exc
    return list(getattr(result, "groups", None) or [])
