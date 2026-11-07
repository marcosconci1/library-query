"""Audited managed research readiness; collection metadata is not a content hash."""
from contextlib import closing, contextmanager
from dataclasses import dataclass
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PayloadSchemaType

from . import config, errors, revisions, store

AUDIT_VERSION = 1
METADATA_KEY = "library_query_readiness"
REQUIRED_INDEXES = ("document_id", "scope", "tags")
IDENTITY_FIELDS = ("corpus_id", "incarnation", "revision", "embedding_fingerprint",
                   "representation_fingerprint")


def not_ready(reason="managed_validation_missing"):
    return errors.ResearchIndexNotReadyError(
        f"{reason}: managed research validation is not ready; "
        "run `library-query readiness --repair` for missing markers or `library-query index` to reconcile")


def collection_info(collection, deadline=None):
    """One live read; the SDK exposes collection timeouts only on the client."""
    if deadline is not None:
        deadline.check()
    try:
        store._validation_call()
        if deadline is None:
            return store.qdrant().get_collection(collection_name=collection)
        with closing(QdrantClient(url=config.QDRANT_URL, timeout=deadline.remaining(),
                                  check_compatibility=False)) as client:
            info = client.get_collection(collection_name=collection)
        deadline.check()
        return info
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            raise not_ready("collection_missing") from None
        raise store._wrap_qdrant_error(exc) from exc
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise store._wrap_qdrant_error(exc) from exc


def remote_validation(info):
    metadata = getattr(getattr(info, "config", None), "metadata", None)
    return metadata.get(METADATA_KEY) if isinstance(metadata, dict) else None


def valid_local(record, ctx):
    verified = revisions.validate_stable_record(record, ctx)
    proof = record.get("research_validation") if isinstance(record, dict) else None
    if verified is None or not isinstance(proof, dict):
        return None
    if (type(proof.get("audit_version")) is not int or proof["audit_version"] != AUDIT_VERSION
            or type(proof.get("revision")) is not int
            or proof.get("state") != "ready"
            or not isinstance(proof.get("generation"), str) or not 1 <= len(proof["generation"]) <= 64
            or any(proof.get(k) != record.get(k) for k in IDENTITY_FIELDS)):
        return None
    return verified


def verify_info(info, proof):
    if remote_validation(info) != proof:
        raise not_ready("remote_validation_missing" if remote_validation(info) is None else "remote_validation_mismatch")
    if store._configured_vector_size(info) != config.EMBEDDING_DIM:
        raise not_ready("vector_configuration_incompatible")
    schema = getattr(info, "payload_schema", None) or {}
    for key in REQUIRED_INDEXES:
        if key not in schema or store._payload_index_data_type(schema[key]) != str(PayloadSchemaType.KEYWORD):
            raise not_ready(f"payload_index_incompatible:{key}")


def update_remote(collection, proof):
    """PATCH only the owned metadata field; preserve other applications' keys."""
    try:
        confirmed = store.qdrant().update_collection(
            collection_name=collection, metadata={METADATA_KEY: proof})
        if confirmed is not True:
            raise errors.IndexRevisionUnavailableError("collection validation publication was not confirmed")
    except errors.LibraryQueryError:
        raise
    except Exception as exc:
        raise store._wrap_qdrant_error(exc) from exc


def invalidate(session, *, recreated):
    session.mark_dirty(recreated=recreated)
    if not recreated:
        update_remote(session.context.collection, {"state": "dirty"})


def publish(session):
    """Called only after a full successful indexing audit under the writer guard."""
    record = session.pending_stable()
    store.audit_research_text(session.context.collection)
    proof = {k: record[k] for k in IDENTITY_FIELDS}
    proof.update(audit_version=AUDIT_VERSION, generation=str(uuid.uuid4()), state="ready")
    update_remote(session.context.collection, proof)
    info = collection_info(session.context.collection)
    verify_info(info, proof)
    return session.publish_stable(validation=proof)


def repair_markers(collection):
    """Explicit audit-only recovery; never infer fingerprints or rebuild documents."""
    ctx = revisions.managed_context(collection)
    # A scheduled recovery must not queue behind or race an active indexer.
    with revisions.writer_session(ctx, blocking=False) as session:
        record = revisions.read_record(ctx)
        if not isinstance(record, dict) or revisions.validate_stable_record(record, ctx) is None:
            raise not_ready("local_revision_missing_dirty_or_incompatible")
        info = collection_info(collection)
        if valid_local(record, ctx) is not None:
            try:
                verify_info(info, record["research_validation"])
                return record, False  # Another maintainer already repaired it.
            except errors.ResearchIndexNotReadyError:
                pass
        remote = remote_validation(info)
        if (remote is not None and valid_local(record, ctx) is not None
                and remote != record["research_validation"]):
            raise not_ready("remote_validation_conflict")
        # A different generation/restore/schema requires deliberate reconciliation.
        if remote is not None and (not isinstance(remote, dict)
                or any(remote.get(k) != record.get(k) for k in IDENTITY_FIELDS)
                or remote.get("audit_version") != AUDIT_VERSION or remote.get("state") != "ready"):
            raise not_ready("remote_validation_conflict")
        # Check configuration without requiring an existing validation marker.
        verify_info(info, remote)
        invalidate(session, recreated=False)
        record = publish(session)  # Full passage/identity audit + remote readback.
        return record, True


def ensure_filter_indexes():
    info = collection_info(config.COLLECTION)
    schema = getattr(info, "payload_schema", None) or {}
    for key in REQUIRED_INDEXES[1:]:
        if key in schema:
            if store._payload_index_data_type(schema[key]) != str(PayloadSchemaType.KEYWORD):
                raise not_ready()
            continue
        store._confirm_write(store.qdrant().create_payload_index(
            collection_name=config.COLLECTION, field_name=key,
            field_schema=PayloadSchemaType.KEYWORD, wait=True))


@dataclass
class Validation:
    revision: revisions.VerifiedRevision
    elapsed_ms: int


@contextmanager
def reader(collection, deadline):
    """Mandatory guard independent of optional caching; no writes or repair."""
    ctx = revisions.managed_context(collection)
    record = revisions.read_record(ctx)
    if record is None:
        if ctx.record_path.exists() or ctx.guard_path.exists():
            raise not_ready("local_revision_missing")
        yield None  # Explicit legacy compatibility: the caller performs a bounded full audit.
        return
    started = time.monotonic()
    # Guard admission is mandatory service work, not part of the cache budget.
    class Budget:
        def exhausted(self):
            deadline.check()
            return False
        def begin(self):
            pass
        def end(self):
            pass
        def remaining_ms(self):
            return min(100.0, max(0.0, (started + 0.1 - time.monotonic()) * 1000),
                       max(0.0, deadline.remaining() * 1000))
    guard = revisions.try_acquire_reader_guard(ctx, Budget())
    if guard is None:
        deadline.check()
        raise not_ready(revisions.reader_bypass_reason(ctx))
    try:
        record = revisions.read_record(ctx)
        verified = valid_local(record, ctx)
        if verified is None:
            if record is None:
                reason = "local_revision_missing"
            elif record.get("state") != "stable":
                reason = "local_revision_dirty"
            elif revisions.validate_stable_record(record, ctx) is None:
                reason = "local_revision_incompatible"
            else:
                reason = "local_validation_missing_or_incompatible"
            raise not_ready(reason)
        info = collection_info(collection, deadline)
        verify_info(info, record["research_validation"])
        deadline.check()
        yield Validation(verified, round((time.monotonic() - started) * 1000))
    finally:
        guard.release()
