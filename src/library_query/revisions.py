"""Managed indexed-corpus revisions and the reader/writer guard protocol.

Explicit indexing publishes one ``index-<context-digest>-revision.json`` record
beside the configured manifest for each credential-free endpoint/collection
context, plus a matching ``index-<context-digest>-guard.lock``. Writers take an
exclusive OS-level guard, durably mark the record dirty, mutate the corpus,
verify completion, then atomically publish a stable next revision. Readers
admitting cached candidates take the shared guard without an unbounded wait,
verify a stable compatible revision, and hold the guard through assembly and
staged publication.

Nothing in this module is created or repaired by query/research/browse/describe
or cache inspection: initial preparation happens only during explicit indexing.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from . import config, errors

try:  # POSIX guard primitive; absence is reported, never faked.
    import fcntl

    HAVE_FCNTL = True
except ImportError:  # pragma: no cover - platform without fcntl
    fcntl = None
    HAVE_FCNTL = False

SCHEMA_VERSION = 1

#: The operator-controlled deployment token identifying the embedding model
#: revision behind a possibly-stable alias. Advance it when replacing a model
#: behind a stable identifier; it does not prove a remote alias is immutable.
EMBEDDING_REVISION_ENV = "LIBRARY_QUERY_EMBEDDING_REVISION"
DEFAULT_EMBEDDING_REVISION = "1"
_MAX_EMBEDDING_REVISION_CHARS = 64

#: How long a reader polls for the shared guard before giving up (bounded by
#: the caller's remaining admission budget, whichever is smaller).
_READER_GUARD_POLL_SECONDS = 0.005

RECORD_SUFFIX = "-revision.json"
GUARD_SUFFIX = "-guard.lock"


def canonical_json(obj) -> str:
    """Stable JSON encoding for digests: sorted keys, tight separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(obj) -> str:
    """SHA-256 of the canonical JSON encoding of a key/fingerprint object."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def credential_free_url(url: str) -> str:
    """Return `url` with any userinfo/query removed so credentials never digest.

    The loopback module configuration carries no credentials today, but a
    deployed endpoint URL can embed them; fingerprints and keys must be
    credential-free by construction, not by assumption.
    """
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
        hostname = parts.hostname or ""
        port = parts.port
    except ValueError:
        raise errors.CacheUnsafeError("invalid endpoint URL for cache identity") from None
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and (parts.scheme, port) not in (("http", 80), ("https", 443)):
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme.lower(), netloc.lower(), parts.path.rstrip("/"), "", ""))


def operator_embedding_revision() -> str:
    """The validated operator-controlled embedding-model revision token."""
    raw = os.environ.get(EMBEDDING_REVISION_ENV, DEFAULT_EMBEDDING_REVISION).strip()
    if (
        not raw
        or len(raw) > _MAX_EMBEDDING_REVISION_CHARS
        or any(ord(ch) < 32 for ch in raw)
    ):
        raise errors.CacheUnsafeError(
            f"{EMBEDDING_REVISION_ENV} must be a non-empty string of at most "
            f"{_MAX_EMBEDDING_REVISION_CHARS} characters"
        )
    return raw


def embedding_identity() -> dict:
    """Credential-free canonical embedding configuration (the key identity)."""
    return {
        "endpoint": credential_free_url(config.EMBEDDING_BASE_URL),
        "model": config.EMBEDDING_MODEL,
        "dimensions": config.EMBEDDING_DIM,
        "model_revision": operator_embedding_revision(),
    }


def representation_identity() -> dict:
    """Canonical passage-representation configuration for the indexed corpus."""
    from . import extract

    return {
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "extensions": sorted(config.SUPPORTED_EXTENSIONS),
        "extractor_version": extract.EXTRACTOR_VERSION,
    }


def embedding_fingerprint() -> str:
    return canonical_digest(embedding_identity())


def representation_fingerprint() -> str:
    return canonical_digest(representation_identity())


@dataclass(frozen=True)
class ManagedContext:
    """One managed endpoint/collection context and its record/guard locations."""

    context_digest: str
    corpus_id: str
    collection: str
    embedding_fingerprint: str
    representation_fingerprint: str
    record_path: Path
    guard_path: Path


def managed_context(collection: str | None = None) -> ManagedContext:
    """Resolve the managed context for a collection without touching storage."""
    coll = collection or config.COLLECTION
    embedding_fp = embedding_fingerprint()
    representation_fp = representation_fingerprint()
    context_obj = {
        "schema": SCHEMA_VERSION,
        "collection": coll,
        "qdrant_endpoint": credential_free_url(config.QDRANT_URL),
    }
    digest = canonical_digest(context_obj)
    try:
        library_root = str(config.LIBRARY_ROOT.resolve())
    except OSError:
        library_root = str(config.LIBRARY_ROOT)
    # The corpus id is a digest of identity, never persisted request/path text:
    # it names "this managed library collection" without exposing the host path.
    corpus_id = canonical_digest({"library_root": library_root, "collection": coll})
    state_dir = Path(config.LEDGER_PATH).parent
    return ManagedContext(
        context_digest=digest,
        corpus_id=corpus_id,
        collection=coll,
        embedding_fingerprint=embedding_fp,
        representation_fingerprint=representation_fp,
        record_path=state_dir / f"index-{digest}{RECORD_SUFFIX}",
        guard_path=state_dir / f"index-{digest}{GUARD_SUFFIX}",
    )


@dataclass(frozen=True)
class VerifiedRevision:
    """A reader-verified stable indexed revision (provenance for one response)."""

    corpus_id: str
    incarnation: str
    revision: int
    representation_fingerprint: str
    embedding_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "corpus_id": self.corpus_id,
            "incarnation": self.incarnation,
            "revision": self.revision,
            "representation_fingerprint": self.representation_fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
        }

    def corpus_context_digest(self) -> str:
        """Digest naming the current corpus incarnation for scoped cache rows.

        Scoped to the corpus incarnation (not the revision) so explicit scoped
        clear retires every cached generation of the current managed corpus.
        """
        return canonical_digest(
            {"corpus_id": self.corpus_id, "incarnation": self.incarnation}
        )


def new_incarnation() -> str:
    return str(uuid.uuid4())


# --- durable record publication -----------------------------------------------------


def _sync_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _sync_dir(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_record(ctx: ManagedContext, record: dict) -> None:
    """Durably replace the context record: sync tmp, atomic replace, sync dir.

    The guard file is never replaced or unlinked here. A failure before the
    replace leaves the previous record (or absence) intact; a failure after the
    replace but before the directory sync still leaves a complete record file.
    """
    try:
        ctx.record_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ctx.record_path.with_name(f".{ctx.record_path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, indent=2, sort_keys=True))
                _sync_file(handle)
            os.replace(tmp, ctx.record_path)
            _sync_dir(ctx.record_path.parent)
        finally:
            tmp.unlink(missing_ok=True)
    except errors.LibraryQueryError:
        raise
    except OSError as exc:
        raise errors.IndexRevisionUnavailableError(
            f"could not durably publish the index revision record: {exc.strerror or type(exc).__name__}"
        ) from exc


def read_record(ctx: ManagedContext) -> dict | None:
    """Read the local context record without creating or repairing anything.

    Returns the parsed record, or None when absent. A corrupt record is
    reported as dirty-shaped unknown state by callers, never trusted as stable.
    """
    try:
        text = ctx.record_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        record = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"state": "dirty", "corrupt": True}
    return record if isinstance(record, dict) else {"state": "dirty", "corrupt": True}


def validate_stable_record(record: dict, ctx: ManagedContext) -> VerifiedRevision | None:
    """A stable, schema-compatible record as verified provenance, else None."""
    if not isinstance(record, dict) or record.get("state") != "stable":
        return None
    if (record.get("context_digest") != ctx.context_digest
            or record.get("corpus_id") != ctx.corpus_id
            or record.get("collection") != ctx.collection):
        return None
    if record.get("schema_version") != SCHEMA_VERSION:
        return None
    if record.get("embedding_fingerprint") != ctx.embedding_fingerprint:
        return None
    if record.get("representation_fingerprint") != ctx.representation_fingerprint:
        return None
    incarnation = record.get("incarnation")
    revision = record.get("revision")
    corpus_id = record.get("corpus_id")
    if not isinstance(incarnation, str) or not incarnation:
        return None
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        return None
    if not isinstance(corpus_id, str) or not corpus_id:
        return None
    return VerifiedRevision(
        corpus_id=corpus_id,
        incarnation=incarnation,
        revision=revision,
        representation_fingerprint=ctx.representation_fingerprint,
        embedding_fingerprint=ctx.embedding_fingerprint,
    )


# --- guard protocol -----------------------------------------------------------------


class _Guard:
    """One held flock on a context guard file; released on close."""

    def __init__(self, ctx: ManagedContext, fd: int):
        self._ctx = ctx
        self._fd = fd

    @property
    def path(self) -> Path:
        return self._ctx.guard_path

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if HAVE_FCNTL:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __del__(self):  # defensive: a leaked handle must never hold the corpus hostage
        with contextlib.suppress(OSError, AttributeError):
            self.release()


def _open_guard_fd(ctx: ManagedContext, flags: int, mode: int = 0o600) -> int:
    if ctx.guard_path.is_symlink():
        raise errors.IndexRevisionUnavailableError("index guard path is a symlink; refusing to follow it")
    return os.open(ctx.guard_path, flags, mode)


def acquire_writer_guard(ctx: ManagedContext, *, blocking: bool = True) -> _Guard:
    """Blocking exclusive guard for an indexing writer; aborts indexing on failure."""
    if not HAVE_FCNTL:
        raise errors.IndexRevisionUnavailableError(
            "no reliable OS-level writer guard is available on this platform; "
            "candidate reuse is not claimable here"
        )
    try:
        ctx.guard_path.parent.mkdir(parents=True, exist_ok=True)
        fd = _open_guard_fd(ctx, os.O_RDWR | os.O_CREAT, 0o600)
    except errors.LibraryQueryError:
        raise
    except OSError as exc:
        raise errors.IndexRevisionUnavailableError(
            f"could not open the index writer guard: {exc.strerror or type(exc).__name__}"
        ) from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError as exc:
        os.close(fd)
        raise errors.ResearchIndexNotReadyError("index_busy: another reader/writer owns the guard; retry maintenance later") from exc
    except OSError as exc:
        os.close(fd)
        raise errors.IndexRevisionUnavailableError(
            f"could not acquire the index writer guard: {exc.strerror or type(exc).__name__}"
        ) from exc
    return _Guard(ctx, fd)


def try_acquire_reader_guard(ctx: ManagedContext, budget) -> _Guard | None:
    """Shared guard for a cache-enabled reader, bounded by the admission budget.

    Returns None (with no guard held) when the guard file is missing, the
    platform lacks flock, the budget is exhausted, or the wait elapses: callers
    map that to the appropriate bounded bypass category. `budget` is a
    cache.AdmissionBudget-like object with begin()/end()/remaining_ms().
    """
    if not HAVE_FCNTL or budget.exhausted():
        return None
    if not ctx.guard_path.exists():
        return None
    try:
        fd = _open_guard_fd(ctx, os.O_RDONLY)
    except (errors.LibraryQueryError, OSError):
        return None
    acquired = False
    budget.begin()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                acquired = True
                return _Guard(ctx, fd)
            except OSError:
                pass
            if budget.remaining_ms() <= 0:
                return None
            time.sleep(min(_READER_GUARD_POLL_SECONDS, max(0.0, budget.remaining_ms()) / 1000.0))
    finally:
        budget.end()
        if not acquired:
            os.close(fd)


def reader_bypass_reason(ctx: ManagedContext) -> str:
    """Why a reader guard cannot even be attempted (before any wait)."""
    if not HAVE_FCNTL:
        return "guard_unavailable"
    if not ctx.guard_path.exists():
        return "unmanaged_index"
    return "index_busy"


# --- writer session used by explicit indexing ---------------------------------------


class WriterSession:
    """Exclusive guard + durable dirty/stable publication for one index pass.

    Corpus mutations may happen only between mark_dirty and publish_stable;
    a raise in between (or a process crash) leaves the durable dirty record
    behind, and the next explicit index pass recovers it.
    """

    def __init__(self, ctx: ManagedContext, guard: _Guard):
        self._ctx = ctx
        self._guard = guard
        self._pending_revision: int | None = None
        self._incarnation: str | None = None

    @property
    def context(self) -> ManagedContext:
        return self._ctx

    def planned_mutation(self, *, recreated: bool) -> tuple[str, int]:
        """The incarnation and next revision a dirty publication would use."""
        record = read_record(self._ctx)
        if (recreated or not isinstance(record, dict) or record.get("corrupt")
                or record.get("embedding_fingerprint") != self._ctx.embedding_fingerprint
                or record.get("representation_fingerprint") != self._ctx.representation_fingerprint):
            return new_incarnation(), 1
        incarnation = record.get("incarnation")
        if not isinstance(incarnation, str) or not incarnation:
            return new_incarnation(), 1
        revision = record.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            revision = 0
        # A dirty record is a pending mutation of the same revision: retry it
        # rather than skipping numbers or trusting its state.
        return incarnation, revision + (0 if record.get("state") == "dirty" else 1)

    def mark_dirty(self, *, recreated: bool) -> None:
        """Durably publish dirty state before the first corpus mutation."""
        incarnation, revision = self.planned_mutation(recreated=recreated)
        record = self._record(incarnation, revision, "dirty")
        publish_record(self._ctx, record)
        self._pending_revision = revision
        self._incarnation = incarnation

    def pending_stable(self) -> dict:
        """Describe the pending stable revision without publishing it."""
        if self._pending_revision is None or self._incarnation is None:
            raise errors.IndexRevisionUnavailableError(
                "cannot publish a stable revision before mark_dirty"
            )
        record = self._record(self._incarnation, self._pending_revision, "stable")
        return record

    def publish_stable(self, *, validation: dict | None = None) -> dict:
        """Publish stable state only after the caller confirms all writes/audits."""
        record = self.pending_stable()
        if validation is not None:
            record["research_validation"] = validation
        publish_record(self._ctx, record)
        return record

    def current_stable(self) -> dict | None:
        record = read_record(self._ctx)
        if isinstance(record, dict) and record.get("state") == "stable" and not record.get("corrupt"):
            return record
        return None

    def _record(self, incarnation: str, revision: int, state: str) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "context_digest": self._ctx.context_digest,
            "corpus_id": self._ctx.corpus_id,
            "collection": self._ctx.collection,
            "incarnation": incarnation,
            "revision": revision,
            "state": state,
            "embedding_fingerprint": self._ctx.embedding_fingerprint,
            "representation_fingerprint": self._ctx.representation_fingerprint,
            "published_at": datetime.now(UTC).isoformat(),
        }

    def close(self) -> None:
        if self._guard is not None:
            guard, self._guard = self._guard, None
            guard.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):  # defensive: see _Guard.__del__
        with contextlib.suppress(OSError, AttributeError):
            self.close()


def writer_session(ctx: ManagedContext, *, blocking: bool = True) -> WriterSession:
    """Acquire the blocking writer guard; indexing aborts if this fails."""
    return WriterSession(ctx, acquire_writer_guard(ctx, blocking=blocking))
