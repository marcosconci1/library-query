"""Opt-in exact retrieval cache: configuration, keys, budget, storage, session.

Two generic tiers, never whole-response replay:

- *embedding*: exact query/probe text vectors, keyed by digested probe text,
  credential-free endpoint identity, model, operator model revision, dimensions,
  input-transform version, cache schema, and caller namespace.
- *evidence*: raw per-probe retrieval candidates (flat locator searches and
  grouped research searches have distinct method fingerprints), additionally
  bound to a reader-verified stable indexed revision and the complete retrieval
  contract (normalized hard filters and candidate-window parameters).

Filtering, fusion, coverage, excerpts, question associations, and budgets are
rebuilt from cached raw candidates for every current request. Storage is a
private rollback-journal SQLite database under a validated cache directory;
`off` mode never touches any of it.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import os
import sqlite3
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Self

from . import config, errors, expansion, revisions, structure

CACHE_SCHEMA_VERSION = 1
CACHE_FILENAME = "retrieval-cache.sqlite3"

#: Explicit cache operating modes; the unset default is `off`.
CACHE_MODES = ("off", "use", "refresh")

#: Bounded, machine-readable bypass categories.
BYPASS_REASONS = (
    "cache_budget_exhausted",
    "cache_unavailable",
    "cache_unsafe",
    "disabled",
    "entry_invalid",
    "entry_oversized",
    "guard_unavailable",
    "index_busy",
    "revision_dirty",
    "revision_incompatible",
    "unmanaged_index",
)

#: Effective default bounds (documented and env-configurable with validation).
DEFAULT_MAX_ENTRIES = 4096
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ENTRY_BYTES = 512 * 1024
DEFAULT_EMBEDDING_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_EVIDENCE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_EMPTY_TTL_SECONDS = 5 * 60
DEFAULT_NAMESPACE = "local"
_MAX_NAMESPACE_CHARS = 128

#: Cumulative optional-cache admission/wait budget shared by both tiers.
ADMISSION_BUDGET_MS = 100.0

#: Retrieval method fingerprints: flat locator vs grouped research never share
#: candidate entries; embeddings are shared across methods.
METHOD_FLAT = "flat"
METHOD_GROUPED = "grouped"
METHOD_VERSION = 1

#: Rows with creation timestamps further than this into the future are rejected
#: rather than turning a backward clock jump into indefinite freshness.
_CLOCK_SKEW_SECONDS = 0

#: Housekeeping is performed in bounded batches inside short transactions.
_EVICT_BATCH = 64

#: Clock sources as module attributes so tests can substitute controllable clocks.
#: The admission budget uses a monotonic clock; TTLs use wall time. Provider and
#: network latency is never charged to the budget — only cache operations are.
MONO_CLOCK = time.monotonic
WALL_CLOCK = time.time

_ENV_MODE = "LIBRARY_QUERY_CACHE_MODE"
_ENV_NAMESPACE = "LIBRARY_QUERY_CACHE_NAMESPACE"
_ENV_DIR = "LIBRARY_QUERY_CACHE_DIR"
_ENV_MAX_ENTRIES = "LIBRARY_QUERY_CACHE_MAX_ENTRIES"
_ENV_MAX_BYTES = "LIBRARY_QUERY_CACHE_MAX_BYTES"
_ENV_MAX_ENTRY_BYTES = "LIBRARY_QUERY_CACHE_MAX_ENTRY_BYTES"
_ENV_EMBEDDING_TTL = "LIBRARY_QUERY_CACHE_EMBEDDING_TTL"
_ENV_EVIDENCE_TTL = "LIBRARY_QUERY_CACHE_EVIDENCE_TTL"
_ENV_EMPTY_TTL = "LIBRARY_QUERY_CACHE_EMPTY_TTL"


# --- configuration ------------------------------------------------------------------


@dataclass(frozen=True)
class CacheSettings:
    mode: str
    namespace: str
    cache_dir: Path
    max_entries: int
    max_bytes: int
    max_entry_bytes: int
    embedding_ttl: int
    evidence_ttl: int
    empty_ttl: int

    def __post_init__(self):
        if self.mode not in CACHE_MODES:
            raise errors.CacheUnsafeError("invalid cache mode")
        _validate_namespace(self.namespace)
        for name in ("max_entries", "max_bytes", "max_entry_bytes", "embedding_ttl", "evidence_ttl", "empty_ttl"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise errors.CacheUnsafeError(f"{name} must be a positive integer")

    def limits_dict(self) -> dict:
        return {
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "max_entry_bytes": self.max_entry_bytes,
            "embedding_ttl_seconds": self.embedding_ttl,
            "evidence_ttl_seconds": self.evidence_ttl,
            "empty_ttl_seconds": self.empty_ttl,
            "admission_budget_ms": int(ADMISSION_BUDGET_MS),
        }


def _env(environ, name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _positive_int(environ, name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw.isdigit():
        raise errors.CacheUnsafeError(f"{name} must be a positive integer")
    try:
        value = int(raw, 10)
    except ValueError:
        raise errors.CacheUnsafeError(f"{name} must be a positive integer") from None
    if value <= 0:
        raise errors.CacheUnsafeError(f"{name} must be a positive integer")
    return value


def _validate_namespace(raw: str) -> str:
    namespace = raw.strip()
    if (
        not namespace
        or len(namespace) > _MAX_NAMESPACE_CHARS
        or any(ord(ch) < 32 for ch in namespace)
    ):
        raise errors.CacheUnsafeError(
            f"{_ENV_NAMESPACE} must be a non-empty string of at most "
            f"{_MAX_NAMESPACE_CHARS} characters without control characters"
        )
    return namespace


def resolve_cache_dir(environ) -> Path:
    """Resolve the configured cache directory without creating anything."""
    override = _env(environ, _ENV_DIR)
    if override is not None:
        return Path(override).expanduser()
    xdg_cache = _env(environ, "XDG_CACHE_HOME")
    if xdg_cache is not None:
        return Path(xdg_cache).expanduser() / "library-query"
    return Path.home() / ".cache" / "library-query"


def resolve_settings(mode_override: str | None = None, environ=None, *, strict: bool = True) -> CacheSettings:
    """Validated effective cache settings; the flag wins over the environment.

    With `strict=False` (retrieval sessions), a resolved `off` mode returns
    immediately: `off` never reads bounds, so unrelated junk configuration
    cannot break an uncached request. Maintenance commands use `strict=True`.
    """
    environ = os.environ if environ is None else environ
    mode = mode_override if mode_override is not None else _env(environ, _ENV_MODE)
    mode = mode or "off"
    if mode not in CACHE_MODES:
        raise errors.CacheUnsafeError(
            f"{_ENV_MODE} must be one of {', '.join(CACHE_MODES)}"
        )
    if mode == "off" and not strict:
        return CacheSettings(
            mode="off",
            namespace=DEFAULT_NAMESPACE,
            cache_dir=resolve_cache_dir(environ),
            max_entries=DEFAULT_MAX_ENTRIES,
            max_bytes=DEFAULT_MAX_BYTES,
            max_entry_bytes=DEFAULT_MAX_ENTRY_BYTES,
            embedding_ttl=DEFAULT_EMBEDDING_TTL_SECONDS,
            evidence_ttl=DEFAULT_EVIDENCE_TTL_SECONDS,
            empty_ttl=DEFAULT_EMPTY_TTL_SECONDS,
        )
    namespace = _validate_namespace(environ.get(_ENV_NAMESPACE, DEFAULT_NAMESPACE))
    return CacheSettings(
        mode=mode,
        namespace=namespace,
        cache_dir=resolve_cache_dir(environ),
        max_entries=_positive_int(environ, _ENV_MAX_ENTRIES, DEFAULT_MAX_ENTRIES),
        max_bytes=_positive_int(environ, _ENV_MAX_BYTES, DEFAULT_MAX_BYTES),
        max_entry_bytes=_positive_int(environ, _ENV_MAX_ENTRY_BYTES, DEFAULT_MAX_ENTRY_BYTES),
        embedding_ttl=_positive_int(environ, _ENV_EMBEDDING_TTL, DEFAULT_EMBEDDING_TTL_SECONDS),
        evidence_ttl=_positive_int(environ, _ENV_EVIDENCE_TTL, DEFAULT_EVIDENCE_TTL_SECONDS),
        empty_ttl=_positive_int(environ, _ENV_EMPTY_TTL, DEFAULT_EMPTY_TTL_SECONDS),
    )


def session_for(mode_override: str | None, collection: str | None = None, clock=None) -> CacheSession | None:
    """Create a session for `use`/`refresh`, or None for `off` (no cache access)."""
    settings = resolve_settings(mode_override, strict=False)
    if settings.mode == "off":
        return None
    return CacheSession(settings, collection or config.COLLECTION, clock=clock)


def off_namespace(environ=None) -> str:
    """The configured namespace label for `off`-mode diagnostics (unvalidated)."""
    environ = os.environ if environ is None else environ
    return _env(environ, _ENV_NAMESPACE) or DEFAULT_NAMESPACE


# --- admission budget -----------------------------------------------------------------


class AdmissionBudget:
    """Cumulative optional-cache admission/wait budget across both tiers.

    Only time spent inside cache operations (between begin()/end()) is charged;
    provider/network latency between operations is not. Remaining lock-wait
    allowances are recomputed from the remainder before each operation and are
    never reset to the full budget per row, statement, or tier.
    """

    def __init__(self, total_ms: float = ADMISSION_BUDGET_MS, clock=None):
        self._total_ms = total_ms
        self._clock = clock or MONO_CLOCK
        self._charged_ms = 0.0
        self._op_start: float | None = None
        self._depth = 0
        self._deadline_remaining = None

    def begin(self) -> None:
        if self._depth == 0:
            self._op_start = self._clock()
        self._depth += 1

    def end(self) -> None:
        self._depth = max(0, self._depth - 1)
        if self._depth == 0 and self._op_start is not None:
            self._charged_ms += (self._clock() - self._op_start) * 1000.0
            self._op_start = None

    def remaining_ms(self) -> float:
        active = (self._clock() - self._op_start) * 1000.0 if self._op_start is not None else 0.0
        remaining = self._total_ms - self._charged_ms - active
        if self._deadline_remaining is not None:
            remaining = min(remaining, self._deadline_remaining() * 1000)
        return remaining

    def exhausted(self) -> bool:
        return self.remaining_ms() <= 0

    def busy_timeout_ms(self) -> int:
        """Whole-millisecond lock wait for the next operation; never refreshed."""
        return max(0, int(self.remaining_ms()))

    def elapsed_ms(self) -> int:
        """Actual cache elapsed time, measured and never clipped to the budget."""
        active = (self._clock() - self._op_start) * 1000 if self._op_start is not None else 0
        return max(0, round(self._charged_ms + active))


# --- exact canonical keys ---------------------------------------------------------------


def probe_digest(text: str) -> str:
    """SHA-256 of the exact probe UTF-8 text; the text itself is never persisted."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_key(settings: CacheSettings, probe_text: str) -> str:
    obj = {
        "v": CACHE_SCHEMA_VERSION,
        "tier": "embedding",
        "namespace": settings.namespace,
        "probe": probe_digest(probe_text),
        "embedding": revisions.embedding_identity(),
        "transform_version": expansion.TRANSFORM_VERSION,
    }
    return revisions.canonical_digest(obj)


def candidate_key(
    settings: CacheSettings,
    *,
    probe_text: str,
    verified: revisions.VerifiedRevision,
    method: str,
    filters: dict,
    window: dict,
) -> str:
    obj = {
        "v": CACHE_SCHEMA_VERSION,
        "tier": "evidence",
        "namespace": settings.namespace,
        "probe_key": embedding_key(settings, probe_text),
        "method": method,
        "method_version": METHOD_VERSION,
        "corpus": {
            "corpus_id": verified.corpus_id,
            "incarnation": verified.incarnation,
            "revision": verified.revision,
        },
        "embedding_fingerprint": verified.embedding_fingerprint,
        "representation_fingerprint": verified.representation_fingerprint,
        "filters": filters,
        "window": window,
    }
    return revisions.canonical_digest(obj)


def flat_filters(scope: str, tags: list[str] | None) -> dict:
    """Canonical flat-locator hard-filter contract (tags normalized, sorted)."""
    normalized = sorted({tag for tag in (structure.normalize_tag(t) for t in (tags or [])) if tag})
    effective_scope = scope if scope in ("inbox", "catalogued", "operations") else "all"
    return {"scope": effective_scope, "tags_all": normalized}


def grouped_filters(scope: str, tags_any, tags_all) -> dict:
    return {
        "scope": scope if scope in ("inbox", "catalogued", "operations") else "all",
        "tags_any": sorted(tags_any or ()),
        "tags_all": sorted(tags_all or ()),
    }


# --- diagnostics ------------------------------------------------------------------------


class CacheDiagnostics:
    """Additive per-invocation cache diagnostics (the top-level `cache` sibling)."""

    def __init__(self, mode: str, namespace: str):
        self.mode = mode
        self.namespace = namespace
        self.index: dict | None = None
        self.embedding_hits = 0
        self.embedding_misses = 0
        self.candidate_hits = 0
        self.candidate_misses = 0
        self.embedding_requests = 0
        self.search_requests = 0
        self.validation_requests = 0
        self.cache_elapsed_ms = 0
        self._bypass: set[str] = set()

    def bypass(self, reason: str | None) -> None:
        if reason:
            self._bypass.add(reason)

    def to_dict(self) -> dict:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "mode": self.mode,
            "namespace": self.namespace,
            "index": self.index,
            "embedding_hits": self.embedding_hits,
            "embedding_misses": self.embedding_misses,
            "candidate_hits": self.candidate_hits,
            "candidate_misses": self.candidate_misses,
            "embedding_request_count": self.embedding_requests,
            "search_request_count": self.search_requests,
            "validation_request_count": self.validation_requests,
            "cache_elapsed_ms": max(0, int(self.cache_elapsed_ms)),
            "bypass_reasons": sorted(self._bypass),
        }


def off_diagnostics(environ=None) -> dict:
    """Zero-counter `cache` block for requests that ran with caching off."""
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "mode": "off",
        "namespace": off_namespace(environ),
        "index": None,
        "embedding_hits": 0,
        "embedding_misses": 0,
        "candidate_hits": 0,
        "candidate_misses": 0,
        "embedding_request_count": 0,
        "search_request_count": 0,
        "validation_request_count": 0,
        "cache_elapsed_ms": 0,
        "bypass_reasons": [],
    }


# --- private SQLite storage ---------------------------------------------------------------


def _path_safety(db_path: Path) -> str | None:
    """Why this cache location is unsafe, or None when safe to use.

    Rejects locations inside the source library, inside any Git worktree, or
    with unsafe ownership/permissions/symlinks. Existing shared parents are
    never chmod'ed; only the owned leaf directory is created (mode 0700).
    """
    try:
        cache_dir = db_path.parent
        resolved_dir = cache_dir.resolve()
        try:
            library_root = config.LIBRARY_ROOT.resolve()
            if resolved_dir == library_root or library_root in resolved_dir.parents:
                return "inside_library"
        except OSError:
            pass
        probe = resolved_dir
        home = Path.home().resolve()
        for _ in range(64):
            # A .git entry at the home root is a personal dotfiles repo, not a
            # project worktree; caches legitimately live under home (XDG dirs).
            git_entry = probe / ".git"
            # Empty .git directories can be stray filesystem entries, not repos.
            is_repo = git_entry.is_file() or (
                git_entry.is_dir() and (git_entry / "HEAD").exists()
            )
            if is_repo and probe != home:
                return "inside_git_worktree"
            if probe.parent == probe:
                break
            probe = probe.parent
        euid = os.geteuid() if hasattr(os, "geteuid") else None
        if any(part.is_symlink() for part in (cache_dir, *cache_dir.parents)):
            return "symlinked_cache_dir"
        if resolved_dir.exists():
            st = resolved_dir.stat()
            if not stat.S_ISDIR(st.st_mode):
                return "not_a_directory"
            if euid is not None and st.st_uid != euid:
                return "wrong_owner"
            if st.st_mode & 0o077:
                return "group_or_world_writable"
        if db_path.is_symlink():
            return "symlinked_database"
        for sidecar in (Path(str(db_path) + suffix) for suffix in ("-journal", "-wal", "-shm")):
            if sidecar.is_symlink():
                return "symlinked_sidecar"
            if sidecar.exists():
                st = sidecar.stat()
                if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077 or (euid is not None and st.st_uid != euid):
                    return "unsafe_sidecar"
        if db_path.exists():
            st = db_path.stat()
            if not stat.S_ISREG(st.st_mode):
                return "not_a_regular_file"
            if euid is not None and st.st_uid != euid:
                return "wrong_owner"
            if st.st_mode & 0o077:
                return "insecure_database_permissions"
    except OSError:
        return "unstatable"
    return None


def _is_readonly_rollback(exc: BaseException) -> bool:
    """True when SQLite refused a read because a hot journal needs rollback."""
    if getattr(exc, "sqlite_errorname", "") == "SQLITE_READONLY_ROLLBACK":
        return True
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        # SQLITE_READONLY_ROLLBACK = SQLITE_READONLY (8) | (3 << 8)
        return code == 776
    message = str(exc).lower()
    return "readonly" in message and "rollback" in message


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    tier TEXT NOT NULL,
    namespace TEXT NOT NULL,
    entry_key TEXT NOT NULL,
    context TEXT NOT NULL,
    row_version INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    accessed_at INTEGER NOT NULL,
    payload BLOB NOT NULL,
    payload_bytes INTEGER NOT NULL,
    PRIMARY KEY (tier, namespace, entry_key)
)
"""


@contextlib.contextmanager
def _transaction(conn: sqlite3.Connection):
    """One short write transaction; rolls back on any failure."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise


class _BudgetConnection(sqlite3.Connection):
    budget = None

    def execute(self, sql, parameters=()):
        if sql == "ROLLBACK":
            self.set_progress_handler(None, 0)
        if self.budget is not None and sql != "ROLLBACK":
            if self.budget.exhausted():
                raise sqlite3.OperationalError("interrupted: cache budget exhausted")
            super().execute(f"PRAGMA busy_timeout={self.budget.busy_timeout_ms()}")
        return super().execute(sql, parameters)


def _cache_operation(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        budget = kwargs.get("budget", args[-1] if args else None)
        if budget.exhausted():
            raise sqlite3.OperationalError("interrupted: cache budget exhausted")
        budget.begin()
        try:
            self.check_safe()
            return method(self, *args, **kwargs)
        finally:
            budget.end()
    return wrapped


class _Storage:
    """Versioned rollback-journal SQLite storage with short transactions."""

    def __init__(self, settings: CacheSettings):
        self.settings = settings
        self.db_path = settings.cache_dir / CACHE_FILENAME
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle
    def check_safe(self) -> None:
        reason = _path_safety(self.db_path)
        if reason is not None:
            raise errors.CacheUnsafeError(f"cache location failed safety checks: {reason}")

    def _connect(self, budget=None) -> sqlite3.Connection:
        if self._conn is None:
            created_dir = not self.settings.cache_dir.exists()
            self.settings.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if created_dir:
                os.chmod(self.settings.cache_dir, 0o700)
            # isolation_level=None: autocommit; explicit _transaction for batches.
            if not self.db_path.exists():
                fd = os.open(self.db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            conn = sqlite3.connect(str(self.db_path), timeout=(budget.busy_timeout_ms() / 1000 if budget else 0), isolation_level=None, factory=_BudgetConnection)
            conn.budget = budget
            if budget is not None:
                conn.set_progress_handler(lambda: int(budget.exhausted()), 1000)
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if version not in (0, CACHE_SCHEMA_VERSION) or (tables and version == 0):
                    raise sqlite3.DatabaseError("unfamiliar cache schema")
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute(_SCHEMA)
                conn.execute(f"PRAGMA user_version={CACHE_SCHEMA_VERSION}")
            except Exception:
                conn.close()
                raise
            self._conn = conn
        if budget is not None:
            self._conn.budget = budget
            self._conn.set_progress_handler(lambda: int(budget.exhausted()), 1000)
        return self._conn

    def connect_readonly(self) -> sqlite3.Connection:
        """Ordinary `mode=ro` connection; never immutable, never read-write."""
        resolved = self.db_path.resolve()
        uri = resolved.as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=0)

    def _apply_busy(self, conn: sqlite3.Connection, budget: AdmissionBudget) -> None:
        conn.execute(f"PRAGMA busy_timeout={budget.busy_timeout_ms()}")

    # -- reads
    @_cache_operation
    def get_payload(self, tier: str, key: str, budget: AdmissionBudget) -> dict | None:
        conn = self._connect(budget)
        self._apply_busy(conn, budget)
        budget.begin()
        try:
            row = conn.execute(
                "SELECT payload, row_version, created_at, expires_at, payload_bytes FROM cache_entries "
                "WHERE tier = ? AND namespace = ? AND entry_key = ? AND length(payload) <= ?",
                (tier, self.settings.namespace, key, self.settings.max_entry_bytes),
            ).fetchone()
            if row is None:
                return None
            raw, row_version, created_at, expires_at, payload_bytes = row
            if not isinstance(raw, bytes) or payload_bytes != len(raw):
                self._delete_keys(conn, [(tier, self.settings.namespace, key)])
                return None
            if not self._timestamps_valid(created_at, expires_at):
                self._delete_keys(conn, [(tier, self.settings.namespace, key)])
                return None
            if row_version != CACHE_SCHEMA_VERSION:
                self._delete_keys(conn, [(tier, self.settings.namespace, key)])
                return None
            if expires_at <= int(WALL_CLOCK()):
                return None  # expired: an ordinary miss, never an error
            try:
                payload = json.loads(bytes(raw).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._delete_keys(conn, [(tier, self.settings.namespace, key)])
                return None
            if not isinstance(payload, dict):
                self._delete_keys(conn, [(tier, self.settings.namespace, key)])
                return None
            # Hits update only LRU access; expiry is never extended.
            conn.execute(
                "UPDATE cache_entries SET accessed_at = ? WHERE tier = ? AND namespace = ? AND entry_key = ?",
                (int(WALL_CLOCK()), tier, self.settings.namespace, key),
            )
            return payload
        finally:
            budget.end()

    def _timestamps_valid(self, created_at, expires_at) -> bool:
        if isinstance(created_at, bool) or isinstance(expires_at, bool):
            return False
        if not isinstance(created_at, int) or not isinstance(expires_at, int):
            return False
        if created_at < 0 or expires_at <= created_at:
            return False
        return not created_at > int(WALL_CLOCK()) + _CLOCK_SKEW_SECONDS

    def _delete_keys(self, conn: sqlite3.Connection, keys: list[tuple]) -> None:
        conn.executemany(
            "DELETE FROM cache_entries WHERE tier = ? AND namespace = ? AND entry_key = ?",
            keys,
        )

    # -- writes
    @_cache_operation
    def put_payload(
        self,
        tier: str,
        key: str,
        context: str,
        payload: dict,
        ttl_seconds: int,
        budget: AdmissionBudget,
    ) -> str | None:
        """Store one validated row; returns a bypass reason or None on success."""
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(data) > min(self.settings.max_entry_bytes, self.settings.max_bytes):
            return "entry_oversized"
        reasons = self.put_payload_batch([(tier, key, context, payload, ttl_seconds)], budget)
        return reasons[0] if reasons else None

    @_cache_operation
    def put_payload_batch(self, rows: list[tuple], budget: AdmissionBudget) -> list[str]:
        """Staged candidate publication: every row in one short transaction."""
        skipped: list[str] = []
        conn = self._connect(budget)
        self._apply_busy(conn, budget)
        budget.begin()
        try:
            with _transaction(conn):
                now = int(WALL_CLOCK())
                for tier, key, context, payload, ttl_seconds in rows:
                    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    if len(data) > min(self.settings.max_entry_bytes, self.settings.max_bytes):
                        skipped.append("entry_oversized")
                        continue
                    conn.execute(
                        "INSERT OR REPLACE INTO cache_entries "
                        "(tier, namespace, entry_key, context, row_version, created_at, expires_at, "
                        " accessed_at, payload, payload_bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (tier, self.settings.namespace, key, context, CACHE_SCHEMA_VERSION,
                         now, now + ttl_seconds, now, data, len(data)),
                    )
                self._evict_within(conn, budget)
        finally:
            budget.end()
        return skipped

    @_cache_operation
    def delete_entry(self, tier: str, key: str, budget: AdmissionBudget) -> None:
        conn = self._connect(budget)
        self._apply_busy(conn, budget)
        budget.begin()
        try:
            self._delete_keys(conn, [(tier, self.settings.namespace, key)])
        finally:
            budget.end()

    # -- bounded global LRU + TTL eviction (deterministic tie-break by key)
    @_cache_operation
    def evict(self, budget: AdmissionBudget) -> None:
        conn = self._connect(budget)
        self._apply_busy(conn, budget)
        budget.begin()
        try:
            with _transaction(conn):
                self._evict_within(conn, budget)
        finally:
            budget.end()

    def _evict_within(self, conn: sqlite3.Connection, budget: AdmissionBudget) -> None:
        def cancel_if_exhausted() -> int:
            return 1 if budget.remaining_ms() <= 0 else 0

        conn.set_progress_handler(cancel_if_exhausted, 1000)
        try:
            now = int(WALL_CLOCK())
            while True:
                expired = conn.execute(
                    "SELECT tier, namespace, entry_key FROM cache_entries WHERE expires_at <= ? "
                    "ORDER BY accessed_at, tier, namespace, entry_key LIMIT ?",
                    (now, _EVICT_BATCH),
                ).fetchall()
                if not expired:
                    break
                self._delete_keys(conn, expired)
                if budget.remaining_ms() <= 0:
                    raise sqlite3.OperationalError("interrupted: cache budget exhausted")
            while True:
                count, total = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0) FROM cache_entries"
                ).fetchone()
                if count <= self.settings.max_entries and total <= self.settings.max_bytes:
                    return
                # Delete only as many oldest rows as needed: a batch delete must
                # never evict live rows beyond the configured pressure.
                need = max(count - self.settings.max_entries, 0)
                if total > self.settings.max_bytes:
                    need = max(need, 1)
                need = max(1, min(need, _EVICT_BATCH))
                oldest = conn.execute(
                    "SELECT tier, namespace, entry_key FROM cache_entries "
                    "ORDER BY accessed_at, tier, namespace, entry_key LIMIT ?",
                    (need,),
                ).fetchall()
                if not oldest:
                    return
                self._delete_keys(conn, oldest)
                if budget.remaining_ms() <= 0:
                    raise sqlite3.OperationalError("interrupted: cache budget exhausted")
        finally:
            conn.set_progress_handler(lambda: int(budget.exhausted()), 1000)

    # -- maintenance
    def scoped_counts(self, conn: sqlite3.Connection, embedding_context: str | None, evidence_context: str | None) -> dict:
        out = {}
        for tier, context in (("embedding", embedding_context), ("evidence", evidence_context)):
            if context is None:
                out[tier] = {"count": 0, "bytes": 0, "oldest_expiry": None, "newest_expiry": None}
                continue
            count, total, oldest, newest = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0), MIN(expires_at), MAX(expires_at) "
                "FROM cache_entries WHERE tier = ? AND namespace = ? AND context = ?",
                (tier, self.settings.namespace, context),
            ).fetchone()
            out[tier] = {
                "count": count,
                "bytes": total,
                "oldest_expiry": oldest,
                "newest_expiry": newest,
            }
        return out

    def clear(self, kind: str, embedding_context: str | None, evidence_context: str | None) -> dict:
        removed = {"embeddings": 0, "evidence": 0}
        if not self.db_path.exists():
            return removed
        conn = self._connect()
        with _transaction(conn):
            if kind in ("embeddings", "all") and embedding_context is not None:
                removed["embeddings"] = conn.execute(
                    "DELETE FROM cache_entries WHERE tier = 'embedding' AND namespace = ? AND context = ?",
                    (self.settings.namespace, embedding_context),
                ).rowcount
            if kind in ("evidence", "all") and evidence_context is not None:
                removed["evidence"] = conn.execute(
                    "DELETE FROM cache_entries WHERE tier = 'evidence' AND namespace = ? AND context = ?",
                    (self.settings.namespace, evidence_context),
                ).rowcount
        return removed


# --- candidate payload contracts ----------------------------------------------------------


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_hit(hit: object, *, require_identity: bool, floor: float | None) -> bool:
    if not isinstance(hit, dict):
        return False
    score = hit.get("score")
    if not _finite_number(score):
        return False
    if floor is not None and not score >= floor:
        return False
    payload = hit.get("payload")
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("text"), str):
        return False
    page = payload.get("page")
    if page is not None and (isinstance(page, bool) or not isinstance(page, int)):
        return False
    for field in ("tags", "authors"):
        value = payload.get(field)
        if value is not None and (
            not isinstance(value, list) or not all(isinstance(item, str) for item in value)
        ):
            return False
    for field in ("path", "doc", "doc_name", "scope", "type", "doc_type", "title", "item_id", "citekey"):
        if payload.get(field) is not None and not isinstance(payload[field], str):
            return False
    if payload.get("year") is not None and (isinstance(payload["year"], bool) or not isinstance(payload["year"], (int, str))):
        return False
    for field in ("subjects",):
        if payload.get(field) is not None and (not isinstance(payload[field], list) or not all(isinstance(x, str) for x in payload[field])):
            return False
    if payload.get("identifier") is not None and not isinstance(payload["identifier"], (str, dict)):
        return False
    point_id = hit.get("point_id")
    if point_id is not None and (isinstance(point_id, bool) or not isinstance(point_id, (int, str))):
        return False
    if require_identity:
        document_id = payload.get("document_id")
        chunk_index = payload.get("chunk_index")
        if not isinstance(document_id, str) or not document_id:
            return False
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            return False
    elif point_id is None and not (
        payload.get("document_id") or payload.get("path") or payload.get("doc")
    ):
        return False
    return True


def _matches_filters(hit, filters):
    payload = hit["payload"]
    if "target" in filters and payload.get("document_id") not in filters["target"]["document_ids_resolved"]:
        return False
    if filters["scope"] != "all" and payload.get("scope") != filters["scope"]:
        return False
    tags = {structure.normalize_tag(tag) for tag in (payload.get("tags") or [])}
    return (set(filters.get("tags_all", [])) <= tags and
            (not filters.get("tags_any") or bool(set(filters["tags_any"]) & tags)))


def validate_candidate_payload(payload: object, *, method: str, filters: dict, window: dict) -> str | None:
    """Why a stored candidate entry is unusable, or None when compatible."""
    if not isinstance(payload, dict) or payload.get("v") != CACHE_SCHEMA_VERSION:
        return "shape"
    if payload.get("method") != method:
        return "method"
    if payload.get("filters") != filters or payload.get("window") != window:
        return "contract"
    if method == METHOD_FLAT:
        hits = payload.get("hits")
        if not isinstance(hits, list):
            return "hits"
        if len(hits) > window["limit"]:
            return "window"
        if any(not _valid_hit(hit, require_identity=False, floor=None) or not _matches_filters(hit, filters) for hit in hits):
            return "hit"
        return None
    floor = window.get("score_threshold") if isinstance(window, dict) else None
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return "groups"
    if len(groups) > window["groups"]:
        return "window"
    for group in groups:
        if not isinstance(group, dict):
            return "group"
        document_id = group.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            return "group"
        hits = group.get("hits")
        if not isinstance(hits, list):
            return "hits"
        if len(hits) > window["group_size"]:
            return "window"
        if any(not _valid_hit(hit, require_identity=True, floor=floor) or
               hit["payload"].get("document_id") != document_id or
               not _matches_filters(hit, filters) for hit in hits):
            return "hit"
    return None


def candidate_is_empty(payload: dict, *, method: str) -> bool:
    if method == METHOD_FLAT:
        return len(payload.get("hits") or []) == 0
    return sum(len(group.get("hits") or []) for group in payload.get("groups") or []) == 0


def flat_payload_from_hits(hits, *, filters: dict, window: dict) -> dict:
    return {
        "v": CACHE_SCHEMA_VERSION,
        "method": METHOD_FLAT,
        "filters": filters,
        "window": window,
        "hits": [
            {
                "point_id": getattr(hit, "id", None),
                "score": getattr(hit, "score", None),
                "payload": getattr(hit, "payload", None),
            }
            for hit in hits
        ],
    }


def grouped_payload_from_groups(groups, *, filters: dict, window: dict) -> dict:
    return {
        "v": CACHE_SCHEMA_VERSION,
        "method": METHOD_GROUPED,
        "filters": filters,
        "window": window,
        "groups": [
            {
                "document_id": getattr(group, "id", None),
                "hits": [
                    {
                        "point_id": getattr(hit, "id", None),
                        "score": getattr(hit, "score", None),
                        "payload": getattr(hit, "payload", None),
                    }
                    for hit in (getattr(group, "hits", None) or [])
                ],
            }
            for group in groups
        ],
    }


def materialize_flat(payload: dict) -> list:
    return [
        SimpleNamespace(id=hit.get("point_id"), score=hit["score"], payload=hit["payload"])
        for hit in payload["hits"]
    ]


def materialize_grouped(payload: dict) -> list:
    groups = []
    for group in payload.get("groups") or []:
        hits = [
            SimpleNamespace(id=hit.get("point_id"), score=hit["score"], payload=hit["payload"])
            for hit in group.get("hits") or []
        ]
        groups.append(SimpleNamespace(id=group["document_id"], hits=hits))
    return groups


# --- per-request cache session --------------------------------------------------------------


class CacheSession:
    """One cached query/research request: lookups, staging, guarded publication.

    The session is a context manager: leaving the body publishes staged
    candidates under the still-held shared guard on success, and discards them
    (releasing the guard) on any exception. Embedding vectors, once validated,
    are stored independently of candidate staging.
    """

    def __init__(self, settings: CacheSettings, collection: str, clock=None):
        self.settings = settings
        self.collection = collection
        self._budget = AdmissionBudget(clock=clock)
        self.diag = CacheDiagnostics(settings.mode, settings.namespace)
        self._storage: _Storage | None = None
        self._storage_bypassed = False
        self._guard: revisions._Guard | None = None
        self._verified: revisions.VerifiedRevision | None = None
        self._verification_attempted = False
        self._staged: list[tuple] = []
        self._finished = False
        self._externally_guarded = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()

    # -- storage access with optional-fail-open isolation
    def _store(self) -> _Storage | None:
        if self._budget.exhausted():
            self.diag.bypass("cache_budget_exhausted")
            return None
        if self._storage is None and not self._storage_bypassed:
            storage = _Storage(self.settings)
            self._budget.begin()
            try:
                storage.check_safe()
                storage._connect(self._budget)
            except errors.CacheUnsafeError:
                self._storage_bypassed = True
                self.diag.bypass("cache_unsafe")
            except (sqlite3.Error, OSError):
                self._storage_bypassed = True
                self.diag.bypass("cache_unavailable")
            except Exception:  # noqa: BLE001 - optional storage failures must not fail retrieval
                self._storage_bypassed = True
                self.diag.bypass("cache_unavailable")
            else:
                self._storage = storage
            finally:
                self._budget.end()
        return self._storage

    def _bypass_for_storage_error(self, exc: BaseException) -> None:
        if isinstance(exc, errors.CacheUnsafeError):
            self.diag.bypass("cache_unsafe")
        elif isinstance(exc, sqlite3.OperationalError) and (
            "interrupted" in str(exc).lower() or self._budget.exhausted()
        ):
            self.diag.bypass("cache_budget_exhausted")
        else:
            self.diag.bypass("cache_unavailable")

    # -- indexed-revision verification (candidate tier precondition)
    def admit_research_revision(self, verified):
        """The mandatory research reader owns the guard through finalization."""
        self._verification_attempted = True
        self._externally_guarded = True
        self._verified = verified
        self.diag.index = verified.to_dict()
        self.diag.validation_requests += 1

    def verify_index(self, live_check=None) -> revisions.VerifiedRevision | None:
        """Verify a stable compatible revision under the shared reader guard.

        `live_check` (collection/dimension/readiness verification) runs under the shared guard
        when given, so a cache hit can never conceal an unready collection or
        an unreachable index service. Returns the verified revision, or None
        with a bounded bypass reason recorded.
        """
        if self._verification_attempted:
            return self._verified
        self._verification_attempted = True
        ctx = revisions.managed_context(self.collection)
        guard = revisions.try_acquire_reader_guard(ctx, self._budget)
        if guard is None:
            if live_check is not None:
                from . import store
                self.diag.validation_requests += store.count_validation_calls(live_check)
            self.diag.bypass(revisions.reader_bypass_reason(ctx))
            return None
        try:
            if live_check is not None:
                from . import store
                self.diag.validation_requests += store.count_validation_calls(live_check)
            if self._budget.exhausted():
                self.diag.bypass("cache_budget_exhausted")
                return None
            self._budget.begin()
            try:
                record = revisions.read_record(ctx)
            finally:
                self._budget.end()
            if not isinstance(record, dict) or record.get("corrupt"):
                self.diag.bypass("revision_dirty")
                return None
            if record.get("state") == "dirty":
                self.diag.bypass("revision_dirty")
                return None
            verified = revisions.validate_stable_record(record, ctx)
            if verified is None:
                self.diag.bypass("revision_incompatible")
                return None
            self._guard = guard
            self._verified = verified
            self.diag.index = verified.to_dict()
            return verified
        finally:
            if self._guard is None:
                guard.release()

    @property
    def verified(self) -> revisions.VerifiedRevision | None:
        return self._verified

    # -- embedding tier
    def _lookup_embedding(self, probe_text: str) -> list[float] | None:
        if self._budget.exhausted():
            self.diag.bypass("cache_budget_exhausted")
            return None
        storage = self._store()
        if storage is None:
            return None
        key = embedding_key(self.settings, probe_text)
        try:
            payload = storage.get_payload("embedding", key, self._budget)
        except (sqlite3.Error, OSError, errors.CacheUnsafeError, ValueError, TypeError, OverflowError) as exc:
            self._storage = None
            self._storage_bypassed = True
            self._bypass_for_storage_error(exc)
            return None
        if payload is None:
            self.diag.embedding_misses += 1
            return None
        vector = _validated_vector(payload)
        if vector is None:
            self.diag.embedding_misses += 1
            self.diag.bypass("entry_invalid")
            try:
                storage.delete_entry("embedding", key, self._budget)
            except (sqlite3.Error, OSError, errors.CacheUnsafeError, ValueError, TypeError, OverflowError) as exc:
                self._bypass_for_storage_error(exc)
            return None
        self.diag.embedding_hits += 1
        return vector

    def embed_probes(self, probes: list[str], embed_callable) -> dict:
        """{probe: vector} for exactly these distinct probes.

        `use` looks up each probe first and embeds only misses; `refresh` skips
        reads and embeds everything. Successful fresh vectors are validated and
        stored independently, before any candidate work happens.
        """
        distinct = list(dict.fromkeys(probes))
        resolved: dict[str, list[float]] = {}
        missing: list[str] = []
        if self.settings.mode == "use":
            for probe in distinct:
                vector = self._lookup_embedding(probe)
                if vector is None:
                    missing.append(probe)
                else:
                    resolved[probe] = vector
        else:
            missing = distinct
        if missing:
            fresh = embed_callable(missing)
            if not isinstance(fresh, list) or len(fresh) != len(missing):
                raise errors.EmbeddingApiError("embedding provider returned an incomplete vector batch")
            self.diag.embedding_requests += math.ceil(len(missing) / config.EMBED_BATCH_SIZE)
            validated = []
            for probe, vector in zip(missing, fresh):
                checked = _validated_vector({"v": CACHE_SCHEMA_VERSION, "vector": vector})
                if checked is None:
                    # Never substitute a malformed vector for a successful one.
                    raise errors.EmbeddingApiError(
                        "embedding provider returned a malformed vector"
                    )
                resolved[probe] = checked
                validated.append((probe, checked))
            self._store_embeddings(validated)
        return resolved

    def _store_embeddings(self, vectors: list[tuple[str, list[float]]]) -> None:
        if not vectors:
            return
        storage = self._store()
        if storage is None:
            return
        context = revisions.embedding_fingerprint()
        for probe, vector in vectors:
            if self._budget.exhausted():
                self.diag.bypass("cache_budget_exhausted")
                return
            key = embedding_key(self.settings, probe)
            try:
                reason = storage.put_payload(
                    "embedding", key, context,
                    {"v": CACHE_SCHEMA_VERSION, "vector": vector},
                    self.settings.embedding_ttl, self._budget,
                )
            except (sqlite3.Error, OSError, errors.CacheUnsafeError, ValueError, TypeError, OverflowError) as exc:
                self._bypass_for_storage_error(exc)
                return
            self.diag.bypass(reason)

    # -- candidate tier
    def lookup_candidates(self, *, probe_text: str, method: str, filters: dict, window: dict):
        """Compatible cached candidates, or None (miss/bypass, reason recorded)."""
        if self._verified is None or self.settings.mode != "use":
            return None
        if self._budget.exhausted():
            self.diag.bypass("cache_budget_exhausted")
            return None
        storage = self._store()
        if storage is None:
            return None
        key = candidate_key(
            self.settings, probe_text=probe_text, verified=self._verified,
            method=method, filters=filters, window=window,
        )
        try:
            payload = storage.get_payload("evidence", key, self._budget)
        except (sqlite3.Error, OSError, errors.CacheUnsafeError, ValueError, TypeError, OverflowError) as exc:
            self._storage = None
            self._storage_bypassed = True
            self._bypass_for_storage_error(exc)
            return None
        if payload is None:
            self.diag.candidate_misses += 1
            return None
        if validate_candidate_payload(payload, method=method, filters=filters, window=window) is not None:
            self.diag.candidate_misses += 1
            self.diag.bypass("entry_invalid")
            try:
                storage.delete_entry("evidence", key, self._budget)
            except (sqlite3.Error, OSError, errors.CacheUnsafeError, ValueError, TypeError, OverflowError) as exc:
                self._bypass_for_storage_error(exc)
            return None
        self.diag.candidate_hits += 1
        return payload

    def stage_candidates(self, *, probe_text: str, method: str, filters: dict, window: dict, payload: dict) -> None:
        """Stage a fresh candidate payload until the whole request succeeds."""
        if self._verified is None or self._finished:
            return
        try:
            json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        if validate_candidate_payload(payload, method=method, filters=filters, window=window) is not None:
            raise errors.LibraryQueryError("internal_error", "index service returned malformed or incompatible candidates")
        key = candidate_key(
            self.settings, probe_text=probe_text, verified=self._verified,
            method=method, filters=filters, window=window,
        )
        ttl = self.settings.empty_ttl if candidate_is_empty(payload, method=method) else self.settings.evidence_ttl
        self._staged.append(("evidence", key, self._verified.corpus_context_digest(), payload, ttl))

    # -- completion
    def finalize(self) -> None:
        """Publish staged candidates under the held guard; release everything."""
        if self._finished:
            return
        self._finished = True
        try:
            if self._staged and self._verified is not None and (self._guard is not None or self._externally_guarded):
                if self._budget.exhausted():
                    self.diag.bypass("cache_budget_exhausted")
                else:
                    storage = self._store()
                    if storage is not None:
                        try:
                            for reason in storage.put_payload_batch(self._staged, self._budget):
                                self.diag.bypass(reason)
                        except (sqlite3.Error, OSError, errors.CacheUnsafeError, ValueError, TypeError, OverflowError) as exc:
                            self._bypass_for_storage_error(exc)
        finally:
            self._release_guard()
            self._staged = []
            self.diag.cache_elapsed_ms = self._budget.elapsed_ms()

    def abort(self) -> None:
        """Discard staged candidates and release the guard (failed request)."""
        if self._finished:
            return
        self._finished = True
        self._staged = []
        self._release_guard()
        self.diag.cache_elapsed_ms = self._budget.elapsed_ms()

    def _release_guard(self) -> None:
        if self._storage is not None and self._storage._conn is not None:
            self._storage._conn.close()
            self._storage._conn = None
        if self._guard is not None:
            guard, self._guard = self._guard, None
            guard.release()

    def diagnostics(self) -> dict:
        if not self._finished:
            self.diag.cache_elapsed_ms = self._budget.elapsed_ms()
        return self.diag.to_dict()


def _validated_vector(payload: dict) -> list[float] | None:
    """A cached embedding vector validated for schema, dimension, type, finiteness."""
    if not isinstance(payload, dict) or payload.get("v") != CACHE_SCHEMA_VERSION:
        return None
    vector = payload.get("vector")
    if not isinstance(vector, list) or len(vector) != config.EMBEDDING_DIM:
        return None
    for value in vector:
        if not _finite_number(value):
            return None
    return [float(value) for value in vector]


# --- maintenance commands -------------------------------------------------------------------


def _unverified_index_report(record: dict | None) -> dict | None:
    """Local revision state for status output: explicitly unverified."""
    if not isinstance(record, dict) or record.get("corrupt"):
        return None
    if record.get("state") not in ("stable", "dirty"):
        return None
    return {
        "corpus_id": record.get("corpus_id"),
        "incarnation": record.get("incarnation"),
        "revision": record.get("revision"),
        "state": record.get("state"),
        "embedding_fingerprint": record.get("embedding_fingerprint"),
        "representation_fingerprint": record.get("representation_fingerprint"),
        "verified": False,
    }


def _record_corpus_context(record: dict | None, ctx=None) -> str | None:
    if not isinstance(record, dict) or record.get("corrupt"):
        return None
    corpus_id = record.get("corpus_id")
    incarnation = record.get("incarnation")
    if ctx is not None and (corpus_id != ctx.corpus_id or record.get("context_digest") != ctx.context_digest):
        return None
    if not isinstance(corpus_id, str) or not isinstance(incarnation, str) or not corpus_id or not incarnation:
        return None
    return revisions.canonical_digest({"corpus_id": corpus_id, "incarnation": incarnation})


def status_report(settings: CacheSettings) -> dict:
    """Read-only local inspection: creates and recovers nothing.

    Uses an ordinary `mode=ro` connection when storage exists; a hot journal
    needing rollback is reported as cache_recovery_required (exit 3) with
    storage bytes and mtimes untouched.
    """
    storage = _Storage(settings)
    ctx = revisions.managed_context()
    record = revisions.read_record(ctx)
    report = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "mode": settings.mode,
        "namespace": settings.namespace,
        "limits": settings.limits_dict(),
        "storage": "absent",
        "database_bytes": 0,
        "entries": {
            "embedding": {"count": 0, "bytes": 0, "oldest_expiry": None, "newest_expiry": None},
            "evidence": {"count": 0, "bytes": 0, "oldest_expiry": None, "newest_expiry": None},
        },
        "index": _unverified_index_report(record),
    }
    if not storage.db_path.exists():
        # An existing but unsafe directory is a real configuration problem even
        # before any database file exists; a fully absent location is not.
        if settings.cache_dir.exists():
            storage.check_safe()
        return report
    storage.check_safe()
    try:
        conn = storage.connect_readonly()
        try:
            counts = storage.scoped_counts(conn, ctx.embedding_fingerprint, _record_corpus_context(record, ctx))
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        if _is_readonly_rollback(exc):
            raise errors.CacheRecoveryRequiredError(
                "cache storage has a hot journal requiring recovery; authorize an "
                "explicit read-write recovery operation and rerun status"
            ) from exc
        raise errors.LibraryQueryError(
            "internal_error",
            f"cache status could not read the storage database ({type(exc).__name__}); "
            "operator recovery is required",
        ) from exc
    except sqlite3.DatabaseError as exc:
        raise errors.LibraryQueryError(
            "internal_error",
            f"cache storage is unreadable ({type(exc).__name__}); operator recovery is required",
        ) from exc
    report["storage"] = "present"
    report["entries"] = counts
    try:
        report["database_bytes"] = storage.db_path.stat().st_size
    except OSError:
        report["database_bytes"] = 0
    return report


def clear_storage(settings: CacheSettings, kind: str) -> dict:
    """Targeted deletion of current-namespace/current-context rows only.

    Never removes directories, guard files, revision records, other namespaces,
    or other corpus/model contexts. Without local corpus identity, evidence
    targeting is reported unavailable and zero evidence rows are removed.
    """
    storage = _Storage(settings)
    ctx = revisions.managed_context()
    record = revisions.read_record(ctx)
    evidence_context = _record_corpus_context(record, ctx)
    absent = not storage.db_path.exists()
    report = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "namespace": settings.namespace,
        "kind": kind,
        "storage": "absent" if absent else "present",
        "evidence_targeting": "ok" if evidence_context is not None else "unavailable",
        "removed": {"embeddings": 0, "evidence": 0},
    }
    if absent:
        return report
    storage.check_safe()
    try:
        removed = storage.clear(kind, ctx.embedding_fingerprint, evidence_context)
    except sqlite3.DatabaseError as exc:
        raise errors.LibraryQueryError(
            "internal_error",
            f"cache clear failed on the storage database ({type(exc).__name__}); "
            "operator recovery is required",
        ) from exc
    report["removed"] = removed
    return report


# --- discovery ------------------------------------------------------------------------------


def describe_schema() -> dict:
    """Typed cache contract derived from the same runtime definitions above."""
    return {
        "modes": list(CACHE_MODES),
        "default_mode": "off",
        "namespace": {"env": _ENV_NAMESPACE, "default": DEFAULT_NAMESPACE, "max_chars": _MAX_NAMESPACE_CHARS},
        "embedding_revision": {
            "env": revisions.EMBEDDING_REVISION_ENV,
            "default": revisions.DEFAULT_EMBEDDING_REVISION,
            "semantics": "operator-controlled deployment token; advance when replacing a model behind a stable identifier; it does not prove a remote alias is immutable",
        },
        "settings": [
            {"name": "max_entries", "env": _ENV_MAX_ENTRIES, "default": DEFAULT_MAX_ENTRIES, "type": "positive_integer"},
            {"name": "max_bytes", "env": _ENV_MAX_BYTES, "default": DEFAULT_MAX_BYTES, "type": "positive_integer"},
            {"name": "max_entry_bytes", "env": _ENV_MAX_ENTRY_BYTES, "default": DEFAULT_MAX_ENTRY_BYTES, "type": "positive_integer"},
            {"name": "embedding_ttl_seconds", "env": _ENV_EMBEDDING_TTL, "default": DEFAULT_EMBEDDING_TTL_SECONDS, "type": "positive_integer"},
            {"name": "evidence_ttl_seconds", "env": _ENV_EVIDENCE_TTL, "default": DEFAULT_EVIDENCE_TTL_SECONDS, "type": "positive_integer"},
            {"name": "empty_ttl_seconds", "env": _ENV_EMPTY_TTL, "default": DEFAULT_EMPTY_TTL_SECONDS, "type": "positive_integer"},
        ],
        "cache_dir": {"env": _ENV_DIR, "fallback": "${XDG_CACHE_HOME}/library-query or ~/.cache/library-query"},
        "storage": {
            "filename": CACHE_FILENAME,
            "journal": "rollback",
            "permissions": {"directory": "0700", "files": "0600"},
            "safety": "rejects source-library and Git-worktree locations, symlinks, unsafe ownership, and group/other-writable paths",
        },
        "admission_budget_ms": int(ADMISSION_BUDGET_MS),
        "budget_semantics": "cumulative optional-work budget shared by both tiers; lock waits are recomputed from the remainder per operation; not a hard bound on SQL or filesystem latency; measured elapsed is never clipped",
        "bypass_reasons": sorted(BYPASS_REASONS),
        "diagnostics_schema": {
            "schema_version": {"type": "integer", "const": CACHE_SCHEMA_VERSION},
            "mode": {"type": "string", "choices": list(CACHE_MODES)},
            "namespace": {"type": "string"},
            "index": {
                "type": ["object", "null"],
                "fields": {
                    "corpus_id": {"type": "string"},
                    "incarnation": {"type": "string"},
                    "revision": {"type": "integer", "min": 1},
                    "representation_fingerprint": {"type": "string"},
                    "embedding_fingerprint": {"type": "string"},
                },
                "semantics": "verified provenance only; null when candidate reuse was bypassed",
            },
            "embedding_hits": {"type": "integer", "min": 0},
            "embedding_misses": {"type": "integer", "min": 0},
            "candidate_hits": {"type": "integer", "min": 0},
            "candidate_misses": {"type": "integer", "min": 0},
            "embedding_request_count": {"type": "integer", "min": 0},
            "search_request_count": {"type": "integer", "min": 0},
            "validation_request_count": {"type": "integer", "min": 0},
            "cache_elapsed_ms": {"type": "integer", "min": 0, "semantics": "measured, never clipped to the admission budget"},
            "bypass_reasons": {"type": "array", "item": {"type": "string", "choices": sorted(BYPASS_REASONS)}},
        },
        "maintenance": {
            "status": {"read_only": True, "creates_nothing": True},
            "clear": {"kinds": ["embeddings", "evidence", "all"], "default_kind": "all"},
        },
        "recovery_error": {
            "category": "cache_recovery_required",
            "exit_code": errors.EXIT_CONFIG_ERROR,
            "semantics": "read-only status met a hot journal; exactly one JSON error line on stderr, no success output, storage untouched",
        },
        "privacy": {
            "keying": "SHA-256 canonical JSON; probe text is digested and never persisted; no credentials or absolute host paths in keys, diagnostics, or errors",
            "residency": "opt-in bounded TTL/LRU rows; deletion is logical eviction, not forensic erasure",
        },
    }
