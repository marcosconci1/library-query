"""The agent-facing failure contract: error categories, exit codes, and classification.

`library-query` is invoked by an autonomous caller that cannot read a traceback or
ask a clarifying question. Every failure therefore has to arrive as one line of JSON
on stderr carrying a stable category, a readable sentence, and the process's real
exit code. This module owns that taxonomy; `cli.main()` owns applying it.
"""
import json
import sys

# Exit code taxonomy. These are contract surface — changing one is a spec change,
# not a code change; tests/test_agent_contract.py pins the taxonomy.
EXIT_OK = 0
EXIT_INVALID_ARGS = 2
EXIT_CONFIG_ERROR = 3
EXIT_SERVICE_ERROR = 4
EXIT_INTERNAL_ERROR = 5

#: Every error category the CLI can emit, mapped to its exit code.
ERROR_EXIT_CODES = {
    "target_not_found": EXIT_INVALID_ARGS,
    "target_not_indexed": EXIT_CONFIG_ERROR,
    "passage_not_found": EXIT_INVALID_ARGS,
    "stale_evidence": EXIT_INVALID_ARGS,
    "missing_api_key": EXIT_CONFIG_ERROR,
    "invalid_library_root": EXIT_CONFIG_ERROR,
    "unsupported_format": EXIT_CONFIG_ERROR,
    "research_index_not_ready": EXIT_CONFIG_ERROR,
    "qdrant_unreachable": EXIT_SERVICE_ERROR,
    "embedding_api_error": EXIT_SERVICE_ERROR,
    "zotero_unreachable": EXIT_SERVICE_ERROR,
    "zotero_invalid_response": EXIT_SERVICE_ERROR,
    "research_timeout": EXIT_SERVICE_ERROR,
    # One vector point serialized larger than the configured upsert-batch
    # ceiling: batching cannot help, and the point must not be truncated.
    # Operator action (raise the ceiling or fix the extractor), not a retry.
    "qdrant_batch_too_large": EXIT_CONFIG_ERROR,
    "invalid_research_request": EXIT_INVALID_ARGS,
    # Read-only planning/generation surfaces found data the caller must
    # resolve (conflicts); the diagnostics themselves ship on stdout.
    "layout_plan_blocked": EXIT_INVALID_ARGS,
    "navigation_write_blocked": EXIT_INVALID_ARGS,
    # Retrieval cache: unsafe/invalid cache configuration refuses to touch the
    # location (exit 3); a read-only status open that meets a hot journal needs
    # an explicitly authorized recovery, which status is not (exit 3).
    "cache_unsafe": EXIT_CONFIG_ERROR,
    "cache_recovery_required": EXIT_CONFIG_ERROR,
    # The managed index-revision guard or durable dirty publication is required
    # before any corpus mutation; without it indexing aborts before writing.
    "index_revision_unavailable": EXIT_CONFIG_ERROR,
    "internal_error": EXIT_INTERNAL_ERROR,
}

#: The complete set of exit codes the CLI is allowed to produce, for tests to assert against.
ALLOWED_EXIT_CODES = {
    EXIT_OK,
    EXIT_INVALID_ARGS,
    EXIT_CONFIG_ERROR,
    EXIT_SERVICE_ERROR,
    EXIT_INTERNAL_ERROR,
}


class LibraryQueryError(Exception):
    """A failure already classified into the agent-facing contract.

    Raised where the cause is unambiguous (e.g. a missing config value); everything
    else is classified after the fact by `classify`.
    """

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.message = message

    @property
    def exit_code(self) -> int:
        return ERROR_EXIT_CODES[self.category]


class MissingApiKeyError(LibraryQueryError):
    def __init__(self, message: str = "OPENROUTER_API_KEY is not set — copy .env.example to .env and fill it in"):
        super().__init__("missing_api_key", message)


class InvalidLibraryRootError(LibraryQueryError):
    def __init__(self, message: str):
        super().__init__("invalid_library_root", message)


class UnsupportedFormatError(LibraryQueryError):
    def __init__(self, message: str):
        super().__init__("unsupported_format", message)


class MissingExtractorError(UnsupportedFormatError):
    pass


class QdrantUnreachableError(LibraryQueryError):
    def __init__(self, message: str):
        super().__init__("qdrant_unreachable", message)


class EmbeddingApiError(LibraryQueryError):
    def __init__(self, message: str):
        super().__init__("embedding_api_error", message)


class ZoteroUnreachableError(LibraryQueryError):
    def __init__(self, message: str):
        super().__init__("zotero_unreachable", message)


class ZoteroInvalidResponseError(LibraryQueryError):
    def __init__(self, message: str):
        super().__init__("zotero_invalid_response", message)


class InvalidResearchRequestError(LibraryQueryError):
    """The research request failed schema/version/limit validation.

    Carries a field-only message (`queries[0].text: ...`): never the caller's
    question text or any secret, so the error line is safe for logs.
    """

    def __init__(self, message: str):
        super().__init__("invalid_research_request", message)


class ResearchIndexNotReadyError(LibraryQueryError):
    """The corpus cannot serve research: missing collection/index or invalid identities.

    Exit 3 (configuration), with the remediation action in the message: the caller
    is expected to run `index`, not retry the request.
    """

    def __init__(self, message: str):
        super().__init__("research_index_not_ready", message)


class ResearchTimeoutError(LibraryQueryError):
    """The total research service-work deadline was exhausted."""

    def __init__(self, message: str):
        super().__init__("research_timeout", message)


class LayoutPlanBlockedError(LibraryQueryError):
    """The read-only layout plan has blocking conflicts.

    The full plan (including the conflicts) is printed to stdout before this
    error is raised; the caller gets machine-readable diagnostics and a
    non-zero exit through the normal error contract.
    """

    def __init__(self, message: str):
        super().__init__("layout_plan_blocked", message)


class NavigationWriteBlockedError(LibraryQueryError):
    """A navigation write was refused before any write happened."""

    def __init__(self, message: str):
        super().__init__("navigation_write_blocked", message)


class CacheUnsafeError(LibraryQueryError):
    """The configured cache location or cache configuration failed a safety check.

    Retrieval treats this as an optional-fail-open bypass; explicit maintenance
    commands surface it as the classified configuration error it is.
    """

    def __init__(self, message: str):
        super().__init__("cache_unsafe", message)


class CacheRecoveryRequiredError(LibraryQueryError):
    """A read-only inspection met a hot journal that requires rollback recovery.

    Emitted exactly once on stderr with exit 3 and no success output; storage
    bytes and mtimes are left untouched for an explicitly authorized recovery.
    """

    def __init__(self, message: str):
        super().__init__("cache_recovery_required", message)


class IndexRevisionUnavailableError(LibraryQueryError):
    """The writer guard or durable revision publication failed before a corpus write.

    Optional-cache fail-open behavior applies to retrieval storage, never to
    authoritative invalidation: indexing reports this and mutates nothing.
    """

    def __init__(self, message: str):
        super().__init__("index_revision_unavailable", message)


class QdrantBatchTooLargeError(LibraryQueryError):
    """One vector point serializes larger than the configured upsert-batch ceiling.

    Batch splitting cannot make the point fit; truncating or dropping it would
    corrupt chunk identity. Reported as a configuration error so the operator
    raises the ceiling or fixes the extractor rather than retrying.
    """

    def __init__(self, message: str):
        super().__init__("qdrant_batch_too_large", message)


def _is_qdrant_exception(exc: BaseException) -> bool:
    """Decide whether an exception originated from the Qdrant client.

    Qdrant's client wraps transport failures in its own exception types and also lets
    bare `httpx` errors through, so this checks both the class lineage's module and
    the `qdrant_client` exception names rather than catching `httpx` broadly — an
    `httpx` failure from the embedding call must not be misreported as Qdrant.
    """
    for klass in type(exc).__mro__:
        module = getattr(klass, "__module__", "") or ""
        if module.startswith("qdrant_client") or module.startswith("grpc"):
            return True
    return False


def _walk_causes(exc: BaseException):
    """Yield the exception and its `__cause__`/`__context__` chain, innermost last."""
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def classify(exc: BaseException) -> tuple[str, str, int]:
    """Map any exception onto `(category, message, exit_code)`.

    Never raises and never returns an unclassified result — an exception this does
    not recognise becomes `internal_error`, which is what keeps a raw traceback from
    ever reaching the caller.
    """
    for current in _walk_causes(exc):
        if isinstance(current, LibraryQueryError):
            return current.category, current.message, current.exit_code

    # Qdrant transport failures, including those raised through the client's wrappers.
    for current in _walk_causes(exc):
        if _is_qdrant_exception(current):
            return (
                "qdrant_unreachable",
                f"Could not reach Qdrant: {current}",
                EXIT_SERVICE_ERROR,
            )

    # Bare httpx/connection failures that reached here unclassified are attributed to
    # the embedding API: every Qdrant path is wrapped by `store` and caught above.
    for current in _walk_causes(exc):
        module = (getattr(type(current), "__module__", "") or "").split(".")[0]
        if module == "httpx":
            return (
                "embedding_api_error",
                f"Embedding API request failed: {current}",
                EXIT_SERVICE_ERROR,
            )

    message = str(exc) or type(exc).__name__
    return "internal_error", message, EXIT_INTERNAL_ERROR


def emit(category: str, message: str, exit_code: int, stream=None) -> int:
    """Print the single-line JSON error object to stderr and return the exit code.

    Structured regardless of whether `--json` was passed: a caller parsing a failure
    shouldn't have to know how the command was invoked to know how to read it.
    """
    stream = stream if stream is not None else sys.stderr
    print(
        json.dumps({"error": category, "message": message, "exit_code": exit_code}, ensure_ascii=False),
        file=stream,
    )
    return exit_code


def report(exc: BaseException, stream=None) -> int:
    """Classify `exc`, emit it as structured JSON, and return its exit code."""
    category, message, exit_code = classify(exc)
    return emit(category, message, exit_code, stream=stream)
