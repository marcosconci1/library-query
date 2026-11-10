"""Versioned multi-question research retrieval: schema, validation, fusion, bundle.

`query` is a single-question locator; `research` is the evidence API for callers
that already planned several independent questions and need source-diverse,
traceable passages back. Everything here is deterministic
post-processing over grouped Qdrant results — no language model, no planning, no
citation prose.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import sys
import time

from . import config, errors, exact, expansion, store, structure, targets

#: Version 1 request limits. These are contract surface: the values are asserted by
#: tests and advertised by `describe --json`, so changing one is a spec change.
SCHEMA_VERSION = 1
MAX_QUESTIONS = 8
MAX_QUESTION_CHARS = 4000
MAX_SOURCES = 10
MAX_PASSAGES_PER_SOURCE = 5
MAX_EXCERPT_CHARS = 8000
MAX_TAGS_PER_ARRAY = 32
MAX_INPUT_BYTES = 65536

DEFAULT_SCOPE = "catalogued"
DEFAULT_EXPAND = False
DEFAULT_MIN_SCORE = None
DEFAULT_MAX_SOURCES = 3
DEFAULT_MAX_PASSAGES_PER_SOURCE = 2
DEFAULT_MAX_EXCERPT_CHARS = 1200

SCOPES = ("inbox", "catalogued", "operations", "all")

#: Reciprocal rank fusion constants (one-based ranks). k=60 is the paper's useful
#: baseline, not a corpus-tuned optimum; the contract pins it so fusion is stable.
RRF_K = 60
RRF_RANK_BASE = 1
FUSION = {"method": "rrf", "k": RRF_K, "rank_base": RRF_RANK_BASE}

COVERAGE_STATUSES = ("selected", "no_eligible_evidence", "budget_excluded")

#: Monotonic clock source for the service-work deadline. A module attribute (not a
#: default argument) so tests can substitute a controllable clock at CLI level too.
CLOCK = time.monotonic

_REQUEST_FIELDS = {
    "schema_version",
    "queries",
    "scope",
    "expand",
    "min_score",
    "max_sources",
    "max_passages_per_source",
    "max_excerpt_chars",
}
_QUERY_FIELDS = {"text", "tags"}
_TAG_FIELDS = {"any", "all"}

SOURCE_OUTPUT_FIELDS = [
    "source_id",
    "doc",
    "doc_name",
    "scope",
    "type",
    "title",
    "authors",
    "year",
    "subjects",
    "tags",
    "identifier",
    "rrf_score",
    "passages",
]

PASSAGE_OUTPUT_FIELDS = [
    "evidence_id",
    "chunk_index",
    "passage_hash",
    "text",
    "text_truncated",
    "page",
    "location",
    "score",
    "query_indexes",
    "query_scores",
]

DOC_METADATA_FIELDS = ("title", "authors", "year", "subjects", "tags", "identifier")


class _DuplicateKey(dict):
    """object_pairs_hook that turns duplicate JSON object keys into a rejection."""


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise errors.InvalidResearchRequestError(f"duplicate field '{key}'")
        out[key] = value
    return out


@dataclass(frozen=True)
class QuerySpec:
    """One validated question with normalized tag constraints."""

    text: str
    tags_any: tuple[str, ...] = ()
    tags_all: tuple[str, ...] = ()
    target: targets.Target | None = None

    @property
    def dedupe_key(self) -> tuple:
        key = (self.text, self.tags_any, self.tags_all)
        return key + (self.target,) if self.target is not None else key

    def to_dict(self) -> dict:
        value = {"text": self.text, "tags": {"any": list(self.tags_any), "all": list(self.tags_all)}}
        if self.target is not None:
            value["target"] = self.target.to_dict()
        return value


@dataclass(frozen=True)
class ResearchRequest:
    """The resolved version 1 request: caller values with defaults applied."""

    queries: tuple[QuerySpec, ...]
    scope: str = DEFAULT_SCOPE
    expand: bool = DEFAULT_EXPAND
    min_score: float | None = DEFAULT_MIN_SCORE
    max_sources: int = DEFAULT_MAX_SOURCES
    max_passages_per_source: int = DEFAULT_MAX_PASSAGES_PER_SOURCE
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS
    schema_version: int = SCHEMA_VERSION
    target: targets.Target | None = None

    def to_dict(self) -> dict:
        value = {
            "schema_version": self.schema_version,
            "queries": [query.to_dict() for query in self.queries],
            "scope": self.scope,
            "expand": self.expand,
            "min_score": self.min_score,
            "max_sources": self.max_sources,
            "max_passages_per_source": self.max_passages_per_source,
            "max_excerpt_chars": self.max_excerpt_chars,
        }
        if self.target is not None:
            value["target"] = self.target.to_dict()
        return value


def _invalid(field: str, expectation: str) -> errors.InvalidResearchRequestError:
    # Field-only messages: never echo question text, file contents, or secrets.
    return errors.InvalidResearchRequestError(f"{field}: {expectation}")


def _check_int(value: object, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(field, f"must be an integer in [{low}, {high}]")
    if not low <= value <= high:
        raise _invalid(field, f"must be in [{low}, {high}]")
    return value


def _normalize_tag_array(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _invalid(field, "must be an array of strings")
    if len(value) > MAX_TAGS_PER_ARRAY:
        raise _invalid(field, f"must contain at most {MAX_TAGS_PER_ARRAY} tags")
    normalized = set()
    for item in value:
        if not isinstance(item, str):
            raise _invalid(field, "must be an array of strings")
        tag = structure.normalize_tag(item)
        if not tag:
            raise _invalid(field, "tags must not be empty or whitespace-only")
        normalized.add(tag)
    return tuple(sorted(normalized))


def _parse_query(value: object, index: int) -> QuerySpec:
    field = f"queries[{index}]"
    if not isinstance(value, dict):
        raise _invalid(field, "must be an object")
    unknown = sorted(set(value) - _QUERY_FIELDS)
    if unknown:
        raise _invalid(field, f"unknown field(s): {', '.join(unknown)}")

    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _invalid(f"{field}.text", "must be a non-empty, non-whitespace string")
    if len(text) > MAX_QUESTION_CHARS:
        raise _invalid(f"{field}.text", f"must be at most {MAX_QUESTION_CHARS} characters")

    tags_any: tuple[str, ...] = ()
    tags_all: tuple[str, ...] = ()
    if "tags" in value:
        tags = value["tags"]
        if not isinstance(tags, dict):
            raise _invalid(f"{field}.tags", "must be an object with any/all arrays")
        unknown_tags = sorted(set(tags) - _TAG_FIELDS)
        if unknown_tags:
            raise _invalid(f"{field}.tags", f"unknown field(s): {', '.join(unknown_tags)}")
        tags_any = _normalize_tag_array(tags.get("any"), f"{field}.tags.any")
        tags_all = _normalize_tag_array(tags.get("all"), f"{field}.tags.all")
    return QuerySpec(text=text, tags_any=tags_any, tags_all=tags_all)


def parse_request(data: bytes) -> ResearchRequest:
    """Validate one bounded UTF-8 JSON request, applying version 1 defaults.

    Runs entirely before any service call: malformed input, duplicate keys,
    unknown fields, wrong types (including booleans standing in for integer
    budgets), non-finite numbers, and out-of-range values all reject here with
    `invalid_research_request` (exit 2) and a field-only message.
    """
    if len(data) > MAX_INPUT_BYTES:
        raise _invalid("input", f"must be at most {MAX_INPUT_BYTES} bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _invalid("input", "must be UTF-8 encoded JSON") from None

    def _non_finite(constant: str):
        raise _invalid("input", f"non-finite number '{constant}' is not allowed")

    try:
        obj = json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_non_finite)
    except errors.InvalidResearchRequestError:
        raise
    except json.JSONDecodeError as exc:
        raise _invalid("input", f"malformed JSON ({exc.msg})") from None
    if not isinstance(obj, dict):
        raise _invalid("input", "must be a single JSON object")

    version = obj.get("schema_version", SCHEMA_VERSION)
    unknown = sorted(set(obj) - (_REQUEST_FIELDS | ({"target"} if version == 2 else set())))
    if unknown:
        raise _invalid("input", f"unknown field(s): {', '.join(unknown)}")

    version = obj.get("schema_version", SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise _invalid("schema_version", "must be an integer")
    if version not in (1, 2):
        raise _invalid("schema_version", f"unsupported version {version}; supported: 1, 2")

    raw_queries = obj.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise _invalid("queries", f"must be a non-empty array of at most {MAX_QUESTIONS} questions")
    if len(raw_queries) > MAX_QUESTIONS:
        raise _invalid("queries", f"must contain at most {MAX_QUESTIONS} questions")
    shared_target = targets.parse(obj["target"]) if "target" in obj else None
    queries_list = []
    for i, item in enumerate(raw_queries):
        target = shared_target
        if version == 2 and isinstance(item, dict) and "target" in item:
            target = targets.parse(item["target"], f"queries[{i}].target")
            item = {k: v for k, v in item.items() if k != "target"}
        query = _parse_query(item, i)
        queries_list.append(replace(query, target=target))
    queries = tuple(queries_list)
    identities = {(q.target.kind, i) for q in queries if q.target for i in q.target.ids}
    if len(identities) > targets.MAX_TARGETS:
        raise _invalid("target", "request-wide identity union exceeds 10")

    scope = obj.get("scope", DEFAULT_SCOPE)
    if not isinstance(scope, str) or scope not in SCOPES:
        raise _invalid("scope", f"must be one of: {', '.join(SCOPES)}")

    expand = obj.get("expand", DEFAULT_EXPAND)
    if not isinstance(expand, bool):
        raise _invalid("expand", "must be a boolean")

    min_score = obj.get("min_score", DEFAULT_MIN_SCORE)
    if min_score is not None:
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
            raise _invalid("min_score", "must be null or a finite number in [-1, 1]")
        if not math.isfinite(min_score):
            raise _invalid("min_score", "must be finite")
        if not -1.0 <= float(min_score) <= 1.0:
            raise _invalid("min_score", "must be within [-1, 1]")
        min_score = float(min_score)

    max_sources = _check_int(obj.get("max_sources", DEFAULT_MAX_SOURCES), "max_sources", 1, MAX_SOURCES)
    max_passages = _check_int(
        obj.get("max_passages_per_source", DEFAULT_MAX_PASSAGES_PER_SOURCE),
        "max_passages_per_source",
        1,
        MAX_PASSAGES_PER_SOURCE,
    )
    max_excerpt = _check_int(
        obj.get("max_excerpt_chars", DEFAULT_MAX_EXCERPT_CHARS),
        "max_excerpt_chars",
        1,
        MAX_EXCERPT_CHARS,
    )

    return ResearchRequest(
        queries=queries,
        scope=scope,
        expand=expand,
        min_score=min_score,
        max_sources=max_sources,
        max_passages_per_source=max_passages,
        max_excerpt_chars=max_excerpt,
        schema_version=version,
        target=shared_target,
    )


def read_input(source: str) -> bytes:
    """Read the bounded request bytes from a named file or stdin (`-`).

    At most `MAX_INPUT_BYTES + 1` bytes are read so an over-limit input is
    detected, not decoded. An unreadable input is an invalid request (exit 2),
    never a silent empty object.
    """
    try:
        if source == "-":
            return sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        with open(source, "rb") as handle:
            return handle.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise _invalid("input", f"could not be read ({exc.strerror or type(exc).__name__})") from None


class _Deadline:
    """Total service-work budget shared by readiness, embedding, and every search."""

    def __init__(self, budget: float, clock):
        self._clock = clock
        self._start = clock()
        self._budget = budget

    def remaining(self) -> float:
        return self._budget - (self._clock() - self._start)

    def service_timeout(self) -> int:
        # Qdrant's per-call timeout is whole seconds; never promise 0.
        return max(1, math.ceil(self.remaining()))

    def check(self) -> None:
        if not math.isfinite(self._budget) or self._budget <= 0:
            raise errors.ResearchTimeoutError(
                f"research deadline must be positive (LIBRARY_QUERY_RESEARCH_TIMEOUT={self._budget})"
            )
        if self.remaining() <= 0:
            raise errors.ResearchTimeoutError(
                f"research service-work deadline of {self._budget:g}s was exhausted"
            )


def rrf_fuse(rankings: list[list[str]]) -> dict[str, tuple[float, int]]:
    """Fuse per-question source rankings into `(rrf_score, best_rank)` per source.

    One vote per ranking in which a source occurs, `sum(1 / (60 + one_based_rank))`;
    absent sources contribute zero. Ties break by best single-question rank, then
    ascending source id (applied by the caller's sort, which also orders by score).
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, source in enumerate(ranking, start=RRF_RANK_BASE):
            scores[source] = scores.get(source, 0.0) + 1.0 / (RRF_K + rank)
            best_rank[source] = min(best_rank.get(source, rank), rank)
    return {source: (scores[source], best_rank[source]) for source in scores}


def _passage_key(hit) -> object:
    point_id = getattr(hit, "id", None)
    if point_id is not None:
        return point_id
    payload = getattr(hit, "payload", None) or {}
    return (payload.get("document_id"), payload.get("chunk_index"), payload.get("text"))


def _passage_score(hit) -> float:
    score = getattr(hit, "score", None)
    return score if isinstance(score, (int, float)) and not isinstance(score, bool) else float("-inf")


def _chunk_index(payload: dict) -> int:
    value = payload.get("chunk_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0  # readiness guarantees a valid non-negative integer; keep sorting total
    return value


def _build_filter(request: ResearchRequest, query: QuerySpec):
    """Compose the Qdrant filter for one question: scope must, all-tags must, any-tags min-should."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue, MinShould

    must = []
    if request.scope != "all":
        must.append(FieldCondition(key="scope", match=MatchValue(value=request.scope)))
    for tag in query.tags_all:
        must.append(FieldCondition(key="tags", match=MatchValue(value=tag)))
    should = [FieldCondition(key="tags", match=MatchValue(value=tag)) for tag in query.tags_any]
    kwargs: dict = {"must": must}
    if should:
        kwargs["should"] = should
        kwargs["min_should"] = MinShould(conditions=should, min_count=1)
    return Filter(**kwargs) if (must or should) else None


def run(request: ResearchRequest, clock=None, collection_name: str | None = None, cache=None, resolved_targets=None) -> dict:
    """Hold mandatory audited readiness independently of optional caching."""
    from . import readiness
    clock = clock or CLOCK
    deadline = _Deadline(config.RESEARCH_TIMEOUT, clock)
    deadline.check()
    collection = collection_name or config.COLLECTION
    resolved_targets = resolve_targets(request) if resolved_targets is None else resolved_targets
    deadline.check()
    token = store.SERVICE_DEADLINE.set(deadline)
    try:
        with readiness.reader(collection, deadline) as validation:
            return _run(request, clock, collection, cache, resolved_targets, deadline, validation)
    finally:
        store.SERVICE_DEADLINE.reset(token)


def _run(request, clock, collection_name, cache, resolved_targets, deadline, validation):
    """Execute one research request and return the evidence bundle.

    Read-only: readiness is inspected, never repaired, and no collection, index,
    or payload is created here. Every service wait is capped by the remaining
    deadline; a failed required probe or an exhausted deadline fails the whole
    request with no partial bundle. With a cache session, candidate lookups run
    before embedding lookups, only missing vectors are obtained, and only
    candidate misses issue searches; staged candidates publish only after the
    whole request succeeds under the still-held reader guard.
    """
    from . import cache as cache_mod

    clock = clock or CLOCK
    deadline.check()
    collection = collection_name or config.COLLECTION
    resolved_targets = resolve_targets(request) if resolved_targets is None else resolved_targets
    if validation is not None:
        validation_count = 1
        if cache is not None:
            cache._budget._deadline_remaining = deadline.remaining
            cache.admit_research_revision(validation.revision)
    elif cache is not None:
        cache._budget._deadline_remaining = deadline.remaining
        cache.verify_index(live_check=lambda: (store.check_research_ready(collection), store.verify_collection_configured(collection)))
    else:
        validation_count = store.count_validation_calls(lambda: store.check_research_ready(collection))
    for resolved in resolved_targets.values():
        deadline.check()
        targets.verify_index(resolved, collection, deadline.service_timeout())
        deadline.check()

    # Distinct normalized questions share retrieval work but keep their original
    # indexes for coverage diagnostics; each contributes one fusion vote.
    distinct: list[QuerySpec] = []
    original_indexes: list[list[int]] = []
    for index, query in enumerate(request.queries):
        if query.dedupe_key in [q.dedupe_key for q in distinct]:
            position = [q.dedupe_key for q in distinct].index(query.dedupe_key)
            original_indexes[position].append(index)
        else:
            distinct.append(query)
            original_indexes.append([index])

    # Planned questions search verbatim by default; explicit expansion keeps the
    # original text as probe 0 and is bounded by the existing probe budget.
    probes_per_question = [
        expansion.probes(query.text, expand=request.expand)[: expansion.MAX_PROBES]
        for query in distinct
    ]
    probe_texts = list(dict.fromkeys(p for probes in probes_per_question for p in probes))

    groups_per_probe = min(3 * request.max_sources, 30)
    window = {
        "groups": groups_per_probe,
        "group_size": request.max_passages_per_source,
        "score_threshold": request.min_score,
    }

    # One grouped search per distinct (probe text, filter) pair: identical probe
    # text with different tag constraints reuses the embedding but not the search.
    work_items: dict[tuple, dict] = {}
    for q_pos, probes in enumerate(probes_per_question):
        resolved = resolved_targets.get(distinct[q_pos].target)
        selections = [None] if resolved is None else [
            {**resolved, "members": [member], "document_ids_resolved": member["document_ids"]}
            for member in resolved["members"] if member["document_ids"]
        ]
        for probe in probes:
            for selection in selections:
                key = (probe, distinct[q_pos].tags_any, distinct[q_pos].tags_all,
                       json.dumps(selection, sort_keys=True) if selection is not None else None)
                query_filter = _build_filter(request, distinct[q_pos])
                if selection is not None:
                    from qdrant_client.models import FieldCondition, Filter, MatchAny
                    conditions = list(query_filter.must or []) if query_filter else []
                    conditions.append(FieldCondition(key="document_id", match=MatchAny(any=selection["document_ids_resolved"])))
                    query_filter = Filter(must=conditions, should=query_filter.should if query_filter else None,
                                          min_should=query_filter.min_should if query_filter else None)
                filters = cache_mod.grouped_filters(request.scope, key[1], key[2])
                if selection is not None:
                    filters["target"] = selection
                    filters["target_contract_version"] = 2
                item = work_items.setdefault(key, {"vector": None, "filter": query_filter,
                                                   "filters": filters, "selection": selection, "questions": []})
                item["questions"].append(q_pos)

    # Candidate tier first: a candidate hit needs no vector and no search. Flat
    # vs grouped fingerprints are distinct, so locator entries never satisfy a
    # grouped lookup.
    cached_payloads: dict[tuple, dict] = {}
    probes_needing_vectors: list[str] = []
    if cache is not None:
        for key, item in work_items.items():
            filters = item["filters"]
            payload = cache.lookup_candidates(
                probe_text=key[0], method=cache_mod.METHOD_GROUPED, filters=filters, window=window
            )
            cached_payloads[key] = payload
            if payload is None and key[0] not in probes_needing_vectors:
                probes_needing_vectors.append(key[0])
    else:
        probes_needing_vectors = list(dict.fromkeys(key[0] for key in work_items))

    deadline.check()
    if cache is not None:
        vector_by_text = cache.embed_probes(
            probes_needing_vectors,
            lambda texts: store.embed(texts, timeout=deadline.remaining()),
        )
        embedding_request_count = cache.diag.embedding_requests
    else:
        vectors = store.embed(probes_needing_vectors, timeout=deadline.remaining()) if probes_needing_vectors else []
        vector_by_text = dict(zip(probes_needing_vectors, vectors))
        embedding_request_count = math.ceil(len(probes_needing_vectors) / config.EMBED_BATCH_SIZE)
    deadline.check()

    grouped_search_count = 0
    # Per effective question: passage key -> best raw score record, per source.
    eligible: list[dict[str, dict]] = [dict() for _ in distinct]
    for key, item in work_items.items():
        payload = cached_payloads.get(key) if cache is not None else None
        if payload is not None:
            groups = cache_mod.materialize_grouped(payload)
        else:
            item["vector"] = vector_by_text[key[0]]
            deadline.check()
            groups = store.query_points_grouped(
                item["vector"],
                groups=groups_per_probe,
                group_size=request.max_passages_per_source,
                query_filter=item["filter"],
                score_threshold=request.min_score,
                timeout=deadline.service_timeout(),
                collection_name=collection,
            )
            grouped_search_count += 1
            if cache is not None:
                filters = item["filters"]
                cache.stage_candidates(
                    probe_text=key[0],
                    method=cache_mod.METHOD_GROUPED,
                    filters=filters,
                    window=window,
                    payload=cache_mod.grouped_payload_from_groups(groups, filters=filters, window=window),
                )
                cache.diag.search_requests += 1
        deadline.check()
        for q_pos in item["questions"]:
            for group in groups:
                document_id = getattr(group, "id", None)
                if not isinstance(document_id, str) or not document_id:
                    raise errors.ResearchIndexNotReadyError("invalid returned group identity; run `library-query index`")
                if item["selection"] is not None and document_id not in item["selection"]["document_ids_resolved"]:
                    raise errors.LibraryQueryError("target_not_indexed", "Search returned an off-target document; run library-query index")
                passages = eligible[q_pos].setdefault(document_id, {})
                for hit in getattr(group, "hits", None) or []:
                    payload = hit.payload or {}
                    if (payload.get("document_id") != document_id or
                            store.invalid_identity_reason({**payload, "text_present":
                                isinstance(payload.get("text"), str) and bool(payload.get("text"))}) is not None):
                        raise errors.ResearchIndexNotReadyError("invalid returned passage; run `library-query index`")
                    if item["selection"] is not None and (hit.payload or {}).get("document_id") != document_id:
                        raise errors.LibraryQueryError("target_not_indexed", "Indexed group identity mismatch; run library-query index")
                    score = _passage_score(hit)
                    # The server applied score_threshold; verify inclusively client-side
                    # so a weak hit can never ride in on a strong sibling passage.
                    if request.min_score is not None and not score >= request.min_score:
                        continue
                    key2 = _passage_key(hit)
                    previous = passages.get(key2)
                    if previous is None or score > previous["score"]:
                        passages[key2] = {"hit": hit, "score": score}

    # Exact-literal leg. Embeddings cannot find `__main__` or `Character.toLowerCase`,
    # so code-like strings in a question are also matched verbatim. Exact chunks join
    # the same eligible pool as semantic ones and still pass the caller's relevance
    # filter. A question without a literal skips this block; a failure here never
    # fails research.
    for q_pos, query in enumerate(distinct):
        wanted = exact.literals(query.text)
        if not wanted:
            continue
        resolved = resolved_targets.get(query.target)
        id_sets = [None] if resolved is None else [
            m["document_ids"] for m in resolved["members"] if m["document_ids"]
        ]
        for ids in id_sets:
            deadline.check()
            query_filter = _build_filter(request, query)
            if ids is not None:
                from qdrant_client.models import FieldCondition, Filter, MatchAny
                base_must = list(query_filter.must or []) if query_filter else []
                query_filter = Filter(
                    must=base_must + [FieldCondition(key="document_id", match=MatchAny(any=ids))],
                    should=query_filter.should if query_filter else None,
                    min_should=query_filter.min_should if query_filter else None,
                )
            try:
                exact_hits = exact.search(store.qdrant(), collection, wanted, query_filter,
                                          timeout=deadline.service_timeout())
            except errors.LibraryQueryError:
                raise
            except Exception:
                exact_hits = []
            deadline.check()
            for hit in exact_hits:
                payload = hit.payload or {}
                document_id = payload.get("document_id")
                if store.invalid_identity_reason({**payload, "text_present": bool(payload.get("text"))}) is not None:
                    continue
                passages = eligible[q_pos].setdefault(document_id, {})
                key2 = _passage_key(hit)
                previous = passages.get(key2)
                if previous is None or hit.score > previous["score"]:
                    passages[key2] = {"hit": hit, "score": hit.score}

    # Per-question source rankings: best eligible passage score desc, then source id asc.
    # A source whose hits all fell below the floor contributes nothing — not even
    # an empty eligibility record.
    rankings: list[list[str]] = []
    eligible_source_counts: list[int] = []
    for q_pos, sources in enumerate(eligible):
        for document_id in [s for s, passages in sources.items() if not passages]:
            del sources[document_id]
        ranking = sorted(
            sources,
            key=lambda source: (-max(p["score"] for p in sources[source].values()), source),
        )
        rankings.append(ranking)
        eligible_source_counts.append(len(ranking))

    fused = rrf_fuse(rankings)
    fused_order = sorted(
        fused,
        key=lambda source: (-fused[source][0], fused[source][1], source),
    )
    pairs = []
    for index, query in enumerate(request.queries):
        effective = next(i for i, originals in enumerate(original_indexes) if index in originals)
        resolved = resolved_targets.get(query.target)
        for member in resolved["members"] if resolved else []:
            pairs.append((index, effective, member))
    reserved = []
    for _, effective, member in pairs:
        ranked = [s for s in rankings[effective] if s in member["document_ids"]]
        if ranked and not any(s in reserved for s in ranked) and len(reserved) < request.max_sources:
            reserved.append(ranked[0])
    selected_set = list(reserved)
    for source in fused_order:
        if source not in selected_set and len(selected_set) < request.max_sources:
            selected_set.append(source)
    selected_sources = [s for s in fused_order if s in selected_set]

    bundle_sources = []
    for source_id in selected_sources:
        # Union eligible passages across questions; attribute the original query
        # indexes so provenance survives fusion.
        by_passage: dict[object, dict] = {}
        for q_pos in range(len(distinct)):
            for key, record in eligible[q_pos].get(source_id, {}).items():
                entry = by_passage.setdefault(
                    key, {"hit": record["hit"], "score": record["score"], "by_question": {}}
                )
                if record["score"] > entry["score"]:
                    entry["score"] = record["score"]
                    entry["hit"] = record["hit"]
                for original in original_indexes[q_pos]:
                    prior = entry["by_question"].get(original)
                    if prior is None or record["score"] > prior:
                        entry["by_question"][original] = record["score"]
        all_ordered = sorted(
            by_passage.values(),
            key=lambda entry: (-entry["score"], _chunk_index((entry["hit"].payload or {}))),
        )
        kept = []
        for index, _, member in pairs:
            if source_id not in member["document_ids"] or any(index in e["by_question"] for e in kept):
                continue
            record = next((e for e in all_ordered if index in e["by_question"]), None)
            if record is not None and len(kept) < request.max_passages_per_source:
                kept.append(record)
        for entry in all_ordered:
            if not any(entry is e for e in kept) and len(kept) < request.max_passages_per_source:
                kept.append(entry)
        ordered = [e for e in all_ordered if any(e is r for r in kept)]
        source_bundle = _source_bundle(source_id, fused[source_id][0], ordered, request)
        if request.schema_version == 2:
            source_bundle["item_ids"] = sorted({m["id"] for _, _, m in pairs
                                                if m["kind"] == "item_ids" and source_id in m["document_ids"]})
        bundle_sources.append(source_bundle)

    # Coverage from the actually emitted passages, not candidate membership.
    selected_indexes = set()
    for source in bundle_sources:
        for passage in source["passages"]:
            selected_indexes.update(passage["query_indexes"])

    query_results = []
    unmatched = []
    for index in range(len(request.queries)):
        effective = next(i for i, originals in enumerate(original_indexes) if index in originals)
        if index in selected_indexes:
            status = "selected"
        elif eligible_source_counts[effective] > 0:
            status = "budget_excluded"
        else:
            status = "no_eligible_evidence"
        evidence_ids = [
            passage["evidence_id"]
            for source in bundle_sources
            for passage in source["passages"]
            if index in passage["query_indexes"]
        ]
        if status != "selected":
            unmatched.append(index)
        query_results.append(
            {
                "query_index": index,
                "status": status,
                "eligible_source_count": eligible_source_counts[effective],
                "selected_evidence_ids": sorted(evidence_ids),
            }
        )
        if request.schema_version == 2:
            target_results = []
            for pair_index, pair_effective, member in pairs:
                if pair_index != index:
                    continue
                count = sum(s in member["document_ids"] for s in rankings[pair_effective])
                ids = sorted(p["evidence_id"] for s in bundle_sources if s["source_id"] in member["document_ids"]
                             for p in s["passages"] if index in p["query_indexes"])
                target_results.append({"kind": member["kind"], "id": member["id"],
                                       "eligible_source_count": count, "selected_evidence_ids": ids,
                                       "status": "selected" if ids else "budget_excluded" if count else "no_eligible_evidence"})
            query_results[-1]["target_results"] = target_results

    if cache is not None:
        # Staged candidates publish only when the whole request succeeded and the
        # ordinary deadline still admits publication work.
        deadline.check()
        cache.finalize()
    deadline.check()

    off_cache = cache_mod.off_diagnostics()
    if cache is None:
        off_cache["embedding_request_count"] = embedding_request_count
        off_cache["search_request_count"] = grouped_search_count
        off_cache["validation_request_count"] = validation_count
    elapsed_ms = max(0, int(round((clock() - deadline._start) * 1000)))
    result = {
        "schema_version": request.schema_version,
        "request": request.to_dict(),
        "retrieval": {
            "embedding_model": config.EMBEDDING_MODEL,
            "collection": collection,
            "expand": request.expand,
            "probe_count": len(probe_texts),
            "embedding_request_count": embedding_request_count,
            "grouped_search_count": grouped_search_count,
            "candidate_groups_per_probe": groups_per_probe,
            "fusion": dict(FUSION),
            "elapsed_ms": elapsed_ms,
        },
        "sources": bundle_sources,
        "query_results": query_results,
        "unmatched_query_indexes": unmatched,
        "cache": cache.diagnostics() if cache is not None else off_cache,
    }
    if request.schema_version == 2:
        result["targets"] = [resolved_targets.get(q.target) for q in request.queries]
    return result


def resolve_targets(request):
    return {t: targets.resolve(t) for t in dict.fromkeys(q.target for q in request.queries if q.target is not None)}


def version2_schema():
    import copy
    request = copy.deepcopy(request_schema())
    request["schema_version"] = {"type": "integer", "const": 2}
    request["target"] = targets.schema()
    request["queries"]["item"]["fields"]["target"] = targets.schema()
    response = copy.deepcopy(response_schema())
    response["schema_version"] = {"type": "integer", "const": 2}
    response["request"]["schema"] = request
    response["sources"]["fields"]["item_ids"] = {"type": "array", "item": {"type": "string"}}
    response["query_results"]["fields"]["target_results"] = {
        "type": "array", "fields": {"kind": {"type": "string", "choices": ["item_ids", "document_ids"]},
        "id": {"type": "string"}, "eligible_source_count": {"type": "integer"},
        "selected_evidence_ids": {"type": "array", "item": {"type": "string"}},
        "status": {"type": "string", "choices": list(COVERAGE_STATUSES)}}}
    response["targets"] = {"type": "array", "semantics": "one resolved effective target or null per question"}
    return {"request": request, "response": response,
            "schema_version": 2, "additional_source_fields": ["item_ids"],
            "additional_query_result_fields": ["target_results"], "additional_response_fields": ["targets"],
            "target_result_fields": ["kind", "id", "eligible_source_count", "selected_evidence_ids", "status"]}


def _passage_bundle(record: dict, request: ResearchRequest) -> dict:
    payload = record["hit"].payload or {}
    full_text = payload.get("text")
    if not isinstance(full_text, str):
        full_text = ""
    passage_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    chunk_index = _chunk_index(payload)
    source_id = payload.get("document_id")
    evidence_id = f"{source_id}:{chunk_index}:{passage_hash}"
    excerpt = full_text[: request.max_excerpt_chars]
    query_scores = [
        {"query_index": index, "score": record["by_question"][index]}
        for index in sorted(record["by_question"])
    ]
    bundle = {
        "evidence_id": evidence_id,
        "chunk_index": chunk_index,
        "passage_hash": passage_hash,
        "text": excerpt,
        "text_truncated": len(full_text) > request.max_excerpt_chars,
        "page": payload.get("page"),
        "location": _location_hint(payload),
        "score": record["score"],
        "query_indexes": sorted(record["by_question"]),
        "query_scores": query_scores,
    }
    if store.has_pdf_location(payload):
        bundle["section"] = section_hint(payload)
    return bundle


def _location_hint(payload: dict) -> dict:
    """Conservative typed anchor: PDF page, EPUB PyMuPDF section, else unknown.

    Markdown and text (including video transcripts) have no structured anchor
    even when a chunk carries a page-like value; timestamps stay verbatim inside
    the text and are never invented here.
    """
    page = payload.get("page")
    if page is None:
        return {"kind": "unknown", "value": None}
    name = str(payload.get("doc_name") or payload.get("path") or payload.get("doc") or "").lower()
    if name.endswith(".epub"):
        return {"kind": "section", "value": page}
    if name.endswith(".pdf"):
        # `value` stays the 1-based PDF page index. The range and printed labels are
        # additive: absent entirely on points indexed before they existed.
        hint = {"kind": "page", "value": page}
        if store.has_pdf_location(payload):
            hint.update({field: payload.get(field) for field in ("page_start", "page_end", "printed_start", "printed_end")})
        return hint
    return {"kind": "unknown", "value": None}


def section_hint(payload: dict) -> dict | None:
    """Outline section of a PDF passage, or None without an outline."""
    path = payload.get("section_path")
    if not path:
        return None
    return {"path": list(path), "start_page": payload.get("section_start_page"), "start_printed": payload.get("section_start_printed")}


def _source_bundle(source_id: str, rrf_score: float, ordered_passages: list, request: ResearchRequest) -> dict:
    first_payload = ordered_passages[0]["hit"].payload or {}
    source = {
        "source_id": source_id,
        "doc": first_payload.get("path") or first_payload.get("doc"),
        "doc_name": first_payload.get("doc_name"),
        "scope": first_payload.get("scope"),
        "type": first_payload.get("type") or first_payload.get("doc_type"),
        "rrf_score": rrf_score,
        "passages": [_passage_bundle(record, request) for record in ordered_passages],
    }
    for field in DOC_METADATA_FIELDS:
        source[field] = first_payload.get(field)
    # Reorder so the advertised contract fields come first and in order.
    ordered = {field: source[field] for field in SOURCE_OUTPUT_FIELDS}
    return ordered


def request_schema() -> dict:
    """The typed version 1 request schema, derived from the same runtime constants
    used for validation — the source `describe --json` advertises."""
    return {
        "schema_version": {"type": "integer", "const": SCHEMA_VERSION, "default": SCHEMA_VERSION},
        "queries": {
            "type": "array",
            "min_items": 1,
            "max_items": MAX_QUESTIONS,
            "item": {
                "type": "object",
                "required": ["text"],
                "fields": {
                    "text": {"type": "string", "min_chars": 1, "max_chars": MAX_QUESTION_CHARS},
                    "tags": {
                        "type": "object",
                        "fields": {
                            "any": tag_array_schema(),
                            "all": tag_array_schema(),
                        },
                        "semantics": "non-empty any = match at least one tag; non-empty all = match every tag; both hold when both supplied; omitted or empty arrays impose no constraint",
                    },
                },
            },
        },
        "scope": {"type": "string", "choices": list(SCOPES), "default": DEFAULT_SCOPE},
        "expand": {"type": "boolean", "default": DEFAULT_EXPAND},
        "min_score": {
            "type": ["number", "null"],
            "range": [-1, 1],
            "finite": True,
            "default": DEFAULT_MIN_SCORE,
            "semantics": "null = ranked candidates with no relevance-floor claim; a floor excludes below-floor passages before source ranking and fusion",
        },
        "max_sources": {"type": "integer", "range": [1, MAX_SOURCES], "default": DEFAULT_MAX_SOURCES},
        "max_passages_per_source": {
            "type": "integer",
            "range": [1, MAX_PASSAGES_PER_SOURCE],
            "default": DEFAULT_MAX_PASSAGES_PER_SOURCE,
        },
        "max_excerpt_chars": {
            "type": "integer",
            "range": [1, MAX_EXCERPT_CHARS],
            "default": DEFAULT_MAX_EXCERPT_CHARS,
        },
        "limits": {
            "max_input_bytes": MAX_INPUT_BYTES,
            "duplicate_object_keys": "rejected",
            "unknown_fields": "rejected",
            "booleans_as_integers": "rejected",
            "non_finite_numbers": "rejected",
        },
    }


def tag_array_schema() -> dict:
    return {
        "type": "array",
        "max_items": MAX_TAGS_PER_ARRAY,
        "item": {"type": "string", "normalization": "trim, collapse internal whitespace, casefold, de-duplicate, sort"},
    }


def response_schema() -> dict:
    """The typed version 1 evidence-bundle schema for `describe --json`."""
    return {
        "schema_version": {"type": "integer", "const": SCHEMA_VERSION},
        "request": {"type": "object", "schema": request_schema()},
        "retrieval": {
            "type": "object",
            "fields": {
                "embedding_model": {"type": "string"},
                "collection": {"type": "string"},
                "expand": {"type": "boolean"},
                "probe_count": {"type": "integer"},
                "embedding_request_count": {"type": "integer"},
                "grouped_search_count": {"type": "integer"},
                "candidate_groups_per_probe": {"type": "integer"},
                "fusion": {"type": "object", "const": dict(FUSION)},
                "elapsed_ms": {"type": "integer", "min": 0},
            },
        },
        "sources": {
            "type": "array",
            "fields": {field: {"type": "string", "nullable": True} for field in SOURCE_OUTPUT_FIELDS[:-2]}
            | {
                "source_id": {"type": "string"},
                "rrf_score": {"type": "number", "semantics": "ranking value, not a relevance probability"},
                "passages": {
                    "type": "array",
                    "fields": {
                        "evidence_id": {
                            "type": "string",
                            "format": "<source_id>:<chunk_index>:<passage_hash>",
                        },
                        "chunk_index": {"type": "integer", "min": 0},
                        "passage_hash": {"type": "string", "format": "lowercase sha256 of the full stored UTF-8 passage, before excerpt truncation"},
                        "text": {"type": "string", "semantics": "verbatim prefix of the stored passage, bounded by max_excerpt_chars"},
                        "text_truncated": {"type": "boolean"},
                        "page": {"type": ["integer", "null"]},
                        "location": {
                            "type": "object",
                            "fields": {
                                "kind": {"type": "string", "choices": ["page", "section", "unknown"]},
                                "value": {"type": ["integer", "null"], "semantics": "1-based PDF page index, not the printed page number"},
                                "page_start": {"type": ["integer", "null"], "semantics": "PDF-only, additive, absent on older points"},
                                "page_end": {"type": ["integer", "null"], "semantics": "PDF-only, additive, absent on older points"},
                                "printed_start": {"type": ["string", "null"], "semantics": "PDF-only; null without a real label scheme"},
                                "printed_end": {"type": ["string", "null"], "semantics": "PDF-only; null without a real label scheme"},
                            },
                        },
                        "section": {
                            "type": ["object", "null"],
                            "fields": {
                                "path": {"type": "array", "item": {"type": "string"}},
                                "start_page": {"type": "integer"},
                                "start_printed": {"type": ["string", "null"]},
                            },
                            "semantics": "PDF-only, present only on points indexed with location fields: outline section containing page_start; null without an outline",
                        },
                        "score": {"type": "number", "semantics": "best raw semantic score across associated questions"},
                        "query_indexes": {"type": "array", "item": {"type": "integer"}, "semantics": "sorted, de-duplicated zero-based indexes of supplied questions"},
                        "query_scores": {
                            "type": "array",
                            "item": {"type": "object", "fields": {"query_index": {"type": "integer"}, "score": {"type": "number"}}},
                        },
                    },
                },
            },
        },
        "query_results": {
            "type": "array",
            "fields": {
                "query_index": {"type": "integer"},
                "status": {"type": "string", "choices": list(COVERAGE_STATUSES)},
                "eligible_source_count": {"type": "integer", "min": 0},
                "selected_evidence_ids": {"type": "array", "item": {"type": "string"}},
            },
        },
        "unmatched_query_indexes": {"type": "array", "item": {"type": "integer"}},
        "cache": {
            "type": "object",
            "semantics": "additive per-invocation cache diagnostics sibling; see the top-level cache contract in describe --json (modes, budget, bypass reasons, typed fields)",
        },
        "semantics": {
            "coverage": "computed from emitted passages; no_eligible_evidence = nothing survived scope/tag/floor filters; budget_excluded = eligible but cut by the final budget",
            "determinism": "identical candidate identities and raw scores produce identical ordering regardless of retrieval completion order; remote embeddings/approximate search repeatability is measured, not promised",
        },
    }
