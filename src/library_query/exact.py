"""Exact-text lookup for code-like literals that embeddings cannot find.

A dense vector places ``__main__`` near "main", not near the string itself. That is
the right trade for prose and the wrong one for identifiers, operators and quoted
snippets. This module adds a literal leg: Qdrant's full-text index narrows the
candidates, then a case-sensitive substring check on the returned text decides.
The index is a candidate filter only; the substring check is the proof.

The leg is additive and fail-open. A question with no literal never touches it. A
missing index or a Qdrant error yields no exact hits, and semantic retrieval answers
as before.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

TEXT_FIELD = "text"
#: Exact hits outrank every cosine score (cosine <= 1) while staying a valid score.
EXACT_SCORE = 1.0
#: Distinct literals taken from one question.
MAX_LITERALS = 3
#: Candidate chunks scanned per literal before the substring check.
SCAN_CAP = 6000
#: Exact hits returned per question, best first.
MAX_HITS = 40
#: Best chunks kept per document, so one book cannot crowd out the others.
PER_DOCUMENT = 3
PAGE = 256

_BACKTICK = re.compile(r"`([^`\n]{2,120})`")
_DOUBLE = re.compile(r'"([^"\n]{2,120})"')
_SINGLE_NOSPACE = re.compile(r"(?<![\w])'([^'\s]{2,120})'(?![\w])")
# Unquoted code-looking tokens.
_CODE_TOKEN = re.compile(
    r"""(?x)
    (?=[A-Za-z0-9_]*[A-Za-z0-9])[A-Za-z0-9_]*_[A-Za-z0-9_]*             # snake_case, __dunder__
    | [A-Za-z0-9_]{2,}(?:\.|::|->)[A-Za-z_][A-Za-z0-9_]*               # Character.toLowerCase, std::map
    | \b[a-z]+[A-Z][A-Za-z0-9]*\b                                      # camelCase
    | \b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b                              # PascalCase
    | [A-Za-z_][A-Za-z0-9_]*\(\)                                       # call()
    | [#@$][A-Za-z_][A-Za-z0-9_]+                                      # #include, @Override, $HOME
    """
)


def literals(question: str) -> list[str]:
    """Return up to MAX_LITERALS exact strings worth matching verbatim.

    Quoted or backticked spans win. Unquoted tokens count only when they look like
    code, so ordinary prose never reaches the exact leg.
    """
    found: list[str] = []

    def add(value: str) -> None:
        value = value.strip().rstrip(".,;:!?")
        if len(value) < 2 or not re.search(r"[A-Za-z0-9]", value) or value in found:
            return
        found.append(value)

    for pattern in (_BACKTICK, _DOUBLE, _SINGLE_NOSPACE):
        for match in pattern.finditer(question):
            add(match.group(1))
    # Mask quoted spans so their pieces are not counted twice as loose tokens.
    loose = _SINGLE_NOSPACE.sub(" ", _DOUBLE.sub(" ", _BACKTICK.sub(" ", question)))
    for match in _CODE_TOKEN.finditer(loose):
        add(match.group(0))
    # Drop a literal wholly contained in a longer one: `main` inside `__main__`.
    found = [lit for lit in found if not any(lit != other and lit in other for other in found)]
    return found[:MAX_LITERALS]


def is_ready(client, collection: str) -> bool:
    """True when the collection has a text index with phrase matching."""
    try:
        schema = getattr(client.get_collection(collection_name=collection), "payload_schema", None) or {}
        entry = schema.get(TEXT_FIELD) if isinstance(schema, dict) else None
        params = getattr(entry, "params", None)
        return entry is not None and str(getattr(entry, "data_type", "")).endswith("text") and \
            getattr(params, "phrase_matching", False) is True
    except Exception:
        return False


def ensure_text_index(client, collection: str) -> str:
    """Create the text index once. Returns 'present', 'created' or 'unavailable'.

    Never raises: an index failure must not break indexing. The index lives on disk
    so it does not add to Qdrant's resident memory.
    """
    from qdrant_client.models import TextIndexParams, TextIndexType, TokenizerType

    if is_ready(client, collection):
        return "present"
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=TEXT_FIELD,
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                lowercase=True,
                phrase_matching=True,
                on_disk=True,
            ),
            wait=True,
        )
        return "created"
    except Exception:
        return "unavailable"


def _with_phrase(base_filter, phrase: str):
    from qdrant_client.models import FieldCondition, Filter, MatchPhrase

    condition = FieldCondition(key=TEXT_FIELD, match=MatchPhrase(phrase=phrase))
    if base_filter is None:
        return Filter(must=[condition])
    return Filter(
        must=[*(base_filter.must or []), condition],
        should=base_filter.should,
        min_should=base_filter.min_should,
        must_not=base_filter.must_not,
    )


def search(client, collection: str, wanted: list[str], base_filter=None, *,
           timeout: int | None = None, max_hits: int = MAX_HITS) -> list:
    """Chunks whose text contains a wanted literal verbatim, best first.

    Ranking: distinct literals present, then total occurrences, then a stable
    document and chunk order. Each hit carries the full Qdrant payload and
    EXACT_SCORE minus a tiny rank offset so the order survives a score sort.
    Returns [] when the index is not ready. Qdrant transport errors from a ready
    index are re-raised as LibraryQueryError subclasses by the caller's policy.
    """
    if not wanted or not is_ready(client, collection):
        return []
    chunks: dict[object, dict] = {}
    for literal in wanted:
        scanned, offset = 0, None
        query_filter = _with_phrase(base_filter, literal)
        while scanned < SCAN_CAP:
            points, offset = client.scroll(
                collection_name=collection, scroll_filter=query_filter, limit=PAGE,
                offset=offset, with_payload=True, with_vectors=False, timeout=timeout,
            )
            for point in points:
                scanned += 1
                text = (point.payload or {}).get(TEXT_FIELD)
                count = text.count(literal) if isinstance(text, str) else 0
                if count:
                    entry = chunks.setdefault(point.id, {"point": point, "literals": 0, "count": 0})
                    entry["literals"] += 1
                    entry["count"] += count
            if offset is None:
                break
    ranked = sorted(
        chunks.values(),
        key=lambda e: (-e["literals"], -e["count"], str(e["point"].payload.get("document_id")),
                       e["point"].payload.get("chunk_index") or 0),
    )
    # One book must not fill every slot: keep a few best chunks per document so each
    # work that contains the literal is represented.
    per_document: dict[object, int] = {}
    ordered = []
    for entry in ranked:
        document_id = entry["point"].payload.get("document_id")
        if per_document.get(document_id, 0) >= PER_DOCUMENT:
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        ordered.append(entry)
        if len(ordered) >= max_hits:
            break
    return [
        SimpleNamespace(id=e["point"].id, payload=e["point"].payload,
                        score=EXACT_SCORE - rank * 1e-4, exact_matches=e["count"])
        for rank, e in enumerate(ordered)
    ]


def snippet(text: str, literal: str, width: int = 240) -> str:
    """A window of `text` centred on the first occurrence of `literal`."""
    at = text.find(literal)
    if at < 0:
        return text[:width]
    start = max(0, at - width // 2)
    end = min(len(text), at + len(literal) + width // 2)
    return ("…" if start else "") + " ".join(text[start:end].split()) + ("…" if end < len(text) else "")
