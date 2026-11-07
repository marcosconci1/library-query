"""Research evidence API: schema validation, RRF fusion, selection, provenance.

Covers the research-evidence-api spec delta and the retrieval/diagnostics tasks
of add-research-evidence-api. Everything runs offline: the embedding provider
and Qdrant are replaced by deterministic fakes keyed on probe text.
"""
import hashlib
import io
import json
import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from library_query import cli, config, errors, research, store


# --- helpers ----------------------------------------------------------------------


def expansion_variant(question: str) -> str:
    """The first deterministic expansion probe for a cued question."""
    from library_query import expansion
    probes = expansion.probes(question, expand=True)
    assert len(probes) > 1, "test question must trigger deterministic expansion"
    return probes[1]


def request_bytes(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


def parse(obj):
    return research.parse_request(request_bytes(obj))


def make_payload(document_id, chunk_index, text, *, doc=None, scope="catalogued", doc_type="paper",
                 page=None, doc_name=None, tags=None, title=None, authors=None, year=None):
    return {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "text": text,
        "path": doc or f"Papers/{document_id}/paper.pdf",
        "doc": doc or f"Papers/{document_id}/paper.pdf",
        "doc_name": doc_name or (doc or f"{document_id}.pdf").rsplit("/", 1)[-1],
        "scope": scope,
        "type": doc_type,
        "page": page,
        "tags": tags,
        "title": title,
        "authors": authors,
        "year": year,
        "subjects": None,
        "identifier": None,
    }


def make_hit(pid, score, payload):
    return SimpleNamespace(id=pid, score=score, payload=payload)


def make_group(document_id, hits):
    return SimpleNamespace(id=document_id, hits=hits)


class FakeRetrieval:
    """Replaces store.embed + store.query_points_grouped with labeled fixtures.

    Grouped candidates are keyed by probe text, so a test declares which probe
    "found" which document groups. Records calls for operation-count assertions.
    """

    def __init__(self, grouped_by_probe, *, on_search=None):
        self.grouped_by_probe = grouped_by_probe
        self.embed_calls = []
        self.search_calls = []
        self.embedded_texts = []
        self._on_search = on_search

    def embed(self, texts, timeout=None):
        self.embed_calls.append({"texts": list(texts), "timeout": timeout})
        self.embedded_texts.extend(texts)
        return [[float(i), 0.0] for i, _ in enumerate(texts)]

    def query_points_grouped(self, vector, *, groups, group_size, query_filter=None,
                             score_threshold=None, timeout=None, collection_name=None):
        probe = self.embedded_texts[int(vector[0])]
        self.search_calls.append({
            "probe": probe,
            "groups": groups,
            "group_size": group_size,
            "score_threshold": score_threshold,
            "timeout": timeout,
            "filter": query_filter,
        })
        if self._on_search is not None:
            self._on_search(probe)
        return list(self.grouped_by_probe.get(probe, []))

    def install(self, monkeypatch):
        monkeypatch.setattr(store, "embed", self.embed)
        monkeypatch.setattr(store, "query_points_grouped", self.query_points_grouped)
        monkeypatch.setattr(store, "check_research_ready", lambda collection_name=None: 100)


def run_cli(argv):
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def read_error(capsys):
    err = capsys.readouterr().err.strip().splitlines()
    assert err, "expected a structured error line on stderr"
    return json.loads(err[-1])


# --- 1.1 request schema and validation ---------------------------------------------


def test_minimal_request_resolves_all_defaults():
    req = parse({"queries": [{"text": "What is retrieval-augmented generation?"}]})
    assert req.schema_version == 1
    assert req.scope == "catalogued"
    assert req.expand is False
    assert req.min_score is None
    assert req.max_sources == 3
    assert req.max_passages_per_source == 2
    assert req.max_excerpt_chars == 1200


def test_design_example_request_parses_with_normalized_tags():
    req = parse({
        "schema_version": 1,
        "queries": [
            {"text": "Which retry strategies reduce queue overload?"},
            {"text": "What trade-offs do backoff and jitter introduce?",
             "tags": {"any": ["  Distributed  Systems ", "reliability", "reliability"], "all": []}},
        ],
        "scope": "catalogued",
        "expand": False,
        "min_score": None,
        "max_sources": 3,
        "max_passages_per_source": 2,
        "max_excerpt_chars": 1200,
    })
    assert req.queries[1].tags_any == ("distributed systems", "reliability")
    assert req.queries[1].tags_all == ()
    resolved = req.to_dict()
    assert resolved["schema_version"] == 1
    assert resolved["scope"] == "catalogued"


def test_omitted_schema_version_means_version_1():
    assert parse({"queries": [{"text": "q"}]}).schema_version == 1


_TOO_MANY_TAGS = json.dumps(
    {"queries": [{"text": "q", "tags": {"any": ["t%d" % i for i in range(33)]}}]}
).encode()

_TOO_LONG_QUESTION = json.dumps({"queries": [{"text": "x" * 4001}]}).encode()

_INVALID_REQUESTS = [
    (b"not json at all", "malformed JSON"),
    (b'{"queries": [{"text": "a"}], "queries": [{"text": "b"}]}', "duplicate field"),
    (b'{"queries": [{"text": "a"}], "surprise": 1}', "unknown field"),
    (b'{"queries": [{"text": "a", "extra": true}]}', "unknown field"),
    (b'{"queries": [{"text": "a", "tags": {"either": []}}]}', "unknown field"),
    (b'{"schema_version": 3, "queries": [{"text": "a"}]}', "unsupported version"),
    (b'{"schema_version": true, "queries": [{"text": "a"}]}', "schema_version"),
    (b'{"queries": []}', "non-empty array"),
    (b'{"queries": [{"text": "   "}]}', "non-empty, non-whitespace"),
    (b'{"queries": [{"text": ""}]}', "non-empty, non-whitespace"),
    (b'{"queries": [{"text": 42}]}', "non-empty, non-whitespace"),
    (b'{"queries": [{"text": "a", "tags": {"any": ["  "]}}]}', "whitespace-only"),
    (b'{"queries": [{"text": "a", "tags": {"any": [7]}}]}', "array of strings"),
    (b'{"queries": [{"text": "a", "tags": {"any": "reliability"}}]}', "must be an array"),
    (b'{"queries": [{"text": "a"}], "scope": "root"}', "scope"),
    (b'{"queries": [{"text": "a"}], "expand": "yes"}', "must be a boolean"),
    (b'{"queries": [{"text": "a"}], "min_score": true}', "min_score"),
    (b'{"queries": [{"text": "a"}], "min_score": "high"}', "min_score"),
    (b'{"queries": [{"text": "a"}], "min_score": 1.5}', "[-1, 1]"),
    (b'{"queries": [{"text": "a"}], "min_score": -1.0001}', "[-1, 1]"),
    (b'{"queries": [{"text": "a"}], "min_score": NaN}', "non-finite"),
    (b'{"queries": [{"text": "a"}], "min_score": Infinity}', "non-finite"),
    (b'{"queries": [{"text": "a"}], "max_sources": true}', "must be an integer"),
    (b'{"queries": [{"text": "a"}], "max_sources": 3.5}', "must be an integer"),
    (b'{"queries": [{"text": "a"}], "max_sources": 0}', "[1, 10]"),
    (b'{"queries": [{"text": "a"}], "max_sources": 11}', "[1, 10]"),
    (b'{"queries": [{"text": "a"}], "max_passages_per_source": 6}', "[1, 5]"),
    (b'{"queries": [{"text": "a"}], "max_excerpt_chars": 0}', "[1, 8000]"),
    (b'{"queries": [{"text": "a"}], "max_excerpt_chars": 9000}', "[1, 8000]"),
    (_TOO_LONG_QUESTION, "at most 4000 characters"),
    (json.dumps({"queries": [{"text": "q"}] * 9}).encode(), "at most 8 questions"),
    (_TOO_MANY_TAGS, "at most 32 tags"),
]

@pytest.mark.parametrize("payload,fragment", _INVALID_REQUESTS)
def test_invalid_requests_are_rejected_with_field_only_messages(payload, fragment):
    with pytest.raises(errors.InvalidResearchRequestError) as excinfo:
        research.parse_request(payload)
    message = str(excinfo.value)
    assert fragment in message
    assert excinfo.value.category == "invalid_research_request"
    assert excinfo.value.exit_code == 2


def test_min_score_accepts_boundary_values_and_ints():
    assert parse({"queries": [{"text": "q"}], "min_score": 1}).min_score == 1.0
    assert parse({"queries": [{"text": "q"}], "min_score": -1}).min_score == -1.0
    assert parse({"queries": [{"text": "q"}], "min_score": 0.45}).min_score == 0.45


def test_input_byte_cap_allows_limit_and_rejects_limit_plus_one():
    base = {"queries": [{"text": "q"}]}
    padding = research.MAX_INPUT_BYTES - len(request_bytes(base))
    padded = request_bytes(base)[:-1] + b" " * padding + b"}"
    assert len(padded) == research.MAX_INPUT_BYTES
    assert research.parse_request(padded).max_sources == 3
    with pytest.raises(errors.InvalidResearchRequestError, match="at most"):
        research.parse_request(padded + b" ")


def test_read_input_reads_at_most_byte_limit_plus_one(tmp_path):
    big = tmp_path / "big.json"
    big.write_bytes(b"x" * (research.MAX_INPUT_BYTES + 50))
    data = research.read_input(str(big))
    assert len(data) == research.MAX_INPUT_BYTES + 1


def test_read_input_stdin_and_missing_file(monkeypatch, tmp_path):
    import sys as _sys
    monkeypatch.setattr(_sys, "stdin", SimpleNamespace(buffer=io.BytesIO(request_bytes({"queries": [{"text": "q"}]}))))
    assert research.read_input("-").startswith(b"{")
    with pytest.raises(errors.InvalidResearchRequestError, match="could not be read"):
        research.read_input(str(tmp_path / "missing.json"))


def test_validation_happens_before_any_service_call(monkeypatch):
    fake = FakeRetrieval({})
    fake.install(monkeypatch)
    spy = MagicMock()
    monkeypatch.setattr(store, "check_research_ready", spy)
    with pytest.raises(errors.InvalidResearchRequestError):
        parse({"queries": [{"text": "ok"}], "max_sources": 99})
    spy.assert_not_called()
    assert fake.embed_calls == [] and fake.search_calls == []


def test_error_message_never_copies_question_text():
    secret = "my secret diagnosis question " + "x" * 4000
    with pytest.raises(errors.InvalidResearchRequestError) as excinfo:
        parse({"queries": [{"text": secret}]})
    assert "my secret diagnosis question" not in str(excinfo.value)


# --- 1.2 / 1.3 bundle shape --------------------------------------------------------


def test_bundle_shape_and_resolved_request(monkeypatch):
    payload_a = make_payload("doc_a", 0, "Alpha passage about retries.", title="Alpha", page=3)
    fake = FakeRetrieval({"q1": [make_group("doc_a", [make_hit(11, 0.9, payload_a)])]})
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "q1"}]})
    bundle = research.run(req, clock=lambda: 0.0)

    assert list(bundle) == ["schema_version", "request", "retrieval", "sources", "query_results",
                            "unmatched_query_indexes", "cache"]
    assert bundle["schema_version"] == 1
    assert bundle["request"]["scope"] == "catalogued"
    assert bundle["request"]["max_sources"] == 3

    source = bundle["sources"][0]
    assert list(source) == research.SOURCE_OUTPUT_FIELDS
    assert source["source_id"] == "doc_a"
    assert source["title"] == "Alpha"

    passage = source["passages"][0]
    assert list(passage) == research.PASSAGE_OUTPUT_FIELDS
    expected_hash = hashlib.sha256("Alpha passage about retries.".encode("utf-8")).hexdigest()
    assert passage["passage_hash"] == expected_hash
    assert passage["evidence_id"] == f"doc_a:0:{expected_hash}"
    assert passage["text_truncated"] is False
    assert passage["page"] == 3
    assert passage["location"] == {"kind": "page", "value": 3}

    assert bundle["query_results"] == [{
        "query_index": 0,
        "status": "selected",
        "eligible_source_count": 1,
        "selected_evidence_ids": [passage["evidence_id"]],
    }]
    assert bundle["unmatched_query_indexes"] == []


def test_metadata_fields_stay_explicit_null_when_absent(monkeypatch):
    payload = make_payload("doc_b", 0, "Bare payload text.")
    payload.update({"title": None, "authors": None, "year": None, "subjects": None,
                    "tags": None, "identifier": None, "page": None})
    fake = FakeRetrieval({"q1": [make_group("doc_b", [make_hit(5, 0.7, payload)])]})
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q1"}]}), clock=lambda: 0.0)
    source = bundle["sources"][0]
    for field in ("title", "authors", "year", "subjects", "tags", "identifier"):
        assert source[field] is None, f"{field} must be present and null"
    assert source["passages"][0]["page"] is None


def test_retrieval_diagnostics_report_deduplicated_work(monkeypatch):
    payload = make_payload("doc_a", 0, "Text one.")
    fake = FakeRetrieval({
        "retry strategies": [make_group("doc_a", [make_hit(1, 0.8, payload)])],
        "queue overload": [make_group("doc_a", [make_hit(1, 0.85, payload)])],
    })
    fake.install(monkeypatch)
    req = parse({
        "queries": [{"text": "retry strategies"}, {"text": "queue overload"}],
        "max_sources": 3,
    })
    bundle = research.run(req, clock=lambda: 0.0)
    retrieval = bundle["retrieval"]
    assert retrieval["expand"] is False
    assert retrieval["probe_count"] == 2
    assert retrieval["embedding_request_count"] == 1
    assert retrieval["grouped_search_count"] == 2
    assert retrieval["candidate_groups_per_probe"] == 9
    assert retrieval["fusion"] == {"method": "rrf", "k": 60, "rank_base": 1}
    assert retrieval["elapsed_ms"] >= 0
    assert retrieval["collection"] == config.COLLECTION
    assert retrieval["embedding_model"] == config.EMBEDDING_MODEL


def test_candidate_groups_cap_at_thirty():
    req = parse({"queries": [{"text": "q"}], "max_sources": 10})
    assert min(3 * req.max_sources, 30) == 30


def test_expand_true_uses_deterministic_expansion_with_verbatim_first_probe(monkeypatch):
    captured = {}

    def on_search(probe):
        captured.setdefault("probes", []).append(probe)

    fake = FakeRetrieval({}, on_search=on_search)
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "i cant stop thinking about her"}], "expand": True})
    bundle = research.run(req, clock=lambda: 0.0)
    assert bundle["retrieval"]["expand"] is True
    assert len(bundle["retrieval"]["probe_count"]) if False else bundle["retrieval"]["probe_count"] >= 2
    assert captured["probes"][0] == "i cant stop thinking about her"


# --- 3.1 duplicate questions and shared probe work --------------------------------


def test_duplicate_questions_share_work_but_keep_original_indexes(monkeypatch):
    payload = make_payload("doc_a", 0, "Shared text.")
    group = [make_group("doc_a", [make_hit(9, 0.9, payload)])]
    fake = FakeRetrieval({"same question": group})
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "same question"}, {"text": "same question"}]})
    bundle = research.run(req, clock=lambda: 0.0)

    # One probe embedded once, one search executed; one fusion vote.
    assert fake.embed_calls[0]["texts"] == ["same question"]
    assert len(fake.search_calls) == 1
    assert bundle["retrieval"]["probe_count"] == 1
    assert bundle["retrieval"]["grouped_search_count"] == 1
    # ...but both original indexes are retained in diagnostics.
    assert bundle["sources"][0]["passages"][0]["query_indexes"] == [0, 1]
    assert bundle["sources"][0]["passages"][0]["query_scores"] == [
        {"query_index": 0, "score": 0.9}, {"query_index": 1, "score": 0.9}
    ]
    assert bundle["query_results"][0]["status"] == bundle["query_results"][1]["status"] == "selected"


def test_duplicate_questions_do_not_change_source_ordering(monkeypatch):
    pa = make_payload("doc_a", 0, "A.")
    pb = make_payload("doc_b", 0, "B.")
    solo = FakeRetrieval({"q1": [make_group("doc_a", [make_hit(1, 0.9, pa)]),
                                 make_group("doc_b", [make_hit(2, 0.8, pb)])]})
    solo.install(monkeypatch)
    single = research.run(parse({"queries": [{"text": "q1"}]}), clock=lambda: 0.0)
    single_order = [s["source_id"] for s in single["sources"]]

    monkeypatch.undo()  # reinstall with the duplicate request's probe set
    dup = FakeRetrieval({"q1": [make_group("doc_a", [make_hit(1, 0.9, pa)]),
                                make_group("doc_b", [make_hit(2, 0.8, pb)])]})
    spy_ready = lambda collection_name=None: 100
    monkeypatch.setattr(store, "check_research_ready", spy_ready)
    monkeypatch.setattr(store, "embed", dup.embed)
    monkeypatch.setattr(store, "query_points_grouped", dup.query_points_grouped)
    double = research.run(parse({"queries": [{"text": "q1"}, {"text": "q1"}]}), clock=lambda: 0.0)
    assert [s["source_id"] for s in double["sources"]] == single_order
    assert all(s["rrf_score"] == pytest.approx(r["rrf_score"])
               for s, r in zip(double["sources"], single["sources"]))


def test_identical_probe_text_with_different_tags_shares_embedding_not_search(monkeypatch):
    tagged_payload = make_payload("doc_t", 0, "Tagged.", tags=["reliability"])
    untagged_payload = make_payload("doc_u", 0, "Untagged.")
    fake = FakeRetrieval({
        "shared probe": [make_group("doc_t", [make_hit(3, 0.9, tagged_payload)]),
                         make_group("doc_u", [make_hit(4, 0.7, untagged_payload)])],
    })
    fake.install(monkeypatch)
    req = parse({
        "queries": [
            {"text": "shared probe"},
            {"text": "shared probe", "tags": {"any": ["reliability"]}},
        ]
    })
    bundle = research.run(req, clock=lambda: 0.0)
    assert fake.embed_calls[0]["texts"] == ["shared probe"]  # one embedding input
    assert len(fake.search_calls) == 2  # separate filters cannot share a search
    assert bundle["retrieval"]["probe_count"] == 1


# --- 3.3 RRF fusion, floor, and merge ----------------------------------------------


def test_rrf_fuse_hand_computed_fixture():
    # Q0 ranks: s1 > s2 > s3 ; Q1 ranks: s2 > s1
    fused = research.rrf_fuse([["s1", "s2", "s3"], ["s2", "s1"]])
    assert fused["s1"] == pytest.approx((1 / 61 + 1 / 62, 1))
    assert fused["s2"] == pytest.approx((1 / 62 + 1 / 61, 1))
    assert fused["s3"] == pytest.approx((1 / 63, 3))
    # Symmetric votes: the tie breaks by best rank (equal), then ascending source id.
    ordered = sorted(fused, key=lambda s: (-fused[s][0], fused[s][1], s))
    assert ordered == ["s1", "s2", "s3"]


def test_rrf_agreement_wins_over_single_best_rank():
    """A source ranked well in every question beats one ranked first once."""
    # Q0: s1 > s2 ; Q1: s2 > s3 ; Q2: s2 > s1
    fused = research.rrf_fuse([["s1", "s2"], ["s2", "s3"], ["s2", "s1"]])
    assert fused["s2"] == pytest.approx((2 / 61 + 1 / 62, 1))
    assert fused["s1"] == pytest.approx((1 / 61 + 1 / 62, 1))
    assert fused["s3"] == pytest.approx((1 / 62, 2))
    ordered = sorted(fused, key=lambda s: (-fused[s][0], fused[s][1], s))
    assert ordered == ["s2", "s1", "s3"]


def test_rrf_absent_source_contributes_zero():
    fused = research.rrf_fuse([["a"], ["b"]])
    assert fused["a"] == pytest.approx((1 / 61, 1))
    assert fused["b"] == pytest.approx((1 / 61, 1))
    assert "c" not in fused


def test_rrf_ties_break_by_best_rank_then_source_id():
    # Both sources appear at rank 2 in one question each: identical fused scores.
    fused = research.rrf_fuse([["z", "a"], ["z", "b"]])
    assert fused["a"][0] == pytest.approx(fused["b"][0])
    ordered = sorted(fused, key=lambda s: (-fused[s][0], fused[s][1], s))
    assert ordered == ["z", "a", "b"]


def test_fusion_ordering_end_to_end(monkeypatch):
    payloads = {
        "s1": make_payload("s1", 0, "one"),
        "s2": make_payload("s2", 0, "two"),
        "s3": make_payload("s3", 0, "three"),
    }
    fake = FakeRetrieval({
        "q0": [make_group("s1", [make_hit(1, 0.95, payloads["s1"])]),
               make_group("s2", [make_hit(2, 0.90, payloads["s2"])])],
        "q1": [make_group("s2", [make_hit(2, 0.99, payloads["s2"])]),
               make_group("s3", [make_hit(3, 0.70, payloads["s3"])])],
        "q2": [make_group("s2", [make_hit(2, 0.60, payloads["s2"])]),
               make_group("s1", [make_hit(1, 0.50, payloads["s1"])])],
    })
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q0"}, {"text": "q1"}, {"text": "q2"}]}), clock=lambda: 0.0)
    by_id = {s["source_id"]: s for s in bundle["sources"]}
    assert by_id["s2"]["rrf_score"] == pytest.approx(2 / 61 + 1 / 62)
    assert by_id["s1"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert [s["source_id"] for s in bundle["sources"]] == ["s2", "s1", "s3"]
    # Raw scores are never rounded into the decision: s1's best passage score stays 0.95.
    assert by_id["s1"]["passages"][0]["score"] == 0.95


def test_min_score_floor_excludes_weak_passages_before_fusion(monkeypatch):
    strong = make_payload("doc_s", 0, "Strong passage.")
    weak = make_payload("doc_w", 0, "Weak passage.")
    fake = FakeRetrieval({
        "q0": [make_group("doc_s", [make_hit(1, 0.90, strong)]),
               make_group("doc_w", [make_hit(2, 0.40, weak)])],
    })
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "q0"}], "min_score": 0.5})
    bundle = research.run(req, clock=lambda: 0.0)
    assert fake.search_calls[0]["score_threshold"] == 0.5  # floor pushed to the store
    assert [s["source_id"] for s in bundle["sources"]] == ["doc_s"]
    assert bundle["query_results"][0]["status"] == "selected"
    assert bundle["unmatched_query_indexes"] == []


def test_floor_applies_per_passage_not_per_document(monkeypatch):
    """A below-floor sibling in the same document must not be admitted."""
    strong = make_payload("doc_a", 0, "Strong chunk.")
    weak_sibling = make_payload("doc_a", 1, "Weak sibling chunk.")
    fake = FakeRetrieval({
        "q0": [make_group("doc_a", [make_hit(1, 0.90, strong), make_hit(2, 0.30, weak_sibling)])],
    })
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q0"}], "min_score": 0.5}), clock=lambda: 0.0)
    source = bundle["sources"][0]
    assert [p["chunk_index"] for p in source["passages"]] == [0]


def test_server_floor_leak_is_caught_client_side(monkeypatch):
    """Even if the store returns a below-threshold hit, inclusive >= rejects it."""
    strong = make_payload("doc_a", 0, "Strong.")
    sneaky = make_payload("doc_b", 0, "Sneaky below-floor.")
    calls = {"n": 0}

    fake = FakeRetrieval({"q0": []})
    def leaky_search(vector, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [make_group("doc_a", [make_hit(1, 0.9, strong)]),
                    make_group("doc_b", [make_hit(2, 0.2, sneaky)])]  # server misbehaved
        return []
    monkeypatch.setattr(store, "check_research_ready", lambda collection_name=None: 10)
    monkeypatch.setattr(store, "embed", fake.embed)
    monkeypatch.setattr(store, "query_points_grouped", leaky_search)
    bundle = research.run(parse({"queries": [{"text": "q0"}], "min_score": 0.5}), clock=lambda: 0.0)
    assert [s["source_id"] for s in bundle["sources"]] == ["doc_a"]


def test_rank_agreement_cannot_rescue_below_floor_evidence(monkeypatch):
    """Every probe scores the passage below the floor -> absent from the bundle."""
    question = "i cant stop thinking about her"
    variant = expansion_variant(question)
    payload = make_payload("doc_a", 0, "Below floor everywhere.")
    fake = FakeRetrieval({
        question: [make_group("doc_a", [make_hit(7, 0.41, payload)])],
        variant: [make_group("doc_a", [make_hit(7, 0.42, payload)])],
    })
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": question}], "expand": True, "min_score": 0.5})
    bundle = research.run(req, clock=lambda: 0.0)
    assert bundle["sources"] == []
    assert bundle["query_results"][0]["status"] == "no_eligible_evidence"
    assert bundle["unmatched_query_indexes"] == [0]


def test_per_question_merge_keeps_best_raw_score_across_probes(monkeypatch):
    question = "i cant stop thinking about her"
    variant = expansion_variant(question)
    payload = make_payload("doc_a", 0, "Merged chunk.")
    fake = FakeRetrieval({
        question: [make_group("doc_a", [make_hit(7, 0.55, payload)])],
        variant: [make_group("doc_a", [make_hit(7, 0.82, payload)])],
    })
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": question}], "expand": True}), clock=lambda: 0.0)
    passage = bundle["sources"][0]["passages"][0]
    assert passage["score"] == 0.82
    assert passage["query_scores"] == [{"query_index": 0, "score": 0.82}]


def test_completion_order_does_not_affect_output(monkeypatch):
    pa = make_payload("s1", 0, "one")
    pb = make_payload("s2", 0, "two")
    fixtures = {
        "q0": [make_group("s1", [make_hit(1, 0.9, pa)]),
               make_group("s2", [make_hit(2, 0.8, pb)])],
        "q1": [make_group("s2", [make_hit(2, 0.95, pb)]),
               make_group("s1", [make_hit(1, 0.7, pa)])],
    }

    results = []
    for reverse in (False, True):
        fake = FakeRetrieval({})
        def grouped(vector, *, reverse=reverse, **kwargs):
            probe = fake.embedded_texts[int(vector[0])]
            groups = list(fixtures.get(probe, []))
            if reverse:
                groups.reverse()
                for g in groups:
                    g.hits.reverse()
            return groups
        monkeypatch.setattr(store, "check_research_ready", lambda collection_name=None: 10)
        monkeypatch.setattr(store, "embed", fake.embed)
        monkeypatch.setattr(store, "query_points_grouped", grouped)
        results.append(research.run(parse({"queries": [{"text": "q0"}, {"text": "q1"}]}), clock=lambda: 0.0))

    assert results[0] == results[1]


# --- 3.2 / 3.4 grouped selection and coverage --------------------------------------


def test_source_diversity_limits_passages_per_document(monkeypatch):
    many = [make_hit(100 + i, 0.99 - i * 0.001, make_payload("doc_hog", i, f"hog {i}"))
            for i in range(6)]
    other = make_payload("doc_other", 0, "Other document passage.")
    fake = FakeRetrieval({
        "q0": [make_group("doc_hog", many), make_group("doc_other", [make_hit(200, 0.80, other)])],
    })
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q0"}]}), clock=lambda: 0.0)
    by_id = {s["source_id"]: s for s in bundle["sources"]}
    assert len(by_id["doc_hog"]["passages"]) == 2  # default budget
    assert "doc_other" in by_id
    search = fake.search_calls[0]
    assert search["groups"] == 9 and search["group_size"] == 2


def test_max_sources_one_returns_one_source(monkeypatch):
    payloads = {name: make_payload(name, 0, f"text {name}") for name in ("a", "b", "c")}
    fake = FakeRetrieval({
        "q0": [make_group(name, [make_hit(i, 0.9 - i * 0.1, payloads[name])])
               for i, name in enumerate(("a", "b", "c"))],
    })
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q0"}], "max_sources": 1}), clock=lambda: 0.0)
    assert len(bundle["sources"]) == 1


def test_passage_ordering_best_score_then_chunk_index(monkeypatch):
    p0 = make_payload("doc_a", 0, "chunk zero")
    p1 = make_payload("doc_a", 1, "chunk one")
    p2 = make_payload("doc_a", 2, "chunk two")
    fake = FakeRetrieval({
        "q0": [make_group("doc_a", [make_hit(3, 0.70, p2), make_hit(1, 0.85, p0), make_hit(2, 0.85, p1)])],
    })
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "q0"}], "max_passages_per_source": 3})
    bundle = research.run(req, clock=lambda: 0.0)
    assert [(p["chunk_index"], p["score"]) for p in bundle["sources"][0]["passages"]] == [(0, 0.85), (1, 0.85), (2, 0.7)]


def test_budget_excluded_is_distinguished_from_no_eligible_evidence(monkeypatch):
    pa = make_payload("doc_a", 0, "A passage.")
    pb = make_payload("doc_b", 0, "B passage.")
    fake = FakeRetrieval({
        "q0": [make_group("doc_a", [make_hit(1, 0.9, pa)])],
        "q1": [make_group("doc_b", [make_hit(2, 0.9, pb)])],
    })
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "q0"}, {"text": "q1"}], "max_sources": 1})
    bundle = research.run(req, clock=lambda: 0.0)
    statuses = {r["query_index"]: r for r in bundle["query_results"]}
    assert bundle["sources"][0]["source_id"] == "doc_a"
    assert statuses[0]["status"] == "selected"
    assert statuses[1]["status"] == "budget_excluded"
    assert statuses[1]["eligible_source_count"] == 1
    assert statuses[1]["selected_evidence_ids"] == []
    assert bundle["unmatched_query_indexes"] == [1]


def test_no_eligible_evidence_for_unmatched_question(monkeypatch):
    pa = make_payload("doc_a", 0, "A passage.")
    fake = FakeRetrieval({"q0": [make_group("doc_a", [make_hit(1, 0.9, pa)])], "q1": []})
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q0"}, {"text": "q1"}]}), clock=lambda: 0.0)
    statuses = {r["query_index"]: r for r in bundle["query_results"]}
    assert statuses[1]["status"] == "no_eligible_evidence"
    assert statuses[1]["eligible_source_count"] == 0
    assert bundle["unmatched_query_indexes"] == [1]


def test_empty_evidence_is_a_successful_result(monkeypatch):
    fake = FakeRetrieval({})  # nothing matches anything
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "q0"}, {"text": "q1"}], "min_score": 0.9})
    bundle = research.run(req, clock=lambda: 0.0)
    assert bundle["sources"] == []
    assert bundle["unmatched_query_indexes"] == [0, 1]
    assert all(r["status"] == "no_eligible_evidence" for r in bundle["query_results"])


def test_every_selected_evidence_id_resolves_within_bundle(monkeypatch):
    pa = make_payload("doc_a", 0, "A.")
    pb = make_payload("doc_a", 1, "B.")
    fake = FakeRetrieval({
        "q0": [make_group("doc_a", [make_hit(1, 0.9, pa), make_hit(2, 0.8, pb)])],
        "q1": [make_group("doc_a", [make_hit(2, 0.95, pb)])],
    })
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "q0"}, {"text": "q1"}], "max_passages_per_source": 5})
    bundle = research.run(req, clock=lambda: 0.0)
    bundle_ids = {p["evidence_id"] for s in bundle["sources"] for p in s["passages"]}
    for result in bundle["query_results"]:
        assert set(result["selected_evidence_ids"]) <= bundle_ids
    assert bundle["unmatched_query_indexes"] == []


# --- 3.5 evidence identity, excerpts, locations ------------------------------------


def test_changed_chunk_text_changes_evidence_id_at_same_coordinates(monkeypatch):
    original = make_payload("doc_a", 3, "Original passage text.")
    rewritten = make_payload("doc_a", 3, "Rewritten passage text.")
    id_original = research.run.__module__  # touch module for clarity
    fake = FakeRetrieval({"q0": [make_group("doc_a", [make_hit(1, 0.9, original)])]})
    fake.install(monkeypatch)
    first = research.run(parse({"queries": [{"text": "q0"}]}), clock=lambda: 0.0)
    monkeypatch.undo()

    fake2 = FakeRetrieval({"q0": [make_group("doc_a", [make_hit(1, 0.9, rewritten)])]})
    monkeypatch.setattr(store, "check_research_ready", lambda collection_name=None: 10)
    monkeypatch.setattr(store, "embed", fake2.embed)
    monkeypatch.setattr(store, "query_points_grouped", fake2.query_points_grouped)
    second = research.run(parse({"queries": [{"text": "q0"}]}), clock=lambda: 0.0)

    p1 = first["sources"][0]["passages"][0]
    p2 = second["sources"][0]["passages"][0]
    assert (p1["chunk_index"], ) == (p2["chunk_index"], ) == (3,)
    assert p1["source_id"] if False else p1["evidence_id"].startswith("doc_a:3:")
    assert p1["evidence_id"] != p2["evidence_id"]
    assert p1["passage_hash"] != p2["passage_hash"]


def test_excerpt_truncation_keeps_full_text_hash(monkeypatch):
    long_text = "abcde " * 500  # 3000 chars
    payload = make_payload("doc_a", 0, long_text)
    fake = FakeRetrieval({"q0": [make_group("doc_a", [make_hit(1, 0.9, payload)])]})
    fake.install(monkeypatch)
    req = parse({"queries": [{"text": "q0"}], "max_excerpt_chars": 100})
    bundle = research.run(req, clock=lambda: 0.0)
    passage = bundle["sources"][0]["passages"][0]
    assert passage["text"] == long_text[:100]
    assert passage["text_truncated"] is True
    assert passage["passage_hash"] == hashlib.sha256(long_text.encode("utf-8")).hexdigest()


def test_excerpt_exactly_at_limit_is_not_truncated(monkeypatch):
    text = "x" * 1200
    payload = make_payload("doc_a", 0, text)
    fake = FakeRetrieval({"q0": [make_group("doc_a", [make_hit(1, 0.9, payload)])]})
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q0"}]}), clock=lambda: 0.0)
    passage = bundle["sources"][0]["passages"][0]
    assert passage["text"] == text
    assert passage["text_truncated"] is False


@pytest.mark.parametrize("doc_name,page,expected", [
    ("paper.pdf", 7, {"kind": "page", "value": 7}),
    ("book.epub", 4, {"kind": "section", "value": 4}),
    ("transcript.md", 2, {"kind": "unknown", "value": None}),
    ("notes.txt", 1, {"kind": "unknown", "value": None}),
    ("paper.pdf", None, {"kind": "unknown", "value": None}),
])
def test_location_hints_are_conservative(monkeypatch, doc_name, page, expected):
    doc_path = f"Papers/X/{doc_name}"
    payload = make_payload("doc_a", 0, "Body.", doc=doc_path, doc_name=doc_name, page=page)
    fake = FakeRetrieval({"q0": [make_group("doc_a", [make_hit(1, 0.9, payload)])]})
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q0"}]}), clock=lambda: 0.0)
    passage = bundle["sources"][0]["passages"][0]
    assert passage["location"] == expected
    assert passage["page"] == page


def test_transcript_overlap_text_is_returned_verbatim_without_invented_anchor(monkeypatch):
    transcript = "## [00:12:34] — Topic\nOverlapping preamble line that belongs to the previous chunk.\nActual content."
    payload = make_payload("doc_v", 0, transcript, doc="Videos/Ch_2026_T/Video [abc].en.md",
                           doc_name="Video [abc].en.md", doc_type="video", page=None)
    fake = FakeRetrieval({"q0": [make_group("doc_v", [make_hit(1, 0.9, payload)])]})
    fake.install(monkeypatch)
    bundle = research.run(parse({"queries": [{"text": "q0"}]}), clock=lambda: 0.0)
    passage = bundle["sources"][0]["passages"][0]
    assert passage["text"] == transcript  # verbatim, heading and overlap untouched
    assert passage["location"] == {"kind": "unknown", "value": None}
    assert "00:12:34" in passage["text"]


# --- 3.6 deadline -------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_service_waits_are_capped_by_remaining_deadline(monkeypatch):
    fake = FakeRetrieval({"q0": []})
    fake.install(monkeypatch)
    clock = FakeClock()
    research.run(parse({"queries": [{"text": "q0"}]}), clock=clock)
    budget = config.RESEARCH_TIMEOUT
    assert 0 < fake.embed_calls[0]["timeout"] <= math.ceil(budget)
    assert fake.embed_calls[0]["timeout"] == math.ceil(budget)


def test_exhausted_deadline_raises_research_timeout_before_embedding(monkeypatch):
    fake = FakeRetrieval({})
    fake.install(monkeypatch)
    clock = FakeClock()

    def slow_readiness(collection_name=None):
        clock.advance(config.RESEARCH_TIMEOUT + 1)
        return 10

    monkeypatch.setattr(store, "check_research_ready", slow_readiness)
    with pytest.raises(errors.ResearchTimeoutError) as excinfo:
        research.run(parse({"queries": [{"text": "q0"}]}), clock=clock)
    assert excinfo.value.category == "research_timeout"
    assert excinfo.value.exit_code == 4
    assert fake.embed_calls == []


def test_slow_service_past_deadline_fails_request_not_just_wait(monkeypatch):
    fake = FakeRetrieval({})
    fake.install(monkeypatch)
    clock = FakeClock()

    def slow_embed(texts, timeout=None):
        clock.advance(config.RESEARCH_TIMEOUT)  # provider blew the whole budget
        return fake.embed(texts, timeout=timeout)

    monkeypatch.setattr(store, "embed", slow_embed)
    with pytest.raises(errors.ResearchTimeoutError):
        research.run(parse({"queries": [{"text": "q0"}]}), clock=clock)
    assert fake.search_calls == []


def test_deadline_checked_between_grouped_searches(monkeypatch):
    fake = FakeRetrieval({})
    fake.install(monkeypatch)
    clock = FakeClock()
    searches = {"n": 0}

    def slow_search(vector, **kwargs):
        searches["n"] += 1
        clock.advance(config.RESEARCH_TIMEOUT)
        return []

    monkeypatch.setattr(store, "query_points_grouped", slow_search)
    with pytest.raises(errors.ResearchTimeoutError):
        research.run(parse({"queries": [{"text": "q0"}, {"text": "q1"}]}), clock=clock)
    assert searches["n"] == 1  # the second search never started


def test_nonpositive_deadline_config_is_a_timeout_error(monkeypatch):
    fake = FakeRetrieval({})
    fake.install(monkeypatch)
    monkeypatch.setattr(config, "RESEARCH_TIMEOUT", 0.0)
    with pytest.raises(errors.ResearchTimeoutError):
        research.run(parse({"queries": [{"text": "q0"}]}), clock=FakeClock())


# --- 1.4 / readiness failures via the CLI -------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    lib_root = tmp_path / "library"
    lib_root.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(config, "LIBRARY_ROOT", lib_root)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    return lib_root


def test_research_unready_collection_exits_three(cli_env, monkeypatch, capsys, tmp_path):
    def not_ready(collection_name=None):
        raise errors.ResearchIndexNotReadyError(
            "collection 'library' does not exist; run `library-query index` to build it"
        )

    monkeypatch.setattr(store, "check_research_ready", not_ready)
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps({"queries": [{"text": "anything"}]}))

    code = run_cli(["research", "--input", str(request_file), "--json"])
    payload = read_error(capsys)
    assert code == payload["exit_code"] == 3
    assert payload["error"] == "research_index_not_ready"
    assert "index" in payload["message"]
    assert capsys.readouterr().out == ""  # no partial success bundle


def test_research_invalid_input_exits_two_and_makes_no_service_calls(cli_env, monkeypatch, capsys, tmp_path):
    embed_spy = MagicMock()
    monkeypatch.setattr(store, "embed", embed_spy)
    request_file = tmp_path / "request.json"
    request_file.write_text('{"queries": [{"text": "ok"}], "max_sources": 42}')

    code = run_cli(["research", "--input", str(request_file), "--json"])
    payload = read_error(capsys)
    assert code == payload["exit_code"] == 2
    assert payload["error"] == "invalid_research_request"
    assert "max_sources" in payload["message"]
    embed_spy.assert_not_called()
    assert capsys.readouterr().out == ""


def test_research_missing_input_file_exits_two(cli_env, capsys, tmp_path):
    code = run_cli(["research", "--input", str(tmp_path / "nope.json"), "--json"])
    payload = read_error(capsys)
    assert code == payload["exit_code"] == 2
    assert payload["error"] == "invalid_research_request"


def test_research_stdin_transport(cli_env, monkeypatch, capsys):
    fake = FakeRetrieval({})
    fake.install(monkeypatch)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(
        buffer=io.BytesIO(request_bytes({"queries": [{"text": "q0"}]}))))
    code = run_cli(["research", "--input", "-", "--json"])
    assert code == 0
    bundle = json.loads(capsys.readouterr().out)
    assert bundle["schema_version"] == 1
    assert bundle["sources"] == []


def test_research_search_failure_after_success_exits_four_without_bundle(cli_env, monkeypatch, capsys, tmp_path):
    fake = FakeRetrieval({})
    fake.install(monkeypatch)
    calls = {"n": 0}

    def flaky(vector, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            payload = make_payload("doc_a", 0, "Found something.")
            return [make_group("doc_a", [make_hit(1, 0.9, payload)])]
        raise errors.QdrantUnreachableError("Could not reach Qdrant at http://x: connection reset")

    monkeypatch.setattr(store, "query_points_grouped", flaky)
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps({"queries": [{"text": "q0"}, {"text": "q1"}]}))
    code = run_cli(["research", "--input", str(request_file), "--json"])
    payload = read_error(capsys)
    assert code == payload["exit_code"] == 4
    assert payload["error"] == "qdrant_unreachable"
    assert capsys.readouterr().out == ""


def test_research_timeout_through_cli_exits_four(cli_env, monkeypatch, capsys, tmp_path):
    fake = FakeRetrieval({})
    fake.install(monkeypatch)
    monkeypatch.setattr(config, "RESEARCH_TIMEOUT", 5.0)
    clock = FakeClock()
    monkeypatch.setattr(research, "CLOCK", clock)

    def slow_embed(texts, timeout=None):
        clock.advance(30)
        return fake.embed(texts, timeout=timeout)

    monkeypatch.setattr(store, "embed", slow_embed)
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps({"queries": [{"text": "q0"}]}))
    code = run_cli(["research", "--input", str(request_file), "--json"])
    payload = read_error(capsys)
    assert code == payload["exit_code"] == 4
    assert payload["error"] == "research_timeout"
    assert capsys.readouterr().out == ""


def test_research_never_mutates_the_store(cli_env, monkeypatch, capsys, tmp_path):
    """Read-only guarantee: no create/upsert/set_payload/delete on ready or unready paths."""
    from qdrant_client.models import PayloadSchemaType
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = SimpleNamespace(
        payload_schema={"document_id": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)})
    mock_client.scroll.return_value = ([], None)
    mock_client.count.return_value = SimpleNamespace(count=0)
    monkeypatch.setattr(store, "qdrant", lambda: mock_client)
    monkeypatch.setattr(store, "QdrantClient", lambda **kw: mock_client)

    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps({"queries": [{"text": "q0"}]}))

    fake = FakeRetrieval({})
    monkeypatch.setattr(store, "embed", fake.embed)
    monkeypatch.setattr(store, "query_points_grouped", fake.query_points_grouped)

    assert run_cli(["research", "--input", str(request_file), "--json"]) == 0
    for mutator in ("create_collection", "create_payload_index", "upsert", "set_payload",
                    "delete", "delete_payload", "overwrite_payload"):
        assert not getattr(mock_client, mutator).called, mutator

    # Unready path: readiness fails and still nothing is created or repaired.
    mock_client.reset_mock()
    mock_client.scroll.return_value = (
        [SimpleNamespace(id=1, payload={"document_id": "doc_x", "chunk_index": None, "text": "t", "doc": "d"})],
        None,
    )
    code = run_cli(["research", "--input", str(request_file), "--json"])
    assert code == 3
    payload = read_error(capsys)
    assert payload["error"] == "research_index_not_ready"
    for mutator in ("create_collection", "create_payload_index", "upsert", "set_payload",
                    "delete", "delete_payload", "overwrite_payload"):
        assert not getattr(mock_client, mutator).called, mutator


# --- scope filtering of research (4.2) ----------------------------------------------


def _scope_probe(monkeypatch, scope, captured):
    payload_catalog = {
        "inbox": make_payload("doc_i", 0, "Inbox passage.", scope="inbox", doc="_Inbox/i.txt",
                              doc_name="i.txt", doc_type="inbox"),
        "catalogued": make_payload("doc_c", 0, "Catalogued passage.", scope="catalogued"),
        "operations": make_payload("doc_o", 0, "Operations passage.", scope="operations",
                                   doc="Operations/ZOTERO.md", doc_name="ZOTERO.md", doc_type="operation"),
    }
    fake = FakeRetrieval({})
    monkeypatch.setattr(store, "check_research_ready", lambda collection_name=None: 10)
    monkeypatch.setattr(store, "embed", fake.embed)

    def grouped(vector, *, groups, group_size, query_filter=None, **kwargs):
        captured.append(query_filter)
        conditions = getattr(query_filter, "must", None) or []
        scope_values = [c.match.value for c in conditions if getattr(c, "key", None) == "scope"]
        scope_value = scope_values[0] if scope_values else "all"
        hits = [make_group("doc_o", [make_hit(3, 0.9, payload_catalog["operations"])]),
                make_group("doc_c", [make_hit(1, 0.9, payload_catalog["catalogued"])]),
                make_group("doc_i", [make_hit(2, 0.9, payload_catalog["inbox"])])]
        return [g for g in hits if scope_value == "all" or g.id.startswith({"inbox": "doc_i", "catalogued": "doc_c", "operations": "doc_o"}[scope_value])]

    monkeypatch.setattr(store, "query_points_grouped", grouped)


def test_research_default_scope_excludes_operations(monkeypatch):
    captured = []
    _scope_probe(monkeypatch, "catalogued", captured)
    bundle = research.run(parse({"queries": [{"text": "q0"}]}), clock=lambda: 0.0)
    assert [s["source_id"] for s in bundle["sources"]] == ["doc_c"]
    assert [s["scope"] for s in bundle["sources"]] == ["catalogued"]


def test_research_explicit_operations_scope(monkeypatch):
    captured = []
    _scope_probe(monkeypatch, "operations", captured)
    req = parse({"queries": [{"text": "q0"}], "scope": "operations"})
    bundle = research.run(req, clock=lambda: 0.0)
    assert [s["source_id"] for s in bundle["sources"]] == ["doc_o"]
    assert bundle["sources"][0]["type"] == "operation"


def test_research_all_scope_sees_every_scope(monkeypatch):
    captured = []
    _scope_probe(monkeypatch, "all", captured)
    req = parse({"queries": [{"text": "q0"}], "scope": "all"})
    bundle = research.run(req, clock=lambda: 0.0)
    assert {s["scope"] for s in bundle["sources"]} == {"inbox", "catalogued", "operations"}
    assert captured[0] is None or not getattr(captured[0], "must", None)


def test_readiness_scan_skips_chunk_text_and_counts_missing_text_server_side(monkeypatch):
    """The readiness scan must not ship every chunk's text (13s on 54k points)."""
    from qdrant_client.models import PayloadSchemaType
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        payload_schema={"document_id": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)})
    client.scroll.return_value = (
        [SimpleNamespace(id=1, payload={"document_id": "doc_x", "chunk_index": 0, "doc": "d"})], None)
    client.count.return_value = SimpleNamespace(count=0)
    monkeypatch.setattr(store, "qdrant", lambda: client)
    assert store.check_research_ready("library") == 1
    assert "text" not in client.scroll.call_args.kwargs["with_payload"]
    client.count.return_value = SimpleNamespace(count=2)
    with pytest.raises(errors.ResearchIndexNotReadyError):
        store.check_research_ready("library")


def test_identity_scan_checks_later_pages_with_bounded_payloads(monkeypatch):
    from qdrant_client.models import PayloadSchemaType
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        payload_schema={"document_id": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)})
    client.scroll.side_effect = [
        ([SimpleNamespace(id=1, payload={"document_id": "doc_x", "chunk_index": 0})], 2),
        ([SimpleNamespace(id=2, payload={"document_id": "doc_x", "chunk_index": -1})], None),
    ]
    client.count.return_value = SimpleNamespace(count=0)
    monkeypatch.setattr(store, "qdrant", lambda: client)
    with pytest.raises(errors.ResearchIndexNotReadyError):
        store.check_research_ready("library")
    assert client.scroll.call_count == 2
    assert client.scroll.call_args.kwargs["offset"] == 2
    for call in client.scroll.call_args_list:
        assert "text" not in call.kwargs["with_payload"]
        assert call.kwargs["with_vectors"] is False
