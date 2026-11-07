"""Cache integration with query/research retrieval: reuse, invalidation, assembly.

Covers tasks.md section 4 (embedding/candidate reuse, current-request assembly,
failure-safe publication, truthful diagnostics) with deterministic mocked
embedding/Qdrant pipelines and real revision records + guards. Provider
counters, not cache claims, decide whether reuse happened.
"""
import json
import os
import sqlite3
from types import SimpleNamespace

import pytest
from conftest import FakeClock, make_settings  # shared helpers

from library_query import (
    cache,
    cli,
    config,
    errors,
    expansion,
    research,
    readiness,
    revisions,
    store,
)


def make_payload(document_id, chunk_index, text, *, doc=None, scope="catalogued", doc_type="paper",
                 page=None, tags=None, title=None):
    path = doc or f"Papers/{document_id}/paper.pdf"
    return {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "text": text,
        "path": path,
        "doc": path,
        "doc_name": path.rsplit("/", 1)[-1],
        "scope": scope,
        "type": doc_type,
        "page": page,
        "tags": tags,
        "title": title,
        "authors": None,
        "year": None,
        "subjects": None,
        "identifier": None,
    }


class FakeEmbedder:
    """Deterministic stand-in for store.embed: vector[i][0] = probe index."""

    def __init__(self, dim=None):
        self.dim = dim or config.EMBEDDING_DIM
        self.calls = []
        self.texts = []

    def __call__(self, texts, timeout=None):
        self.calls.append(list(texts))
        start = len(self.texts)
        self.texts.extend(texts)
        return [[float(start + i)] + [0.0] * (self.dim - 1) for i in range(len(texts))]

    def probe_for(self, vector):
        return self.texts[int(vector[0])]


class FakeQueryClient:
    """query_points keyed by probe text; optionally fails everything."""

    def __init__(self, embedder, hits_by_probe):
        self.embedder = embedder
        self.hits_by_probe = hits_by_probe
        self.search_calls = 0
        self.broken = False

    def query_points(self, collection_name=None, query=None, limit=None, query_filter=None):
        self.search_calls += 1
        if self.broken:
            raise RuntimeError("connection reset by peer")
        probe = self.embedder.probe_for(query)
        hits = self.hits_by_probe.get(probe, [])
        if isinstance(hits, Exception):
            raise hits
        if query_filter is not None:
            for condition in query_filter.must or []:
                hits = [h for h in hits if (condition.match.value in (h[2].get("tags") or [])
                        if condition.key == "tags" else h[2].get(condition.key) == condition.match.value)]
        return SimpleNamespace(points=[
            SimpleNamespace(id=pid, score=score, payload=dict(payload))
            for pid, score, payload in hits
        ])

    def collection_exists(self, collection_name):
        if self.broken:
            raise RuntimeError("connection reset by peer")
        return True

    def get_collection(self, collection_name=None):
        if self.broken:
            raise errors.QdrantUnreachableError("connection reset by peer")
        record = revisions.read_record(revisions.managed_context(collection_name)) or {}
        return SimpleNamespace(config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=config.EMBEDDING_DIM)),
            metadata={readiness.METADATA_KEY: record.get("research_validation")}),
            payload_schema={k: {"data_type": "keyword"} for k in readiness.REQUIRED_INDEXES})


def hit(pid, score, payload):
    return (pid, score, payload)


def publish_index(rev_env, revision=1, incarnation="inc-1", collection=None, state="stable"):
    ctx = revisions.managed_context(collection)
    record = {
        "schema_version": 1,
        "context_digest": ctx.context_digest,
        "corpus_id": ctx.corpus_id,
        "collection": ctx.collection,
        "incarnation": incarnation,
        "revision": revision,
        "state": state,
        "embedding_fingerprint": ctx.embedding_fingerprint,
        "representation_fingerprint": ctx.representation_fingerprint,
        "published_at": "2026-10-01T00:00:00+00:00",
    }
    record["research_validation"] = {k: record[k] for k in readiness.IDENTITY_FIELDS}
    record["research_validation"].update(audit_version=1, state="ready", generation=f"generation-{revision}")
    revisions.publish_record(ctx, record)
    revisions.acquire_writer_guard(ctx).release()  # guard file exists for readers
    return ctx


@pytest.fixture
def query_env(cache_env, monkeypatch):
    """Published stable revision + fake embedder + fake query client."""
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    ctx = publish_index(cache_env)
    embedder = FakeEmbedder()
    client = FakeQueryClient(embedder, {})
    monkeypatch.setattr(store, "embed", embedder)
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(readiness, "collection_info", lambda collection, deadline=None:
                        client.get_collection(collection_name=collection))

    def make_session(mode="use", namespace="local", clock=None):
        settings = make_settings(cache_env.cache_dir, mode=mode, namespace=namespace)
        return cache.CacheSession(settings, config.COLLECTION, clock=clock)

    return SimpleNamespace(
        cache_env=cache_env, ctx=ctx, embedder=embedder, client=client,
        make_session=make_session,
    )


@pytest.fixture
def research_env(query_env, monkeypatch):
    """research.run with faked readiness + grouped search keyed by probe."""
    grouped_calls = {"count": 0}
    groups_by_probe = {}

    def grouped(vector, *, groups, group_size, query_filter=None, score_threshold=None,
                timeout=None, collection_name=None):
        grouped_calls["count"] += 1
        if query_env.client.broken:
            raise RuntimeError("connection reset by peer")
        return list(groups_by_probe.get(query_env.embedder.probe_for(vector), []))

    monkeypatch.setattr(store, "check_research_ready", lambda collection_name=None: 10)
    monkeypatch.setattr(store, "query_points_grouped", grouped)
    query_env.grouped_calls = grouped_calls
    query_env.groups_by_probe = groups_by_probe
    return query_env


def group(document_id, hits):
    return SimpleNamespace(
        id=document_id,
        hits=[SimpleNamespace(id=pid, score=score, payload=payload) for pid, score, payload in hits],
    )


def run_query(query_env, question, *, mode="use", session=None, **kwargs):
    session = session or query_env.make_session(mode)
    with session:
        places = store.query_places(question, cache=session, **kwargs)
    return places, session.diagnostics()


def parse(request_obj):
    return research.parse_request(json.dumps(request_obj).encode("utf-8"))


def run_research(research_env, request_obj, *, mode="use", session=None):
    session = session or research_env.make_session(mode)
    with session:
        bundle = research.run(parse(request_obj), clock=lambda: 0.0, cache=session)
    return bundle, session.diagnostics()


def evidence_rows(cache_env):
    db = cache_env.cache_dir / cache.CACHE_FILENAME
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT tier, entry_key, expires_at, created_at, payload FROM cache_entries WHERE tier = 'evidence'").fetchall()
    finally:
        conn.close()


# --- 4.1 embedding reuse across locator and research -----------------------------------------


def test_cold_then_warm_query_requires_zero_provider_work(query_env):
    hits = [hit(11, 0.92, make_payload("doc_a", 0, "Passage about retry strategies.", page=3))]
    query_env.client.hits_by_probe["what retry strategies work?"] = hits

    cold_places, cold_diag = run_query(query_env, "what retry strategies work?")
    assert cold_diag["candidate_misses"] == 1
    assert cold_diag["embedding_misses"] == 1
    assert cold_diag["embedding_request_count"] == 1
    assert cold_diag["search_request_count"] == 1
    assert cold_diag["index"]["revision"] == 1
    assert query_env.client.search_calls == 1
    assert len(cold_places) == 1

    warm_places, warm_diag = run_query(query_env, "what retry strategies work?")
    assert warm_places == cold_places, "cached raw candidates rebuild identical output"
    assert warm_diag["candidate_hits"] == 1
    assert warm_diag["candidate_misses"] == 0
    # Candidate-hit probes need no embedding lookup at all.
    assert warm_diag["embedding_hits"] == 0 and warm_diag["embedding_misses"] == 0
    assert warm_diag["embedding_request_count"] == 0
    assert warm_diag["search_request_count"] == 0
    assert warm_diag["bypass_reasons"] == []
    assert query_env.client.search_calls == 1, "warm lookup issues no search"
    assert query_env.embedder.calls == [["what retry strategies work?"]], "warm lookup embeds nothing"


def test_mixed_batch_embeds_only_missing_probes(query_env, monkeypatch):
    question = "i cant stop thinking about her"
    probes = expansion.probes(question, expand=True)
    assert len(probes) == 2
    variant = probes[1]
    # Pre-seed only the expansion variant's embedding.
    session = query_env.make_session()
    storage = session._store()
    budget = cache.AdmissionBudget()
    storage.put_payload("embedding", cache.embedding_key(session.settings, variant),
                        revisions.embedding_fingerprint(),
                        {"v": 1, "vector": [0.5] * config.EMBEDDING_DIM},
                        session.settings.embedding_ttl, budget)
    query_env.client.hits_by_probe = {p: [] for p in probes}

    _places, diag = run_query(query_env, question, session=session)
    assert query_env.embedder.calls == [probes[:1]], "only probe 0 is embedded"
    assert diag["embedding_hits"] == 1 and diag["embedding_misses"] == 1
    assert diag["embedding_request_count"] == 1


def test_embedding_vectors_survive_a_later_search_failure(query_env):
    probes = expansion.probes("i cant stop thinking about her", expand=True)
    query_env.client.hits_by_probe = {
        probes[0]: [hit(1, 0.9, make_payload("a", 0, "A."))],
        probes[1]: RuntimeError("later search failed"),
    }

    session = query_env.make_session()
    with pytest.raises(errors.QdrantUnreachableError), session:
        store.query_places("i cant stop thinking about her", cache=session)
    # The second probe's search failed; both embeddings were stored independently.
    db = query_env.cache_env.cache_dir / cache.CACHE_FILENAME
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT COUNT(*) FROM cache_entries WHERE tier = 'embedding'").fetchone()[0]
    conn.close()
    assert rows == 2
    assert evidence_rows(query_env.cache_env) == [], "no partial candidate publication"


def test_refresh_skips_reads_and_replaces_entries(query_env):
    question = "anything about backoff?"
    payload = make_payload("doc_b", 0, "Backoff and jitter trade-offs.")
    query_env.client.hits_by_probe[question] = [hit(5, 0.8, payload)]

    _, cold_diag = run_query(query_env, question)
    assert cold_diag["candidate_misses"] == 1

    warm_places, warm_diag = run_query(query_env, question)
    assert warm_diag["candidate_hits"] == 1
    assert query_env.client.search_calls == 1

    # refresh: no reads, fresh provider work, successful rows replaced.
    _, refresh_diag = run_query(query_env, question, mode="refresh")
    assert refresh_diag["candidate_hits"] == 0 and refresh_diag["candidate_misses"] == 0
    assert refresh_diag["embedding_request_count"] == 1
    assert refresh_diag["search_request_count"] == 1
    assert query_env.client.search_calls == 2

    # The refreshed row is now what a warm `use` run serves.
    again_places, again_diag = run_query(query_env, question)
    assert again_places == warm_places
    assert again_diag["candidate_hits"] == 1


def test_embedding_sharing_across_flat_and_grouped_methods(query_env, research_env):
    question = "shared probe text"
    query_env.client.hits_by_probe[question] = [hit(7, 0.9, make_payload("doc_f", 0, "Flat hit."))]
    research_env.groups_by_probe[question] = [group("doc_g", [hit(8, 0.85, make_payload("doc_g", 0, "Grouped hit."))])]

    _, flat_diag = run_query(query_env, question)
    assert flat_diag["candidate_misses"] == 1 and flat_diag["embedding_misses"] == 1

    bundle, grouped_diag = run_research(research_env, {"queries": [{"text": question}]})
    # Embeddings are method-agnostic; candidate entries are not.
    assert grouped_diag["embedding_hits"] == 1 and grouped_diag["embedding_misses"] == 0
    assert grouped_diag["candidate_misses"] == 1
    assert bundle["sources"][0]["source_id"] == "doc_g"
    assert research_env.grouped_calls["count"] == 1


# --- 4.2 candidate invalidation ---------------------------------------------------------------


def test_changed_tags_do_not_reuse_candidates(query_env):
    question = "tag sensitive question"
    payload_tagged = make_payload("doc_t", 0, "Tagged passage.", tags=["reliability"])
    payload_untagged = make_payload("doc_u", 0, "Untagged passage.")
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, payload_tagged), hit(2, 0.8, payload_untagged)]

    _, cold = run_query(query_env, question, tags=["reliability"])
    assert cold["candidate_misses"] == 1

    _, warm_same = run_query(query_env, question, tags=["reliability"])
    assert warm_same["candidate_hits"] == 1

    _, changed = run_query(query_env, question, tags=["distributed systems"])
    assert changed["candidate_hits"] == 0 and changed["candidate_misses"] == 1
    assert query_env.client.search_calls == 2


def test_changed_scope_invalidates_research_candidates(research_env):
    payload_cat = make_payload("doc_c", 0, "Catalogued passage.")
    payload_ops = make_payload("doc_o", 0, "Operations passage.", scope="operations",
                               doc="Operations/ZOTERO.md", doc_type="operation")
    research_env.groups_by_probe["q1"] = [group("doc_c", [hit(1, 0.9, payload_cat)])]
    _, cold = run_research(research_env, {"queries": [{"text": "q1"}]})
    assert cold["candidate_misses"] == 1

    _, warm = run_research(research_env, {"queries": [{"text": "q1"}]})
    assert warm["candidate_hits"] == 1

    research_env.groups_by_probe["q1"] = [group("doc_o", [hit(2, 0.9, payload_ops)])]
    bundle, changed = run_research(research_env, {"queries": [{"text": "q1"}], "scope": "operations"})
    assert changed["candidate_misses"] == 1
    assert bundle["sources"][0]["source_id"] == "doc_o"
    assert research_env.grouped_calls["count"] == 2


def test_changed_candidate_window_invalidates(query_env):
    question = "window sensitive"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("w", 0, "W."))]
    run_query(query_env, question, top_k=5)
    _, same = run_query(query_env, question, top_k=5)
    assert same["candidate_hits"] == 1
    _, wider = run_query(query_env, question, top_k=10)
    assert wider["candidate_hits"] == 0 and wider["candidate_misses"] == 1
    assert query_env.client.search_calls == 2


def test_min_score_stays_post_merge_and_out_of_the_candidate_key(query_env):
    question = "floor behavior check"
    query_env.client.hits_by_probe[question] = [
        hit(1, 0.7, make_payload("doc_low", 0, "Weak passage.")),
        hit(2, 0.95, make_payload("doc_high", 0, "Strong passage.")),
    ]
    cold_places, _ = run_query(query_env, question, min_score=0.5)
    assert [p["score"] for p in cold_places] == [0.95, 0.7]

    # Same window: the cached candidates are reused even though the floor changed;
    # the locator's post-merge floor is applied to the merged ranking fresh.
    warm_places, warm_diag = run_query(query_env, question, min_score=0.9)
    assert warm_diag["candidate_hits"] == 1
    assert [p["score"] for p in warm_places] == [0.95]
    assert query_env.client.search_calls == 1


def test_successful_empty_result_is_cached(query_env):
    question = "nothing in the library matches this"
    query_env.client.hits_by_probe[question] = []
    cold_places, cold_diag = run_query(query_env, question)
    assert cold_places == []
    assert cold_diag["candidate_misses"] == 1
    warm_places, warm_diag = run_query(query_env, question)
    assert warm_places == []
    assert warm_diag["candidate_hits"] == 1
    assert warm_diag["search_request_count"] == 0


def test_advanced_revision_invalidates_candidates_but_not_embeddings(query_env):
    question = "revision bound probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("r", 0, "R."))]
    run_query(query_env, question)
    publish_index(query_env.cache_env, revision=2, incarnation="inc-1")

    places, diag = run_query(query_env, question)
    assert diag["candidate_misses"] == 1, "old revision candidates are ineligible"
    assert diag["embedding_hits"] == 1, "vectors outlive corpus revisions"
    assert diag["embedding_request_count"] == 0
    assert diag["index"]["revision"] == 2
    assert len(places) == 1


def test_unmanaged_index_bypasses_candidates_but_caches_embeddings(query_env):
    # No published record: candidate tier bypasses, embedding tier still works.
    question = "unmanaged probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("u", 0, "U."))]
    query_env.ctx.record_path.unlink()
    query_env.ctx.guard_path.unlink()

    places, diag = run_query(query_env, question)
    assert len(places) == 1
    assert diag["index"] is None
    assert diag["bypass_reasons"] == ["unmanaged_index"]
    assert diag["candidate_hits"] == 0 and diag["candidate_misses"] == 0

    _, second = run_query(query_env, question)
    assert second["embedding_hits"] == 1, "embedding reuse works without a managed index"
    assert second["embedding_request_count"] == 0
    assert query_env.client.search_calls == 2


def test_dirty_revision_never_admits_cached_candidates(query_env):
    question = "dirty probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("d", 0, "D."))]
    run_query(query_env, question)
    publish_index(query_env.cache_env, revision=2, incarnation="inc-1", state="dirty")

    places, diag = run_query(query_env, question)
    assert len(places) == 1, "ordinary uncached retrieval still works"
    assert diag["index"] is None
    assert diag["bypass_reasons"] == ["revision_dirty"]
    assert diag["search_request_count"] == 1


def test_index_service_outage_on_a_potential_hit_is_not_hidden(query_env):
    question = "outage probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("o", 0, "O."))]
    run_query(query_env, question)
    query_env.client.broken = True

    session = query_env.make_session()
    with pytest.raises(errors.QdrantUnreachableError), session:
        store.query_places(question, cache=session)
    diag = session.diagnostics()
    assert diag["candidate_hits"] == 0, "the live check precedes any cache read"
    assert diag["index"] is None


def test_guard_unavailable_platform_still_embeds(query_env, monkeypatch):
    monkeypatch.setattr(revisions, "HAVE_FCNTL", False)
    question = "guardless probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("g", 0, "G."))]
    places, diag = run_query(query_env, question)
    assert len(places) == 1
    assert "guard_unavailable" in diag["bypass_reasons"]
    _, second = run_query(query_env, question)
    assert second["embedding_hits"] == 1, "embedding caching needs no guard"


# --- 4.3 current-request assembly -------------------------------------------------------------


def test_reordered_questions_use_new_indexes(research_env):
    payloads = {
        "s1": make_payload("s1", 0, "one"),
        "s2": make_payload("s2", 0, "two"),
    }
    research_env.groups_by_probe["qa"] = [group("s1", [hit(1, 0.9, payloads["s1"])])]
    research_env.groups_by_probe["qb"] = [group("s2", [hit(2, 0.9, payloads["s2"])])]

    _cold, cold_diag = run_research(research_env, {"queries": [{"text": "qa"}, {"text": "qb"}]})
    assert cold_diag["candidate_misses"] == 2
    assert research_env.grouped_calls["count"] == 2

    warm, warm_diag = run_research(research_env, {"queries": [{"text": "qb"}, {"text": "qa"}]})
    assert warm_diag["candidate_hits"] == 2
    assert research_env.grouped_calls["count"] == 2, "no searches on the warm run"
    assert warm["retrieval"]["embedding_request_count"] == 0
    by_index = {row["query_index"]: row for row in warm["query_results"]}
    assert by_index[0]["status"] == "selected" and by_index[1]["status"] == "selected"
    # The swapped order attributes votes to the new positions.
    s1_warm = next(s for s in warm["sources"] if s["source_id"] == "s1")
    s2_warm = next(s for s in warm["sources"] if s["source_id"] == "s2")
    assert s1_warm["passages"][0]["query_indexes"] == [1]
    assert s2_warm["passages"][0]["query_indexes"] == [0]


def test_duplicate_questions_share_cached_work(research_env):
    payload = make_payload("dup", 0, "Duplicate target.")
    research_env.groups_by_probe["same question"] = [group("dup", [hit(1, 0.9, payload)])]
    _cold, cold_diag = run_research(research_env, {"queries": [{"text": "same question"}]})
    assert cold_diag["candidate_misses"] == 1

    warm, warm_diag = run_research(research_env, {"queries": [{"text": "same question"}, {"text": "same question"}]})
    assert warm_diag["candidate_hits"] == 1, "one distinct probe key, not per-question misses"
    assert warm_diag["embedding_hits"] == 0 and warm_diag["embedding_misses"] == 0
    passage = warm["sources"][0]["passages"][0]
    assert passage["query_indexes"] == [0, 1]
    assert warm["query_results"][0]["status"] == warm["query_results"][1]["status"] == "selected"


def test_excerpt_budget_change_reexcerpts_without_a_fresh_search(research_env):
    text = "A long stored passage that will be excerpted at different budgets."
    research_env.groups_by_probe["excerpt q"] = [
        group("doc_e", [hit(1, 0.9, make_payload("doc_e", 0, text))])
    ]
    cold, _ = run_research(research_env, {"queries": [{"text": "excerpt q"}], "max_excerpt_chars": 10})
    cold_passage = cold["sources"][0]["passages"][0]
    assert cold_passage["text"] == text[:10]
    assert cold_passage["text_truncated"] is True

    warm, warm_diag = run_research(
        research_env, {"queries": [{"text": "excerpt q"}], "max_excerpt_chars": 40}
    )
    assert warm_diag["candidate_hits"] == 1, "excerpt budgets are not candidate-window keys"
    assert research_env.grouped_calls["count"] == 1
    warm_passage = warm["sources"][0]["passages"][0]
    assert warm_passage["text"] == text[:40]
    assert warm_passage["text_truncated"] is True
    assert warm_passage["passage_hash"] == cold_passage["passage_hash"], "hash is over the full passage"


# --- 4.4 failure-safe publication --------------------------------------------------------------


def test_later_probe_failure_publishes_nothing_and_preserves_prior_rows(research_env, monkeypatch):
    research_env.groups_by_probe["ok probe"] = [
        group("doc_ok", [hit(1, 0.9, make_payload("doc_ok", 0, "Fine."))])
    ]
    cold, _ = run_research(research_env, {"queries": [{"text": "ok probe"}]})
    assert cold["sources"][0]["source_id"] == "doc_ok"
    before = evidence_rows(research_env.cache_env)
    assert len(before) == 1

    def explode_late(vector, **kwargs):
        probe = research_env.embedder.probe_for(vector)
        if probe == "bad probe":
            raise errors.QdrantUnreachableError("Could not reach Qdrant at http://x: reset")
        return list(research_env.groups_by_probe.get(probe, []))

    monkeypatch.setattr(store, "query_points_grouped", explode_late)
    session = research_env.make_session()
    with pytest.raises(errors.QdrantUnreachableError), session:
        research.run(parse({"queries": [{"text": "ok probe"}, {"text": "bad probe"}]}),
                     clock=lambda: 0.0, cache=session)

    after = evidence_rows(research_env.cache_env)
    assert len(after) == 1, "no new candidate rows from the failed request"
    assert after[0][1] == before[0][1], "prior valid rows are untouched"
    assert after[0][3] == before[0][3], "prior rows are not overwritten"


def test_research_timeout_discards_staged_candidates(research_env, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(research, "CLOCK", clock)
    research_env.groups_by_probe["slow probe"] = [
        group("doc_s", [hit(1, 0.9, make_payload("doc_s", 0, "Slow."))])
    ]

    def slow_grouped(vector, **kwargs):
        clock.advance(30)  # burn the whole service-work deadline inside the search
        return list(research_env.groups_by_probe.get(research_env.embedder.probe_for(vector), []))

    monkeypatch.setattr(store, "query_points_grouped", slow_grouped)
    session = research_env.make_session()
    with pytest.raises(errors.ResearchTimeoutError), session:
        research.run(parse({"queries": [{"text": "slow probe"}]}), cache=session)
    assert evidence_rows(research_env.cache_env) == [], "timeout publishes nothing"


def test_failed_refresh_keeps_prior_valid_rows(query_env):
    question = "refresh failure probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("rf", 0, "RF."))]
    warm_places, _ = run_query(query_env, question)

    query_env.embedder.dim = 2  # make every fresh vector malformed on the refresh run
    session = query_env.make_session(mode="refresh")
    with pytest.raises(errors.EmbeddingApiError), session:
        store.query_places(question, cache=session)
    query_env.embedder.dim = config.EMBEDDING_DIM

    served, diag = run_query(query_env, question)
    assert diag["candidate_hits"] == 1, "the failed refresh did not destroy the prior row"
    assert served == warm_places


# --- 4.5 truthful diagnostics -------------------------------------------------------------------


def test_warm_research_counts_agree_with_retrieval_block(research_env):
    research_env.groups_by_probe["counted q"] = [
        group("doc_ct", [hit(1, 0.9, make_payload("doc_ct", 0, "Counted."))])
    ]
    cold, cold_diag = run_research(research_env, {"queries": [{"text": "counted q"}]})
    assert cold["retrieval"]["embedding_request_count"] == cold_diag["embedding_request_count"] == 1
    assert cold["retrieval"]["grouped_search_count"] == cold_diag["search_request_count"] == 1
    assert cold_diag["validation_request_count"] == 1  # one live configuration and validation-record read

    warm, warm_diag = run_research(research_env, {"queries": [{"text": "counted q"}]})
    assert warm["retrieval"]["embedding_request_count"] == warm_diag["embedding_request_count"] == 0
    assert warm["retrieval"]["grouped_search_count"] == warm_diag["search_request_count"] == 0
    assert warm_diag["candidate_hits"] == 1
    assert warm_diag["cache_elapsed_ms"] >= 0
    assert warm_diag["bypass_reasons"] == []


def test_cache_block_leaks_no_question_text_or_paths(query_env, capsys):
    question = "a very private question about sleep and rumination"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("leak", 0, "Leak check."))]
    session = query_env.make_session()
    with session:
        store.query_places(question, cache=session)
    diag_blob = json.dumps(session.diagnostics())
    assert question not in diag_blob
    assert str(query_env.cache_env.cache_dir) not in diag_blob
    assert "OPENROUTER" not in diag_blob
    reasons = session.diagnostics()["bypass_reasons"]
    assert reasons == sorted(set(reasons)), "bypass reasons are sorted and unique"


def test_thirteen_field_legacy_shape_is_preserved_on_cache_hits(query_env):
    question = "shape check"
    query_env.client.hits_by_probe[question] = [
        hit(1, 0.9, make_payload("doc_shape", 0, "Shape.", page=4, title="Shape Title")),
    ]
    run_query(query_env, question)
    places, diag = run_query(query_env, question)
    assert diag["candidate_hits"] == 1
    assert len(places) == 1
    assert list(places[0].keys()) == cli.QUERY_OUTPUT_FIELDS
    assert len(places[0]) == 13
    assert places[0]["page"] == 4


def test_budget_exhaustion_bypasses_but_never_changes_the_result(query_env):
    question = "budget probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("bp", 0, "BP."))]
    expected, _ = run_query(query_env, question)

    query_env.client.search_calls = 0
    clock = FakeClock()
    session = query_env.make_session(clock=clock)
    session._budget._charged_ms = 99_000  # pre-exhausted: no optional cache work starts
    places, diag = run_query(query_env, question, session=session)
    assert places == expected, "ordinary result contract is unchanged"
    assert "cache_budget_exhausted" in diag["bypass_reasons"]
    assert diag["candidate_hits"] == 0 and diag["embedding_hits"] == 0
    assert query_env.client.search_calls == 1


def test_corrupt_cache_database_bypasses_and_is_never_deleted(query_env):
    question = "corrupt probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("cp", 0, "CP."))]
    db = query_env.cache_env.cache_dir / cache.CACHE_FILENAME
    run_query(query_env, question)
    db.write_bytes(b"garbage")
    os.chmod(db, 0o600)

    places, diag = run_query(query_env, question)
    assert len(places) == 1
    assert "cache_unavailable" in diag["bypass_reasons"]
    assert db.exists(), "retrieval never auto-deletes an unfamiliar database"


def test_namespaces_are_isolated(query_env):
    question = "namespace probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("ns", 0, "NS."))]
    run_query(query_env, question, session=query_env.make_session(namespace="caller-a"))

    _, isolated = run_query(query_env, question, session=query_env.make_session(namespace="caller-b"))
    assert isolated["candidate_misses"] == 1 and isolated["embedding_misses"] == 1
    assert isolated["embedding_request_count"] == 1
    assert isolated["namespace"] == "caller-b"


def test_embedding_revision_change_severs_reuse(query_env, monkeypatch):
    question = "model revision probe"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("mr", 0, "MR."))]
    run_query(query_env, question)
    monkeypatch.setenv("LIBRARY_QUERY_EMBEDDING_REVISION", "2")
    # The same collection guard protects both deployments; fingerprint mismatch bypasses.
    _, diag = run_query(query_env, question)
    assert diag["bypass_reasons"] == ["revision_incompatible"]
    assert diag["embedding_misses"] == 1, "old vectors cannot satisfy the new fingerprint"
    assert diag["embedding_request_count"] == 1


def test_cached_candidates_cannot_contradict_hard_filters(query_env):
    question = "filtered evidence"
    query_env.client.hits_by_probe[question] = [hit(1, 0.9, make_payload("d", 0, "D", tags=["allowed"]))]
    _, _cold = run_query(query_env, question, tags=["allowed"])
    db = query_env.cache_env.cache_dir / cache.CACHE_FILENAME
    conn = sqlite3.connect(db)
    key, raw = conn.execute("SELECT entry_key, payload FROM cache_entries WHERE tier='evidence'").fetchone()
    value = json.loads(raw)
    value["hits"][0]["payload"]["tags"] = ["forbidden"]
    raw = json.dumps(value).encode()
    conn.execute("UPDATE cache_entries SET payload=?, payload_bytes=? WHERE entry_key=?", (raw, len(raw), key))
    conn.commit()
    conn.close()
    places, diag = run_query(query_env, question, tags=["allowed"])
    assert places[0]["tags"] == ["allowed"]
    assert diag["candidate_hits"] == 0 and "entry_invalid" in diag["bypass_reasons"]


def test_live_validation_runs_under_shared_guard(query_env):
    session = query_env.make_session()
    def check():
        import fcntl
        fd = os.open(query_env.ctx.guard_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    with session:
        assert session.verify_index(check) is not None


def test_real_qdrant_dimension_shape_is_checked(query_env):
    query_env.client.get_collection = lambda **kwargs: SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=config.EMBEDDING_DIM + 1))))
    with pytest.raises(errors.ResearchIndexNotReadyError):
        run_query(query_env, "dimension mismatch")


def test_off_mode_reports_actual_work_without_cache_creation(query_env, capsys):
    assert cli.main(["query", "plain question", "--json", "--cache-mode", "off"]) == 0
    diag = json.loads(capsys.readouterr().out)["cache"]
    assert diag["embedding_request_count"] == diag["search_request_count"] == 1
    assert diag["embedding_hits"] == diag["embedding_misses"] == 0
    assert not query_env.cache_env.cache_dir.exists()
