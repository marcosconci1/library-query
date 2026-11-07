"""Work targeting, balanced comparisons, and content-verified reading."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from library_query import browse, cache, cli, config, errors, inspection, research, store, targets


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    monkeypatch.setattr(config, "LIBRARY_ROOT", root)
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "ledger.json")
    documents, payloads, ids = {}, {}, []
    for n in range(2):
        item_id = "item-" + str(n + 1) * 32
        ids.append(item_id)
        attachments = []
        for role, shelf in (("source", "Books"), ("note", "Notes")):
            relative = f"{shelf}/work-{n}-{role}.md"
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"Work {n} {role} body")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            doc_id = "doc_" + digest
            documents[doc_id] = {"path": relative, "content_hash": digest, "chunks": 3,
                                 "point_ids": [store.make_point_id(doc_id, i) for i in range(3)]}
            payloads[doc_id] = [{"document_id": doc_id, "chunk_index": i, "text": f"{role} passage {n} {i}",
                                 "doc": relative, "doc_name": path.name, "scope": "catalogued", "type": "book",
                                 "page": None, "tags": []} for i in range(3)]
            attachments.append({"path": relative, "role": role})
        record = root / "Metadata" / f"{item_id}.md"
        record.parent.mkdir(exist_ok=True)
        import yaml
        record.write_text("---\n" + yaml.safe_dump({"schema_version": 1, "item_id": item_id, "type": "book",
                                                    "title": f"Shared Title {n}", "attachments": attachments}) + "---\n")
    config.LEDGER_PATH.write_text(json.dumps({"version": 2, "documents": documents}))
    monkeypatch.setattr(store, "retrieve_chunks", lambda d, indexes, *args, **kw: [SimpleNamespace(payload=p) for p in payloads.get(d, []) if p["chunk_index"] in indexes])
    return SimpleNamespace(root=root, ids=ids, documents=documents, payloads=payloads)


def request(obj):
    return research.parse_request(json.dumps(obj).encode())


def test_discover_inspect_and_roles(library):
    matches = browse.browse_json(library.root, " shared   TITLE ")["items"]
    assert len(matches) == 2
    assert browse.browse_json(library.root, "missing")["items"] == []
    item = inspection.inspect_item(library.ids[0])["item"]
    assert len(item["sources"]) == len(item["notes"]) == 1
    source = targets.resolve(targets.parse({"item_ids": [library.ids[0]]}))
    notes = targets.resolve(targets.parse({"item_ids": [library.ids[0]], "roles": ["note"]}))
    assert set(source["document_ids_resolved"]).isdisjoint(notes["document_ids_resolved"])
    empty = targets.resolve(targets.parse({"item_ids": [library.ids[0]], "roles": ["supplement"]}))
    assert empty["document_ids_resolved"] == []


@pytest.mark.parametrize("value", [None, {}, {"item_ids": []}, {"item_ids": ["title"]},
                                        {"document_ids": ["doc_" + "a" * 64], "roles": ["source"]},
                                        {"item_ids": ["item-" + "a" * 32], "roles": [False]}])
def test_invalid_targets_never_become_unrestricted(value):
    with pytest.raises(errors.InvalidResearchRequestError):
        request({"schema_version": 2, "target": value, "queries": [{"text": "question"}]})


def test_version_and_override(library):
    with pytest.raises(errors.InvalidResearchRequestError):
        request({"target": {"item_ids": [library.ids[0]]}, "queries": [{"text": "question"}]})
    r = request({"schema_version": 2, "target": {"item_ids": [library.ids[0]]}, "queries": [
        {"text": "question"}, {"text": "question", "target": {"item_ids": [library.ids[1]], "roles": ["note"]}}]})
    assert r.queries[0].target.ids == (library.ids[0],)
    assert r.queries[1].target.ids == (library.ids[1],)
    assert r.queries[0].dedupe_key != r.queries[1].dedupe_key


def test_unknown_stale_and_duplicate_content(library):
    with pytest.raises(errors.LibraryQueryError, match="not found") as exc:
        targets.resolve(targets.parse({"item_ids": ["item-" + "f" * 32]}))
    assert exc.value.category == "target_not_found"
    original = config.LEDGER_PATH.read_bytes()
    (library.root / "Books/work-0-source.md").write_text("changed content")
    with pytest.raises(errors.LibraryQueryError) as exc:
        targets.resolve(targets.parse({"item_ids": [library.ids[0]]}))
    assert exc.value.category == "target_not_indexed"
    assert config.LEDGER_PATH.read_bytes() == original
    (library.root / "Books/work-0-source.md").write_bytes((library.root / "Books/work-1-source.md").read_bytes())
    resolved = targets.resolve(targets.parse({"item_ids": library.ids}))
    assert resolved["members"][0]["document_ids"] == resolved["members"][1]["document_ids"]


def pipeline(library, monkeypatch):
    monkeypatch.setattr(store, "check_research_ready", lambda *a: 12)
    monkeypatch.setattr(store, "embed", lambda texts, **kw: [[0.0] for _ in texts])
    calls = []
    def search(vector, query_filter=None, **kw):
        ids = next(c.match.any for c in query_filter.must if c.key == "document_id")
        calls.append(ids)
        return [SimpleNamespace(id=d, hits=[SimpleNamespace(id=store.make_point_id(d, i),
                    score=0.9 if "work-0" in p["doc"] else 0.5, payload=p)
                    for i, p in enumerate(library.payloads[d])][:kw["group_size"]]) for d in ids]
    monkeypatch.setattr(store, "query_points_grouped", search)
    return calls


def test_comparison_reserves_work_and_reports_budget(library, monkeypatch):
    calls = pipeline(library, monkeypatch)
    obj = {"schema_version": 2, "target": {"item_ids": library.ids}, "queries": [{"text": "compare"}], "max_sources": 2}
    result = research.run(request(obj))
    assert len(calls) == 2
    assert len(result["sources"]) == 2
    assert [r["status"] for r in result["query_results"][0]["target_results"]] == ["selected", "selected"]
    obj["max_sources"] = 1
    result = research.run(request(obj))
    assert [r["status"] for r in result["query_results"][0]["target_results"]] == ["selected", "budget_excluded"]


def test_empty_role_and_off_target(library, monkeypatch):
    pipeline(library, monkeypatch)
    obj = {"schema_version": 2, "target": {"item_ids": [library.ids[0]], "roles": ["supplement"]}, "queries": [{"text": "q"}]}
    assert research.run(request(obj))["sources"] == []
    obj["target"]["roles"] = ["source"]
    monkeypatch.setattr(store, "query_points_grouped", lambda *a, **kw: [SimpleNamespace(id="wrong", hits=[])])
    with pytest.raises(errors.LibraryQueryError):
        research.run(request(obj))


def test_source_and_verified_context(library):
    source = inspection.source(path="Books/work-0-source.md", limit=1)
    assert source["total_chunks"] == 3 and source["next_offset"] == 1
    doc = source["document_id"]
    center = inspection.descriptor(library.payloads[doc][1], 200)["evidence_id"]
    result = inspection.passage(center, max_chars=5)
    assert result["passages"][0]["chunk_index"] == 1
    assert result["passages"][0]["text_truncated"]
    assert result["omitted_chunk_indexes"] == [0, 2]
    result = inspection.passage(center)
    assert [p["chunk_index"] for p in result["passages"]] == [0, 1, 2]
    library.payloads[doc][1]["text"] = "new text"
    with pytest.raises(errors.LibraryQueryError) as exc:
        inspection.passage(center)
    assert exc.value.category == "stale_evidence"


def test_candidate_membership_validation(library):
    doc = next(iter(library.documents))
    filters = {"scope": "all", "target": {"document_ids_resolved": [doc]}}
    hit = {"payload": {"document_id": "other"}}
    assert not cache._matches_filters(hit, filters)
    hit["payload"]["document_id"] = doc
    assert cache._matches_filters(hit, filters)


def test_cli_new_errors_and_discovery(library, capsys):
    assert cli.main(["browse", "--search", "title", "--write"]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_research_request"
    assert cli.main(["inspect", "--item", "item-" + "f" * 32, "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "target_not_found"
    assert cli.main(["describe", "--json"]) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["commands"]["research"]["supported_versions"] == [1, 2]
    assert {"inspect", "source", "passage"} <= description["commands"].keys()


def test_generated_notes_missing_attachment_and_legacy(library):
    record = library.root / "Metadata" / (library.ids[0] + ".md")
    original = record.read_text()
    generated_path = f"Notes/{library.ids[0]}-zotero.md"
    record.write_text(original.replace("Notes/work-0-note.md", generated_path))
    old = library.root / "Notes/work-0-note.md"
    old.rename(library.root / generated_path)
    # Deterministic generated mapping also applies when the record does not declare the note.
    import yaml
    data = yaml.safe_load(record.read_text().split("---")[1])
    data["attachments"] = [a for a in data["attachments"] if a["role"] != "note"]
    record.write_text("---\n" + yaml.safe_dump(data) + "---\n")
    selected = targets.resolve(targets.parse({"item_ids": [library.ids[0]], "roles": ["note"]}))
    assert selected["members"][0]["paths"] == [generated_path]
    (library.root / "Books/work-0-source.md").unlink()
    with pytest.raises(errors.LibraryQueryError) as exc:
        targets.resolve(targets.parse({"item_ids": [library.ids[0]]}))
    assert exc.value.category == "target_not_indexed"
    record.unlink()
    doc = next(d for d, r in library.documents.items() if r["path"] == "Books/work-1-source.md")
    assert targets.resolve(targets.parse({"document_ids": [doc]}))["document_ids_resolved"] == [doc]


def test_targeted_cache_keys_and_old_entry_isolation(library, tmp_path):
    from conftest import make_settings
    verified = SimpleNamespace(corpus_id="corpus", incarnation="inc", revision=1,
                               embedding_fingerprint="emb", representation_fingerprint="rep")
    settings = make_settings(tmp_path / "cache")
    resolved = [targets.resolve(targets.parse({"item_ids": [identity]})) for identity in library.ids]
    keys = []
    for selection in [None, *resolved, targets.resolve(targets.parse({"item_ids": [library.ids[0]], "roles": ["note"]}))]:
        filters = cache.grouped_filters("catalogued", [], [])
        if selection is not None:
            filters["target"] = selection
        keys.append(cache.candidate_key(settings, probe_text="same", verified=verified, method=cache.METHOD_GROUPED,
                                        filters=filters, window={"groups": 3, "group_size": 2}))
    assert len(set(keys)) == 4


def test_selected_document_absent_from_live_index(library, monkeypatch):
    monkeypatch.setattr(store, "retrieve_chunks", lambda *a, **kw: [])
    selected = targets.resolve(targets.parse({"item_ids": [library.ids[0]]}))
    with pytest.raises(errors.LibraryQueryError) as exc:
        targets.verify_index(selected)
    assert exc.value.category == "target_not_indexed"


def test_multiple_source_formats_resolve_together(library):
    # Text bytes stand in for extractable attachments: resolution never extracts.
    path = library.root / "Books/work-0.epub"
    path.write_text("alternate source bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state = json.loads(config.LEDGER_PATH.read_text())
    state["documents"]["doc_" + digest] = {"path": "Books/work-0.epub", "content_hash": digest,
                                         "chunks": 1, "point_ids": [123]}
    config.LEDGER_PATH.write_text(json.dumps(state))
    import yaml
    record = library.root / "Metadata" / (library.ids[0] + ".md")
    data = yaml.safe_load(record.read_text().split("---")[1])
    data["attachments"].append({"path": "Books/work-0.epub", "role": "source"})
    record.write_text("---\n" + yaml.safe_dump(data) + "---\n")
    selected = targets.resolve(targets.parse({"item_ids": [library.ids[0]]}))
    assert len(selected["document_ids_resolved"]) == 2


def test_per_question_passage_budget_is_honest(library, monkeypatch):
    doc = next(iter(library.documents))
    monkeypatch.setattr(store, "check_research_ready", lambda *a: 12)
    monkeypatch.setattr(store, "embed", lambda texts, **kw: [[float(i)] for i in range(len(texts))])
    def search(vector, **kw):
        index = int(vector[0])
        return [SimpleNamespace(id=doc, hits=[SimpleNamespace(id=index, score=0.7,
                payload=library.payloads[doc][index])])]
    monkeypatch.setattr(store, "query_points_grouped", search)
    obj = {"schema_version": 2, "target": {"document_ids": [doc]},
           "queries": [{"text": "first"}, {"text": "second"}], "max_passages_per_source": 1}
    result = research.run(request(obj))
    assert [r["status"] for r in result["query_results"]] == ["selected", "budget_excluded"]
    obj["max_passages_per_source"] = 2
    result = research.run(request(obj))
    assert [r["status"] for r in result["query_results"]] == ["selected", "selected"]
