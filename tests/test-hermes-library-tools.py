"""Hermes tool registration and exact-selection propagation without a gateway."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from library_query import browse, config, inspection, research, store

PLUGIN = Path(__file__).resolve().parents[1] / "integrations/hermes/library-query/__init__.py"
SKILL = Path(__file__).resolve().parents[1] / "integrations/hermes/library-query/skills/library-consult/SKILL.md"
spec = importlib.util.spec_from_file_location("library_query_native_tests", PLUGIN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(autouse=True)
def hermes_scope(monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_constants",
                        SimpleNamespace(hermes_home_key=lambda: "test-scope"))


def test_tool_registration():
    registered = {}
    ctx = SimpleNamespace(
        get_config=lambda key, default=None: default,
        register_tool=lambda **kw: registered.update({kw["name"]: kw}),
        register_skill=lambda name, path: None,
    )
    m.register(ctx)

    assert set(registered) == {
        "library_query", "library_browse", "library_inspect", "library_source", "library_passage"
    }
    assert "target" in registered["library_query"]["schema"]["parameters"]["properties"]
    assert all(tool["toolset"] == "library" for tool in registered.values())


def test_query_propagates_exact_target_and_weight(monkeypatch):
    monkeypatch.setattr(m, "preflight", lambda scope: None)
    monkeypatch.setattr(m, "judge", lambda scope, question, candidates: ("model", []))
    calls = []

    def call_cli(command, args=None, request=None):
        calls.append((command, args, request))
        return {"sources": []}

    monkeypatch.setattr(m, "call_cli", call_cli)
    target = {"item_ids": ["item-" + "a" * 32], "roles": ["note"]}
    result = json.loads(m.handler("test-scope", "research")({
        "question": "q", "weight": "important", "target": target
    }))

    assert result["receipt"]["weight"] == "important"
    assert len(calls) == 1
    command, args, request = calls[0]
    assert command == "research" and args is None
    assert request["target"] == target
    assert request["scope"] == "catalogued"
    assert request["schema_version"] == 2


def test_invalid_target_never_calls_cli(monkeypatch):
    monkeypatch.setattr(m, "call_cli", lambda *a, **kw: pytest.fail("No CLI call expected"))
    result = json.loads(m.handler("test-scope", "research")({
        "question": "q", "weight": "important", "target": {"item_ids": []}
    }))
    assert result["sources"] == []
    assert result["receipt"]["selected_count"] == 0
    assert result["receipt"]["filtering_status"] == "consultation_unavailable"


def test_invalid_passage_arguments_never_call_cli(monkeypatch):
    monkeypatch.setattr(m, "call_cli", lambda *a, **kw: pytest.fail("No CLI call expected"))
    result = json.loads(m.handler("test-scope", "passage")({
        "evidence_id": "evidence", "question": "q", "before": True
    }))
    assert result["sources"] == []
    assert result["receipt"]["selected_count"] == 0
    assert result["receipt"]["filtering_status"] == "consultation_unavailable"


def test_cli_errors_return_bounded_categories_without_raw_diagnostics(monkeypatch):
    def fail(*args, **kwargs):
        raise m.RetrievalError("retrieval_failed")

    monkeypatch.setattr(m, "call_cli", fail)
    result = json.loads(m.handler("test-scope", "browse")({"search": "title"}))
    assert result["sources"] == []
    assert result["receipt"]["filtering_status"] == "consultation_unavailable"
    assert result["receipt"]["filtering_reason"] == "retrieval_failed"
    assert "stderr" not in json.dumps(result)


@pytest.fixture
def synthetic_library(tmp_path, monkeypatch):
    """Isolated synthetic library with legacy, ambiguous, and catalogued control works."""
    root = tmp_path / "library"
    (root / "Books").mkdir(parents=True)
    monkeypatch.setattr(config, "LIBRARY_ROOT", root)
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "ledger.json")

    documents = {}
    payloads = {}

    def register_doc(rel_path: str, text: str, title: str, author: str, year: int, item_id: str | None = None):
        file_path = root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        doc_id = f"doc_{digest}"
        documents[doc_id] = {
            "path": rel_path,
            "content_hash": digest,
            "chunks": 1,
            "point_ids": [store.make_point_id(doc_id, 0)],
        }
        payloads[doc_id] = [{
            "document_id": doc_id,
            "chunk_index": 0,
            "text": text,
            "doc": rel_path,
            "doc_name": file_path.name,
            "scope": "catalogued",
            "type": "book",
            "title": title,
            "authors": [author],
            "year": year,
            "page": 42,
            "tags": [],
            "item_id": item_id,
        }]
        return doc_id

    legacy_doc_id = register_doc(
        "Books/Dijkstra_1976_Discipline_of_Programming/book.md",
        "Weakest preconditions and predicate transformers define guarded command semantics.",
        "Discipline of Programming",
        "Edsger W. Dijkstra",
        1976,
    )
    ambiguous_first_doc_id = register_doc(
        "Books/Alpha_2018_Systems_Design/book.md",
        "First systems design volume on distributed queues.",
        "Systems Design",
        "Alpha Author",
        2018,
    )
    ambiguous_second_doc_id = register_doc(
        "Books/Beta_2021_Systems_Design/book.md",
        "Second systems design volume on consensus protocols.",
        "Systems Design",
        "Beta Author",
        2021,
    )
    control_item_id = "item-" + "1" * 32
    control_doc_id = register_doc(
        "Books/catalogued-control.md",
        "Control catalogued book passage verifying item-targeted evidence flow.",
        "Catalogued Control",
        "Control Author",
        2024,
        item_id=control_item_id,
    )
    record_path = root / "Metadata" / f"{control_item_id}.md"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        "---\n"
        + yaml.safe_dump({
            "schema_version": 1,
            "item_id": control_item_id,
            "type": "book",
            "title": "Catalogued Control",
            "attachments": [{"path": "Books/catalogued-control.md", "role": "source"}],
        })
        + "---\n",
        encoding="utf-8",
    )
    (root / "catalog.bib").write_text(
        "@book{Control2024,\n"
        "  title = {Catalogued Control},\n"
        "  author = {Control Author},\n"
        "  year = {2024},\n"
        "  file = {Books/catalogued-control.md}\n"
        "}\n",
        encoding="utf-8",
    )
    config.LEDGER_PATH.write_text(json.dumps({"version": 2, "documents": documents}), encoding="utf-8")

    monkeypatch.setattr(
        store,
        "retrieve_chunks",
        lambda doc_id, indexes, *a, **kw: [
            SimpleNamespace(payload=p) for p in payloads.get(doc_id, []) if p["chunk_index"] in indexes
        ],
    )
    monkeypatch.setattr(store, "check_research_ready", lambda *a, **kw: len(documents))
    monkeypatch.setattr(store, "embed", lambda texts, **kw: [[0.1] for _ in texts])

    def fake_grouped(vector, query_filter=None, **kw):
        ids = next(c.match.any for c in query_filter.must if c.key == "document_id")
        return [
            SimpleNamespace(
                id=doc_id,
                hits=[
                    SimpleNamespace(id=store.make_point_id(doc_id, i), score=0.95, payload=p)
                    for i, p in enumerate(payloads[doc_id])
                ][:kw["group_size"]],
            )
            for doc_id in ids
            if doc_id in payloads
        ]

    monkeypatch.setattr(store, "query_points_grouped", fake_grouped)
    monkeypatch.setattr(m, "preflight", lambda scope: False)

    calls = []

    def fake_call_cli(command, args=None, request=None):
        calls.append((command, args, request))
        if command == "browse":
            return browse.browse_json(root, (args or {}).get("search"))
        if command == "inspect":
            return inspection.inspect_item(args["item"], root=root)
        if command == "source":
            return inspection.source(
                document_id=(args or {}).get("document"),
                path=(args or {}).get("path"),
                offset=(args or {}).get("offset", 0),
                limit=(args or {}).get("limit", inspection.SOURCE_LIMIT),
            )
        if command == "research":
            req = research.parse_request(json.dumps(request).encode("utf-8"))
            resolved = research.resolve_targets(req)
            extra = {"resolved_targets": resolved} if req.schema_version == 2 else {}
            return research.run(req, **extra)
        if command == "passage":
            return inspection.passage(
                args["evidence_id"],
                before=args.get("before", inspection.CONTEXT_COUNT),
                after=args.get("after", inspection.CONTEXT_COUNT),
                max_chars=args.get("max_chars", inspection.CONTEXT_CHARS),
            )
        raise AssertionError(f"Unexpected CLI command: {command}")

    monkeypatch.setattr(m, "call_cli", fake_call_cli)
    return SimpleNamespace(
        root=root,
        calls=calls,
        legacy_doc_id=legacy_doc_id,
        ambiguous_doc_ids=(ambiguous_first_doc_id, ambiguous_second_doc_id),
        control_item_id=control_item_id,
        control_doc_id=control_doc_id,
    )


def test_legacy_browse_to_source_to_document_targeted_evidence(synthetic_library):
    browse_out = json.loads(m.handler("test-scope", "browse")({"search": "Discipline Programming"}))
    assert browse_out["total_matches"] == 1
    assert browse_out["budget_omitted"] == 0
    match = browse_out["items"][0]
    assert match["item_id"] is None
    assert match["sources"] == ["Books/Dijkstra_1976_Discipline_of_Programming/book.md"]

    source_out = json.loads(m.handler("test-scope", "source")({"path": match["sources"][0], "limit": 1}))
    doc_id = source_out["document_id"]
    assert doc_id == synthetic_library.legacy_doc_id
    assert source_out["item_ids"] == []
    assert len(source_out["chunks"]) == 1
    assert "text" not in source_out["chunks"][0]

    query_out = json.loads(m.handler("test-scope", "research")({
        "question": "What defines guarded command semantics?",
        "weight": "important",
        "target": {"document_ids": [doc_id]},
    }))
    assert len(query_out["sources"]) == 1
    assert query_out["sources"][0]["source_id"] == doc_id
    passage = query_out["sources"][0]["passages"][0]
    assert "Weakest preconditions" in passage["text"]
    assert query_out["coverage"] == [{"identity": doc_id, "returned_evidence_ids": [passage["evidence_id"]]}]
    assert [c[0] for c in synthetic_library.calls] == ["browse", "source", "research"]


def test_ambiguous_title_browse_preserves_candidates_and_guidance_forbids_silent_choice(synthetic_library):
    browse_out = json.loads(m.handler("test-scope", "browse")({"search": "Systems Design"}))
    assert browse_out["total_matches"] == 2
    assert len(browse_out["items"]) == 2
    assert [item["year"] for item in browse_out["items"]] == [2018, 2021]
    assert [item["sources"] for item in browse_out["items"]] == [
        ["Books/Alpha_2018_Systems_Design/book.md"],
        ["Books/Beta_2021_Systems_Design/book.md"],
    ]

    skill_text = SKILL.read_text(encoding="utf-8")
    assert "item_id: null" in skill_text or "item_id` is `null`" in skill_text
    assert "library_source(path" in skill_text
    assert "document_ids" in skill_text
    assert "silently" in skill_text

    browse_desc = m.TOOLS["library_browse"][3]
    inspect_desc = m.TOOLS["library_inspect"][3]
    assert "library_source" in browse_desc and "document_ids" in browse_desc
    assert "silently" in browse_desc
    assert "item-" in inspect_desc and "library_source" in inspect_desc


@pytest.mark.parametrize(
    "bad_item",
    [
        "Books/Dijkstra_1976_Discipline_of_Programming/book.md",
        "null",
        None,
        "Discipline of Programming",
        "doc_" + "a" * 64,
        "item-not-valid",
    ],
)
def test_invalid_inspect_item_identifiers_return_distinct_actionable_guidance_without_cli(monkeypatch, bad_item):
    monkeypatch.setattr(m, "call_cli", lambda *a, **kw: pytest.fail("No CLI call expected"))
    result = json.loads(m.handler("test-scope", "inspect")({"item": bad_item}))

    assert result["sources"] == []
    assert result["receipt"]["candidate_count"] == 0
    assert result["receipt"]["selected_count"] == 0
    assert result["receipt"]["filtering_status"] == "consultation_unavailable"
    assert result["receipt"]["filtering_reason"] == "invalid_inspect_item"
    assert "library_source" in result["guidance"]
    assert "Library consultation remains available" not in result["guidance"]
    assert "document_ids" in result["guidance"]
    assert "item_id" in result["guidance"]
    assert "do not retry automatically" not in result["guidance"]
    assert "Continue from other context" not in result["guidance"]
    if isinstance(bad_item, str) and bad_item not in ("null",):
        assert bad_item not in json.dumps(result)


def test_valid_catalogued_control_browse_inspect_and_item_targeted_query(synthetic_library):
    browse_out = json.loads(m.handler("test-scope", "browse")({"search": "Catalogued Control"}))
    assert browse_out["total_matches"] == 1
    item_id = browse_out["items"][0]["item_id"]
    assert item_id == synthetic_library.control_item_id

    inspect_out = json.loads(m.handler("test-scope", "inspect")({"item": item_id}))
    assert len(inspect_out["items"]) == 1
    assert inspect_out["items"][0]["item_id"] == item_id
    assert inspect_out["items"][0]["attachments"] == [{"path": "Books/catalogued-control.md", "role": "source"}]

    query_out = json.loads(m.handler("test-scope", "research")({
        "question": "What does the control book verify?",
        "weight": "important",
        "target": {"item_ids": [item_id]},
    }))
    assert len(query_out["sources"]) == 1
    assert query_out["sources"][0]["item_ids"] == [item_id]
    passage = query_out["sources"][0]["passages"][0]
    assert "Control catalogued book passage" in passage["text"]
    assert query_out["coverage"] == [{"identity": item_id, "returned_evidence_ids": [passage["evidence_id"]]}]
    assert [c[0] for c in synthetic_library.calls] == ["browse", "inspect", "research"]
