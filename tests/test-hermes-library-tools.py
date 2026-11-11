"""Hermes tool registration and exact-selection propagation without a gateway."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "integrations/hermes/library-query/__init__.py"
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
