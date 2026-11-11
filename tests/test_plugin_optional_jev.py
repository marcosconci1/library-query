"""Optional Jev behavior in the Hermes library-query plugin."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "integrations/hermes/library-query"
PLUGIN = PLUGIN_DIR / "__init__.py"
spec = importlib.util.spec_from_file_location("library_query_optional_jev_tests", PLUGIN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def install_hermes_stubs(monkeypatch, jev_handler=None, configured=True):
    scope = "optional-jev-test-scope"
    registry = SimpleNamespace(snapshot_registration=lambda name, *, scope: (
        SimpleNamespace(handler=jev_handler) if jev_handler is not None else None))
    monkeypatch.setitem(sys.modules, "hermes_constants",
                        SimpleNamespace(hermes_home_key=lambda: scope))
    monkeypatch.setitem(sys.modules, "agent",
                        SimpleNamespace())
    monkeypatch.setitem(sys.modules, "agent.secret_scope", SimpleNamespace(
        current_secret_scope_home=lambda: None,
        get_secret=lambda name: "configured credential",
        is_multiplex_active=lambda: False,
    ))
    monkeypatch.setitem(sys.modules, "hermes_cli", SimpleNamespace())
    connection = {"backend": "typesafe", "model": "jev-latest"} if configured else {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", SimpleNamespace(
        load_config_readonly=lambda: {"plugins": {"entries": {"jev": {"settings": {
            "connection": connection}}}}}))
    monkeypatch.setitem(sys.modules, "tools", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tools.registry", SimpleNamespace(registry=registry))
    return scope


def raw_candidates(count=6, text_size=1600, per_source=1):
    return {"sources": [{
        "source_id": f"doc{i}", "title": f"Title {i}", "authors": ["Author"],
        "passages": [{"evidence_id": f"e{i}-{j}", "chunk_index": i * per_source + j,
                      "text": str(i) * (text_size - 1) + str(j), "text_truncated": True}
                     for j in range(per_source)],
    } for i in range(count)]}


def test_missing_jev_returns_unfiltered_ranked_evidence_with_caps(monkeypatch):
    scope = install_hermes_stubs(monkeypatch)
    assert m.preflight(scope) is False
    calls = []

    def retrieve(command, request=None, **kwargs):
        calls.append((command, request))
        return raw_candidates(per_source=2)

    monkeypatch.setattr(m, "call_cli", retrieve)
    monkeypatch.setattr(m, "judge", lambda *a: pytest.fail("judge must not run without Jev"))

    data = json.loads(m.handler(scope, "research")({"question": "focused", "weight": "important"}))
    passages = [p for source in data["sources"] for p in source["passages"]]

    assert [p["evidence_id"] for p in passages] == ["e0-0", "e0-1", "e1-0"]
    assert calls[0][0] == "research"
    assert calls[0][1]["max_sources"] == m.FULL_BUDGET["max_sources"]
    assert calls[0][1]["max_passages_per_source"] == m.FULL_BUDGET["max_passages_per_source"]
    assert calls[0][1]["max_excerpt_chars"] == m.FULL_BUDGET["max_excerpt_chars"]
    assert data["receipt"]["candidate_count"] == 12
    assert data["receipt"]["judged_candidate_count"] == 0
    assert data["receipt"]["selected_count"] == len(passages) == m.MAX_PASSAGES
    assert data["receipt"]["filtering_status"] == "unfiltered"
    assert data["receipt"]["jev_model"] is None
    assert sum(len(p["text"]) for p in passages) <= m.MAX_TEXT
    assert all(len(s["passages"]) <= m.FULL_BUDGET["max_passages_per_source"] for s in data["sources"])
    assert len(json.dumps(data, ensure_ascii=False, separators=(",", ":"))) <= m.MAX_RESPONSE
    assert "Relevance is not judged" in data["guidance"]


def test_registered_configured_jev_still_filters(monkeypatch):
    calls = []

    def jev_handler(args):
        calls.append(args)
        return {"ok": True, "model": "jev-latest", "answers": {
            "p0": {"type": "choice", "choice": "directly_useful"}}}

    scope = install_hermes_stubs(monkeypatch, jev_handler)
    monkeypatch.setattr(m, "call_cli", lambda *a, **k: raw_candidates(count=1))
    judge_calls = []
    original_judge = m.judge

    def tracked_judge(*args):
        judge_calls.append(args)
        return original_judge(*args)

    monkeypatch.setattr(m, "judge", tracked_judge)
    data = json.loads(m.handler(scope, "research")({"question": "focused", "weight": "important"}))

    assert len(judge_calls) == 1 and len(calls) == 1
    assert data["receipt"]["filtering_status"] == "filtered"
    assert data["receipt"]["jev_model"] == "jev-latest"
    assert data["receipt"]["selected_count"] == 1


def test_registered_unconfigured_jev_still_fails_closed(monkeypatch):
    scope = install_hermes_stubs(monkeypatch, lambda args: {}, configured=False)
    monkeypatch.setattr(m, "call_cli", lambda *a, **k: pytest.fail("must fail before retrieval"))

    data = json.loads(m.handler(scope, "research")({"question": "focused", "weight": "important"}))

    assert data["sources"] == []
    assert data["receipt"]["filtering_status"] == "filter_unavailable"
    assert data["receipt"]["filtering_reason"] == "filter_not_configured"


def test_plugin_manifest_has_no_jev_dependency():
    manifest = (PLUGIN_DIR / "plugin.yaml").read_text()
    assert "requires_plugins:" not in manifest
    assert not any("jev" in line.lower() for line in manifest.splitlines())


def test_plugin_can_register_without_jev_dependency(monkeypatch):
    manifest = (PLUGIN_DIR / "plugin.yaml").read_text()
    assert "requires_plugins:" not in manifest
    install_hermes_stubs(monkeypatch)
    registered = {}
    ctx = SimpleNamespace(
        get_config=lambda key, default=None: default,
        register_tool=lambda **kw: registered.update({kw["name"]: kw}),
        register_skill=lambda name, path: None,
    )

    m.register(ctx)

    assert set(registered) == {"library_query", "library_browse", "library_inspect",
                               "library_source", "library_passage"}

