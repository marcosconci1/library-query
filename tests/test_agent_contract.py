"""The agent-tool-contract capability: structured errors, exit codes, output shape, describe.

These tests assert the *contract* an autonomous caller depends on — a failure here is an
interface break, not a refactor regression.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from conftest import configure_readiness_client

from library_query import cli, config, errors, ledger, store


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch):
    """A valid library root and state dir, with Qdrant and embeddings stubbed out."""
    lib_root = tmp_path / "library"
    lib_root.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    monkeypatch.setattr(config, "LIBRARY_ROOT", lib_root)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")

    mock_client = MagicMock()
    configure_readiness_client(mock_client)
    mock_client.collection_exists.return_value = True
    monkeypatch.setattr(store, "qdrant", lambda: mock_client)

    return SimpleNamespace(lib_root=lib_root, state_dir=state_dir, client=mock_client)


def run_cli(argv: list[str]) -> int:
    """Invoke the CLI the way a caller does, returning the real exit code."""
    try:
        return cli.main(argv)
    except SystemExit as exc:  # argparse's own exit path
        return int(exc.code or 0)


def read_error(capsys) -> dict:
    """Parse the single JSON error line the contract requires on stderr."""
    err = capsys.readouterr().err.strip().splitlines()
    assert err, "expected a structured error line on stderr"
    payload = json.loads(err[-1])
    assert set(payload) >= {"error", "message", "exit_code"}
    return payload


# --- Section 1 / 6: structured errors ---------------------------------------------


def test_qdrant_unreachable_during_query(isolated_env, monkeypatch, capsys):
    """Scenario 1: Qdrant down during query -> qdrant_unreachable / exit 4."""
    monkeypatch.setattr(store, "embed", lambda texts: [[0.1] * 1536 for _ in texts])
    isolated_env.client.query_points.side_effect = httpx.ConnectError("connection refused")

    code = run_cli(["query", "anything", "--json"])
    payload = read_error(capsys)

    assert payload["error"] == "qdrant_unreachable"
    assert code == payload["exit_code"] == 4
    assert "qdrant" in payload["message"].lower()


def test_missing_api_key_during_index(isolated_env, monkeypatch, capsys):
    """Scenario 2: index needing embedding without an API key -> missing_api_key / exit 3."""
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    inbox = isolated_env.lib_root / "_Inbox"
    inbox.mkdir()
    (inbox / "doc.txt").write_text("Content that must be embedded to be indexed. " * 30)

    code = run_cli(["index"])
    payload = read_error(capsys)

    assert payload["error"] == "missing_api_key"
    assert code == payload["exit_code"] == 3


def test_internal_error_never_leaks_a_traceback(isolated_env, monkeypatch, capsys):
    """Scenario 3: an unclassified exception -> internal_error / exit 5, structured."""
    monkeypatch.setattr(ledger, "stats", MagicMock(side_effect=ValueError("something unexpected")))

    code = run_cli(["status"])
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip().splitlines()[-1])

    assert payload["error"] == "internal_error"
    assert code == payload["exit_code"] == 5
    assert payload["message"] == "something unexpected"
    assert "Traceback" not in captured.err


def test_invalid_scope_argument_exits_two(isolated_env, capsys):
    """Scenario 4: argparse rejects a bad --scope with exit 2, unchanged by the wrapper."""
    assert run_cli(["query", "x", "--scope", "invalid"]) == 2


def test_missing_library_root_exits_three(isolated_env, monkeypatch, capsys):
    """Scenario 5: LIBRARY_ROOT pointing nowhere -> invalid_library_root / exit 3."""
    monkeypatch.setattr(config, "LIBRARY_ROOT", isolated_env.lib_root / "does-not-exist")

    code = run_cli(["status"])
    payload = read_error(capsys)

    assert payload["error"] == "invalid_library_root"
    assert code == payload["exit_code"] == 3


def test_library_root_that_is_a_file_exits_three(isolated_env, monkeypatch, capsys):
    """A root that exists but is not a directory is equally a configuration error."""
    not_a_dir = isolated_env.lib_root / "regular-file"
    not_a_dir.write_text("x")
    monkeypatch.setattr(config, "LIBRARY_ROOT", not_a_dir)

    code = run_cli(["index"])
    payload = read_error(capsys)

    assert payload["error"] == "invalid_library_root"
    assert code == 3


def test_unregistered_extension_in_index_exits_three(isolated_env, monkeypatch, capsys):
    """Scenario: Configured suffix lacks an extractor -> unsupported_format / exit 3."""
    monkeypatch.setattr(config, "SUPPORTED_EXTENSIONS", {".pdf", ".epub", ".md", ".txt", ".odt"})

    code = run_cli(["index"])
    payload = read_error(capsys)

    assert payload["error"] == "unsupported_format"
    assert code == payload["exit_code"] == 3
    assert ".odt" in payload["message"]


def test_embedding_api_non_2xx_exits_four(isolated_env, monkeypatch, capsys):
    """Scenario 6: a non-2xx from the embedding endpoint -> embedding_api_error / exit 4."""
    monkeypatch.setattr(
        store.httpx,
        "post",
        lambda *a, **kw: httpx.Response(500, request=httpx.Request("POST", "http://embed")),
    )

    code = run_cli(["query", "anything", "--json"])
    payload = read_error(capsys)

    assert payload["error"] == "embedding_api_error"
    assert code == payload["exit_code"] == 4
    assert "500" in payload["message"]


def test_embedding_api_unreachable_exits_four(isolated_env, monkeypatch, capsys):
    """A transport failure reaching the embedding API is also exit 4."""
    def boom(*a, **kw):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(store.httpx, "post", boom)

    code = run_cli(["query", "anything", "--json"])
    payload = read_error(capsys)

    assert payload["error"] == "embedding_api_error"
    assert code == 4


def test_error_output_is_exactly_one_json_line(isolated_env, monkeypatch, capsys):
    """The contract is one line of JSON, not a multi-line dump."""
    monkeypatch.setattr(ledger, "stats", MagicMock(side_effect=RuntimeError("boom")))

    run_cli(["status"])
    lines = [line for line in capsys.readouterr().err.strip().splitlines() if line.strip()]

    assert len(lines) == 1
    json.loads(lines[0])


def test_structured_errors_do_not_depend_on_json_flag(isolated_env, monkeypatch, capsys):
    """Errors are structured whether or not --json was passed for successful output."""
    monkeypatch.setattr(store, "embed", lambda texts: [[0.1] * 1536 for _ in texts])
    isolated_env.client.query_points.side_effect = httpx.ConnectError("refused")

    run_cli(["query", "anything"])  # note: no --json
    payload = read_error(capsys)

    assert payload["error"] == "qdrant_unreachable"


# --- Section 2: exit code enforcement ---------------------------------------------


def test_every_documented_exit_code_is_reachable(isolated_env, monkeypatch, capsys):
    """All five documented codes are produced, and nothing outside the taxonomy is."""
    observed = set()

    # 0 — success (zero results is still success)
    monkeypatch.setattr(store, "embed", lambda texts: [[0.1] * 1536 for _ in texts])
    isolated_env.client.query_points.return_value = SimpleNamespace(points=[])
    observed.add(run_cli(["query", "nothing", "--json"]))

    # 2 — invalid arguments
    observed.add(run_cli(["query", "x", "--scope", "nope"]))

    # 3 — configuration error
    monkeypatch.setattr(config, "LIBRARY_ROOT", isolated_env.lib_root / "missing")
    observed.add(run_cli(["status"]))
    monkeypatch.setattr(config, "LIBRARY_ROOT", isolated_env.lib_root)

    # 4 — external service error
    isolated_env.client.query_points.side_effect = httpx.ConnectError("refused")
    observed.add(run_cli(["query", "x", "--json"]))
    isolated_env.client.query_points.side_effect = None

    # 5 — internal error
    monkeypatch.setattr(ledger, "stats", MagicMock(side_effect=ValueError("unexpected")))
    observed.add(run_cli(["status"]))

    assert observed == {0, 2, 3, 4, 5}
    assert observed <= errors.ALLOWED_EXIT_CODES


def test_no_error_category_maps_outside_the_taxonomy():
    """Every declared category resolves to a documented non-zero exit code."""
    assert set(errors.ERROR_EXIT_CODES.values()) <= errors.ALLOWED_EXIT_CODES - {0}
    for category, code in errors.ERROR_EXIT_CODES.items():
        assert code in (2, 3, 4, 5), f"{category} maps to undocumented exit code {code}"


def test_unknown_exception_types_all_classify(isolated_env):
    """classify() is total: no exception type escapes unclassified."""
    for exc in (ValueError("v"), KeyError("k"), OSError("o"), Exception()):
        category, message, code = errors.classify(exc)
        assert category in errors.ERROR_EXIT_CODES
        assert isinstance(message, str) and message
        assert code in errors.ALLOWED_EXIT_CODES


# --- Section 3: complete query output fields --------------------------------------


DOCUMENTED_FIELDS = [
    "doc",
    "doc_name",
    "page",
    "score",
    "text",
    "scope",
    "type",
    "title",
    "authors",
    "year",
    "subjects",
    "tags",
    "identifier",
]


def test_documented_field_list_is_frozen():
    """Schema-shape guard: renaming or removing a field requires a deliberate bump here.

    If this fails, the CLI's output contract changed. Update DOCUMENTED_FIELDS *and*
    the agent-tool-contract spec together — never one without the other.
    """
    assert cli.QUERY_OUTPUT_FIELDS == DOCUMENTED_FIELDS
    assert len(cli.QUERY_OUTPUT_FIELDS) == 13


def test_result_from_bare_payload_still_has_every_field(isolated_env, monkeypatch, capsys):
    """Scenarios 7 & 8: no page and no metadata still yield all keys, valued null."""
    monkeypatch.setattr(store, "embed", lambda texts: [[0.1] * 1536 for _ in texts])
    hit = SimpleNamespace(score=0.42, payload={"path": "_Inbox/note.md", "text": "body"})
    isolated_env.client.query_points.return_value = SimpleNamespace(points=[hit])

    assert run_cli(["query", "anything", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["places"][0]

    assert list(result) == DOCUMENTED_FIELDS
    assert result["page"] is None
    for field in ("title", "authors", "year", "subjects", "tags", "identifier"):
        assert result[field] is None, f"{field} should be null, not omitted"


def test_result_with_entirely_empty_payload_has_every_field(isolated_env, monkeypatch):
    """Even a payload-less hit produces the full, stable shape."""
    monkeypatch.setattr(store, "embed", lambda texts: [[0.1] * 1536 for _ in texts])
    hit = SimpleNamespace(score=0.1, payload=None)
    isolated_env.client.query_points.return_value = SimpleNamespace(points=[hit])

    place = store.query_places("q")[0]
    assert set(place) == set(DOCUMENTED_FIELDS)


def test_cli_restores_a_field_the_store_drops(isolated_env, monkeypatch, capsys):
    """The CLI enforces the shape even if the store regresses and omits a key."""
    monkeypatch.setattr(store, "query_places", lambda *a, **kw: [{"doc": "x", "score": 0.5}])

    assert run_cli(["query", "anything", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["places"][0]

    assert list(result) == DOCUMENTED_FIELDS
    assert result["identifier"] is None


def test_human_query_output_formatting_for_formats(isolated_env, monkeypatch, capsys):
    """Task 2.3: Human-readable query formats EPUB as section <n>, PDF as p.<n>, and txt without anchor."""
    epub_place = {
        "doc": "Books/Test/book.epub",
        "doc_name": "book.epub",
        "title": "EPUB Book",
        "authors": ["Author A"],
        "year": 2024,
        "page": 7,
        "score": 0.85,
        "scope": "catalogued",
        "type": "book",
        "text": "EPUB chunk text content",
        "subjects": None,
        "identifier": None,
    }
    pdf_place = {
        "doc": "Papers/Test/paper.pdf",
        "doc_name": "paper.pdf",
        "title": "PDF Paper",
        "authors": ["Author B"],
        "year": 2023,
        "page": 3,
        "score": 0.90,
        "scope": "catalogued",
        "type": "paper",
        "text": "PDF chunk text content",
        "subjects": None,
        "identifier": None,
    }
    txt_place = {
        "doc": "_Inbox/notes.txt",
        "doc_name": "notes.txt",
        "title": "Text Notes",
        "authors": None,
        "year": None,
        "page": None,
        "score": 0.75,
        "scope": "inbox",
        "type": "other",
        "text": "Plain text chunk content",
        "subjects": None,
        "identifier": None,
    }

    monkeypatch.setattr(store, "query_places", lambda *a, **kw: [epub_place, pdf_place, txt_place])

    # Human-readable output
    assert run_cli(["query", "question"]) == 0
    out = capsys.readouterr().out
    assert "EPUB Book (Author A | 2024) section 7" in out
    assert "PDF Paper (Author B | 2023) p.3" in out
    assert "Text Notes" in out
    assert "Text Notes p." not in out
    assert "Text Notes section" not in out

    # Machine-readable JSON output
    assert run_cli(["query", "question", "--json"]) == 0
    json_out = json.loads(capsys.readouterr().out)
    places = json_out["places"]
    assert places[0]["page"] == 7
    assert places[1]["page"] == 3
    assert places[2]["page"] is None


# --- Section 4: zero results is success -------------------------------------------


def test_zero_results_exits_zero_with_empty_places(isolated_env, monkeypatch, capsys):
    """Scenario 9: no matches is a successful, empty response — never an error."""
    monkeypatch.setattr(store, "embed", lambda texts: [[0.1] * 1536 for _ in texts])
    isolated_env.client.query_points.return_value = SimpleNamespace(points=[])

    code = run_cli(["query", "a topic the library never discusses", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    assert json.loads(captured.out)["places"] == []
    assert captured.err.strip() == "", "zero results must not be routed through the error path"


# --- Section 5: describe --json ----------------------------------------------------


def test_describe_enumerates_all_subcommands(capsys):
    """Scenario 10: describe covers every subcommand and query's output fields."""
    assert run_cli(["describe", "--json"]) == 0
    schema = json.loads(capsys.readouterr().out)

    assert set(schema["commands"]) == {
        "index", "readiness", "query", "research", "status", "sync-zotero-notes", "describe",
        "browse", "plan-layout", "cache", "inspect", "source", "passage", "find",
    }
    assert schema["commands"]["query"]["output_fields"] == DOCUMENTED_FIELDS
    assert schema["commands"]["browse"]["inventory_json_version"] == 1
    assert schema["commands"]["plan-layout"]["plan_version"] == 1
    assert schema["commands"]["plan-layout"]["naming"]["applying_host_verified"] is False


def test_describe_matches_the_real_parser(capsys):
    """Scenario 10 / task 5.4: described arguments cannot drift from the live parser."""
    assert run_cli(["describe", "--json"]) == 0
    schema = json.loads(capsys.readouterr().out)
    query_args = {arg["name"]: arg for arg in schema["commands"]["query"]["arguments"]}

    assert query_args["scope"]["choices"] == ["all", "inbox", "catalogued", "operations"]
    assert query_args["scope"]["default"] == "all"
    assert query_args["scope"]["type"] == "string"
    assert query_args["tag"]["default"] == []
    assert query_args["tag"]["type"] == "string"
    assert query_args["top_k"]["type"] == "integer"
    assert query_args["top_k"]["default"] == 5
    assert query_args["json"]["type"] == "boolean"
    assert query_args["question"]["positional"] is True

    # Cross-check against the parser itself, so adding a flag without describing it fails.
    parser = cli.build_parser()
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for name, subparser in sub_action.choices.items():
        real = {
            a.dest
            for a in subparser._actions
            if not isinstance(a, argparse._HelpAction)
        }
        described = {arg["name"] for arg in schema["commands"][name]["arguments"]}
        assert described == real, f"describe drifted from the parser for `{name}`"


def test_describe_exposes_the_error_contract(capsys):
    """An agent can read the exit-code taxonomy and error categories from describe."""
    assert run_cli(["describe", "--json"]) == 0
    schema = json.loads(capsys.readouterr().out)

    assert set(schema["exit_codes"]) == {"0", "2", "3", "4", "5"}
    assert schema["error_categories"] == dict(sorted(errors.ERROR_EXIT_CODES.items()))


def test_describe_has_no_side_effects(tmp_path, monkeypatch, capsys):
    """Scenario 11 / task 5.3: no filesystem scan, no Qdrant, no embedding call.

    Dependencies are pointed at values that would fail loudly if touched: a missing
    library root, an unset API key, and a Qdrant client that raises on any access.
    """
    monkeypatch.setattr(config, "LIBRARY_ROOT", tmp_path / "definitely-missing")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")

    def explode(*a, **kw):
        raise AssertionError("describe must not perform any I/O")

    monkeypatch.setattr(store, "qdrant", explode)
    monkeypatch.setattr(store, "embed", explode)
    monkeypatch.setattr(store.httpx, "post", explode)
    monkeypatch.setattr(ledger, "scan_content_files", explode)
    monkeypatch.setattr(ledger, "reconcile", explode)
    monkeypatch.setattr(ledger, "stats", explode)

    code = run_cli(["describe", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.err.strip() == ""
    json.loads(captured.out)  # stdout is the schema and nothing else


def test_query_normalizes_and_forwards_zotero_tags(isolated_env, monkeypatch, capsys):
    """Repeated --tag values become a deduplicated, normalized filter list."""
    captured = {}

    def fake_query_places(question, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return []

    monkeypatch.setattr(store, "query_places", fake_query_places)

    assert run_cli(["query", "attention", "--tag", "Machine Learning", "--tag", "machine learning", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert captured["tags"] == ["machine learning"]
    assert output["tags"] == ["machine learning"]
