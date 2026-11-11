"""Exact-literal leg: embeddings miss `__main__`; verbatim text matching must not."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from library_query import cli, errors, exact, research, store


# --- literal detection ------------------------------------------------------------


@pytest.mark.parametrize("question, expected", [
    ("O que é __main__ em Python?", ["__main__"]),
    ("what does `finally` do in Java?", ["finally"]),
    ("if __name__ == '__main__': explain", ["__main__", "__name__"]),   # quoted spans first
    ("Character.toLowerCase and isUpperCase", ["Character.toLowerCase", "isUpperCase"]),
    ("use `std::vector` with push_back", ["std::vector", "push_back"]),
    ('the phrase "make it stick" in the book', ["make it stick"]),
    ("what does main() return", ["main()"]),
    ("@Override annotation", ["@Override"]),
    # Prose must never trigger the exact leg.
    ("how do I stay calm before exams", []),
    ("e.g. the idea of i.e. a trade-off", []),
    ("A reader's books and Bob's notes", []),
    ("", []),
])
def test_literals(question, expected):
    assert exact.literals(question) == expected


def test_literals_cap_and_containment():
    assert len(exact.literals("a_b c_d e_f g_h")) == exact.MAX_LITERALS
    assert exact.literals("`__main__` and `main`") == ["__main__"]


# --- search -----------------------------------------------------------------------


def point(pid, document_id, chunk_index, text, **payload):
    return SimpleNamespace(id=pid, payload={"document_id": document_id, "chunk_index": chunk_index,
                                            "text": text, "doc": f"Books/{document_id}.pdf", **payload})


def fake_client(points, *, ready=True, page=2):
    client = MagicMock()
    pages = [points[i:i + page] for i in range(0, len(points), page)] or [[]]
    state = {"i": 0}

    def scroll(**kwargs):
        i = state["i"]
        state["i"] += 1
        return pages[i], (i + 1 if i + 1 < len(pages) else None)

    client.scroll.side_effect = scroll
    entry = SimpleNamespace(data_type="text", params=SimpleNamespace(phrase_matching=ready))
    client.get_collection.return_value = SimpleNamespace(payload_schema={"text": entry} if ready else {})
    return client


def test_search_verifies_the_literal_and_ignores_token_only_matches():
    # The word tokenizer reduces `__main__` to `main`; the substring check must reject these.
    client = fake_client([
        point(1, "java", 0, "public static void main(String[] args)"),
        point(2, "py", 0, "if __name__ == '__main__': run()"),
        point(3, "py", 1, "the __main__ module and __main__ again"),
    ])
    hits = exact.search(client, "c", ["__main__"])
    assert [h.id for h in hits] == [3, 2]          # more occurrences first
    assert hits[0].exact_matches == 2 and hits[0].score == exact.EXACT_SCORE
    assert hits[0].score > hits[1].score            # order survives a score sort


def test_search_is_case_sensitive():
    client = fake_client([point(1, "d", 0, "isuppercase is not isUpperCase")])
    assert [h.id for h in exact.search(client, "c", ["isUpperCase"])] == [1]
    client = fake_client([point(1, "d", 0, "only isuppercase here")])
    assert exact.search(client, "c", ["isUpperCase"]) == []


def test_search_keeps_a_few_chunks_per_document(monkeypatch):
    monkeypatch.setattr(exact, "PER_DOCUMENT", 2)
    client = fake_client([point(i, "big", i, "__main__ " * (10 - i)) for i in range(1, 6)]
                         + [point(9, "small", 0, "__main__")])
    hits = exact.search(client, "c", ["__main__"])
    assert sorted(h.payload["document_id"] for h in hits) == ["big", "big", "small"]


def test_search_returns_nothing_when_index_is_not_ready():
    client = fake_client([point(1, "d", 0, "__main__")], ready=False)
    assert exact.search(client, "c", ["__main__"]) == []
    client.scroll.assert_not_called()


def test_search_requires_all_literal_filters_in_the_query():
    client = fake_client([point(1, "d", 0, "__main__")])
    exact.search(client, "c", ["__main__"], None)
    flt = client.scroll.call_args.kwargs["scroll_filter"]
    assert flt.must[0].key == "text" and flt.must[0].match.phrase == "__main__"


def test_ensure_text_index_states():
    ready = fake_client([])
    assert exact.ensure_text_index(ready, "c") == "present"
    ready.create_payload_index.assert_not_called()

    missing = fake_client([], ready=False)
    assert exact.ensure_text_index(missing, "c") == "created"
    schema = missing.create_payload_index.call_args.kwargs["field_schema"]
    assert schema.phrase_matching is True and schema.on_disk is True

    broken = fake_client([], ready=False)
    broken.create_payload_index.side_effect = RuntimeError("qdrant down")
    assert exact.ensure_text_index(broken, "c") == "unavailable"   # never raises


def test_snippet_centres_on_the_literal():
    text = "x" * 500 + " if __name__ == '__main__': go() " + "y" * 500
    out = exact.snippet(text, "__main__", width=80)
    assert "__main__" in out and out.startswith("…") and out.endswith("…") and len(out) < 120


# --- research integration -----------------------------------------------------------


def test_research_adds_exact_hits_and_leaves_prose_questions_alone(monkeypatch):
    from test_research import FakeRetrieval, make_group, make_hit, make_payload, parse

    semantic = make_payload("doc_java", 0, "main method in Java", doc="Books/java.pdf", doc_type="book", page=4)
    FakeRetrieval({"what does __main__ do": [make_group("doc_java", [make_hit(1, 0.42, semantic)])]}).install(monkeypatch)
    wanted = make_payload("doc_py", 3, "if __name__ == '__main__': run()", doc="Books/py.pdf", doc_type="book", page=267)
    calls = []

    def fake_search(client, collection, literals, base_filter, **kwargs):
        calls.append(literals)
        return [SimpleNamespace(id=77, payload=wanted, score=1.0, exact_matches=1)]

    monkeypatch.setattr(exact, "search", fake_search)
    monkeypatch.setattr(store, "qdrant", lambda: MagicMock())

    bundle = research.run(parse({"queries": [{"text": "what does __main__ do"}], "max_sources": 3}), clock=lambda: 0.0)
    assert calls == [["__main__"]]
    first = bundle["sources"][0]
    assert first["source_id"] == "doc_py" and first["passages"][0]["score"] == 1.0
    assert first["passages"][0]["page"] == 267
    assert {s["source_id"] for s in bundle["sources"]} == {"doc_py", "doc_java"}

    calls.clear()
    FakeRetrieval({"how to stay calm": []}).install(monkeypatch)
    research.run(parse({"queries": [{"text": "how to stay calm"}]}), clock=lambda: 0.0)
    assert calls == []   # no literal, no exact leg


def test_research_survives_an_exact_leg_failure(monkeypatch):
    from test_research import FakeRetrieval, make_group, make_hit, make_payload, parse

    semantic = make_payload("doc_a", 0, "about modules", page=1)
    FakeRetrieval({"explain __main__": [make_group("doc_a", [make_hit(1, 0.7, semantic)])]}).install(monkeypatch)
    monkeypatch.setattr(store, "qdrant", lambda: MagicMock())

    def boom(*args, **kwargs):
        raise RuntimeError("qdrant exploded")

    monkeypatch.setattr(exact, "search", boom)
    bundle = research.run(parse({"queries": [{"text": "explain __main__"}]}), clock=lambda: 0.0)
    assert [s["source_id"] for s in bundle["sources"]] == ["doc_a"]


# --- find command -------------------------------------------------------------------


@pytest.mark.parametrize(("place", "expected"), [
    ({"doc_name": "book.epub", "page": 234}, " section 234"),
    ({"doc_name": "paper.pdf", "page": 234}, " PDF p.234"),
    ({"doc_name": "notes.md", "page": 234}, ""),
    ({"doc_name": "notes.txt", "page": 234}, ""),
    ({"doc_name": "book.epub", "page": None}, ""),
])
def test_location_label(place, expected):
    assert cli._location_label(place) == expected


def test_find_reports_verbatim_places(monkeypatch, capsys):
    client = fake_client([point(1, "py", 4, "if __name__ == '__main__': go()", doc_name="py.pdf",
                                title="Py Book", authors=["A. Writer"], year=2020, page=12, scope="catalogued")])
    monkeypatch.setattr(store, "qdrant", lambda: client)
    assert cli.main(["find", "__main__", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["literals"] == ["__main__"]
    assert out["places"][0]["page"] == 12 and out["places"][0]["matches"] == 1
    assert "__main__" in out["places"][0]["text"]


def test_find_human_output_labels_epub_and_pdf(monkeypatch, capsys):
    client = fake_client([
        point(1, "epub", 0, "__main__ in an EPUB", doc_name="book.epub",
              title="EPUB Book", page=234),
        point(2, "pdf", 0, "__main__ in a PDF", doc_name="paper.pdf",
              title="PDF Paper", page=234),
    ])
    monkeypatch.setattr(store, "qdrant", lambda: client)

    assert cli.main(["find", "__main__"]) == 0
    out = capsys.readouterr().out
    assert "EPUB Book section 234" in out
    assert "EPUB Book PDF p." not in out
    assert "PDF Paper PDF p.234" in out


def test_find_says_so_when_nothing_matches(monkeypatch, capsys):
    monkeypatch.setattr(store, "qdrant", lambda: fake_client([point(1, "d", 0, "nothing here")]))
    assert cli.main(["find", "__main__"]) == 0
    assert "No verbatim match" in capsys.readouterr().out


def test_find_fails_clearly_when_index_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(store, "qdrant", lambda: fake_client([], ready=False))
    code = cli.main(["find", "__main__"])
    assert code != 0 and "exact-text index is not built" in capsys.readouterr().err
