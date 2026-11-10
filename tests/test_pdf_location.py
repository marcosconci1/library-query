from types import SimpleNamespace

import pymupdf
import pytest

from library_query import extract, inspection, research, store

FILLER = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor. "


def make_pdf(path, pages=6, labels=None, toc=None, text=None):
    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page()
        body = text(number) if text else f"Page {number + 1} content. " + FILLER * 3
        page.insert_textbox(pymupdf.Rect(36, 36, 560, 800), body, fontsize=9)
    if labels:
        doc.set_page_labels(labels)
    if toc:
        doc.set_toc(toc)
    doc.save(path)
    doc.close()
    return path


def chunks_for(path, **kw):
    return extract.chunk(extract.extract(path), **kw)


def test_labels_and_three_level_outline(tmp_path):
    pdf = make_pdf(
        tmp_path / "a.pdf", pages=6,
        labels=[{"startpage": 0, "style": "r", "firstpagenum": 1},
                {"startpage": 2, "style": "D", "firstpagenum": 1}],
        toc=[[1, "Part", 3], [2, "Chapter", 3], [3, "Section", 5]],
    )
    by_start = {c["page_start"]: c for c in chunks_for(pdf, chunk_size=300, overlap=0)}
    first = by_start[1]
    assert first["printed_start"] == "i"
    assert first["section_path"] == [] and first["section_start_page"] is None
    third = by_start[3]
    assert third["printed_start"] == "1"
    assert third["section_path"] == ["Part", "Chapter"]
    assert third["section_start_page"] == 3 and third["section_start_printed"] == "1"
    fifth = by_start[5]
    assert fifth["section_path"] == ["Part", "Chapter", "Section"]
    assert fifth["section_start_page"] == 5 and fifth["section_start_printed"] == "3"
    assert all(c["page_end"] >= c["page_start"] for c in by_start.values())


def test_no_labels_still_has_section(tmp_path):
    pdf = make_pdf(tmp_path / "b.pdf", toc=[[1, "Intro", 1]])
    for c in chunks_for(pdf):
        assert c["printed_start"] is None and c["printed_end"] is None
        assert c["section_path"] == ["Intro"]


def test_plain_decimal_rule_is_not_labelled(tmp_path):
    pdf = make_pdf(tmp_path / "c.pdf", labels=[{"startpage": 0, "style": "D", "firstpagenum": 1}])
    for c in chunks_for(pdf):
        assert c["printed_start"] is None and c["printed_end"] is None


def test_no_outline_means_no_section(tmp_path):
    pdf = make_pdf(tmp_path / "d.pdf")
    chunk = chunks_for(pdf)[0]
    assert chunk["section_path"] == []
    payload = {"text": chunk["text"], "page": chunk["page"], "doc_name": "d.pdf", "chunk_index": 0,
               "document_id": "x", **{f: chunk[f] for f in store.PDF_LOCATION_FIELDS}}
    bundle = research._passage_bundle(
        {"hit": SimpleNamespace(payload=payload), "by_question": {0: 0.5}, "score": 0.5},
        research.ResearchRequest(queries=()),
    )
    assert "section" in bundle and bundle["section"] is None
    assert inspection.descriptor(payload, 100)["section"] is None


def test_chunk_spanning_pages_and_oversized_split(tmp_path):
    pdf = make_pdf(tmp_path / "e.pdf", pages=3, text=lambda n: f"P{n + 1} " + "word " * 30)
    spanning = chunks_for(pdf, chunk_size=400, overlap=0)
    assert any(c["page_end"] > c["page_start"] for c in spanning)
    blocks = [{"text": "x" * 500, "page": 4, "printed": "iv", "section_path": [],
               "section_start_page": None, "section_start_printed": None},
              {"text": "y" * 500, "page": 5, "printed": "v", "section_path": [],
               "section_start_page": None, "section_start_printed": None}]
    pieces = extract.chunk(blocks, chunk_size=200, overlap=20)
    for piece in pieces:
        assert piece["page_start"] == piece["page_end"] == piece["page"]
        assert piece["printed_start"] == {4: "iv", 5: "v"}[piece["page"]]


def test_overlap_carry_over_page_start():
    def block(page, ch):
        return {"text": ch * 60, "page": page, "printed": None, "section_path": [],
                "section_start_page": None, "section_start_printed": None}
    out = extract.chunk([block(1, "a"), block(2, "b"), block(3, "c")], chunk_size=130, overlap=50)
    # chunk 1 = pages 1-2; its overlap tail is page 2 text only, so chunk 2 starts on page 2
    assert out[0]["page_start"] == 1 and out[0]["page_end"] == 2
    assert out[1]["page_start"] == 2 and out[1]["page_end"] == 3


@pytest.mark.parametrize("suffix", [".md", ".txt"])
def test_text_formats_gain_no_pdf_fields(tmp_path, suffix):
    path = tmp_path / f"n{suffix}"
    path.write_text("some plain words here. " * 20)
    chunks = chunks_for(path)
    assert chunks
    for c in chunks:
        assert not set(extract.PDF_FIELDS) & set(c)


def test_old_payload_has_no_section():
    payload = {"text": "t", "page": 3, "doc_name": "old.pdf", "chunk_index": 0, "document_id": "d"}
    assert research._location_hint(payload) == {"kind": "page", "value": 3}
    assert "section" not in inspection.descriptor(payload, 10)
    bundle = research._passage_bundle(
        {"hit": SimpleNamespace(payload=payload), "by_question": {0: 0.1}, "score": 0.1},
        research.ResearchRequest(queries=()),
    )
    assert "section" not in bundle


def test_extractor_version():
    assert extract.EXTRACTOR_VERSION == 3


def test_query_place_keeps_thirteen_fields():
    payload = {"path": "a.pdf", "doc_name": "a.pdf", "page": 2, "text": "t", "page_start": 2,
               "printed_start": "ii", "section_path": ["X"]}
    place = store._place_from_hit(SimpleNamespace(payload=payload, score=0.5))
    assert len(place) == 13
    assert not set(store.PDF_LOCATION_FIELDS) & set(place)


def test_legacy_page_can_be_later_than_page_start(tmp_path):
    # Page 1 is long enough that the second chunk starts with overlap text from page 1
    # and adds its own text from page 2. The legacy `page` then names page 2 while
    # `page_start` names the page where the chunk text really starts.
    pdf = make_pdf(
        tmp_path / "f.pdf", pages=3,
        text=lambda n: f"Page {n + 1} marker. " + FILLER * 4,
    )
    chunks = chunks_for(pdf, chunk_size=600, overlap=150)
    carried = [c for c in chunks if c["page_start"] == 1 and c["page"] == 2]
    assert carried, [(c["page"], c["page_start"], c["page_end"]) for c in chunks]
    chunk = carried[0]
    assert chunk["page_end"] == 2
    assert "Page 2 marker" in chunk["text"]
