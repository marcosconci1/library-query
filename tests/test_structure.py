"""Tests for structure classification, scope/type derivation, and metadata resolution."""
from pathlib import Path
from library_query import structure


def test_classify_file_roles(tmp_path: Path):
    # Setup simulated directory
    (tmp_path / "Collections").mkdir()
    (tmp_path / "Collections" / "README.md").write_text("Collections index")
    (tmp_path / "Collections" / "Math.md").write_text("Math topic")
    (tmp_path / "Templates" / "video-transcript").mkdir(parents=True)
    (tmp_path / "Templates" / "video-transcript" / "inbox-transcript.md").write_text("Template")
    (tmp_path / "README.md").write_text("Root README")
    (tmp_path / "catalog.bib").write_text("@book{...}")
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / ".DS_Store").write_text("")

    inbox = tmp_path / "_Inbox"
    inbox.mkdir()
    (inbox / "paper.pdf").write_text("%PDF-dummy")
    (inbox / "note.md").write_text("# Notes")
    (inbox / ".gitkeep").write_text("")

    book_dir = tmp_path / "Books" / "Martin_2008_Clean-Code"
    book_dir.mkdir(parents=True)
    (book_dir / "book.pdf").write_text("%PDF-clean-code")
    (book_dir / "README.md").write_text("---\ntitle: Clean Code\n---")

    # Navigation files
    assert structure.classify_file(tmp_path / "Collections" / "README.md", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / "Collections" / "Math.md", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / "Templates" / "video-transcript" / "inbox-transcript.md", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / "README.md", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / ".gitkeep", tmp_path) == "navigation"
    assert structure.classify_file(tmp_path / ".DS_Store", tmp_path) == "navigation"
    assert structure.classify_file(inbox / ".gitkeep", tmp_path) == "navigation"

    # Metadata files
    assert structure.classify_file(tmp_path / "catalog.bib", tmp_path) == "metadata"
    assert structure.classify_file(book_dir / "README.md", tmp_path) == "metadata"
    # Inbox markdown is pipeline bookkeeping, never library content.
    assert structure.classify_file(inbox / "note.md", tmp_path) == "metadata"

    # Content files
    (inbox / "story.epub").write_bytes(b"PK\x03\x04")
    (inbox / "memo.txt").write_text("plain memo")
    assert structure.classify_file(inbox / "paper.pdf", tmp_path) == "content"
    assert structure.classify_file(inbox / "story.epub", tmp_path) == "content"
    assert structure.classify_file(inbox / "memo.txt", tmp_path) == "content"
    assert structure.classify_file(book_dir / "book.pdf", tmp_path) == "content"


def test_inbox_markdown_is_bookkeeping_not_content(tmp_path: Path):
    """`_Inbox/*.md` are the processing skill's own notes about its work.

    Indexed as content, `_Inbox/UNSORTED.md` was retrievable at 0.64 for "unsorted
    inbox items" -- operational bookkeeping served back as library
    evidence. Excluded at the role boundary so it is never embedded and, on the next
    pass, is purged from the manifest as a non-content ghost.
    """
    inbox = tmp_path / "_Inbox"
    inbox.mkdir()

    for name in ("UNSORTED.md", "notes.md", "TODO.md", "readme.md"):
        (inbox / name).write_text("parking note")
        assert structure.classify_file(inbox / name, tmp_path) == "metadata", name

    # Scope is still the inbox; only the role changed.
    assert structure.get_scope_and_type(inbox / "UNSORTED.md", tmp_path) == ("inbox", "inbox")

    # Real inbox arrivals are unaffected: books land as PDF/EPUB, not markdown.
    for name in ("book.pdf", "story.epub", "memo.txt"):
        (inbox / name).write_bytes(b"data")
        assert structure.classify_file(inbox / name, tmp_path) == "content", name

    # Markdown elsewhere in the library is still content.
    other = tmp_path / "articles"
    other.mkdir()
    (other / "essay.md").write_text("a real essay")
    assert structure.classify_file(other / "essay.md", tmp_path) == "content"


def test_scope_and_type_derivation(tmp_path: Path):
    inbox_file = tmp_path / "_Inbox" / "doc.pdf"
    book_file = tmp_path / "Books" / "Martin_2008_Clean-Code" / "book.pdf"
    paper_file = tmp_path / "Papers" / "Vaswani_2017_Attention" / "paper.pdf"
    video_file = tmp_path / "Videos" / "Channel_2026_Retrieval" / "Retrieval [abc123def45].en.md"
    misc_file = tmp_path / "articles" / "blog.md"

    assert structure.get_scope_and_type(inbox_file, tmp_path) == ("inbox", "inbox")
    assert structure.get_scope_and_type(book_file, tmp_path) == ("catalogued", "book")
    assert structure.get_scope_and_type(paper_file, tmp_path) == ("catalogued", "paper")
    assert structure.get_scope_and_type(video_file, tmp_path) == ("catalogued", "video")
    assert structure.get_scope_and_type(misc_file, tmp_path) == ("catalogued", "other")


def test_root_catalog_operator_guide_uses_operations_scope(tmp_path):
    guide = tmp_path / "ZOTERO.md"
    assert structure.classify_file(guide, tmp_path) == "content"
    assert structure.get_scope_and_type(guide, tmp_path) == ("operations", "operation")
    # A source named the same way inside a shelf still follows its shelf.
    assert structure.get_scope_and_type(tmp_path / "Papers" / "ZOTERO.md", tmp_path) == ("catalogued", "paper")


def test_folder_name_parsing_fallback():
    # Standard format
    res1 = structure.parse_folder_name("Martin_2008_Clean-Code")
    assert res1["authors"] == ["Martin"]
    assert res1["year"] == 2008
    assert res1["title"] == "Clean Code"

    # Complex authors with underscores or hyphens
    res2 = structure.parse_folder_name("García_Márquez_1967_Cien-Años-de-Soledad")
    assert res2["authors"] == ["García Márquez"]
    assert res2["year"] == 1967
    assert res2["title"] == "Cien Años de Soledad"

    res3 = structure.parse_folder_name("Cormen-et-al_2022_Introduction-to-Algorithms")
    assert res3["authors"] == ["Cormen-et-al"]
    assert res3["year"] == 2022
    assert res3["title"] == "Introduction to Algorithms"

    # Malformed / non-standard format
    res4 = structure.parse_folder_name("Algorithms-Handbook")
    assert res4["authors"] is None
    assert res4["year"] is None
    assert res4["title"] == "Algorithms Handbook"


def test_metadata_resolution_hierarchy(tmp_path: Path):
    # Scenario: item with sidecar README frontmatter when no catalog exists.
    book_dir = tmp_path / "Books" / "Martin_2008_Clean-Code"
    book_dir.mkdir(parents=True)
    pdf_file = book_dir / "book.pdf"
    pdf_file.write_text("content")

    # Frontmatter in README
    readme = book_dir / "README.md"
    readme.write_text("""---
title: "Clean Code: A Handbook of Agile Software Craftsmanship"
authors:
  - Robert C. Martin
year: 2008
subjects:
  - software-craftsmanship
  - refactoring
identifier:
  isbn: "9780132350884"
status: read
---
# Notes
Detailed summary.
""")

    meta = structure.resolve_metadata(pdf_file, tmp_path)
    assert meta["title"] == "Clean Code: A Handbook of Agile Software Craftsmanship"
    assert meta["authors"] == ["Robert C. Martin"]
    assert meta["year"] == 2008
    assert meta["subjects"] == ["software-craftsmanship", "refactoring"]
    assert meta["identifier"] == {"isbn": "9780132350884"}


def test_metadata_resolution_bib_precedence(tmp_path: Path):
    # Scenario: item with catalog.bib entry but no README frontmatter.
    paper_dir = tmp_path / "Papers" / "Vaswani_2017_Attention"
    paper_dir.mkdir(parents=True)
    pdf_file = paper_dir / "paper.pdf"
    pdf_file.write_text("content")

    bib = tmp_path / "catalog.bib"
    bib.write_text("""@article{Vaswani_2017_Attention,
  title = {Attention Is All You Need},
  author = {Ashish Vaswani and Noam Shazeer},
  year = {2017},
  keywords = {transformers, machine-learning},
  doi = {10.48550/arXiv.1706.03762}
}
""")

    meta = structure.resolve_metadata(pdf_file, tmp_path)
    assert meta["title"] == "Attention Is All You Need"
    assert meta["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert meta["year"] == 2017
    assert meta["subjects"] == ["transformers", "machine-learning"]
    assert meta["tags"] == ["transformers", "machine-learning"]
    assert meta["citekey"] == "Vaswani_2017_Attention"
    assert meta["identifier"] == {"doi": "10.48550/arXiv.1706.03762"}


def test_video_transcript_resolves_catalog_metadata_by_linked_path(tmp_path: Path):
    video_dir = tmp_path / "Videos" / "Channel_2026_Retrieval"
    video_dir.mkdir(parents=True)
    transcript = video_dir / "Retrieval [abc123def45].transcript.en.md"
    transcript.write_text("## [00:00:00] — Opening\n\nTranscript text.", encoding="utf-8")
    (tmp_path / "catalog.bib").write_text(
        """@video{Channel2026Retrieval,
  title = {Retrieval Design},
  author = {Example Channel},
  year = {2026},
  keywords = {retrieval, video},
  url = {https://www.youtube.com/watch?v=abc123def45},
  file = {Videos/Channel_2026_Retrieval/Retrieval [abc123def45].transcript.en.md}
}
""",
        encoding="utf-8",
    )

    meta = structure.resolve_metadata(transcript, tmp_path)

    assert meta["title"] == "Retrieval Design"
    assert meta["authors"] == ["Example Channel"]
    assert meta["year"] == 2026
    assert meta["tags"] == ["retrieval", "video"]
    assert meta["citekey"] == "Channel2026Retrieval"


def test_catalog_export_overrides_bibliographic_readme_and_keeps_local_fields(tmp_path: Path):
    """A Zotero/Better BibTeX export is the source for bibliographic facts."""
    book_dir = tmp_path / "Books" / "Martin_2008_Clean-Code"
    book_dir.mkdir(parents=True)
    pdf_file = book_dir / "book.pdf"
    pdf_file.write_text("content")
    (book_dir / "README.md").write_text("""---
citekey: Martin2008CleanCode
title: "Stale Local Title"
authors: ["Stale Local Author"]
year: 2007
subjects: [software-craftsmanship]
identifier:
  isbn: "9780132350884"
status: reading
---
""")
    (tmp_path / "catalog.bib").write_text("""@book{Martin2008CleanCode,
  title = {Clean Code: A Handbook of Agile Software Craftsmanship},
  author = {Robert C. Martin},
  year = {2008},
  isbn = {9780132350884},
  file = {Clean Code:/srv/library/Books/Martin_2008_Clean-Code/book.pdf:application/pdf}
}
""")

    meta = structure.resolve_metadata(pdf_file, tmp_path)

    assert meta["title"] == "Clean Code: A Handbook of Agile Software Craftsmanship"
    assert meta["authors"] == ["Robert C. Martin"]
    assert meta["year"] == 2008
    assert meta["identifier"] == {"isbn": "9780132350884"}
    assert meta["subjects"] == ["software-craftsmanship"]


def test_epub_embedded_metadata_resolution(tmp_path: Path, make_epub):
    epub_file = make_epub(
        tmp_path / "book.epub",
        title="Embedded EPUB Title",
        author="Embedded EPUB Author",
    )
    meta = structure.resolve_metadata(epub_file, tmp_path)
    assert meta["title"] == "Embedded EPUB Title"
    assert meta["authors"] == ["Embedded EPUB Author"]


def test_epub_sidecar_precedence_over_embedded(tmp_path: Path, make_epub):
    book_dir = tmp_path / "Books" / "TestBook"
    book_dir.mkdir(parents=True)
    epub_file = make_epub(
        book_dir / "book.epub",
        title="Embedded EPUB Title",
        author="Embedded EPUB Author",
    )
    # Sidecar README with higher precedence
    readme = book_dir / "README.md"
    readme.write_text("""---
title: "Sidecar Overriding Title"
authors:
  - "Sidecar Author"
---
""")

    meta = structure.resolve_metadata(epub_file, tmp_path)
    assert meta["title"] == "Sidecar Overriding Title"
    assert meta["authors"] == ["Sidecar Author"]


def test_plain_text_metadata_resolution(tmp_path: Path):
    txt_file = tmp_path / "plain_doc.txt"
    txt_file.write_text("just text")
    meta = structure.resolve_metadata(txt_file, tmp_path)
    assert meta["title"] == "plain_doc"
    assert meta["authors"] is None


def test_graphify_output_is_not_library_content(tmp_path: Path):
    """Knowledge-graph tool output must never be embedded as a library document."""
    out = tmp_path / "graphify-out"
    out.mkdir()
    report = out / "GRAPH_REPORT.md"
    report.write_text("# Graph Report")
    assert structure.classify_file(report, tmp_path) == "navigation"
