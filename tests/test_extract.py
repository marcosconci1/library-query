"""Unit tests for document extraction, suffix registry, and validation."""
import io
from pathlib import Path
import zipfile
import pytest

from library_query import config, errors, extract


def create_test_epub(
    path: Path,
    title: str = "Test EPUB Title",
    author: str = "Test EPUB Author",
    sections: list[str] | None = None,
) -> Path:
    """Generate a valid minimal EPUB fixture without network access."""
    if sections is None:
        sections = [
            "Section 1: The foundational concepts of systems architecture and modular software design.",
            "Section 2: Practical applications and patterns for reliable autonomous agent tool contracts.",
        ]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )

        manifest_items = ['<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>']
        spine_items = []
        for i in range(len(sections)):
            manifest_items.append(f'<item id="s{i+1}" href="s{i+1}.xhtml" media-type="application/xhtml+xml"/>')
            spine_items.append(f'<itemref idref="s{i+1}"/>')

        opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="BookId" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{title}</dc:title>
    <dc:creator>{author}</dc:creator>
    <dc:identifier id="BookId">urn:uuid:12345</dc:identifier>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    {' '.join(manifest_items)}
  </manifest>
  <spine toc="ncx">
    {' '.join(spine_items)}
  </spine>
</package>"""
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr(
            "OEBPS/toc.ncx",
            f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="urn:uuid:12345"/></head>
  <docTitle><text>{title}</text></docTitle>
  <navMap/>
</ncx>""",
        )

        for i, text in enumerate(sections):
            xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Section {i+1}</title></head>
<body><p>{text}</p></body>
</html>"""
            zf.writestr(f"OEBPS/s{i+1}.xhtml", xhtml)

    return path


def create_test_pdf(
    path: Path,
    title: str = "Test PDF Title",
    author: str = "Test PDF Author",
    pages: list[str] | None = None,
) -> Path:
    """Generate a minimal PDF fixture using PyMuPDF."""
    import pymupdf

    if pages is None:
        pages = [
            "Page 1: Principles of distributed systems and robust data ingestion pipelines.",
            "Page 2: Vector search indexing strategies and semantic retrieval benchmarks.",
        ]
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    doc.set_metadata({"title": title, "author": author})
    doc.save(str(path))
    doc.close()
    return path


# --- Task 1.1: Registry and suffix dispatch --------------------------------------


def test_registry_contains_all_advertised_formats():
    """Registry explicitly contains .pdf, .epub, .md, and .txt."""
    expected = {".pdf", ".epub", ".md", ".txt"}
    assert expected.issubset(extract.REGISTRY.keys())
    assert extract.EXTRACTORS is extract.REGISTRY


def test_dispatch_mixed_case_suffixes(tmp_path: Path):
    """Extraction normalizes suffix and dispatches mixed-case extensions properly."""
    epub_file = create_test_epub(tmp_path / "Book.EPUB")
    pdf_file = create_test_pdf(tmp_path / "Document.Pdf")
    md_file = tmp_path / "Notes.Md"
    md_file.write_text("Markdown notes content for testing mixed-case dispatch.", encoding="utf-8")
    txt_file = tmp_path / "Data.TXT"
    txt_file.write_text("Text data content for testing mixed-case dispatch.", encoding="utf-8")

    epub_blocks = extract.extract(epub_file)
    assert len(epub_blocks) == 2
    assert epub_blocks[0]["page"] == 1

    pdf_blocks = extract.extract(pdf_file)
    assert len(pdf_blocks) == 2
    assert pdf_blocks[0]["page"] == 1

    md_blocks = extract.extract(md_file)
    assert len(md_blocks) == 1
    assert md_blocks[0]["page"] is None

    txt_blocks = extract.extract(txt_file)
    assert len(txt_blocks) == 1
    assert txt_blocks[0]["page"] is None


# --- Task 1.2: Plain-text extraction --------------------------------------------


def test_plain_text_extractor_emits_null_anchor(tmp_path: Path):
    """Plain-text extractor returns a single block with null page anchor."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Plain text content without page numbers.", encoding="utf-8")
    blocks = extract.extract(txt_file)
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Plain text content without page numbers."
    assert blocks[0]["page"] is None


def test_timestamped_markdown_becomes_video_location_blocks(tmp_path: Path):
    """Transcript headings stay with their text when later chunks are formed."""
    transcript = tmp_path / "Retrieval [abc123def45].en.md"
    transcript.write_text(
        "# Retrieval Design\n\n## [00:00:00] — Opening\n\nFirst passage.\n"
        "\n## [00:12:34] — Retrieval\n\nSecond passage.",
        encoding="utf-8",
    )

    blocks = extract.extract(transcript)

    assert [block["page"] for block in blocks] == [None, None]
    assert blocks[0]["text"].startswith("# Retrieval Design")
    assert blocks[1]["text"].startswith("## [00:12:34]")


def test_plain_text_extractor_preserves_replacement_handling(tmp_path: Path):
    """Non-UTF8 bytes in plain text files are replaced with replacement characters."""
    bad_file = tmp_path / "corrupt.txt"
    bad_file.write_bytes(b"Valid prefix \xff\xfe invalid suffix")
    blocks = extract.extract(bad_file)
    assert len(blocks) == 1
    assert "\ufffd" in blocks[0]["text"]
    assert blocks[0]["page"] is None


# --- Task 1.3: PyMuPDF extraction for PDF and EPUB ------------------------------


def test_epub_extraction_yields_sequential_one_based_anchors(tmp_path: Path):
    """EPUB extraction emits blocks with sequential 1-based page/section anchors."""
    epub_file = create_test_epub(
        tmp_path / "multi.epub",
        sections=[
            "Section one content describing algorithmic principles.",
            "Section two content analyzing asymptotic complexity.",
            "Section three content detailing benchmark results.",
        ],
    )
    blocks = extract.extract(epub_file)
    assert len(blocks) == 3
    assert [b["page"] for b in blocks] == [1, 2, 3]
    assert "Section one" in blocks[0]["text"]
    assert "Section two" in blocks[1]["text"]
    assert "Section three" in blocks[2]["text"]


def test_epub_empty_whitespace_units_are_omitted(tmp_path: Path):
    """Empty or whitespace-only sections in EPUB are omitted."""
    epub_file = create_test_epub(
        tmp_path / "with_empty.epub",
        sections=[
            "Section one with substantive text.",
            "   \n\t   ",  # whitespace only
            "Section three with more substantive text.",
        ],
    )
    blocks = extract.extract(epub_file)
    assert len(blocks) == 2
    assert blocks[0]["page"] == 1
    assert blocks[1]["page"] == 3


# --- Task 1.4: Direct unregistered-suffix extraction raises domain error --------


def test_unregistered_suffix_raises_unsupported_format_error(tmp_path: Path):
    """Direct extraction of an unregistered extension raises UnsupportedFormatError."""
    bin_file = tmp_path / "binary.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

    with pytest.raises(errors.UnsupportedFormatError) as exc_info:
        extract.extract(bin_file)

    assert "Unsupported format" in exc_info.value.message
    assert ".bin" in exc_info.value.message
    assert exc_info.value.category == "unsupported_format"
    assert exc_info.value.exit_code == errors.EXIT_CONFIG_ERROR


def test_unregistered_binary_never_decoded_as_text(tmp_path: Path):
    """Binary files with unregistered suffixes are never opened or decoded as text."""
    odt_file = tmp_path / "document.odt"
    odt_file.write_bytes(b"PK\x03\x04 fake zip binary data")

    with pytest.raises(errors.UnsupportedFormatError):
        extract.extract(odt_file)


# --- Task 1.5: Pre-reconciliation validation ------------------------------------


def test_validate_supported_extensions_success():
    """Default SUPPORTED_EXTENSIONS all have registered extractors."""
    extract.validate_supported_extensions()


def test_validate_supported_extensions_failure(monkeypatch):
    """Configured supported extension lacking an extractor raises classified error."""
    monkeypatch.setattr(config, "SUPPORTED_EXTENSIONS", {".pdf", ".epub", ".md", ".txt", ".odt"})

    with pytest.raises(errors.UnsupportedFormatError) as exc_info:
        extract.validate_supported_extensions()

    assert ".odt" in exc_info.value.message
    assert exc_info.value.category == "unsupported_format"


# --- Task 2.2: Chunking preserves EPUB anchor -----------------------------------


def test_chunking_preserves_first_contributing_epub_anchor():
    """Chunking preserves the 1-based section anchor of the first contributing block."""
    blocks = [
        {"text": "A" * 700, "page": 4},
        {"text": "B" * 700, "page": 5},
    ]
    chunks = extract.chunk(blocks, chunk_size=1000, overlap=100)
    assert len(chunks) >= 2
    assert chunks[0]["page"] == 4
    assert chunks[1]["page"] == 5


def test_chunking_splits_one_oversized_block_without_a_tail_duplicate():
    """One oversized Markdown/plain-text block must become bounded overlapping chunks."""
    text = "abcdefghijklmnopqrstuvwxyz" * 100

    chunks = extract.chunk([{"text": text, "page": None}], chunk_size=500, overlap=100)

    assert len(chunks) == 7
    assert all(len(chunk["text"]) <= 500 for chunk in chunks)
    assert all(chunk["page"] is None for chunk in chunks)
    assert all(
        previous["text"][-100:] == current["text"][:100]
        for previous, current in zip(chunks, chunks[1:])
    )
