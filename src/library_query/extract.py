"""Document extraction and chunking.

Dispatches document extraction through a registry keyed by normalized lowercase suffix.
PDFs and EPUBs are extracted page/section-aware via PyMuPDF so every chunk
carries its source page or section anchor.
Plain text (.md, .txt) files are read as UTF-8 text with null anchors.
"""
import re
from collections.abc import Callable
from pathlib import Path

from . import config
from .errors import UnsupportedFormatError


def _labels_are_meaningful(doc) -> bool:
    """True when the PDF declares a real page-label scheme.

    No rules, or a single plain decimal rule starting at 1, labels every page with
    its own PDF index: that adds nothing and is not treated as printed pagination.
    Rules are only inspected for that check; the labels themselves always come from
    page.get_label(), which resolves prefixes and styles.
    """
    try:
        rules = doc.get_page_labels()
    except Exception:
        return False
    if not rules:
        return False
    if len(rules) == 1:
        rule = rules[0]
        if (rule.get("startpage") == 0 and rule.get("style") in ("D", "")
                and not rule.get("prefix") and rule.get("firstpagenum", 1) == 1):
            return False
    return True


def _outline(doc) -> list[tuple[int, str, int]]:
    """Outline entries (level, title, page) in document order, page >= 1 only."""
    try:
        toc = doc.get_toc()
    except Exception:
        return []
    return [(int(lvl), str(title), int(page)) for lvl, title, page, *_ in toc if int(page) >= 1]


def _section_for(outline, page: int) -> tuple[list[str], int | None]:
    """Ancestor title path and start page of the last outline entry at or before page."""
    stack: list[tuple[int, str]] = []
    start = None
    for lvl, title, entry_page in outline:
        if entry_page > page:
            continue
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        stack.append((lvl, title))
        start = entry_page
    return [title for _, title in stack], start


def _extract_pymupdf(path: Path) -> list[dict]:
    import pymupdf

    is_pdf = path.suffix.lower() == ".pdf"
    pages = []
    with pymupdf.open(path) as doc:
        labelled = is_pdf and _labels_are_meaningful(doc)
        outline = _outline(doc) if is_pdf else []

        def printed(number: int) -> str | None:
            if not labelled:
                return None
            return doc[number - 1].get_label() or None

        for number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            block = {"text": text, "page": number}
            if is_pdf:
                section_path, section_page = _section_for(outline, number)
                block["printed"] = printed(number)
                block["section_path"] = section_path
                block["section_start_page"] = section_page
                block["section_start_printed"] = printed(section_page) if section_page else None
            pages.append(block)
    return pages


# Retain backwards compatibility alias
_extract_pdf = _extract_pymupdf


def _extract_text(path: Path) -> list[dict]:
    return [{"text": path.read_text(encoding="utf-8", errors="replace"), "page": None}]


TIMESTAMP_HEADING = re.compile(
    r"(?=^##\s+\[\d{1,2}:\d{2}(?::\d{2})?\]\s*[-—])",
    re.MULTILINE,
)


def _extract_markdown(path: Path) -> list[dict]:
    """Keep timestamped transcript sections together as text blocks.

    Ordinary Markdown remains a single block. A video transcript formatted with
    ``## [HH:MM:SS] — topic`` headings becomes one block per timestamp, so a
    chunk starts with the source location it needs for later verification.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [block.strip() for block in TIMESTAMP_HEADING.split(text) if block.strip()]
    if len(blocks) > 1 and blocks[1].startswith("## ["):
        # Keep a transcript's title/provenance preamble with its first location.
        blocks[1] = f"{blocks[0]}\n\n{blocks[1]}"
        blocks.pop(0)
    return [{"text": block, "page": None} for block in blocks]


REGISTRY: dict[str, Callable[[Path], list[dict]]] = {
    ".pdf": _extract_pymupdf,
    ".epub": _extract_pymupdf,
    ".md": _extract_markdown,
    ".txt": _extract_text,
}

EXTRACTORS = REGISTRY

#: Passage-representation version: part of the indexed-representation fingerprint.
#: Bump when extraction/chunking output for identical bytes changes meaningfully.
#: 3: PDF chunks carry page range, printed page labels and outline section path.
EXTRACTOR_VERSION = 3

PDF_FIELDS = ("page_start", "page_end", "printed_start", "printed_end", "section_path", "section_start_page", "section_start_printed")


def validate_supported_extensions(extensions: set[str] | None = None) -> None:
    """Validate that every configured extension has a registered extractor."""
    if extensions is None:
        extensions = config.SUPPORTED_EXTENSIONS
    missing = sorted(ext.lower() for ext in extensions if ext.lower() not in REGISTRY)
    if missing:
        raise UnsupportedFormatError(
            f"Configured supported extensions lack registered extractors: {', '.join(missing)}"
        )


def extract(path: Path) -> list[dict]:
    """Return [{"text": str, "page": int | None}, ...] page/section blocks."""
    suffix = path.suffix.lower()
    extractor = REGISTRY.get(suffix)
    if extractor is None:
        raise UnsupportedFormatError(
            f"Unsupported format: '{path.suffix}' (file: {path.name}) has no registered extractor"
        )
    return extractor(path)


def _located(text: str, spans: list[tuple[int, dict]]) -> dict:
    """Chunk dict for text plus PDF location fields from the blocks it spans.

    spans maps buffer offsets to the block that contributed the characters from
    that offset on; the chunk is the stripped buffer, so its first and last
    characters decide the page range (overlap carried over included).
    """
    lead = len(text) - len(text.lstrip())
    last = len(text.rstrip()) - 1
    first_block = last_block = spans[0][1]
    for offset, block in spans:
        if offset <= lead:
            first_block = block
        if offset <= last:
            last_block = block
    return {
        "text": text.strip(),
        "page_start": first_block["page"],
        "page_end": last_block["page"],
        "printed_start": first_block.get("printed"),
        "printed_end": last_block.get("printed"),
        "section_path": list(first_block.get("section_path") or []),
        "section_start_page": first_block.get("section_start_page"),
        "section_start_printed": first_block.get("section_start_printed"),
    }


def chunk(pages: list[dict], chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """Merge page blocks into overlapping chunks.

    Each chunk's `page` is the page of the first block added to the buffer after
    the previous flush (kept for compatibility). PDF blocks (those carrying
    `printed`) additionally give each chunk page_start/page_end, printed_start/
    printed_end and the outline section of page_start. Other formats gain none of
    these keys. An oversized block is split into pieces that all carry that
    block's own page.
    """
    chunks = []
    buf, buf_page = "", None
    spans: list[tuple[int, dict]] = []  # (buffer offset, contributing block)

    def emit(text: str, page, text_spans) -> None:
        item = {"text": text.strip(), "page": page}
        if text_spans and "printed" in text_spans[0][1]:
            item.update(_located(text, text_spans), text=item["text"])
        chunks.append(item)

    for block in pages:
        page = block["page"]
        text = block["text"]
        if len(text) > chunk_size:
            if buf.strip():
                emit(buf, buf_page, spans)
                buf, spans = "", []
            step = max(1, chunk_size - overlap)
            for start in range(0, len(text), step):
                piece = text[start : start + chunk_size]
                if piece.strip():
                    emit(piece, page, [(0, block)])
            buf_page = None
            continue
        if buf and len(buf) + len(text) > chunk_size:
            emit(buf, buf_page, spans)
            kept = buf[-overlap:] if overlap else ""
            cut = len(buf) - len(kept)
            tail = [(max(off - cut, 0), blk) for off, blk in spans]
            # Drop spans that end before the cut: keep the last one starting at or before it.
            keep_from = max((i for i, (off, _) in enumerate(spans) if off <= cut), default=0)
            spans = tail[keep_from:] if kept else []
            buf = kept
            buf_page = page
        if buf:
            buf += "\n"
        spans.append((len(buf), block))
        buf += text
        if buf_page is None:
            buf_page = page
    if buf.strip():
        emit(buf, buf_page, spans)
    return [c for c in chunks if len(c["text"]) > 80]
