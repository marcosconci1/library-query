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


def _extract_pymupdf(path: Path) -> list[dict]:
    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "page": number})
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
EXTRACTOR_VERSION = 2


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


def chunk(pages: list[dict], chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """Merge page blocks into overlapping chunks, keeping the dominant page number."""
    chunks = []
    buf, buf_page = "", None
    for block in pages:
        page = block["page"]
        text = block["text"]
        if len(text) > chunk_size:
            if buf.strip():
                chunks.append({"text": buf.strip(), "page": buf_page})
                buf = ""
            step = max(1, chunk_size - overlap)
            for start in range(0, len(text), step):
                piece = text[start : start + chunk_size].strip()
                if piece:
                    chunks.append({"text": piece, "page": page})
            buf_page = None
            continue
        if buf and len(buf) + len(text) > chunk_size:
            chunks.append({"text": buf.strip(), "page": buf_page})
            buf = buf[-overlap:] if overlap else ""
            buf_page = page
        buf += ("\n" if buf else "") + text
        if buf_page is None:
            buf_page = page
    if buf.strip():
        chunks.append({"text": buf.strip(), "page": buf_page})
    return [c for c in chunks if len(c["text"]) > 80]
