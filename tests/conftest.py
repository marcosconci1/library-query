"""Shared pytest fixtures and test helpers for library-query."""
from pathlib import Path
import zipfile
import pytest


@pytest.fixture(autouse=True)
def isolated_revision_state(tmp_path, request):
    """Offline tests must never consult the operator's live revision records."""
    if request.node.get_closest_marker("live"):
        yield
        return
    from library_query import config
    with pytest.MonkeyPatch.context() as isolated:
        isolated.setattr(config, "LEDGER_PATH", tmp_path / "isolated-state" / "ledger.json")
        yield


def configure_readiness_client(client):
    """Collection metadata fake for indexing tests; state persists across calls."""
    from types import SimpleNamespace
    from library_query import config, readiness
    from qdrant_client.models import PayloadSchemaType
    metadata = {}
    def update(**kwargs):
        metadata.update(kwargs["metadata"])
        return True
    client.update_collection.side_effect = update
    client.get_collection.side_effect = lambda **kwargs: SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=config.EMBEDDING_DIM)),
                               metadata=metadata),
        payload_schema={key: SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)
                        for key in readiness.REQUIRED_INDEXES})
    client.scroll.return_value = ([], None)
    client.count.return_value = SimpleNamespace(count=0)


class FakeClock:
    """Controllable monotonic/wall clock for budget and TTL tests."""

    def __init__(self, start=1_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_settings(cache_dir, mode="use", **overrides):
    """A CacheSettings over a temporary directory with sane test defaults."""
    from library_query import cache

    params = dict(
        mode=mode,
        namespace="local",
        cache_dir=cache_dir,
        max_entries=cache.DEFAULT_MAX_ENTRIES,
        max_bytes=cache.DEFAULT_MAX_BYTES,
        max_entry_bytes=cache.DEFAULT_MAX_ENTRY_BYTES,
        embedding_ttl=cache.DEFAULT_EMBEDDING_TTL_SECONDS,
        evidence_ttl=cache.DEFAULT_EVIDENCE_TTL_SECONDS,
        empty_ttl=cache.DEFAULT_EMPTY_TTL_SECONDS,
    )
    params.update(overrides)
    return cache.CacheSettings(**params)


CACHE_ENV_VARS = [
    "LIBRARY_QUERY_CACHE_MODE",
    "LIBRARY_QUERY_CACHE_NAMESPACE",
    "LIBRARY_QUERY_CACHE_DIR",
    "LIBRARY_QUERY_CACHE_MAX_ENTRIES",
    "LIBRARY_QUERY_CACHE_MAX_BYTES",
    "LIBRARY_QUERY_CACHE_MAX_ENTRY_BYTES",
    "LIBRARY_QUERY_CACHE_EMBEDDING_TTL",
    "LIBRARY_QUERY_CACHE_EVIDENCE_TTL",
    "LIBRARY_QUERY_CACHE_EMPTY_TTL",
    "LIBRARY_QUERY_EMBEDDING_REVISION",
    "XDG_CACHE_HOME",
]


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """Isolated library/state/cache locations with all cache env removed."""
    from types import SimpleNamespace

    from library_query import cache, config

    monkeypatch.setattr(cache, "MONO_CLOCK", FakeClock())
    lib_root = tmp_path / "library"
    lib_root.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    cache_dir = tmp_path / "cache-dir"
    monkeypatch.setattr(config, "LIBRARY_ROOT", lib_root)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "LEDGER_PATH", state_dir / "ledger.json")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    for name in CACHE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIBRARY_QUERY_CACHE_DIR", str(cache_dir))
    return SimpleNamespace(lib_root=lib_root, state_dir=state_dir, cache_dir=cache_dir)


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


@pytest.fixture
def make_epub():
    return create_test_epub


@pytest.fixture
def make_pdf():
    return create_test_pdf
