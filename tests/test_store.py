"""Tests for store module: deterministic point IDs, payload operations, and filtering."""
from unittest.mock import MagicMock
import json
import pytest
from qdrant_client.models import PointsList, PointStruct
from library_query import config, errors, store


def test_qdrant_client_is_built_with_an_explicit_timeout(monkeypatch):
    """The 5s client default times out under bulk upsert; pin the generous one."""
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(store, "QdrantClient", FakeClient)
    monkeypatch.setattr(store, "_client", None)

    store.qdrant()

    assert captured["timeout"] == config.QDRANT_TIMEOUT
    assert config.QDRANT_TIMEOUT >= 60


def test_make_point_id_determinism():
    doc_id = "doc_d47eb83a5e78b8f2cae0a836602213fe"

    id0_path_a = store.make_point_id(doc_id, 0)
    id1_path_a = store.make_point_id(doc_id, 1)

    # Point IDs must be non-negative 63-bit integers
    assert isinstance(id0_path_a, int)
    assert id0_path_a >= 0
    assert id0_path_a < 2**63

    # Different chunks must have different IDs
    assert id0_path_a != id1_path_a

    # Moving a file keeps document_id identical, so point ID must be invariant to path!
    id0_path_b = store.make_point_id(doc_id, 0)
    assert id0_path_a == id0_path_b


def test_store_upsert_and_payload_updates(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(store, "qdrant", lambda: mock_client)

    chunks = [{"text": "Sample text", "page": 1}]
    vectors = [[0.1] * 1536]
    metadata = {
        "title": "Clean Code",
        "authors": ["Robert C. Martin"],
        "year": 2008,
        "subjects": ["programming"],
        "identifier": {"isbn": "9780132350884"},
    }

    point_ids = store.upsert_chunks(
        doc_id="doc_123",
        content_hash="123",
        file_rel="Books/Martin_2008_Clean-Code/book.pdf",
        scope="catalogued",
        doc_type="book",
        metadata=metadata,
        chunks=chunks,
        vectors=vectors,
    )

    assert len(point_ids) == 1
    assert mock_client.upsert.called
    upserted_points = mock_client.upsert.call_args[1]["points"]
    assert len(upserted_points) == 1
    p = upserted_points[0]
    assert p.id == point_ids[0]
    assert p.payload["scope"] == "catalogued"
    assert p.payload["title"] == "Clean Code"
    assert p.payload["authors"] == ["Robert C. Martin"]

    # Test update_payloads
    store.update_payloads(
        point_ids=point_ids,
        payload={"scope": "catalogued", "title": "Updated Title"},
    )
    assert mock_client.set_payload.called

    # Test delete_points
    store.delete_points(point_ids=point_ids)
    assert mock_client.delete.called


def test_store_upsert_epub_chunk_preserves_section_anchor(monkeypatch):
    """Task 2.2: store.upsert_chunks persists EPUB section anchor in the existing page payload field."""
    mock_client = MagicMock()
    monkeypatch.setattr(store, "qdrant", lambda: mock_client)

    chunks = [
        {"text": "Section 7 content from EPUB document", "page": 7},
    ]
    vectors = [[0.05] * 1536]
    metadata = {"title": "EPUB Guide", "authors": ["Guide Author"]}

    point_ids = store.upsert_chunks(
        doc_id="doc_epub_456",
        content_hash="hash456",
        file_rel="Books/Guide/book.epub",
        scope="catalogued",
        doc_type="book",
        metadata=metadata,
        chunks=chunks,
        vectors=vectors,
    )

    assert len(point_ids) == 1
    upserted_points = mock_client.upsert.call_args[1]["points"]
    assert len(upserted_points) == 1
    p = upserted_points[0]
    assert p.payload["page"] == 7
    assert p.payload["doc_name"] == "book.epub"


def test_collection_name_defaults_to_library_and_honors_env_override(monkeypatch):
    """LIBRARY_QUERY_COLLECTION selects the vector collection (disposable smokes)."""
    import importlib

    try:
        monkeypatch.delenv("LIBRARY_QUERY_COLLECTION", raising=False)
        importlib.reload(config)
        assert config.COLLECTION == "library"

        monkeypatch.setenv("LIBRARY_QUERY_COLLECTION", "research-smoke-test")
        importlib.reload(config)
        assert config.COLLECTION == "research-smoke-test"
    finally:
        monkeypatch.delenv("LIBRARY_QUERY_COLLECTION", raising=False)
        importlib.reload(config)
    assert config.COLLECTION == "library"


def _upsert_fixture(chunks):
    return dict(doc_id="doc_batch", content_hash="batch", file_rel="Books/Example/source.txt",
                scope="catalogued", doc_type="book", metadata={}, chunks=chunks,
                vectors=[[0.123456789] * 1536 for _ in chunks])


def test_large_upsert_batches_preserve_every_chunk_and_wire_limit(monkeypatch):
    client = MagicMock()
    client.upsert.return_value.status = "completed"
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(config, "QDRANT_MAX_BATCH_BYTES", 50000)
    chunks = [{"text": "Unicode source: 漢字 café 🧠 " * 20, "page": i + 1} for i in range(7)]
    ids = store.upsert_chunks(**_upsert_fixture(chunks))
    assert client.upsert.call_count > 1
    points = []
    for call in client.upsert.call_args_list:
        batch = call.kwargs["points"]
        assert call.kwargs["wait"] is True
        assert len(PointsList(points=batch).model_dump_json().encode("utf-8")) <= 50000
        points.extend(batch)
    assert [p.id for p in points] == ids == [store.make_point_id("doc_batch", i) for i in range(7)]
    assert [p.payload["chunk_index"] for p in points] == list(range(7))
    assert [p.payload["text"] for p in points] == [c["text"] for c in chunks]


def test_batch_boundary_fits_and_next_point_starts_new_batch():
    points = [PointStruct(id=i, vector=[0.5], payload={"text": "sample"}) for i in range(3)]
    ceiling = len('{"points":[]}') + 1 + sum(store._point_wire_size(p) for p in points[:2])
    batches = store._batches_by_wire_size(points, ceiling, "source")
    assert [[p.id for p in b] for b in batches] == [[0, 1], [2]]
    assert all(len(json.dumps({"points": [p.model_dump() for p in b]}).encode()) <= ceiling for b in batches)


def test_oversize_point_fails_before_any_write(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(config, "QDRANT_MAX_BATCH_BYTES", 50000)
    chunks = [{"text": "small", "page": 1}, {"text": "x" * 100000, "page": 2}]
    with pytest.raises(errors.QdrantBatchTooLargeError) as failure:
        store.upsert_chunks(**_upsert_fixture(chunks))
    assert errors.classify(failure.value)[0::2] == ("qdrant_batch_too_large", 3)
    assert "Books/Example/source.txt" in str(failure.value)
    client.upsert.assert_not_called()


def test_partial_batch_failure_retry_overwrites_same_point_ids(monkeypatch):
    persisted = {}
    writes = []
    fail = True

    def upsert(**kwargs):
        nonlocal fail
        writes.append([p.id for p in kwargs["points"]])
        if fail and len(writes) == 2:
            fail = False
            raise RuntimeError("interrupted batch")
        for point in kwargs["points"]:
            persisted[point.id] = point
        result = MagicMock()
        result.status = "completed"
        return result

    client = MagicMock()
    client.upsert.side_effect = upsert
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(config, "QDRANT_MAX_BATCH_BYTES", 30000)
    args = _upsert_fixture([{"text": str(i), "page": 1} for i in range(4)])
    with pytest.raises(RuntimeError, match="interrupted batch"):
        store.upsert_chunks(**args)
    assert 0 < len(persisted) < 4
    first_batch = writes[0]
    ids = store.upsert_chunks(**args)
    assert writes[2] == first_batch
    assert set(persisted) == set(ids)
    assert len(persisted) == 4


def test_default_batch_ceiling_has_margin_below_server_limit(monkeypatch):
    import importlib
    monkeypatch.delenv("LIBRARY_QUERY_QDRANT_MAX_BATCH_BYTES", raising=False)
    importlib.reload(config)
    assert config.QDRANT_MAX_BATCH_BYTES == 16 * 1024 * 1024
