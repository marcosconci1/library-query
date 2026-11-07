"""Opt-in real Qdrant proof-loss/recovery test; never touches the live collection."""
import os
import uuid

import pytest
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from library_query import cli, config, readiness, revisions, store


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("LIBRARY_QUERY_LIVE_READINESS") != "1", reason="explicit live readiness test opt-in required")
def test_real_qdrant_missing_marker_recovers_after_full_audit(tmp_path, monkeypatch, capsys):
    collection = "readiness-test-" + uuid.uuid4().hex
    monkeypatch.setattr(config, "COLLECTION", collection)
    monkeypatch.setattr(config, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "state" / "ledger.json")
    client = store.qdrant()
    try:
        client.create_collection(collection, vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE))
        for key in readiness.REQUIRED_INDEXES:
            client.create_payload_index(collection, key, PayloadSchemaType.KEYWORD, wait=True)
        # Clearly synthetic fixture, stored only in the disposable test collection.
        client.upsert(collection, points=[PointStruct(id=1, vector=[1.0] + [0.0] * (config.EMBEDDING_DIM - 1),
            payload={"document_id": "test-document", "chunk_index": 0, "text": "Synthetic readiness test passage."})], wait=True)
        ctx = revisions.managed_context(collection)
        with revisions.writer_session(ctx) as writer:
            readiness.invalidate(writer, recreated=True)
            original = readiness.publish(writer)
        assert cli.main(["readiness", "--json"]) == 0
        capsys.readouterr()
        assert client.update_collection(collection, metadata={readiness.METADATA_KEY: None}) is True
        assert readiness.remote_validation(readiness.collection_info(collection)) is None
        assert cli.main(["readiness", "--json"]) == 3
        assert "remote_validation_missing" in capsys.readouterr().err
        assert cli.main(["readiness", "--repair", "--json"]) == 0
        capsys.readouterr()
        restored = revisions.read_record(ctx)
        assert isinstance(restored, dict)
        assert restored["revision"] == original["revision"] + 1
        assert restored["research_validation"] == readiness.remote_validation(readiness.collection_info(collection))
        assert client.get_collection(collection).points_count == 1
        assert cli.main(["readiness", "--json"]) == 0
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        assert not client.collection_exists(collection)
