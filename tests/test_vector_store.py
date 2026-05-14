from __future__ import annotations

from models import Chunk
from rag.embedding_core import FakeEmbeddingModel
from vectorstore.fake_store import FakeVectorStore


def _chunk(chunk_id: str, text: str, file_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"chunk_id": chunk_id, "file_id": file_id, "file_name": f"{file_id}.pdf"},
    )


def test_add_and_search_similar_chunks() -> None:
    emb = FakeEmbeddingModel()
    store = FakeVectorStore(embedding_model=emb)
    chunks = [
        _chunk("c1", "multa por atraso", "f1"),
        _chunk("c2", "pagamento do contrato", "f1"),
        _chunk("c3", "ferias coletivas", "f2"),
    ]
    store.add(chunks)

    query = emb.embed("qual a multa")
    result = store.search(query_embedding=query, top_k=2)

    assert len(result) == 2
    assert result[0].chunk_id == "c1"


def test_filter_by_file_id() -> None:
    emb = FakeEmbeddingModel()
    store = FakeVectorStore(embedding_model=emb)
    store.add(
        [
            _chunk("c1", "multa por atraso", "f1"),
            _chunk("c2", "multa e contrato", "f2"),
        ]
    )
    query = emb.embed("multa")
    result = store.search(query_embedding=query, top_k=5, filters={"file_id": "f2"})
    assert [chunk.chunk_id for chunk in result] == ["c2"]
