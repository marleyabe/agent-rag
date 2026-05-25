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


def test_add_uses_embed_many_for_indexing() -> None:
    class BatchOnlyEmbedding(FakeEmbeddingModel):
        def __init__(self) -> None:
            self.embed_calls = 0
            self.embed_many_calls = 0

        def embed(self, text: str) -> list[float]:
            self.embed_calls += 1
            return super().embed(text)

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            self.embed_many_calls += 1
            return [FakeEmbeddingModel.embed(self, text) for text in texts]

    emb = BatchOnlyEmbedding()
    store = FakeVectorStore(embedding_model=emb)
    store.add([_chunk("c1", "multa", "f1"), _chunk("c2", "contrato", "f1")])
    assert emb.embed_many_calls == 1
    assert emb.embed_calls == 0


def test_fetch_filters_and_limits_chunks() -> None:
    emb = FakeEmbeddingModel()
    store = FakeVectorStore(embedding_model=emb)
    store.add(
        [
            _chunk("c1", "multa", "f1"),
            _chunk("c2", "contrato", "f2"),
            _chunk("c3", "prazo", "f1"),
        ]
    )

    assert [chunk.chunk_id for chunk in store.fetch(filters={"file_id": "f1"}, limit=1)] == ["c1"]
