from __future__ import annotations

from models import Chunk
from rag.embedding_core import FakeEmbeddingModel
from rag.retriever import Retriever
from vectorstore.fake_store import FakeVectorStore


def test_retriever_embeds_query_and_preserves_chunk_metadata() -> None:
    emb = FakeEmbeddingModel()
    store = FakeVectorStore(embedding_model=emb)
    store.add(
        [
            Chunk(
                chunk_id="c1",
                text="A multa por atraso e de 2%",
                metadata={"file_id": "doc1", "file_name": "a.pdf", "chunk_id": "c1"},
            ),
            Chunk(
                chunk_id="c2",
                text="Tema sem relacao",
                metadata={"file_id": "doc2", "file_name": "b.pdf", "chunk_id": "c2"},
            ),
        ]
    )

    retriever = Retriever(embedding_model=emb, vector_store=store)
    chunks = retriever.retrieve("qual a multa", top_k=1)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].metadata["file_name"] == "a.pdf"


def test_retriever_reranks_by_hybrid_score() -> None:
    emb = FakeEmbeddingModel()
    store = FakeVectorStore(embedding_model=emb)
    store._items = [  # type: ignore[attr-defined]
        (
            Chunk(
                chunk_id="c_high_vec_low_lex",
                text="topico generico sem termos da pergunta",
                metadata={"file_id": "doc1", "chunk_id": "c_high_vec_low_lex"},
            ),
            [1.0, 1.0, 1.0, 1.0],
        ),
        (
            Chunk(
                chunk_id="c_good_lex",
                text="a multa por atraso no pagamento e 2 por cento",
                metadata={"file_id": "doc1", "chunk_id": "c_good_lex"},
            ),
            [0.9, 0.9, 0.9, 0.9],
        ),
    ]
    retriever = Retriever(embedding_model=emb, vector_store=store)
    chunks = retriever.retrieve("qual a multa por atraso", top_k=2)
    assert chunks[0].chunk_id == "c_good_lex"


def test_retriever_merges_lexical_matches_outside_vector_top_k() -> None:
    emb = FakeEmbeddingModel()
    store = FakeVectorStore(embedding_model=emb)
    store.add(
        [
            Chunk(
                chunk_id="c1",
                text="texto inicial sem relacao",
                metadata={"file_id": "doc1", "chunk_id": "c1"},
            ),
            Chunk(
                chunk_id="c2",
                text="Dos Direitos e Garantias Fundamentais",
                metadata={"file_id": "doc1", "chunk_id": "c2"},
            ),
        ]
    )

    retriever = Retriever(embedding_model=emb, vector_store=store)
    chunks = retriever.retrieve(
        "Quais sao os direitos fundamentais?",
        top_k=1,
        filters={"file_id": "doc1"},
    )

    assert [chunk.chunk_id for chunk in chunks] == ["c2"]


def test_retriever_full_lexical_scan_is_opt_in() -> None:
    class CountingStore(FakeVectorStore):
        def __init__(self, embedding_model: FakeEmbeddingModel) -> None:
            super().__init__(embedding_model)
            self.fetch_calls = 0

        def fetch(self, filters: dict | None = None, limit: int | None = None) -> list[Chunk]:
            self.fetch_calls += 1
            return super().fetch(filters=filters, limit=limit)

    emb = FakeEmbeddingModel()
    store = CountingStore(embedding_model=emb)
    store.add(
        [
            Chunk(
                chunk_id="c1",
                text="A multa por atraso e de 2%",
                metadata={"file_id": "doc1", "chunk_id": "c1"},
            )
        ]
    )

    Retriever(embedding_model=emb, vector_store=store).retrieve("qual a multa", top_k=1)
    assert store.fetch_calls == 0

    Retriever(embedding_model=emb, vector_store=store, full_lexical_scan=True).retrieve(
        "qual a multa", top_k=1
    )
    assert store.fetch_calls == 1
