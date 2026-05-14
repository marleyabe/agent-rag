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
