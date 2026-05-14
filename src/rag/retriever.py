from __future__ import annotations

from models import Chunk
from rag.embedding_core import EmbeddingModel
from vectorstore.fake_store import FakeVectorStore


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: FakeVectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int, filters: dict | None = None) -> list[Chunk]:
        query_embedding = self.embedding_model.embed(question)
        return self.vector_store.search(query_embedding=query_embedding, top_k=top_k, filters=filters)
