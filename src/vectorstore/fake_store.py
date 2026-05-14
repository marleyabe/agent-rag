from __future__ import annotations

from models import Chunk
from rag.embedding_core import EmbeddingModel


class FakeVectorStore:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._items.append((chunk, self.embedding_model.embed(chunk.text)))

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[Chunk]:
        filtered: list[tuple[Chunk, list[float]]] = []
        for item in self._items:
            chunk, _ = item
            if not filters:
                filtered.append(item)
                continue
            if "file_id" in filters and chunk.metadata.get("file_id") != filters["file_id"]:
                continue
            filtered.append(item)

        ranked = sorted(
            filtered,
            key=lambda item: self._dot(query_embedding, item[1]),
            reverse=True,
        )
        return [chunk for chunk, _ in ranked[:top_k]]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
