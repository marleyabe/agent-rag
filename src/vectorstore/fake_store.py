from __future__ import annotations

from models import Chunk
from rag.embedding_core import EmbeddingModel


class FakeVectorStore:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeds = self.embedding_model.embed_many([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeds):
            self._items.append((chunk, embedding))

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
        out: list[Chunk] = []
        for chunk, emb in ranked[:top_k]:
            metadata = dict(chunk.metadata or {})
            metadata["retrieval_score"] = self._dot(query_embedding, emb)
            out.append(Chunk(chunk_id=chunk.chunk_id, text=chunk.text, metadata=metadata))
        return out

    def fetch(self, filters: dict | None = None, limit: int | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        for chunk, _ in self._items:
            if filters and "file_id" in filters and chunk.metadata.get("file_id") != filters["file_id"]:
                continue
            chunks.append(chunk)
            if limit is not None and len(chunks) >= limit:
                break
        return chunks

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
